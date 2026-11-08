"""D/ST boom odds: the chance a defense posts a big game (BOOM+ points), and why.

Big D/ST games are rare and an average projection always under-calls them, so
this models the probability directly. Every feature is what was known before
kickoff: each team's rates from earlier weeks of that season, blended with the
prior season (counted as dst.PRIOR_GAMES games), the betting line, and whether
the opposing QB is a backup/new starter. A logistic regression is fitted on
every team-game since TRAIN_FROM and checked season by season on games it did
not see (backtest()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import dst, fetch

BOOM = 15                  # D/ST points that count as a boom game
SMASH = 20                 # ... and as a smash game
TRAIN_FROM = 2024          # first season with a prior season on disk to blend with
NEW_QB_STARTS = 3          # opposing QB with fewer starts for that team (this season + last) = new/backup
L2 = 1.0                   # ridge penalty on standardized coefficients

FEATURES = {
    "opp_implied": "Opponent's implied points (betting line)",
    "opp_giveaways_pg": "Opponent giveaways per game",
    "opp_sack_rate": "Opponent sack rate allowed",
    "opp_epa_pp": "Opponent offensive EPA/play",
    "opp_new_qb": "Opponent starting a backup/new QB",
    "def_takeaways_pg": "Defense takeaways per game",
    "def_sack_rate": "Defense sack rate",
    "def_epa_pp": "Defense EPA/play allowed",
    "home": "Defense at home",
}


def _rates(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    """Per-team totals for one side ('offense' or 'defense') of team-game rows."""
    a = rows.groupby(side).agg(games=("game_id", "nunique"), takeaways=("takeaways", "sum"),
                               sacks=("sacks", "sum"), dropbacks=("dropbacks", "sum"),
                               epa=("epa", "sum"), plays=("plays", "sum"))
    return pd.DataFrame({"games": a["games"], "to_pg": a["takeaways"] / a["games"],
                         "sack_rate": a["sacks"] / a["dropbacks"], "epa_pp": a["epa"] / a["plays"]})


def _blend(cur: pd.DataFrame, prior: pd.DataFrame, team: str, col: str) -> float:
    pri = prior[col].get(team, np.nan)
    if team not in cur.index:
        return pri
    g, c = cur.at[team, "games"], cur.at[team, col]
    return c if np.isnan(pri) else (g * c + dst.PRIOR_GAMES * pri) / (g + dst.PRIOR_GAMES)


def _qb_starts(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: season, week, team, starting QB id."""
    s = sched[sched["game_type"] == "REG"]
    return pd.concat([
        s[["season", "week", "home_team", "home_qb_id"]].set_axis(["season", "week", "team", "qb"], axis=1),
        s[["season", "week", "away_team", "away_qb_id"]].set_axis(["season", "week", "team", "qb"], axis=1),
    ])


def features(tg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Pre-kickoff features for each (defense, offense) row of `games`
    (columns season, week, game_id, defense, offense, home), using team-game
    history `tg` strictly before that week."""
    sched = fetch.load_schedule()
    sched = sched[sched["game_type"] == "REG"].set_index("game_id")
    starts = _qb_starts(sched.reset_index())
    out = []
    for (season, week), grp in games.groupby(["season", "week"]):
        cur = tg[(tg["season"] == season) & (tg["week"] < week)]
        pri = tg[tg["season"] == season - 1]
        co, cd = _rates(cur, "offense"), _rates(cur, "defense")
        po, pd_ = _rates(pri, "offense"), _rates(pri, "defense")
        st = starts[((starts["season"] == season) & (starts["week"] < week)) | (starts["season"] == season - 1)]
        for r in grp.itertuples():
            g = sched.loc[r.game_id]
            o_home = g["home_team"] == r.offense
            spread, total = g["spread_line"], g["total_line"]
            implied = total / 2 + (spread / 2 if o_home else -spread / 2)
            qb = g["home_qb_id"] if o_home else g["away_qb_id"]
            prior_starts = int(((st["team"] == r.offense) & (st["qb"] == qb)).sum()) if pd.notna(qb) else np.nan
            out.append({
                "season": season, "week": week, "game_id": r.game_id, "defense": r.defense, "offense": r.offense,
                "home": float(r.home), "opp_implied": implied,
                "opp_giveaways_pg": _blend(co, po, r.offense, "to_pg"),
                "opp_sack_rate": _blend(co, po, r.offense, "sack_rate"),
                "opp_epa_pp": _blend(co, po, r.offense, "epa_pp"),
                "opp_new_qb": float(prior_starts < NEW_QB_STARTS) if pd.notna(prior_starts) else np.nan,
                "opp_qb": (g["home_qb_name"] if o_home else g["away_qb_name"]), "opp_qb_starts": prior_starts,
                "def_takeaways_pg": _blend(cd, pd_, r.defense, "to_pg"),
                "def_sack_rate": _blend(cd, pd_, r.defense, "sack_rate"),
                "def_epa_pp": _blend(cd, pd_, r.defense, "epa_pp"),
            })
    return pd.DataFrame(out)


def played_games(tg: pd.DataFrame) -> pd.DataFrame:
    sched = fetch.load_schedule().set_index("game_id")
    g = tg[tg["season"] >= TRAIN_FROM][["season", "week", "game_id", "defense", "offense", "dst_fp"]].copy()
    g["home"] = [sched.at[gid, "home_team"] == d for gid, d in zip(g["game_id"], g["defense"])]
    return g


# ---- logistic regression (ridge, Newton's method) on standardized features ----

class Logit:
    def __init__(self, cols: list[str], l2: float = L2):
        self.cols, self.l2 = cols, l2

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Logit":
        self.mu, self.sd = X[self.cols].mean(), X[self.cols].std().replace(0, 1)
        Z = self._z(X)
        w = np.zeros(Z.shape[1])
        pen = np.full(Z.shape[1], self.l2); pen[0] = 0        # no penalty on the intercept
        for _ in range(50):
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y) + pen * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen)
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        Z = ((X[self.cols] - self.mu) / self.sd).fillna(0).to_numpy()
        return np.column_stack([np.ones(len(Z)), Z])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self._z(X) @ self.w))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Each feature's push on the log-odds vs. an average matchup."""
        return pd.DataFrame(self._z(X)[:, 1:] * self.w[1:], columns=self.cols, index=X.index)

    def coefs(self) -> dict:
        return dict(zip(self.cols, np.round(self.w[1:], 3)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def backtest(data: pd.DataFrame, cols: list[str], threshold: int = BOOM) -> pd.DataFrame:
    """Hold out each season in turn: fit on the others, score the held-out games."""
    rows = []
    y_all = (data["dst_fp"] >= threshold).astype(float).to_numpy()
    for s in sorted(data["season"].unique()):
        tr, te = data["season"] != s, data["season"] == s
        m = Logit(cols).fit(data[tr], y_all[tr.to_numpy()])
        p = m.predict(data[te]); y = y_all[te.to_numpy()]
        top = p >= np.quantile(p, 0.9)
        rows.append({"season": s, "games": int(te.sum()), "base_rate": y.mean(), "auc": auc(y, p),
                     "top10_hit_rate": y[top].mean(), "bottom50_hit_rate": y[p <= np.median(p)].mean()})
    return pd.DataFrame(rows)


# ---- what the page uses ----

MODEL_COLS = ["opp_implied"]    # backtest(): team stats add nothing out of sample once the line is known
FLAG_GIVEAWAYS = 1.5            # opponent giveaways per game worth flagging ...
FLAG_SACK_RATE = 0.08           # ... and sack rate allowed


def flags(r) -> list[str]:
    out = []
    if r["opp_new_qb"] == 1:
        out.append(f"{r['opp_qb']} ({int(r['opp_qb_starts'])} starts)")
    if r["opp_giveaways_pg"] >= FLAG_GIVEAWAYS:
        out.append(f"Opp {r['opp_giveaways_pg']:.1f} giveaways/G")
    if r["opp_sack_rate"] >= FLAG_SACK_RATE:
        out.append(f"Opp sacked {r['opp_sack_rate'] * 100:.0f}%")
    return out


def boom(tg: pd.DataFrame, upcoming: pd.DataFrame) -> dict:
    """Boom (BOOM+) and smash (SMASH+) odds for `upcoming` (season, week, game_id,
    defense, offense, home), plus the backtest and history the page explains them with."""
    played = played_games(tg)
    X = features(tg, played).merge(played[["game_id", "defense", "dst_fp"]], on=["game_id", "defense"])
    X = X[X["opp_implied"].notna()]
    y15, y20 = (X["dst_fp"] >= BOOM).astype(float).to_numpy(), (X["dst_fp"] >= SMASH).astype(float).to_numpy()
    m15, m20 = Logit(MODEL_COLS).fit(X, y15), Logit(MODEL_COLS).fit(X, y20)

    U = features(tg, upcoming)
    has_line = U["opp_implied"].notna()
    U["boom_pct"] = np.where(has_line, m15.predict(U), np.nan)
    U["smash_pct"] = np.where(has_line, m20.predict(U), np.nan)
    U["flags"] = [flags(r) for _, r in U.iterrows()]

    bt = backtest(X, MODEL_COLS, BOOM)
    w = bt["games"]
    buckets = pd.cut(X["opp_implied"], [0, 17, 19.5, 22, 25, 99], labels=["17 or less", "17.5–19.5", "20–22", "22.5–25", "25.5+"])
    by_bucket = X.groupby(buckets, observed=True).agg(games=("dst_fp", "size"), avg=("dst_fp", "mean"),
                                                      boom=("dst_fp", lambda s: (s >= BOOM).mean()),
                                                      smash=("dst_fp", lambda s: (s >= SMASH).mean()))
    hist = X[X["dst_fp"] >= SMASH].sort_values(["season", "week"], ascending=False).copy()
    hist["implied_pct"] = [(X["opp_implied"] <= v).mean() for v in hist["opp_implied"]]
    hist["flags"] = [flags(r) for _, r in hist.iterrows()]
    return {
        "upcoming": U[["defense", "offense", "opp_implied", "boom_pct", "smash_pct", "flags",
                       "opp_qb", "opp_new_qb", "opp_giveaways_pg", "opp_sack_rate"]],
        "summary": {
            "boom": BOOM, "smash": SMASH, "games": int(len(X)), "seasons": sorted(int(s) for s in X["season"].unique()),
            "base_boom": float(y15.mean()), "base_smash": float(y20.mean()),
            "auc": float(np.average(bt["auc"], weights=w)),
            "top10_hit": float(np.average(bt["top10_hit_rate"], weights=w)),
            "bottom50_hit": float(np.average(bt["bottom50_hit_rate"], weights=w)),
            "smash_low_implied_share": float((X.loc[y20 == 1, "opp_implied"] <= 19.5).mean()),
            "low_implied_share": float((X["opp_implied"] <= 19.5).mean()),
        },
        "buckets": [{"implied": str(k), **{c: round(float(v), 3) for c, v in r.items()}} for k, r in by_bucket.iterrows()],
        "history": [{"season": int(r.season), "week": int(r.week), "defense": r.defense, "offense": r.offense,
                     "dst_fp": float(r.dst_fp), "opp_implied": float(r.opp_implied), "implied_pct": round(float(r.implied_pct), 3),
                     "opp_qb": r.opp_qb, "flags": r.flags} for r in hist.itertuples()],
    }
