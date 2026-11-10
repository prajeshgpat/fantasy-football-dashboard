"""Defense-style vs. offense-style D/ST matchups over 20 seasons.

Ranking sites usually rate a D/ST matchup by how many points defenses have
scored against that offense so far. That number is skewed by schedule: an
offense that opened against four strong defenses looks generous, one that
opened against four weak ones looks stingy. This module compares like with
like instead:

1. Opponent-adjusted stats. For every team, each per-play rate (EPA, success,
   sacks, pressure, takeaways, explosive plays, run stuffs, 3rd downs) is
   measured against what its opponents normally allow/produce.
2. Standardized within season, so a 2008 unit is compared with 2008, not
   with today's passing environment.
3. Clusters known before kickoff. Each team gets a defense style and an
   offense style from k-means centroids fitted on full team-seasons; every
   game uses the team's profile from earlier games only (this season, blended
   with last season counted as PRIOR_GAMES games).

Then every D/ST game since FIRST_SEASON lands in a cell of the style matrix,
and backtest() checks, season by season with everything fitted on earlier
seasons only, whether the cell predicts D/ST points better than the
ranking-site number, a schedule-adjusted version of it, and the betting line.

The 20-season pass is heavy, so `python -m src.cluster_matchups` runs it once
and writes OUTPUT (committed); the weekly build only assigns this week's
styles with assign_current().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import def_clusters, dst, fetch

FIRST_SEASON = 2006                # first season scored (2005 is its prior)
K = 4
PRIOR_GAMES = 4
Z_CAP = 2.0
BOOM = 15
MIN_TRAIN_SEASONS = 6
OUTPUT = Path(__file__).resolve().parents[1] / "static" / "cluster_matchups.json"

RATES = ["pass_epa_pp", "rush_epa_pp", "success_rate", "first_down_rate", "sack_rate", "hit_rate",
         "takeaway_rate", "exp_pass_rate", "exp_run_rate", "stuff_rate", "third_rate", "pass_rate"]
UNADJUSTED = {"pass_rate"}         # a tendency, not a result: no opponent adjustment
# (label, higher is better for that unit?)
DEF_FEATURES = {
    "pass_epa_pp": ("Dropback EPA allowed", False), "rush_epa_pp": ("Rush EPA allowed", False),
    "success_rate": ("Success rate allowed", False), "sack_rate": ("Sack rate", True),
    "hit_rate": ("Pressure (sack/QB hit) rate", True), "takeaway_rate": ("Takeaways per 100 plays", True),
    "exp_pass_rate": ("20+ yd passes allowed", False), "exp_run_rate": ("10+ yd runs allowed", False),
    "stuff_rate": ("Run stuff rate", True), "third_rate": ("3rd-down conv. allowed", False),
}
OFF_FEATURES = {
    "pass_epa_pp": ("Dropback EPA", True), "rush_epa_pp": ("Rush EPA", True),
    "success_rate": ("Success rate", True), "pass_rate": ("Neutral pass rate", None),
    "sack_rate": ("Sack rate taken", False), "takeaway_rate": ("Giveaways per 100 plays", False),
    "exp_pass_rate": ("20+ yd pass rate", True), "exp_run_rate": ("10+ yd run rate", True),
    "third_rate": ("3rd-down conv.", True),
}


def _alias(s: pd.Series) -> pd.Series:
    return s.replace(fetch.TEAM_ALIASES)


# ---------------------------------------------------------------- game rows

def game_stats(year: int) -> pd.DataFrame:
    """One row per game per (offense, defense): play counts and sums for RATES, plus D/ST points."""
    p = fetch.load_pbp(year)
    p = p[p["play_type"].isin(["pass", "run"]) & p["posteam"].notna() & p["epa"].notna()].copy()
    db, ru = p["pass"] == 1, p["rush"] == 1
    neutral = (p["qtr"] <= 4) & p["wp"].between(0.20, 0.80)
    third = (p["third_down_converted"] == 1) | (p["third_down_failed"] == 1)
    yd = p["yards_gained"].fillna(0)
    p = p.assign(
        n=1, db=db, ru=ru, pass_epa=p["epa"].where(db, 0), rush_epa=p["epa"].where(ru, 0),
        succ=p["success"].fillna(0), fd=p["first_down"].fillna(0), sk=p["sack"].fillna(0).where(db, 0),
        hit=((p["qb_hit"].fillna(0) + p["sack"].fillna(0)) > 0) & db,
        to=p["interception"].fillna(0) + p["fumble_lost"].fillna(0),
        xp=db & (yd >= 20), xr=ru & (yd >= 10), stuff=ru & (yd <= 0),
        t_att=third, t_conv=third & (p["third_down_converted"] == 1),
        neu=neutral, neu_db=neutral & db,
    )
    cols = ["n", "db", "ru", "pass_epa", "rush_epa", "succ", "fd", "sk", "hit", "to", "xp", "xr", "stuff",
            "t_att", "t_conv", "neu", "neu_db"]
    g = p.groupby(["game_id", "season", "week", "posteam", "defteam"])[cols].sum().reset_index()
    g = g.rename(columns={"posteam": "offense", "defteam": "defense"})
    fp = dst.team_games(year)[["game_id", "defense", "dst_fp"]]
    g = g.merge(fp, on=["game_id", "defense"], how="left")
    sched = fetch.load_schedule().set_index("game_id")
    s = sched.reindex(g["game_id"])
    half = s["spread_line"].to_numpy() / 2
    g["implied"] = s["total_line"].to_numpy() / 2 + np.where(s["home_team"].to_numpy() == g["offense"].to_numpy(), half, -half)
    g["offense"], g["defense"] = _alias(g["offense"]), _alias(g["defense"])
    return g


def _rates(sums: pd.DataFrame) -> pd.DataFrame:
    s = sums
    return pd.DataFrame({
        "pass_epa_pp": s["pass_epa"] / s["db"], "rush_epa_pp": s["rush_epa"] / s["ru"],
        "success_rate": s["succ"] / s["n"], "first_down_rate": s["fd"] / s["n"],
        "sack_rate": s["sk"] / s["db"], "hit_rate": s["hit"] / s["db"],
        "takeaway_rate": s["to"] / s["n"] * 100, "exp_pass_rate": s["xp"] / s["db"],
        "exp_run_rate": s["xr"] / s["ru"], "stuff_rate": s["stuff"] / s["ru"],
        "third_rate": s["t_conv"] / s["t_att"], "pass_rate": s["neu_db"] / s["neu"],
    }, index=s.index)


SUM_COLS = ["n", "db", "ru", "pass_epa", "rush_epa", "succ", "fd", "sk", "hit", "to", "xp", "xr", "stuff",
            "t_att", "t_conv", "neu", "neu_db"]


def adjusted_profiles(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Opponent-adjusted RATES for each team as offense and as defense, from game `rows`.
    Offense T's rate minus the average (over its games) of how far each opposing defense's
    allowed rate sits from league average, and vice versa. Also returns games and the
    per-game D/ST points scored (defense) / allowed (offense)."""
    O_sum, D_sum = rows.groupby("offense")[SUM_COLS].sum(), rows.groupby("defense")[SUM_COLS].sum()
    O, D = _rates(O_sum), _rates(D_sum)
    lg = _rates(rows[SUM_COLS].sum().to_frame().T).iloc[0]
    dev_D, dev_O = D - lg, O - lg
    adj_cols = [c for c in RATES if c not in UNADJUSTED]
    o_opp = rows[["offense", "defense"]].join(dev_D[adj_cols], on="defense").groupby("offense")[adj_cols].mean()
    d_opp = rows[["offense", "defense"]].join(dev_O[adj_cols], on="offense").groupby("defense")[adj_cols].mean()
    O_adj, D_adj = O.copy(), D.copy()
    O_adj[adj_cols] = O[adj_cols] - o_opp
    D_adj[adj_cols] = D[adj_cols] - d_opp
    O_adj["games"] = rows.groupby("offense")["game_id"].nunique()
    D_adj["games"] = rows.groupby("defense")["game_id"].nunique()
    O_adj["dst_allowed_pg"] = rows.groupby("offense")["dst_fp"].mean()
    D_adj["dst_pg"] = rows.groupby("defense")["dst_fp"].mean()
    # Schedule-adjusted D/ST allowed: how far above the defenses' usual output this offense let them score.
    d_dst_dev = D_adj["dst_pg"] - rows["dst_fp"].mean()
    O_adj["dst_allowed_adj_pg"] = O_adj["dst_allowed_pg"] - rows.join(d_dst_dev.rename("dd"), on="defense").groupby("offense")["dd"].mean()
    return O_adj, D_adj


def _blend(cur: pd.DataFrame, prior: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    teams = cur.index.union(prior.index)
    c, p = cur.reindex(teams), prior.reindex(teams)
    g = c["games"].fillna(0)
    out = pd.DataFrame(index=teams)
    for col in cols:
        cv, pv = c[col], p[col]
        out[col] = np.where(pv.isna(), cv, np.where(cv.isna(), pv, (g * cv.fillna(0) + PRIOR_GAMES * pv) / (g + PRIOR_GAMES)))
    out["games"] = g
    return out


def _z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return ((df[cols] - df[cols].mean()) / df[cols].std(ddof=0)).clip(-Z_CAP, Z_CAP)


# --------------------------------------------------------------- clustering

def fit_centroids(seasons: dict[int, pd.DataFrame]) -> dict:
    """k-means centroids for defense and offense styles on full team-seasons (z within season)."""
    out = {}
    for side, feats in (("defense", list(DEF_FEATURES)), ("offense", list(OFF_FEATURES))):
        Zs = []
        for yr, rows in seasons.items():
            O, D = adjusted_profiles(rows)
            Zs.append(_z(D if side == "defense" else O, feats))
        Z = pd.concat(Zs).dropna()
        lab, C = def_clusters.kmeans(Z.to_numpy(), K)
        # Order clusters best to worst for the unit (defense: lowest EPA allowed; offense: highest EPA).
        quality = Z["pass_epa_pp"] * 0.6 + Z["rush_epa_pp"] * 0.4
        q = pd.Series(quality.to_numpy()).groupby(lab).mean()
        order = q.sort_values(ascending=(side == "defense")).index.tolist()
        C = C[order]
        out[side] = {"features": feats, "centroids": C.round(4).tolist(), "team_seasons": int(len(Z))}
    return out


def assign(z: pd.DataFrame, centroids: list[list[float]]) -> pd.Series:
    C = np.array(centroids)
    d = ((z.to_numpy()[:, None, :] - C[None]) ** 2).sum(2)
    return pd.Series(np.argmin(d, axis=1) + 1, index=z.index)


TRAIT = {
    "defense": {"pass_epa_pp": "stopping the pass", "rush_epa_pp": "stopping the run", "success_rate": "winning downs",
                "sack_rate": "sacks", "hit_rate": "pressure", "takeaway_rate": "takeaways",
                "exp_pass_rate": "limiting big passes", "exp_run_rate": "limiting big runs",
                "stuff_rate": "stuffing runs", "third_rate": "3rd downs"},
    "offense": {"pass_epa_pp": "passing", "rush_epa_pp": "running", "success_rate": "staying on schedule",
                "sack_rate": "avoiding sacks", "takeaway_rate": "ball security", "exp_pass_rate": "big passes",
                "exp_run_rate": "big runs", "third_rate": "3rd downs"},
}


def _cluster_name(side: str, centroid: np.ndarray, feats: list[str], rank: int) -> str:
    """'<tier> D/O: best at X, worst at Y' from the centroid's z-scores (signed so + = good for that unit)."""
    fmap = DEF_FEATURES if side == "defense" else OFF_FEATURES
    good = pd.Series({f: centroid[i] * (1 if fmap[f][1] else -1) for i, f in enumerate(feats) if fmap[f][1] is not None})
    tier = ["Elite", "Above average", "Below average", "Weak"][rank]
    name = f"{tier} {side[0].upper()}: best at {TRAIT[side][good.idxmax()]}, worst at {TRAIT[side][good.idxmin()]}"
    if "pass_rate" in feats:
        pr = centroid[feats.index("pass_rate")]
        name += " (pass-heavy)" if pr >= 0.5 else " (run-heavy)" if pr <= -0.5 else ""
    return name


# --------------------------------------------------------- pre-game table

def pregame_table(seasons: dict[int, pd.DataFrame], cents: dict) -> pd.DataFrame:
    """Every team-game since FIRST_SEASON with both teams' pre-kickoff styles and comparison numbers."""
    out = []
    years = sorted(seasons)
    full = {y: adjusted_profiles(seasons[y]) for y in years}
    dcols, ocols = cents["defense"]["features"], cents["offense"]["features"]
    for y in years:
        if y < FIRST_SEASON or (y - 1) not in seasons:
            continue
        rows = seasons[y]
        pO, pD = full[y - 1]
        for w in sorted(rows["week"].unique()):
            cur = rows[rows["week"] < w]
            if len(cur):
                cO, cD = adjusted_profiles(cur)
            else:
                cO = cD = pd.DataFrame(columns=RATES + ["games", "dst_pg", "dst_allowed_pg", "dst_allowed_adj_pg"])
            O = _blend(cO, pO, RATES + ["dst_allowed_pg", "dst_allowed_adj_pg"])
            D = _blend(cD, pD, RATES + ["dst_pg"])
            dz, oz = _z(D, dcols).fillna(0), _z(O, ocols).fillna(0)
            d_cl = assign(dz, cents["defense"]["centroids"])
            o_cl = assign(oz, cents["offense"]["centroids"])
            g = rows[rows["week"] == w]
            zcols = {f"dz_{c}": dz[c].reindex(g["defense"]).to_numpy() for c in dcols}
            zcols.update({f"oz_{c}": oz[c].reindex(g["offense"]).to_numpy() for c in ocols})
            lg_dst = D["dst_pg"].mean()
            out.append(pd.DataFrame({
                "season": y, "week": w, "game_id": g["game_id"].to_numpy(),
                "defense": g["defense"].to_numpy(), "offense": g["offense"].to_numpy(),
                "dst_fp": g["dst_fp"].to_numpy(), "implied": g["implied"].to_numpy(),
                "d_cluster": d_cl.reindex(g["defense"]).to_numpy(), "o_cluster": o_cl.reindex(g["offense"]).to_numpy(),
                "def_dst_pg": D["dst_pg"].reindex(g["defense"]).to_numpy(),
                "opp_allowed_pg": O["dst_allowed_pg"].reindex(g["offense"]).to_numpy(),
                "opp_allowed_adj_pg": O["dst_allowed_adj_pg"].reindex(g["offense"]).to_numpy(),
                "lg_dst": lg_dst, **zcols,
            }))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- backtest

def _auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def _matrix(tr: pd.DataFrame) -> pd.DataFrame:
    return tr.groupby(["d_cluster", "o_cluster"])["dst_fp"].mean().unstack()


def _additive(tr: pd.DataFrame) -> tuple[float, pd.Series, pd.Series]:
    """D/ST = mu + a[d] + b[o], fitted by alternating means (no interaction)."""
    mu = tr["dst_fp"].mean()
    a = pd.Series(0.0, index=range(1, K + 1)); b = pd.Series(0.0, index=range(1, K + 1))
    for _ in range(50):
        a = (tr["dst_fp"] - mu - tr["o_cluster"].map(b)).groupby(tr["d_cluster"]).mean().reindex(a.index, fill_value=0)
        b = (tr["dst_fp"] - mu - tr["d_cluster"].map(a)).groupby(tr["o_cluster"]).mean().reindex(b.index, fill_value=0)
    return mu, a, b


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def predictions(tr: pd.DataFrame, te: pd.DataFrame) -> dict[str, np.ndarray]:
    M = _matrix(tr)
    mu, a, b = _additive(tr)
    cell = np.array([M.at[d, o] for d, o in zip(te["d_cluster"], te["o_cluster"])])
    add = mu + te["d_cluster"].map(a).to_numpy() + te["o_cluster"].map(b).to_numpy()
    preds = {
        "Ranking-site: D/ST pts opponent allowed so far": te["opp_allowed_pg"].to_numpy(),
        "Same, schedule-adjusted": te["opp_allowed_adj_pg"].to_numpy(),
        "Defense scored + opponent allowed (results)": (te["def_dst_pg"] + te["opp_allowed_pg"] - te["lg_dst"]).to_numpy(),
        "Style clusters, no matchup effect (additive)": add,
        "Style matchup matrix": cell,
    }
    ok_tr = tr["implied"].notna()
    if ok_tr.sum() > 100 and te["implied"].notna().all():
        trc = np.array([M.at[d, o] for d, o in zip(tr["d_cluster"], tr["o_cluster"])])
        w = _ols(tr.loc[ok_tr, ["implied"]].to_numpy(), tr.loc[ok_tr, "dst_fp"].to_numpy())
        preds["Betting line"] = w[0] + w[1] * te["implied"].to_numpy()
        X = np.column_stack([tr["implied"], trc])[ok_tr.to_numpy()]
        w2 = _ols(X, tr.loc[ok_tr, "dst_fp"].to_numpy())
        preds["Betting line + style matrix"] = w2[0] + np.column_stack([te["implied"], cell]) @ w2[1:]
    return preds


def backtest(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seasons = sorted(T["season"].unique())
    for s in seasons[MIN_TRAIN_SEASONS:]:
        tr, te = T[T["season"] < s], T[T["season"] == s]
        te = te[te["implied"].notna()] if te["implied"].notna().mean() > 0.95 else te
        y = te["dst_fp"].to_numpy()
        for name, p in predictions(tr, te).items():
            p = np.asarray(p, dtype=float)
            ok = ~np.isnan(p)
            top = p[ok] >= np.quantile(p[ok], 0.9)
            rows.append({"season": int(s), "model": name, "games": int(ok.sum()),
                         "corr": float(np.corrcoef(p[ok], y[ok])[0, 1]),
                         "rmse": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))),
                         "boom_auc": _auc((y[ok] >= BOOM).astype(int), p[ok]),
                         "top10_boom": float((y[ok][top] >= BOOM).mean())})
    return pd.DataFrame(rows)


# The style matchups the theory is about, as continuous interactions of pre-game z-scores.
# (defense z, offense z, sign): sign makes "+" mean the defense's strength meets the offense's weakness.
STYLE_PAIRS = {
    "Pressure D vs sack-prone O": ("dz_hit_rate", "oz_sack_rate", 1),
    "Ball-hawking D vs turnover-prone O": ("dz_takeaway_rate", "oz_takeaway_rate", 1),
    "Strong pass D vs pass-heavy O": ("dz_pass_epa_pp", "oz_pass_rate", -1),
    "Strong run D vs run-heavy O": ("dz_rush_epa_pp", "oz_pass_rate", 1),
    "Big-play-limiting D vs explosive pass O": ("dz_exp_pass_rate", "oz_exp_pass_rate", -1),
}


def style_interactions(T: pd.DataFrame) -> list[dict]:
    """For each style pair: D/ST points per 1-SD x 1-SD match, controlling for both units'
    full profiles (all z-scores as main effects). Fitted on seasons before the backtest
    window, then checked for holdout gain in correlation."""
    main = [c for c in T.columns if c.startswith(("dz_", "oz_"))]
    seasons = sorted(T["season"].unique())
    cut = seasons[MIN_TRAIN_SEASONS]
    out = []
    for name, (dcol, ocol, sign) in STYLE_PAIRS.items():
        X = T[main].to_numpy(dtype=float)
        inter = (sign * T[dcol].astype(float) * T[ocol].astype(float)).to_numpy()
        y = T["dst_fp"].to_numpy(dtype=float)
        full = _ols(np.column_stack([X, inter]), y)
        resid = y - np.column_stack([np.ones(len(X)), X, inter]) @ full
        XtX = np.linalg.pinv(np.column_stack([np.ones(len(X)), X, inter]).T @ np.column_stack([np.ones(len(X)), X, inter]))
        se = float(np.sqrt(resid.var() * XtX[-1, -1]))
        tr, te = (T["season"] < cut).to_numpy(), (T["season"] >= cut).to_numpy()
        b0 = _ols(X[tr], y[tr]); b1 = _ols(np.column_stack([X, inter])[tr], y[tr])
        p0 = np.column_stack([np.ones(te.sum()), X[te]]) @ b0
        p1 = np.column_stack([np.ones(te.sum()), X[te], inter[te]]) @ b1
        out.append({"pair": name, "effect": round(float(full[-1]), 3), "se": round(se, 3),
                    "t": round(float(full[-1]) / se, 2),
                    "holdout_corr_gain": round(float(np.corrcoef(p1, y[te])[0, 1] - np.corrcoef(p0, y[te])[0, 1]), 4)})
    return out


# ------------------------------------------------------------------ output

def run() -> dict:
    years = list(range(FIRST_SEASON - 1, config.SEASON + 1))
    seasons = {}
    for y in years:
        print(f"  season {y}", file=sys.stderr)
        # Completed seasons never change: cache their game rows next to the other data.
        seasons[y] = game_stats(y) if y >= config.SEASON else fetch._cached(f"matchup_games_{y}", lambda y=y: game_stats(y))
    full_seasons = {y: r for y, r in seasons.items() if y < config.SEASON}     # centroids: completed seasons only
    cents = fit_centroids(full_seasons)
    T = pregame_table(seasons, cents)
    T = T.dropna(subset=["d_cluster", "o_cluster", "dst_fp"])
    T[["d_cluster", "o_cluster"]] = T[["d_cluster", "o_cluster"]].astype(int)

    bt = backtest(T)
    styles = style_interactions(T[T["season"] < config.SEASON])
    summary = bt.groupby("model").apply(lambda d: pd.Series({
        "games": int(d["games"].sum()),
        **{m: float(np.average(d[m], weights=d["games"])) for m in ("corr", "rmse", "boom_auc", "top10_boom")}}),
        include_groups=False).sort_values("corr", ascending=False)

    hist = T[T["season"] < config.SEASON]
    mu, a, b = _additive(hist)
    cells = []
    for (d, o), g in hist.groupby(["d_cluster", "o_cluster"]):
        cells.append({"d": int(d), "o": int(o), "games": int(len(g)), "avg": round(float(g["dst_fp"].mean()), 2),
                      "boom": round(float((g["dst_fp"] >= BOOM).mean()), 3),
                      "bust": round(float((g["dst_fp"] <= 0).mean()), 3),
                      "vs_additive": round(float(g["dst_fp"].mean() - (mu + a[d] + b[o])), 2),
                      "se": round(float(g["dst_fp"].std() / np.sqrt(len(g))), 2)})
    names = {}
    for side, feats_map in (("defense", DEF_FEATURES), ("offense", OFF_FEATURES)):
        feats = cents[side]["features"]
        names[side] = [{"id": i + 1, "name": _cluster_name(side, np.array(c), feats, i),
                        "profile": {f: round(float(v), 2) for f, v in zip(feats, c)},
                        "team_games": int((hist["d_cluster" if side == "defense" else "o_cluster"] == i + 1).sum())}
                       for i, c in enumerate(cents[side]["centroids"])]
    # Does the matchup (interaction) beat good-vs-bad (additive)? Compare holdout correlation.
    s = summary["corr"]
    return {
        "first_season": FIRST_SEASON, "last_season": int(hist["season"].max()), "k": K, "boom": BOOM,
        "prior_games": PRIOR_GAMES, "z_cap": Z_CAP, "games": int(len(hist)),
        "centroids": cents,
        "features": {"defense": {f: list(v) for f, v in DEF_FEATURES.items()},
                     "offense": {f: list(v) for f, v in OFF_FEATURES.items()}},
        "clusters": names, "cells": cells,
        "additive": {"mu": round(float(mu), 2), "defense": {int(k): round(float(v), 2) for k, v in a.items()},
                     "offense": {int(k): round(float(v), 2) for k, v in b.items()}},
        "backtest": {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for m, r in summary.iterrows()},
        "backtest_seasons": [int(x) for x in sorted(bt["season"].unique())],
        "style_pairs": styles,
        "matrix_beats_additive": bool(s["Style matchup matrix"] > s["Style clusters, no matchup effect (additive)"]),
    }


def assign_current(cents_file: Path = OUTPUT) -> dict | None:
    """This week's styles for every team (current season to date, blended with last season)."""
    if not cents_file.exists():
        return None
    res = json.loads(cents_file.read_text())
    cents = res["centroids"]
    cur_rows, pri_rows = game_stats(config.SEASON), game_stats(config.PRIOR_SEASON)
    pO, pD = adjusted_profiles(pri_rows)
    cO, cD = adjusted_profiles(cur_rows)
    O = _blend(cO, pO, RATES + ["dst_allowed_pg", "dst_allowed_adj_pg"])
    D = _blend(cD, pD, RATES + ["dst_pg"])
    d_cl = assign(_z(D, cents["defense"]["features"]).fillna(0), cents["defense"]["centroids"])
    o_cl = assign(_z(O, cents["offense"]["features"]).fillna(0), cents["offense"]["centroids"])
    teams = {t: {"d": int(d_cl[t]), "o": int(o_cl[t]),
                 "opp_allowed_pg": round(float(O.at[t, "dst_allowed_pg"]), 2),
                 "opp_allowed_adj_pg": round(float(O.at[t, "dst_allowed_adj_pg"]), 2)}
             for t in d_cl.index if t in o_cl.index}
    return {**res, "teams": teams}


if __name__ == "__main__":
    out = run()
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUTPUT}", file=sys.stderr)
    for r in out["style_pairs"]:
        print(f"  {r['pair']:42s} effect {r['effect']:+.3f} pts (t {r['t']:+.2f}) holdout corr gain {r['holdout_corr_gain']:+.4f}", file=sys.stderr)
    for m, r in out["backtest"].items():
        print(f"  {m:48s} corr {r['corr']:.3f} rmse {r['rmse']:.2f} boomAUC {r['boom_auc']:.3f} top10 {r['top10_boom']:.3f}", file=sys.stderr)
