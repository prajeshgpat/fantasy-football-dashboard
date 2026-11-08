"""D/ST matchups: which fantasy defenses to start this week.

A defense's outlook depends on the opponent as much as on the defense. Four
pre-kickoff projections of D/ST points, each built only from what was known
before the game:

  results    D/ST points this defense scores per game + D/ST points this
             offense gives up per game - league average
  EPA proj   expected offensive EPA/play in this game (each side's pass and
             rush EPA vs. league, weighted by the opponent's neutral-script
             pass rate), converted with the slope of D/ST points on EPA/play
  yards proj expected yards allowed (opponent pass/rush yards/G + defense
             pass/rush yards allowed/G - league), scored with the D/ST
             yards-allowed tiers smoothed over the game-to-game spread
  line proj  D/ST points from the opponent's implied points (betting line),
             a straight-line fit on past games

The projection is a weighted blend of the four. The weights are refitted every
build (non-negative, summing to 1) to past games since dst_boom.TRAIN_FROM,
and backtest() checks each season with weights fitted on the others. Team
rates blend this season with last (PRIOR_GAMES games). D/ST points use
config.DST_SCORING (ESPN standard by default).

  fit        within the EPA projection, the part that comes from tendency
             meeting strength: (opponent pass rate - league) x (defense pass
             EPA - rush EPA), in D/ST points
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import dst_boom, fetch, rankings

NEUTRAL_WP = (0.20, 0.80)
PRIOR_GAMES = 4            # last season's rates count as this many games


def _tier(value: float, tiers: list[tuple[float, float]]) -> float:
    """tiers: [(upper bound inclusive, points), ...] ascending."""
    return next(pts for cap, pts in tiers if value <= cap)


def team_games(year: int) -> pd.DataFrame:
    """One row per game per defense: the D/ST box score and the opposing offense's EPA split."""
    pbp = fetch.load_pbp(year)
    sched = fetch.load_schedule()
    sched = sched[(sched["season"] == year) & (sched["game_type"] == "REG") & sched["result"].notna()]
    sc = config.DST_SCORING
    rows = []
    for g in sched.itertuples():
        d = pbp[pbp["game_id"] == g.game_id]
        for defense, offense, pts_allowed in ((g.home_team, g.away_team, g.away_score),
                                              (g.away_team, g.home_team, g.home_score)):
            off = d[d["posteam"] == offense]
            scrim = off[off["play_type"].isin(["pass", "run"]) & off["epa"].notna()]
            dropbacks = scrim[scrim["pass"] == 1]
            neutral = scrim[(scrim["qtr"] <= 4) & scrim["wp"].between(*NEUTRAL_WP)]
            rushes = scrim[scrim["rush"] == 1]
            sacks = off["sack"].fillna(0).sum()
            ints = off["interception"].fillna(0).sum()
            fum = off["fumble_lost"].fillna(0).sum()
            # Defense/return TDs: scored by the defense while the offense had the ball
            # (pick-six, fumble or blocked-punt return), or on a kickoff it received.
            tds = (((d["touchdown"] == 1) & (d["td_team"] == defense) & (d["posteam"] == offense)).sum()
                   + ((d["return_touchdown"] == 1) & (d["td_team"] == defense) & (d["posteam"] == defense)
                      & (d["play_type"] == "kickoff")).sum())
            safeties = ((d["safety"] == 1) & (d["posteam"] == offense)).sum()
            blocks = (off["punt_blocked"].fillna(0).sum() + (off["field_goal_result"] == "blocked").sum()
                      + (off["extra_point_result"] == "blocked").sum())
            yards = off.loc[off["play_type"].isin(["pass", "run"]), "yards_gained"].fillna(0).sum()
            pass_yds = off.loc[off["play_type"].isin(["pass", "run"]) & (off["pass"] == 1), "yards_gained"].fillna(0).sum()
            fp = (sacks * sc["sack"] + ints * sc["int"] + fum * sc["fumble_rec"] + tds * sc["td"]
                  + safeties * sc["safety"] + blocks * sc["block"]
                  + _tier(pts_allowed, sc["points_allowed"]) + _tier(yards, sc["yards_allowed"]))
            rows.append({
                "season": year, "week": g.week, "game_id": g.game_id, "defense": defense, "offense": offense,
                "dst_fp": fp, "sacks": sacks, "takeaways": ints + fum, "def_tds": tds,
                "pts_allowed": pts_allowed, "yds_allowed": yards,
                "plays": len(scrim), "dropbacks": len(dropbacks), "rushes": len(rushes),
                "pass_epa": dropbacks["epa"].sum(), "rush_epa": rushes["epa"].sum(),
                "pass_yds": pass_yds, "rush_yds": yards - pass_yds,
                "epa": scrim["epa"].sum(),
                "neutral_plays": len(neutral), "neutral_dropbacks": int((neutral["pass"] == 1).sum()),
            })
    return pd.DataFrame(rows)


def _profiles(tg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per team: as a defense (what it does) and as an offense (what it gives up to defenses)."""
    def agg(by):
        a = tg.groupby(by).agg(games=("game_id", "nunique"), dst_fp=("dst_fp", "sum"),
                               sacks=("sacks", "sum"), takeaways=("takeaways", "sum"),
                               dropbacks=("dropbacks", "sum"), rushes=("rushes", "sum"),
                               pass_epa=("pass_epa", "sum"), rush_epa=("rush_epa", "sum"),
                               pass_yds=("pass_yds", "sum"), rush_yds=("rush_yds", "sum"),
                               neutral_plays=("neutral_plays", "sum"), neutral_dropbacks=("neutral_dropbacks", "sum"))
        return a.assign(dst_pg=a["dst_fp"] / a["games"], sack_rate=a["sacks"] / a["dropbacks"],
                        takeaways_pg=a["takeaways"] / a["games"],
                        pass_epa_pp=a["pass_epa"] / a["dropbacks"], rush_epa_pp=a["rush_epa"] / a["rushes"],
                        pass_ypg=a["pass_yds"] / a["games"], rush_ypg=a["rush_yds"] / a["games"],
                        pass_rate=a["neutral_dropbacks"] / a["neutral_plays"])
    return agg("defense"), agg("offense")


def _blend(cur: pd.DataFrame, prior: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Current-season rate shrunk toward last season's: last season counts as PRIOR_GAMES games."""
    out = cur[["games"]].copy()
    p = prior.reindex(cur.index)
    for c in cols:
        has = p[c].notna()
        out[c] = np.where(has, (cur["games"] * cur[c] + PRIOR_GAMES * p[c]) / (cur["games"] + PRIOR_GAMES), cur[c])
    return out


def yards_sd(tg: pd.DataFrame) -> float:
    """Game-to-game spread of yards allowed around the additive offense + defense expectation."""
    parts = []
    for _, g in tg.groupby("season"):
        lg = g["yds_allowed"].mean()
        exp = g["offense"].map(g.groupby("offense")["yds_allowed"].mean()) \
            + g["defense"].map(g.groupby("defense")["yds_allowed"].mean()) - lg
        parts.append(g["yds_allowed"] - exp)
    return float(pd.concat(parts).std())


def expected_tier(mu: float, sd: float, tiers: list[tuple[float, float]]) -> float:
    """Expected tier points when the stat is roughly Normal(mu, sd)."""
    z = np.linspace(-3, 3, 61)
    w = np.exp(-z ** 2 / 2)
    return float(np.average([_tier(mu + sd * zi, tiers) for zi in z], weights=w))


def epa_slope(tg: pd.DataFrame) -> float:
    """D/ST points per 1.0 of offensive EPA/play in that game (least squares, negative)."""
    x = tg["epa"] / tg["plays"]
    ok = tg["plays"] > 0
    return float(np.polyfit(x[ok], tg.loc[ok, "dst_fp"], 1)[0])


COLS = ["dst_pg", "sack_rate", "takeaways_pg", "pass_epa_pp", "rush_epa_pp", "pass_ypg", "rush_ypg", "pass_rate"]
COMPONENTS = ["results", "epa_proj", "yards_proj", "line_proj"]
WEIGHT_STEP = 0.05         # grid for fitting the projection weights


def state(tg: pd.DataFrame, season: int, week: int, sd: float) -> dict:
    """Team profiles as of the start of `week`: this season's earlier games blended with last season."""
    cur = tg[(tg["season"] == season) & (tg["week"] < week)]
    pri = tg[tg["season"] == season - 1]
    d_cur, o_cur = _profiles(cur)
    d_pri, o_pri = _profiles(pri)
    if cur.empty:                               # week 1: last season only
        D, O = d_pri[COLS + ["games"]].copy(), o_pri[COLS + ["games"]].copy()
    else:
        D, O = _blend(d_cur, d_pri, COLS), _blend(o_cur, o_pri, COLS)
    lg = {c: D[c].mean() for c in COLS}
    lg["pass_rate"] = O["pass_rate"].mean()
    lg["epa"] = lg["pass_rate"] * lg["pass_epa_pp"] + (1 - lg["pass_rate"]) * lg["rush_epa_pp"]
    lg["ya_pts"] = expected_tier(lg["pass_ypg"] + lg["rush_ypg"], sd, config.DST_SCORING["yards_allowed"])
    return {"D": D, "O": O, "lg": lg}


def implied_points(sched: pd.DataFrame, game_id: str, offense: str) -> float:
    g = sched.loc[game_id]
    half = g["spread_line"] / 2
    return g["total_line"] / 2 + (half if g["home_team"] == offense else -half)


def components(st: dict, defense: str, offense: str, slope: float, sd: float) -> dict | None:
    D, O, lg = st["D"], st["O"], st["lg"]
    if defense not in D.index or offense not in O.index:
        return None
    d, o = D.loc[defense], O.loc[offense]
    exp_pass = o["pass_epa_pp"] + d["pass_epa_pp"] - lg["pass_epa_pp"]
    exp_rush = o["rush_epa_pp"] + d["rush_epa_pp"] - lg["rush_epa_pp"]
    matchup_epa = o["pass_rate"] * exp_pass + (1 - o["pass_rate"]) * exp_rush
    exp_pass_yds = o["pass_ypg"] + d["pass_ypg"] - lg["pass_ypg"]
    exp_rush_yds = o["rush_ypg"] + d["rush_ypg"] - lg["rush_ypg"]
    exp_yds = exp_pass_yds + exp_rush_yds
    return {
        "results": d["dst_pg"] + o["dst_pg"] - lg["dst_pg"],
        "epa_proj": lg["dst_pg"] + slope * (matchup_epa - lg["epa"]),
        "yards_proj": lg["dst_pg"] + expected_tier(exp_yds, sd, config.DST_SCORING["yards_allowed"]) - lg["ya_pts"],
        "matchup_epa": matchup_epa,
        "fit_epa": slope * (o["pass_rate"] - lg["pass_rate"]) * (d["pass_epa_pp"] - d["rush_epa_pp"]),
        "exp_pass_yds": exp_pass_yds, "exp_rush_yds": exp_rush_yds, "exp_yds": exp_yds,
    }


def history(tg: pd.DataFrame, slope: float, sd: float) -> pd.DataFrame:
    """Every played game since dst_boom.TRAIN_FROM with its pre-kickoff components and the result."""
    sched = fetch.load_schedule().set_index("game_id")
    rows = []
    for (season, week), grp in tg[tg["season"] >= dst_boom.TRAIN_FROM].groupby(["season", "week"]):
        st = state(tg, season, week, sd)
        for r in grp.itertuples():
            c = components(st, r.defense, r.offense, slope, sd)
            if c is None:
                continue
            rows.append({"season": season, "week": week, "dst_fp": r.dst_fp,
                         "implied": implied_points(sched, r.game_id, r.offense), **c})
    return pd.DataFrame(rows)


def fit_line(H: pd.DataFrame) -> tuple[float, float]:
    """D/ST points = a + b x opponent implied points (least squares)."""
    h = H[H["implied"].notna()]
    b, a = np.polyfit(h["implied"], h["dst_fp"], 1)
    return float(a), float(b)


def fit_weights(H: pd.DataFrame) -> dict:
    """Non-negative weights summing to 1 over COMPONENTS that minimize squared error."""
    X, y = H[COMPONENTS].to_numpy(), H["dst_fp"].to_numpy()
    n = int(round(1 / WEIGHT_STEP))
    best, best_w = np.inf, None
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                w = np.array([i, j, k, n - i - j - k]) / n
                err = np.mean((X @ w - y) ** 2)
                if err < best:
                    best, best_w = err, w
    return dict(zip(COMPONENTS, best_w.round(2)))


def _with_line(H: pd.DataFrame, line: tuple[float, float]) -> pd.DataFrame:
    return H.assign(line_proj=line[0] + line[1] * H["implied"])


def backtest(H: pd.DataFrame) -> pd.DataFrame:
    """Hold out each season: fit the line and weights on the others, score the held-out season."""
    H = H[H["implied"].notna()]
    rows = []
    for s in sorted(H["season"].unique()):
        tr, te = H[H["season"] != s], H[H["season"] == s]
        line = fit_line(tr)
        tr, te = _with_line(tr, line), _with_line(te, line)
        w = fit_weights(tr)
        blends = {"blend": w, "old (thirds, no line)": {"results": 1 / 3, "epa_proj": 1 / 3, "yards_proj": 1 / 3, "line_proj": 0},
                  "equal quarters": {c: 0.25 for c in COMPONENTS},
                  "half line, rest equal": {"results": 1 / 6, "epa_proj": 1 / 6, "yards_proj": 1 / 6, "line_proj": 0.5}}
        blends.update({c: {k: float(k == c) for k in COMPONENTS} for c in COMPONENTS})
        for name, wt in blends.items():
            pred = te[COMPONENTS].to_numpy() @ np.array([wt[c] for c in COMPONENTS])
            rows.append({"season": s, "model": name, "games": len(te),
                         "rmse": float(np.sqrt(np.mean((pred - te["dst_fp"]) ** 2))),
                         "corr": float(np.corrcoef(pred, te["dst_fp"])[0, 1])})
    return pd.DataFrame(rows)


def matchups() -> dict:
    cur_y, pri_y = config.SEASON, config.PRIOR_SEASON
    tg_all = pd.concat([team_games(y) for y in range(dst_boom.TRAIN_FROM - 1, cur_y + 1)])
    both = tg_all[tg_all["season"] >= pri_y]
    slope, sd = epa_slope(both), yards_sd(both)

    H = history(tg_all, slope, sd)
    line = fit_line(H)
    H = _with_line(H, line)
    weights = fit_weights(H[H["implied"].notna()])
    bt = backtest(H)
    bt_summary = bt.groupby("model").apply(
        lambda d: pd.Series({"rmse": np.average(d["rmse"], weights=d["games"]),
                             "corr": np.average(d["corr"], weights=d["games"])}), include_groups=False)

    opp = rankings.next_opponents(cur_y)
    week = int(opp["week"].min()) if len(opp) else None
    st = state(tg_all, cur_y, week, sd)
    D, O, lg = st["D"], st["O"], st["lg"]
    # Ranks: 1 = best for the defense (lowest EPA/yards allowed); offense ranks 1 = best offense.
    D["pass_rank"] = D["pass_epa_pp"].rank(method="min").astype(int)
    D["rush_rank"] = D["rush_epa_pp"].rank(method="min").astype(int)
    O["pass_rank"] = O["pass_epa_pp"].rank(ascending=False, method="min").astype(int)
    O["rush_rank"] = O["rush_epa_pp"].rank(ascending=False, method="min").astype(int)
    for k in ("pass", "rush"):
        D[f"{k}_yds_rank"] = D[f"{k}_ypg"].rank(method="min").astype(int)
        O[f"{k}_yds_rank"] = O[f"{k}_ypg"].rank(ascending=False, method="min").astype(int)

    sched_all = fetch.load_schedule()
    sched = sched_all[(sched_all["season"] == cur_y) & (sched_all["week"] == week) & (sched_all["game_type"] == "REG")]
    gid = {**{(h, a): g for g, h, a in zip(sched["game_id"], sched["home_team"], sched["away_team"])},
           **{(a, h): g for g, h, a in zip(sched["game_id"], sched["home_team"], sched["away_team"])}}
    sched_idx = sched_all.set_index("game_id")
    rows = []
    # Only this week's unplayed games (teams that already played this week show their next game in `opp`).
    for r in opp[~opp["bye"].astype(bool) & (opp["week"] == week)].itertuples():
        dfn, ofn = r.team, r.opp
        c = components(st, dfn, ofn, slope, sd)
        if c is None:
            continue
        d, o = D.loc[dfn], O.loc[ofn]
        implied = implied_points(sched_idx, gid[(dfn, ofn)], ofn)
        c["line_proj"] = line[0] + line[1] * implied if pd.notna(implied) else np.nan
        w = dict(weights)
        if pd.isna(implied):                    # no line yet: spread its weight over the rest
            rest = 1 - w["line_proj"]
            w = {k: (v / rest if k != "line_proj" else 0.0) for k, v in w.items()}
        proj = sum(w[k] * (c[k] if k != "line_proj" or pd.notna(c[k]) else 0) for k in COMPONENTS)
        rows.append({
            "defense": dfn, "opp": ofn, "home": bool(r.home), "week": int(r.week), "game_id": gid[(dfn, ofn)],
            "proj": proj, **{k: c[k] for k in COMPONENTS}, "fit": c["fit_epa"],
            "matchup_epa": c["matchup_epa"], "exp_pass_yds": c["exp_pass_yds"], "exp_rush_yds": c["exp_rush_yds"],
            "exp_yds": c["exp_yds"],
            "def_pass_ypg": d["pass_ypg"], "def_pass_yds_rank": int(d["pass_yds_rank"]),
            "def_rush_ypg": d["rush_ypg"], "def_rush_yds_rank": int(d["rush_yds_rank"]),
            "opp_pass_ypg": o["pass_ypg"], "opp_pass_yds_rank": int(o["pass_yds_rank"]),
            "opp_rush_ypg": o["rush_ypg"], "opp_rush_yds_rank": int(o["rush_yds_rank"]),
            "def_dst_pg": d["dst_pg"], "opp_dst_allowed_pg": o["dst_pg"],
            "opp_pass_rate": o["pass_rate"],
            "def_pass_epa": d["pass_epa_pp"], "def_pass_rank": int(d["pass_rank"]),
            "def_rush_epa": d["rush_epa_pp"], "def_rush_rank": int(d["rush_rank"]),
            "opp_pass_epa": o["pass_epa_pp"], "opp_pass_rank": int(o["pass_rank"]),
            "opp_rush_epa": o["rush_epa_pp"], "opp_rush_rank": int(o["rush_rank"]),
            "sack_edge": (d["sack_rate"] + o["sack_rate"]) / 2,
            "takeaway_edge": (d["takeaways_pg"] + o["takeaways_pg"]) / 2,
        })
    df = pd.DataFrame(rows).sort_values("proj", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    up = df[["defense", "opp", "home", "game_id"]].rename(columns={"opp": "offense"}).assign(season=cur_y, week=week)
    bm = dst_boom.boom(tg_all, up)
    df = df.drop(columns="game_id").merge(bm["upcoming"].rename(columns={"offense": "opp"}), on=["defense", "opp"], how="left")
    out = df.drop(columns="flags").round(4)
    return {
        "week": week,
        "slope": round(slope, 1),
        "weights": weights,
        "line": {"intercept": round(line[0], 2), "per_point": round(line[1], 3)},
        "backtest": {m: {"rmse": round(float(r["rmse"]), 2), "corr": round(float(r["corr"]), 3)} for m, r in bt_summary.iterrows()},
        "backtest_games": int(H["implied"].notna().sum()),
        "yards_sd": round(sd, 1),
        "league": {"dst_pg": round(lg["dst_pg"], 2), "pass_rate": round(lg["pass_rate"], 3),
                   "pass_epa": round(lg["pass_epa_pp"], 3), "rush_epa": round(lg["rush_epa_pp"], 3),
                   "epa": round(lg["epa"], 3),
                   "pass_ypg": round(lg["pass_ypg"], 1), "rush_ypg": round(lg["rush_ypg"], 1),
                   "sack_rate": round(lg["sack_rate"], 3), "takeaways_pg": round(lg["takeaways_pg"], 2)},
        "prior_games": PRIOR_GAMES,
        "rows": out.astype(object).where(out.notna(), None).assign(flags=df["flags"]).to_dict("records"),
        "boom": {k: v for k, v in bm.items() if k != "upcoming"},
    }
