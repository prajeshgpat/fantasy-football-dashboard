"""D/ST matchups: which fantasy defenses to start this week.

A defense's outlook depends on the opponent as much as on the defense, and on
how they fit: a pass-heavy offense matters most against a defense that is good
(or bad) against the pass. For each upcoming game:

  results    D/ST points this defense scores per game + D/ST points this
             offense gives up per game - league average (each blended with
             last season, which counts as PRIOR_GAMES games)
  matchup    expected offensive EPA/play in this game: the opponent's pass EPA
  EPA        and the defense's pass EPA allowed (each vs. league), weighted by
             the opponent's neutral-script pass rate, plus the same for runs
  EPA proj   league D/ST points + slope x (matchup EPA - league EPA), where
             slope is D/ST points per EPA/play, fitted on every team-game this
             season and last
  proj       half results, half EPA proj
  fit        the part of matchup EPA that comes from tendency meeting strength:
             (opponent pass rate - league) x (defense pass EPA - rush EPA),
             shown in D/ST points

EPA per play comes from nflverse play-by-play (the same source as nfelo's EPA
tiers). D/ST points use config.DST_SCORING (ESPN standard by default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import fetch, rankings

NEUTRAL_WP = (0.20, 0.80)
PRIOR_GAMES = 4            # last season's rates count as this many games
PROJ_EPA_WEIGHT = 0.5      # projection = (1 - w) x fantasy results + w x EPA projection


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
            fp = (sacks * sc["sack"] + ints * sc["int"] + fum * sc["fumble_rec"] + tds * sc["td"]
                  + safeties * sc["safety"] + blocks * sc["block"]
                  + _tier(pts_allowed, sc["points_allowed"]) + _tier(yards, sc["yards_allowed"]))
            rows.append({
                "season": year, "week": g.week, "game_id": g.game_id, "defense": defense, "offense": offense,
                "dst_fp": fp, "sacks": sacks, "takeaways": ints + fum, "def_tds": tds,
                "pts_allowed": pts_allowed, "yds_allowed": yards,
                "plays": len(scrim), "dropbacks": len(dropbacks), "rushes": len(rushes),
                "pass_epa": dropbacks["epa"].sum(), "rush_epa": rushes["epa"].sum(),
                "epa": scrim["epa"].sum(),
            })
    return pd.DataFrame(rows)


def neutral_pass_rate(year: int) -> pd.Series:
    """Share of an offense's plays that are dropbacks with win probability 20-80% (quarters 1-4)."""
    p = fetch.load_pbp(year)
    p = p[p["play_type"].isin(["pass", "run"]) & (p["qtr"] <= 4) & p["wp"].between(*NEUTRAL_WP)]
    return p.groupby("posteam")["pass"].mean()


def _profiles(tg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per team: as a defense (what it does) and as an offense (what it gives up to defenses)."""
    def agg(by):
        a = tg.groupby(by).agg(games=("game_id", "nunique"), dst_fp=("dst_fp", "sum"),
                               sacks=("sacks", "sum"), takeaways=("takeaways", "sum"),
                               dropbacks=("dropbacks", "sum"), rushes=("rushes", "sum"),
                               pass_epa=("pass_epa", "sum"), rush_epa=("rush_epa", "sum"))
        return a.assign(dst_pg=a["dst_fp"] / a["games"], sack_rate=a["sacks"] / a["dropbacks"],
                        takeaways_pg=a["takeaways"] / a["games"],
                        pass_epa_pp=a["pass_epa"] / a["dropbacks"], rush_epa_pp=a["rush_epa"] / a["rushes"])
    return agg("defense"), agg("offense")


def _blend(cur: pd.DataFrame, prior: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Current-season rate shrunk toward last season's: last season counts as PRIOR_GAMES games."""
    out = cur[["games"]].copy()
    p = prior.reindex(cur.index)
    for c in cols:
        has = p[c].notna()
        out[c] = np.where(has, (cur["games"] * cur[c] + PRIOR_GAMES * p[c]) / (cur["games"] + PRIOR_GAMES), cur[c])
    return out


def epa_slope(tg: pd.DataFrame) -> float:
    """D/ST points per 1.0 of offensive EPA/play in that game (least squares, negative)."""
    x = tg["epa"] / tg["plays"]
    ok = tg["plays"] > 0
    return float(np.polyfit(x[ok], tg.loc[ok, "dst_fp"], 1)[0])


def matchups() -> dict:
    cur_y, pri_y = config.SEASON, config.PRIOR_SEASON
    tg_cur, tg_pri = team_games(cur_y), team_games(pri_y)
    slope = epa_slope(pd.concat([tg_cur, tg_pri]))

    cols = ["dst_pg", "sack_rate", "takeaways_pg", "pass_epa_pp", "rush_epa_pp"]
    d_cur, o_cur = _profiles(tg_cur)
    d_pri, o_pri = _profiles(tg_pri)
    D, O = _blend(d_cur, d_pri, cols), _blend(o_cur, o_pri, cols)
    pr_cur, pr_pri = neutral_pass_rate(cur_y), neutral_pass_rate(pri_y)
    O["pass_rate"] = (O["games"] * pr_cur.reindex(O.index) + PRIOR_GAMES * pr_pri.reindex(O.index).fillna(pr_cur.mean())) \
        / (O["games"] + PRIOR_GAMES)

    lg = {c: D[c].mean() for c in cols}
    lg_pr = O["pass_rate"].mean()
    lg_epa = lg_pr * lg["pass_epa_pp"] + (1 - lg_pr) * lg["rush_epa_pp"]
    # Ranks: 1 = best for the defense (lowest EPA allowed, most sacks/takeaways/points).
    D["pass_rank"] = D["pass_epa_pp"].rank(method="min").astype(int)
    D["rush_rank"] = D["rush_epa_pp"].rank(method="min").astype(int)
    O["pass_rank"] = O["pass_epa_pp"].rank(ascending=False, method="min").astype(int)   # 1 = best offense
    O["rush_rank"] = O["rush_epa_pp"].rank(ascending=False, method="min").astype(int)

    opp = rankings.next_opponents(cur_y)
    week = int(opp["week"].min()) if len(opp) else None
    rows = []
    for r in opp[~opp["bye"].astype(bool)].itertuples():
        dfn, ofn = r.team, r.opp
        if dfn not in D.index or ofn not in O.index:
            continue
        d, o = D.loc[dfn], O.loc[ofn]
        results = d["dst_pg"] + o["dst_pg"] - lg["dst_pg"]
        # Expected opponent EPA/play, weighting pass vs run by the opponent's tendency.
        exp_pass = o["pass_epa_pp"] + d["pass_epa_pp"] - lg["pass_epa_pp"]
        exp_rush = o["rush_epa_pp"] + d["rush_epa_pp"] - lg["rush_epa_pp"]
        matchup_epa = o["pass_rate"] * exp_pass + (1 - o["pass_rate"]) * exp_rush
        epa_proj = lg["dst_pg"] + slope * (matchup_epa - lg_epa)
        fit = PROJ_EPA_WEIGHT * slope * (o["pass_rate"] - lg_pr) * (d["pass_epa_pp"] - d["rush_epa_pp"])
        rows.append({
            "defense": dfn, "opp": ofn, "home": bool(r.home), "week": int(r.week),
            "proj": (1 - PROJ_EPA_WEIGHT) * results + PROJ_EPA_WEIGHT * epa_proj,
            "results": results, "epa_proj": epa_proj, "fit": fit, "matchup_epa": matchup_epa,
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
    out = df.round(4)
    return {
        "week": week,
        "slope": round(slope, 1),
        "epa_weight": PROJ_EPA_WEIGHT,
        "league": {"dst_pg": round(lg["dst_pg"], 2), "pass_rate": round(lg_pr, 3),
                   "pass_epa": round(lg["pass_epa_pp"], 3), "rush_epa": round(lg["rush_epa_pp"], 3),
                   "epa": round(lg_epa, 3),
                   "sack_rate": round(lg["sack_rate"], 3), "takeaways_pg": round(lg["takeaways_pg"], 2)},
        "prior_games": PRIOR_GAMES,
        "rows": out.astype(object).where(out.notna(), None).to_dict("records"),
    }
