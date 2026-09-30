"""Top-200 tiered rankings.

Skill PPG = config.RANKING_WEIGHTS over this season and the two before it
(70/20/10, renormalized over seasons played), where each season's PPG mixes
actual and expected fantasy points (config.XFP_PPG_WEIGHT; xFP covers
rushing and receiving, so QB passing points stay actual).

proj_pts = skill_adj * matchup_mult, where skill_adj is skill PPG moved by
the other tabs' signals (Plays/G, separation, CPOE/RYOE; see
src/signals.py) and matchup_mult comes from Def vs Pos (WR1/WR2 rows, funnel
confidence) and the slot/perimeter fit. ros_pts and po_pts apply the SOS tab's
rest-of-season and fantasy-playoff schedule multipliers instead.

Ranks by raw projected points — no positional-scarcity adjustment, so a
QB-heavy top 12 is expected and is not draft advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import defense, expected_points, fetch, scoring, signals

INJURED_DISCOUNT = 0.92       # zero-game ACT players with history
TOP_N = 200
TIERS = [(12, "Elite"), (24, "Locks"), (40, "Strong Starters"), (60, "Solid Starters"),
         (80, "Flex"), (100, "Streamers"), (130, "Bench Stash"), (160, "Deep Bench"),
         (200, "Waiver Watch")]


def next_opponents(year: int = config.SEASON) -> pd.DataFrame:
    """Each team's next unplayed regular-season game.

    `bye` is True when a team has no game at all in the league's current week
    (the earliest week with an unplayed game); its row then shows the game after
    the bye, and the matchup multiplier treats it as neutral.
    """
    g = fetch.load_schedule()
    g = g[(g["season"] == year) & (g["game_type"] == "REG")]
    pending = g[g["result"].isna()]
    if pending.empty:
        return pd.DataFrame(columns=["team", "opp", "week", "home", "bye"])
    current_week = pending["week"].min()
    long = pd.concat([
        pending.assign(team=pending["home_team"], opp=pending["away_team"], home=True),
        pending.assign(team=pending["away_team"], opp=pending["home_team"], home=False),
    ])[["team", "opp", "week", "home"]]
    long = long.sort_values("week").drop_duplicates("team")
    playing = set(g.loc[g["week"] == current_week, ["home_team", "away_team"]].values.ravel())
    long["bye"] = ~long["team"].isin(playing)
    return long.reset_index(drop=True)


def season_ppg(year: int) -> pd.DataFrame:
    """player_id -> games, actual ppg, expected ppg (xppg) and the scored ppg the
    blend uses. xppg swaps rushing/receiving points for their xFP value; players
    below the xFP opportunity minimum (and QB passing) keep actual points."""
    s = scoring.player_seasons(year)
    x = expected_points.expected_vs_actual(year)[["player_id", "xfp", "actual"]]
    s = s.merge(x, on="player_id", how="left")
    s["xppg"] = (s["fp"] - s["actual"] + s["xfp"]) / s["games"]
    s["score_ppg"] = np.where(s["xppg"].notna(),
                              (1 - config.XFP_PPG_WEIGHT) * s["ppg"] + config.XFP_PPG_WEIGHT * s["xppg"], s["ppg"])
    return s[s["games"] > 0][["player_id", "games", "ppg", "xppg", "score_ppg"]]


def _weighted(df: pd.DataFrame) -> pd.DataFrame:
    """Adds skill_ppg and w_current from the per-season score_ppg_<year> columns."""
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for yr, w in config.RANKING_WEIGHTS.items():
        has = df[f"games_{yr}"] > 0
        num += np.where(has, w * df[f"score_ppg_{yr}"].fillna(0), 0)
        den += np.where(has, w, 0)
    df["skill_ppg"] = num / den.where(den > 0)
    cur = df[f"games_{config.SEASON}"] > 0
    df["w_current"] = np.where(cur, config.RANKING_WEIGHTS[config.SEASON] / den.where(den > 0), 0.0)
    # Active but hasn't played this season (injury, suspension): small discount.
    df.loc[~cur, "skill_ppg"] *= INJURED_DISCOUNT
    return df


def _starting_qbs(df: pd.DataFrame) -> set[str]:
    """Per roster team, the ACT QB with the most current-season pass attempts
    (last season's games/PPG break ties, e.g. preseason)."""
    qbs = df[df["position"] == "QB"].copy()
    prev = config.SEASON - 1
    qbs = qbs.sort_values(["pass_att", f"games_{prev}", f"ppg_{prev}"], ascending=False)
    return set(qbs.drop_duplicates("team")["player_id"])


def _tier(rank: int) -> str:
    return next((name for cap, name in TIERS if rank <= cap), "Outside Top 200")


def build_rankings(dvp: pd.DataFrame | None = None, tables: dict | None = None,
                   proj: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (rankings, rookie/no-data gaps). `rank`/`tier` are by this week's
    proj_pts; ros/po ranks are included so the page can re-rank by schedule.
    Rows are the union of each view's top 200. Pass `proj` to reuse project_players output."""
    df, gaps = proj if proj is not None else project_players(dvp, tables)
    df = df.copy()
    for col, key in (("proj_pts", ""), ("ros_pts", "ros_"), ("po_pts", "po_")):
        df[key + "rank"] = df[col].rank(ascending=False, method="first").astype(int)
    keep = (df[["rank", "ros_rank", "po_rank"]] <= TOP_N).any(axis=1)
    df = df[keep].sort_values("rank").reset_index(drop=True)
    df["tier"] = df["rank"].map(_tier)
    return df, gaps


def project_players(dvp: pd.DataFrame | None = None, tables: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every eligible player (ACT, QBs = starters) with skill_ppg, signals, matchup,
    proj_pts (this week), ros_pts and po_pts, sorted by proj_pts.
    Returns (projections, rookie/no-data gaps)."""
    ros = scoring.roster_lookup(config.SEASON)
    ros = ros[(ros["status"] == "ACT") & ros["position"].isin(scoring.SKILL_POSITIONS)]

    cur = scoring.player_seasons(config.SEASON)[["player_id", "pass_att"]]
    df = ros.merge(cur, on="player_id", how="left")
    for yr in config.RANKING_WEIGHTS:
        sp = season_ppg(yr).rename(columns=lambda c: c if c == "player_id" else f"{c}_{yr}")
        df = df.merge(sp, on="player_id", how="left")
        df[f"games_{yr}"] = df[f"games_{yr}"].fillna(0)
    df["pass_att"] = df["pass_att"].fillna(0)
    y = config.SEASON
    df["games"], df["ppg"], df["xppg"] = df[f"games_{y}"], df[f"ppg_{y}"], df[f"xppg_{y}"]

    played = sum(df[f"games_{yr}"] for yr in config.RANKING_WEIGHTS) > 0
    gaps = df[~played]
    df = df[played]

    df = df[(df["position"] != "QB") | df["player_id"].isin(_starting_qbs(df))].copy()
    df = _weighted(df)

    t = tables or signals.tables()
    if dvp is None:
        dvp = t["dvp"]
    t = {**t, "dvp": dvp}
    df = signals.season_signals(df.reset_index(drop=True), t)
    df = signals.schedule_signals(df, t)

    # Matchup multiplier, shrunk toward neutral by confidence. WRs face the
    # opponent's WR1/WR2 row when they hold that role.
    opp = next_opponents()
    df = df.merge(opp[["team", "opp", "bye"]], on="team", how="left")
    df.loc[df["bye"].fillna(False).astype(bool), "opp"] = None
    df["group"] = signals.matchup_group(df)
    d = dvp[["team", "position", "fp_pg", "league_fp_pg", "confidence", "rank"]].rename(
        columns={"team": "opp", "position": "group", "rank": "opp_rank", "confidence": "opp_confidence"})
    df = df.merge(d, on=["opp", "group"], how="left")
    df = signals.matchup_signals(df, t)
    raw = df["fp_pg"] / df["league_fp_pg"]
    mult = ((1.0 + (raw - 1.0) * df["opp_confidence"] / 100) * df["slot_fit"]).clip(0.75, 1.30)
    df["matchup_mult"] = mult.fillna(1.0)   # bye week / unknown opponent -> neutral
    df["proj_pts"] = df["skill_adj"] * df["matchup_mult"]

    return df.sort_values("proj_pts", ascending=False).reset_index(drop=True), gaps


def to_json(df: pd.DataFrame) -> list[dict]:
    cols = ["rank", "ros_rank", "po_rank", "tier", "player_id", "full_name", "position", "team", "opp", "bye",
            "group", "games", "ppg", "xppg", f"ppg_{config.SEASON - 1}", f"ppg_{config.SEASON - 2}",
            "w_current", "skill_ppg", "fpoe_pg", "sig_pace", "sig_eff", "sig_total", "skill_adj", "opp_rank", "opp_confidence",
            "slot_fit", "matchup_mult", "proj_pts", "ros_mult", "ros_pts", "po_mult", "po_pts"]
    out = df[cols].round(2)
    out = out.astype(object).where(out.notna(), None)
    out["chips"] = df["chips"]
    return out.to_dict("records")


def gaps_to_json(gaps: pd.DataFrame) -> list[dict]:
    return gaps[["full_name", "position", "team"]].sort_values(["position", "team"]).to_dict("records")
