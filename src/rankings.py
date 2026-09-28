"""Top-200 tiered rankings: proj_pts = skill_ppg * matchup_mult.

Ranks by raw projected points — no positional-scarcity adjustment, so a
QB-heavy top 12 is expected and is not draft advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import defense, fetch, scoring

W_CURRENT_PER_GAME = 0.10     # current-season weight grows 10%/game ...
W_CURRENT_CAP = 0.50          # ... capped at 50% (~5 games)
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


def _history() -> pd.DataFrame:
    rows = []
    for yr, w in zip(config.HISTORY_SEASONS, config.HISTORY_WEIGHTS):
        s = scoring.player_seasons(yr)
        s["opp"] = scoring.opportunity(s)
        rows.append(s.assign(w=w)[["player_id", "games", "fp", "opp", "w"]])
    h = pd.concat(rows)
    h = h[h["games"] > 0]
    h["ppg"] = h["fp"] / h["games"]
    h["opp_pg"] = h["opp"] / h["games"]
    agg = h.groupby("player_id").apply(lambda d: pd.Series({
        "ppg_3yr": np.average(d["ppg"], weights=d["w"]),
        "opp_3yr_pg": np.average(d["opp_pg"], weights=d["w"]),
        "games_3yr": d["games"].sum(),
    }), include_groups=False)
    return agg.reset_index()


def _starting_qbs(df: pd.DataFrame) -> set[str]:
    """Per roster team, the ACT QB with the most current-season pass attempts
    (prior 3-year PPG breaks ties, e.g. preseason)."""
    qbs = df[df["position"] == "QB"].copy()
    qbs = qbs.sort_values(["pass_att", "games_3yr", "ppg_3yr"], ascending=False)
    return set(qbs.drop_duplicates("team")["player_id"])


def build_rankings(dvp: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (top-200 rankings, rookie/no-data gaps)."""
    df, gaps = project_players(dvp)
    df = df.head(TOP_N).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["tier"] = df["rank"].map(lambda r: next(name for cap, name in TIERS if r <= cap))
    return df, gaps


def project_players(dvp: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every eligible player (ACT, QBs = starters) with skill_ppg, matchup and proj_pts,
    sorted by proj_pts. Returns (projections, rookie/no-data gaps)."""
    ros = scoring.roster_lookup(config.SEASON)
    ros = ros[(ros["status"] == "ACT") & ros["position"].isin(scoring.SKILL_POSITIONS)]

    cur = scoring.player_seasons(config.SEASON)
    cur["opp_cur"] = scoring.opportunity(cur)
    cur = cur[["player_id", "games", "fp", "ppg", "opp_cur", "pass_att"]]
    df = ros.merge(cur, on="player_id", how="left").merge(_history(), on="player_id", how="left")
    df[["games", "fp", "opp_cur", "pass_att", "games_3yr"]] = \
        df[["games", "fp", "opp_cur", "pass_att", "games_3yr"]].fillna(0)

    gaps = df[(df["games"] == 0) & df["ppg_3yr"].isna()]
    df = df.drop(gaps.index)

    df = df[(df["position"] != "QB") | df["player_id"].isin(_starting_qbs(df))]

    w_cur = np.minimum(df["games"] * W_CURRENT_PER_GAME, W_CURRENT_CAP)
    opp_cur_pg = np.where(df["games"] > 0, df["opp_cur"] / df["games"].where(df["games"] > 0), 0)
    vol = np.where(df["opp_3yr_pg"] > 0, np.minimum(opp_cur_pg / df["opp_3yr_pg"], 1.0), 1.0)
    w_eff = w_cur * (0.5 + 0.5 * vol)
    blended = df["ppg_3yr"] * (1 - w_eff) + df["ppg"].fillna(0) * w_eff
    df["skill_ppg"] = np.select(
        [df["ppg_3yr"].isna(), df["games"] == 0],
        [df["ppg"], df["ppg_3yr"] * INJURED_DISCOUNT],
        default=blended,
    )
    df["volume_ratio"] = vol
    df["w_current"] = w_eff

    # Matchup multiplier, shrunk toward neutral by confidence.
    dvp = defense.defense_vs_position() if dvp is None else dvp
    opp = next_opponents()
    df = df.merge(opp[["team", "opp", "bye"]], on="team", how="left")
    df.loc[df["bye"].fillna(False).astype(bool), "opp"] = None
    d = dvp[["team", "position", "fp_pg", "league_fp_pg", "confidence", "rank"]].rename(
        columns={"team": "opp", "rank": "opp_rank", "confidence": "opp_confidence"})
    df = df.merge(d, on=["opp", "position"], how="left")
    raw = df["fp_pg"] / df["league_fp_pg"]
    mult = (1.0 + (raw - 1.0) * df["opp_confidence"] / 100).clip(0.75, 1.30)
    df["matchup_mult"] = mult.fillna(1.0)   # bye week / unknown opponent -> neutral
    df["proj_pts"] = df["skill_ppg"] * df["matchup_mult"]

    return df.sort_values("proj_pts", ascending=False).reset_index(drop=True), gaps


def to_json(df: pd.DataFrame) -> list[dict]:
    cols = ["rank", "tier", "player_id", "full_name", "position", "team", "opp", "bye", "games",
            "ppg", "ppg_3yr", "volume_ratio", "w_current", "skill_ppg", "opp_rank",
            "opp_confidence", "matchup_mult", "proj_pts"]
    out = df[cols].round(2)
    return out.astype(object).where(out.notna(), None).to_dict("records")


def gaps_to_json(gaps: pd.DataFrame) -> list[dict]:
    return gaps[["full_name", "position", "team"]].sort_values(["position", "team"]).to_dict("records")
