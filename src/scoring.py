"""PPR fantasy scoring from play-by-play.

Each play is split into per-player "credit" rows (passer, rusher, receiver,
fumbler) so every downstream module can aggregate points and opportunity the
same way. pbp has no position column, so positions come from the roster file of
the matching season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import fetch

PTS = {
    "pass_yd": 0.04, "pass_td": 4, "int": -2,
    "rush_yd": 0.1, "rush_td": 6,
    "rec_yd": 0.1, "rec_td": 6, "rec": 1,
    "fumble_lost": -2,
}
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
STAT_COLS = ["dropbacks", "pass_att", "pass_yds", "pass_td", "int",
             "rush_att", "rush_yds", "rush_td",
             "targets", "rec", "rec_yds", "rec_td", "fumbles_lost",
             "pass_fp", "rush_fp", "rec_fp", "fp"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def credit_rows(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (play, player, role) with stats and fantasy points."""
    base = ["game_id", "play_id", "season", "week", "posteam", "defteam", "yardline_100",
            "pass_location", "air_yards"]
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])]
    parts = []

    # Passer: attempts, yards, TDs, INTs. Scrambles are run plays with the QB as rusher.
    p = pbp[pbp["passer_player_id"].notna()]
    parts.append(p[base].assign(
        player_id=p["passer_player_id"], name=p["passer_player_name"], role="pass",
        dropbacks=_num(p["qb_dropback"]),
        pass_att=_num(p["pass_attempt"]) * (1 - _num(p["sack"])),
        pass_yds=_num(p["passing_yards"]), pass_td=_num(p["pass_touchdown"]),
        int=_num(p["interception"]),
    ))

    r = pbp[(pbp["play_type"] == "run") & pbp["rusher_player_id"].notna()]
    parts.append(r[base].assign(
        player_id=r["rusher_player_id"], name=r["rusher_player_name"], role="rush",
        dropbacks=_num(r["qb_scramble"]),
        rush_att=1.0, rush_yds=_num(r["rushing_yards"]), rush_td=_num(r["rush_touchdown"]),
    ))

    t = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()]
    parts.append(t[base].assign(
        player_id=t["receiver_player_id"], name=t["receiver_player_name"], role="target",
        targets=1.0, rec=_num(t["complete_pass"]),
        rec_yds=_num(t["receiving_yards"]), rec_td=_num(t["pass_touchdown"]),
    ))

    f = pbp[(_num(pbp["fumble_lost"]) == 1) & pbp["fumbled_1_player_id"].notna()]
    parts.append(f[base].assign(
        player_id=f["fumbled_1_player_id"], name=np.nan, role="fumble", fumbles_lost=1.0,
    ))

    out = pd.concat(parts, ignore_index=True)
    for c in STAT_COLS:
        if c not in out:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    out["pass_fp"] = out.pass_yds * PTS["pass_yd"] + out.pass_td * PTS["pass_td"] + out["int"] * PTS["int"]
    out["rush_fp"] = out.rush_yds * PTS["rush_yd"] + out.rush_td * PTS["rush_td"]
    out["rec_fp"] = out.rec_yds * PTS["rec_yd"] + out.rec_td * PTS["rec_td"] + out.rec * PTS["rec"]
    out["fp"] = out.pass_fp + out.rush_fp + out.rec_fp + out.fumbles_lost * PTS["fumble_lost"]
    return out


def roster_lookup(year: int) -> pd.DataFrame:
    """gsis_id -> full_name, position, team, status (latest week row per player)."""
    r = fetch.load_roster(year)
    r = r[r["gsis_id"].notna()].sort_values("week").drop_duplicates("gsis_id", keep="last")
    r = r.rename(columns={"gsis_id": "player_id"})
    r["position"] = r["position"].replace({"FB": "RB"})
    return r[["player_id", "full_name", "position", "team", "status", "headshot_url"]]


def credits_with_position(year: int) -> pd.DataFrame:
    c = credit_rows(fetch.load_pbp(year))
    ros = roster_lookup(year)[["player_id", "full_name", "position"]]
    return c.merge(ros, on="player_id", how="left")


def player_games(year: int) -> pd.DataFrame:
    """Per player per game stat lines for skill positions."""
    c = credits_with_position(year)
    c = c[c["position"].isin(SKILL_POSITIONS)]
    # pbp's abbreviated name is a fallback only; full_name is what users see.
    g = (c.groupby(["player_id", "game_id", "week", "season"], as_index=False)
          .agg(**{k: (k, "sum") for k in STAT_COLS},
               team=("posteam", "first"), opp=("defteam", "first"),
               position=("position", "first"), full_name=("full_name", "first")))
    return g


def player_seasons(year: int) -> pd.DataFrame:
    """Per player season totals with games played and PPG."""
    g = player_games(year)
    s = (g.groupby("player_id", as_index=False)
          .agg(**{k: (k, "sum") for k in STAT_COLS},
               games=("game_id", "nunique"), team=("team", "last"),
               position=("position", "first"), full_name=("full_name", "first")))
    # Guard against a player surfacing twice (e.g. gadget snaps at another
    # position): keep the row with the most games.
    s = s.sort_values("games", ascending=False).drop_duplicates("player_id")
    s["ppg"] = s["fp"] / s["games"]
    s["season"] = year
    return s


def opportunity(df: pd.DataFrame) -> pd.Series:
    """Volume measure per position: targets (WR/TE), rush+targets (RB), dropbacks (QB)."""
    return np.select(
        [df["position"] == "QB", df["position"] == "RB"],
        [df["dropbacks"], df["rush_att"] + df["targets"]],
        default=df["targets"],
    )
