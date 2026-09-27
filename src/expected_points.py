"""Expected fantasy points (xFP): calibrate on the last completed season, apply
to every current-season target and carry.

Large negative delta (actual - xFP) = buy-low; large positive = regression candidate.
"""
from __future__ import annotations

import pandas as pd

import config
from src import scoring

AIR_BINS = [-100, 0, 5, 10, 15, 20, 100]
YL_BINS = [-1, 5, 10, 20, 50, 100]
MIN_OPPORTUNITIES = 10


def _bucket(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["air_bin"] = pd.cut(c["air_yards"].fillna(0).clip(-99, 99), AIR_BINS).astype(str)
    c["yl_bin"] = pd.cut(c["yardline_100"], YL_BINS).astype(str)
    return c


def calibrate(year: int = config.PRIOR_SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = _bucket(scoring.credit_rows(scoring.fetch.load_pbp(year)))
    t = c[c["role"] == "target"]
    rec = t.groupby("air_bin").apply(lambda d: pd.Series({
        "catch_rate": d["rec"].mean(),
        "avg_yards_if_complete": d.loc[d["rec"] == 1, "rec_yds"].mean(),
        "td_rate": d["rec_td"].mean(),
        "n": len(d),
    }), include_groups=False).fillna(0)
    r = c[c["role"] == "rush"]
    rush = r.groupby("yl_bin").apply(lambda d: pd.Series({
        "avg_yards": d["rush_yds"].mean(),
        "td_rate": d["rush_td"].mean(),
        "n": len(d),
    }), include_groups=False).fillna(0)
    return rec, rush


def expected_vs_actual(year: int = config.SEASON) -> pd.DataFrame:
    rec, rush = calibrate()
    c = _bucket(scoring.credits_with_position(year))
    c = c[c["position"].isin(["RB", "WR", "TE"])]

    t = c[c["role"] == "target"].join(rec, on="air_bin")
    t["xfp"] = t["catch_rate"] * (1 + t["avg_yards_if_complete"] * 0.1) + t["td_rate"] * 6
    r = c[c["role"] == "rush"].join(rush, on="yl_bin")
    r["xfp"] = r["avg_yards"] * 0.1 + r["td_rate"] * 6

    plays = pd.concat([t, r])
    plays["actual"] = plays["rec_fp"] + plays["rush_fp"]
    df = plays.groupby("player_id").agg(
        name=("full_name", "first"), position=("position", "first"), team=("posteam", "last"),
        games=("game_id", "nunique"), targets=("targets", "sum"), carries=("rush_att", "sum"),
        xfp=("xfp", "sum"), actual=("actual", "sum"),
    ).reset_index()
    df = df[df["targets"] + df["carries"] >= MIN_OPPORTUNITIES]
    df["delta"] = df["actual"] - df["xfp"]
    df["xfp_pg"] = df["xfp"] / df["games"]
    df["actual_pg"] = df["actual"] / df["games"]
    return df.sort_values("xfp", ascending=False).reset_index(drop=True)


def to_json(df: pd.DataFrame) -> list[dict]:
    return df.round(2).to_dict("records")
