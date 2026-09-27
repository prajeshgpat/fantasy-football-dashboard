"""Red zone opportunity: targets/carries inside the 20 and 10, plus TDs.
High opportunity with zero TDs = positive TD regression candidate."""
from __future__ import annotations

import pandas as pd

import config
from src import scoring

REGRESSION_MIN_OPPS = 6


def red_zone(year: int = config.SEASON) -> pd.DataFrame:
    c = scoring.credits_with_position(year)
    c = c[c["position"].isin(scoring.SKILL_POSITIONS) & c["role"].isin(["rush", "target"])
          & (c["yardline_100"] <= 20)]
    c = c.assign(i10=c["yardline_100"] <= 10, td=c["rush_td"] + c["rec_td"])
    df = c.groupby("player_id").agg(
        name=("full_name", "first"), position=("position", "first"), team=("posteam", "last"),
        rz_targets=("targets", "sum"), rz_carries=("rush_att", "sum"), rz_tds=("td", "sum"),
    )
    i10 = c[c["i10"]].groupby("player_id").agg(i10_targets=("targets", "sum"), i10_carries=("rush_att", "sum"))
    df = df.join(i10).fillna({"i10_targets": 0, "i10_carries": 0}).reset_index()
    df["rz_opps"] = df["rz_targets"] + df["rz_carries"]
    df["i10_opps"] = df["i10_targets"] + df["i10_carries"]
    df["td_regression"] = (df["rz_opps"] >= REGRESSION_MIN_OPPS) & (df["rz_tds"] == 0)
    return df.sort_values(["rz_opps", "i10_opps"], ascending=False).reset_index(drop=True)


def to_json(df: pd.DataFrame) -> list[dict]:
    return df.round(2).to_dict("records")
