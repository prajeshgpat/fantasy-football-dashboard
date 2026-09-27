"""Slot/middle vs. perimeter WR+TE fantasy points allowed (proxy).

nflverse publishes no verified WR alignment in-season, so pass_location stands
in: 'middle' -> slot/middle, 'left'/'right' -> perimeter. Perimeter volume is
higher league-wide, so every team skews negative — read the numbers relative to
the league, not in absolute terms.
"""
from __future__ import annotations

import pandas as pd

import config
from src import scoring


def slot_vs_perimeter(year: int = config.SEASON) -> pd.DataFrame:
    c = scoring.credits_with_position(year)
    c = c[(c["role"] == "target") & c["position"].isin(["WR", "TE"]) & c["pass_location"].notna()]
    c = c.assign(zone=c["pass_location"].map({"middle": "slot", "left": "perimeter", "right": "perimeter"}))
    games = c.groupby("defteam")["game_id"].nunique()
    pts = c.pivot_table(index="defteam", columns="zone", values="rec_fp", aggfunc="sum", fill_value=0)
    df = pd.DataFrame({
        "slot_fp_pg": pts.get("slot", 0) / games,
        "perim_fp_pg": pts.get("perimeter", 0) / games,
        "games": games,
    }).reset_index().rename(columns={"defteam": "team"})
    df["diff"] = df["slot_fp_pg"] - df["perim_fp_pg"]
    # Relative read: how far this defense's slot-minus-perimeter sits from the league mean.
    z = (df["diff"] - df["diff"].mean()) / (df["diff"].std(ddof=0) or 1)
    df["rel_z"] = z
    df["tendency"] = pd.cut(z, [-float("inf"), -0.5, 0.5, float("inf")],
                            labels=["Perimeter-vulnerable", "Balanced", "Slot-vulnerable"]).astype(str)
    return df.sort_values("diff", ascending=False).reset_index(drop=True)


def to_json(df: pd.DataFrame) -> list[dict]:
    return df.round(2).to_dict("records")
