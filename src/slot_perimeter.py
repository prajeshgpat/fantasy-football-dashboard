"""Slot/middle vs. perimeter WR+TE fantasy points allowed (proxy).

nflverse publishes no verified WR alignment in-season, so pass_location stands
in: 'middle' -> slot/middle, 'left'/'right' -> perimeter. Perimeter volume is
higher league-wide, so every team skews negative — read the numbers relative to
the league, not in absolute terms.
"""
from __future__ import annotations

import numpy as np
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


TAG_QUANTILE = 0.25      # top quarter of middle-target share at the position = "Slot", bottom = "Perimeter"
MIN_TAG_TARGETS = 15


def receiver_tags(year: int = config.SEASON) -> pd.DataFrame:
    """WR/TE alignment proxy from where they're targeted (middle vs outside),
    using this season and last season together for sample size. Team = current team."""
    frames = []
    for y in (year, year - 1):
        c = scoring.credits_with_position(y)
        frames.append(c[(c["role"] == "target") & c["position"].isin(["WR", "TE"]) & c["pass_location"].notna()])
    t = pd.concat(frames)
    t = t.assign(mid=(t["pass_location"] == "middle").astype(float))
    agg = t.groupby("player_id").agg(name=("full_name", "last"), position=("position", "last"),
                                     tgts=("mid", "size"), middle_share=("mid", "mean")).reset_index()
    cur = scoring.credits_with_position(year)
    cur = cur[cur["role"] == "target"].groupby("player_id").agg(team=("posteam", "last"), tgts_cur=("targets", "sum"))
    agg = agg.join(cur, on="player_id", how="inner")
    q = agg[agg["tgts"] >= MIN_TAG_TARGETS].groupby("position")["middle_share"]
    hi, lo = agg["position"].map(q.quantile(1 - TAG_QUANTILE)), agg["position"].map(q.quantile(TAG_QUANTILE))
    agg["tag"] = np.select(
        [agg["tgts"] < MIN_TAG_TARGETS, agg["middle_share"] >= hi, agg["middle_share"] <= lo],
        ["", "Slot", "Perimeter"], "Mixed")
    return agg.sort_values("tgts_cur", ascending=False)


def opponent_receivers(slot: pd.DataFrame, tags: pd.DataFrame, opp: pd.DataFrame, n: int = 3) -> dict:
    """defense -> top-n receivers (by current targets) of the offense it faces next, with fit."""
    tend = slot.set_index("team")["tendency"]
    nxt = opp.set_index("team")
    out = {}
    for d in slot["team"]:
        if d not in nxt.index:
            continue
        o = nxt.loc[d]
        recv = tags[tags["team"] == o["opp"]].head(n)
        fit = lambda tag: (("Slot" in tend[d] and tag == "Slot") or ("Perimeter" in tend[d] and tag == "Perimeter"))
        out[d] = {"opp": o["opp"], "week": int(o["week"]), "home": bool(o["home"]), "bye": bool(o["bye"]),
                  "receivers": [{"name": r["name"], "position": r["position"], "tag": r["tag"],
                                 "middle_share": round(float(r["middle_share"]), 3), "fit": bool(fit(r["tag"]))}
                                for _, r in recv.iterrows()]}
    return out


def to_json(df: pd.DataFrame) -> list[dict]:
    return df.round(2).to_dict("records")
