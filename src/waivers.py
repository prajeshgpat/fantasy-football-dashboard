"""Waiver targets: the best players at each position who are probably unrostered.

There's no league API, so "probably rostered" is estimated: the top N at each
position by current projection or by 3-year reputation (what got drafted),
early-round rookies, your roster, and anyone listed in ROSTERED_ELSEWHERE.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config
from src import fetch, scoring

TOP_PER_POSITION = 5
ROOKIE_PICK_CUTOFF = {"QB": 32, "RB": 40, "WR": 40, "TE": 40}
RISING = 1.25              # last-2-games opportunity vs earlier games
MIN_RISING_OPP = {"QB": 30, "RB": 10, "WR": 5, "TE": 4}
BUY_LOW_XFP = -5.0         # actual minus expected FP, season to date


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def usage_trend(year: int = config.SEASON) -> pd.DataFrame:
    """player_id -> opportunity per game in the last 2 games vs the games before."""
    g = scoring.player_games(year)
    g["opp"] = scoring.opportunity(g)
    g = g.sort_values(["player_id", "week"])
    g["n_from_end"] = g.groupby("player_id").cumcount(ascending=False)
    last2 = g[g["n_from_end"] < 2].groupby("player_id")["opp"].mean().rename("opp_last2")
    earlier = g[g["n_from_end"] >= 2].groupby("player_id")["opp"].mean().rename("opp_earlier")
    return pd.concat([last2, earlier], axis=1).reset_index()


def waiver_targets(proj: pd.DataFrame, xfp: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    ros = fetch.load_roster(config.SEASON)[["gsis_id", "rookie_year", "draft_number"]].drop_duplicates("gsis_id")
    df = proj.merge(ros.rename(columns={"gsis_id": "player_id"}), on="player_id", how="left")
    df = df.merge(usage_trend(), on="player_id", how="left")
    df = df.merge(xfp[["player_id", "xfp_pg", "delta"]].rename(columns={"delta": "xfp_delta"}),
                  on="player_id", how="left")
    vol = volume.set_index("team")
    df["team_volume"] = df["team"].map(vol["verdict"])
    df["team_volume_tone"] = df["team"].map(vol["tone"])
    df["team_reg_delta"] = df["team"].map(vol["reg_delta"])

    depth = config.WAIVER_ROSTERED_DEPTH
    df["proj_pos_rank"] = df.groupby("position")["proj_pts"].rank(ascending=False, method="first")
    df["rep_pos_rank"] = df.groupby("position")["ppg_3yr"].rank(ascending=False, method="first")
    lim = df["position"].map(depth)
    rookie = df["rookie_year"] == config.SEASON
    early_pick = rookie & (df["draft_number"] <= df["position"].map(ROOKIE_PICK_CUTOFF))
    taken = {_key(n) for n in list(config.MY_ROSTER) + list(getattr(config, "ROSTERED_ELSEWHERE", []))}
    df["assumed_rostered"] = ((df["proj_pos_rank"] <= lim) | (df["rep_pos_rank"] <= lim) | early_pick
                              | df["full_name"].map(_key).isin(taken))
    pool = df[~df["assumed_rostered"]].copy()

    def tags(r):
        t = []
        if r["games"] == 0:
            t.append("Back from injury")
        if (pd.notna(r["opp_earlier"]) and r["opp_earlier"] > 0 and r["opp_last2"] >= RISING * r["opp_earlier"]
                and r["opp_last2"] >= MIN_RISING_OPP[r["position"]]):
            t.append("Rising usage")
        if pd.notna(r["xfp_delta"]) and r["xfp_delta"] <= BUY_LOW_XFP:
            t.append("Buy-low (under xFP)")
        if pd.notna(r["opp_rank"]) and r["opp_rank"] >= 23:
            t.append("Plus matchup")
        if r["team_volume_tone"] == "real" and r["team_reg_delta"] > 0:
            t.append("Team volume up")
        if r["games"] > 0 and pd.notna(r["opp_last2"]) and r["opp_last2"] < 0.5 * MIN_RISING_OPP[r["position"]]:
            t.append("Low usage")
        if r["rookie_year"] == config.SEASON:
            t.append("Rookie")
        return t
    pool["tags"] = pool.apply(tags, axis=1)
    pool = pool.sort_values("proj_pts", ascending=False)
    out = pool.groupby("position", group_keys=False).head(TOP_PER_POSITION)
    out["pos_rank"] = out.groupby("position").cumcount() + 1
    order = {p: i for i, p in enumerate(scoring.SKILL_POSITIONS)}
    return out.sort_values(["position", "pos_rank"], key=lambda s: s.map(order) if s.name == "position" else s)


def to_json(df: pd.DataFrame) -> list[dict]:
    cols = ["pos_rank", "full_name", "position", "team", "opp", "bye", "opp_rank", "games", "ppg", "ppg_3yr",
            "skill_ppg", "proj_pts", "proj_pos_rank", "opp_last2", "opp_earlier", "xfp_pg", "xfp_delta",
            "team_volume", "team_volume_tone", "tags"]
    out = df[cols].copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out.astype(object).where(out.notna(), None).to_dict("records")
