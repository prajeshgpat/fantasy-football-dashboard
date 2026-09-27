"""WR/TE NGS separation vs. targets per route run.

Prior (completed) season: true TPRR from pbp_participation — a route is a
qb_dropback play with the player on the field. Current season: participation
is not published in-season, so targets / offensive snaps stands in (labeled as a
proxy everywhere it is shown).

NGS avg_separation is tracking-derived (yards to nearest defender at catch or
incompletion). It is NOT Fantasy Points' charted "Average Separation Score".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import fetch, scoring

MIN_ROUTES = 50
MIN_SNAPS = 20           # below this the per-snap proxy is too noisy to show
SEP_SWING = 1.0          # yards
RATE_UP, RATE_DOWN = 0.50, -0.30


def _ngs_season(year: int) -> pd.DataFrame:
    n = fetch.load_ngs("receiving")
    n = n[(n["season"] == year) & (n["week"] == 0) & (n["season_type"] == "REG")
          & n["player_position"].isin(["WR", "TE"])]
    return n[["player_gsis_id", "player_display_name", "player_position", "team_abbr",
              "avg_separation", "avg_cushion", "targets"]].rename(
        columns={"player_gsis_id": "player_id", "targets": "ngs_targets"}).assign(
        team_abbr=lambda d: d["team_abbr"].replace(fetch.TEAM_ALIASES))


def _targets(year: int) -> pd.Series:
    c = scoring.credit_rows(fetch.load_pbp(year))
    return c[c["role"] == "target"].groupby("player_id")["targets"].sum()


def true_tprr(year: int) -> pd.DataFrame | None:
    part = fetch.load_participation(year)
    if part is None:
        return None
    pbp = fetch.load_pbp(year)
    db = pbp.loc[pbp["qb_dropback"] == 1, ["game_id", "play_id"]]
    part = part.rename(columns={"nflverse_game_id": "game_id"})
    part["play_id"] = part["play_id"].astype(float)
    db = db.assign(play_id=db["play_id"].astype(float)).merge(part, on=["game_id", "play_id"])
    players = db["offense_players"].dropna().str.split(";").explode().str.strip()
    routes = players[players != ""].value_counts().rename("routes")
    df = routes.to_frame().join(_targets(year).rename("targets"), how="left").fillna({"targets": 0})
    df = df[df["routes"] >= MIN_ROUTES]
    df["tprr"] = df["targets"] / df["routes"]
    return df.rename_axis("player_id").reset_index()


def targets_per_snap(year: int) -> pd.DataFrame:
    """Targets / offensive snaps. Snap counts carry no gsis_id: join on the
    roster's pfr_id, falling back to (unique) full name."""
    snaps = fetch.load_snap_counts(year)
    snaps = snaps[snaps["game_type"] == "REG"]
    snaps = snaps.groupby(["pfr_player_id", "player"], as_index=False)["offense_snaps"].sum()
    ros = fetch.load_roster(year)[["gsis_id", "pfr_id", "full_name"]].dropna(subset=["gsis_id"])
    by_pfr = ros.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    by_name = ros.drop_duplicates(["full_name", "gsis_id"]).drop_duplicates("full_name", keep=False) \
                 .set_index("full_name")["gsis_id"]
    snaps["player_id"] = snaps["pfr_player_id"].map(by_pfr).fillna(snaps["player"].map(by_name))
    m = snaps.dropna(subset=["player_id"]).groupby("player_id", as_index=False)["offense_snaps"].sum()
    m = m.rename(columns={"offense_snaps": "snaps"})
    t = _targets(year).rename("targets").reset_index()
    m = m.merge(t, on="player_id", how="left").fillna({"targets": 0})
    m["tps"] = np.where(m["snaps"] >= MIN_SNAPS, m["targets"] / m["snaps"], np.nan)
    return m


def separation_table() -> pd.DataFrame:
    cur_y, pri_y = config.SEASON, config.PRIOR_SEASON
    cur = _ngs_season(cur_y).merge(targets_per_snap(cur_y), on="player_id", how="left")
    pri = _ngs_season(pri_y).merge(targets_per_snap(pri_y)[["player_id", "tps", "snaps"]],
                                   on="player_id", how="left")
    tprr = true_tprr(pri_y)
    if tprr is not None:
        pri = pri.merge(tprr[["player_id", "routes", "tprr"]], on="player_id", how="left")
    else:
        pri["routes"], pri["tprr"] = np.nan, np.nan

    df = cur.merge(pri[["player_id", "avg_separation", "tps", "tprr", "routes"]],
                   on="player_id", how="outer", suffixes=("", "_prior"))
    names = pd.concat([cur, pri])[["player_id", "player_display_name", "player_position", "team_abbr"]]
    names = names.drop_duplicates("player_id")  # current season listed first
    df = df.drop(columns=["player_display_name", "player_position", "team_abbr"]).merge(names, on="player_id")
    df = df.rename(columns={"avg_separation": "sep", "avg_separation_prior": "sep_prior",
                            "tps": "tps", "tps_prior": "tps_prior",
                            "player_display_name": "name", "player_position": "position", "team_abbr": "team"})
    df["sep_delta"] = df["sep"] - df["sep_prior"]
    # YoY rate change is proxy-vs-proxy so the two seasons are measured the same way.
    df["rate_change"] = df["tps"] / df["tps_prior"] - 1

    def flags(r):
        out = []
        if r.sep_delta >= SEP_SWING: out.append("Sep ↑")
        if r.sep_delta <= -SEP_SWING: out.append("Sep ↓")
        if r.rate_change >= RATE_UP: out.append("Tgt rate ↑")
        if r.rate_change <= RATE_DOWN: out.append("Tgt rate ↓")
        return ", ".join(out)
    df["flags"] = df.apply(flags, axis=1)
    return df


def to_json(df: pd.DataFrame) -> list[dict]:
    cols = ["player_id", "name", "position", "team", "sep", "tps", "targets", "snaps",
            "sep_prior", "tprr", "routes", "tps_prior", "sep_delta", "rate_change", "flags"]
    out = df[cols].replace([np.inf, -np.inf], np.nan).round(3)
    return out.astype(object).where(out.notna(), None).to_dict("records")
