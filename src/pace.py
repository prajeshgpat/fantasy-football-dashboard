"""Offensive volume and whether it will last.

A jump in plays per game can come from things that persist (a faster neutral-
script tempo, often after a coaching change) or from things that don't
(overtime, or spending the game trailing and throwing to catch up). Each team's
change is split into those pieces and given a verdict plus an expected rest-of-
season plays/G that leans on this season only as much as the evidence supports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import fetch

NEUTRAL_WP = (0.20, 0.80)
BIG_MARGIN = 9            # "trailing/leading big" = two scores
PLAYS_DELTA = 3.0         # plays/G change worth explaining
PACE_DELTA = 1.5          # seconds per snap change in neutral script = real tempo change
SCRIPT_DELTA = 0.08       # 8-point swing in share of snaps played from well behind/ahead
OT_SHARE = 0.03           # OT snaps above 3% of a team's snaps is a noticeable bump
K_REAL, K_NOISE = 2.0, 6.0  # prior-season "games" of weight in the ROS expectation


def team_volume(year: int) -> pd.DataFrame:
    pbp = fetch.load_pbp(year)
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()].copy()
    games = p.groupby("posteam")["game_id"].nunique()
    reg = p[p["qtr"] <= 4]
    out = pd.DataFrame({
        "games": games,
        "plays_pg": p.groupby("posteam").size() / games,
        "reg_plays_pg": reg.groupby("posteam").size() / games,
        "ot_plays": p[p["qtr"] > 4].groupby("posteam").size(),
        "trail_share": reg.assign(x=reg["score_differential"] <= -BIG_MARGIN).groupby("posteam")["x"].mean(),
        "lead_share": reg.assign(x=reg["score_differential"] >= BIG_MARGIN).groupby("posteam")["x"].mean(),
    }).fillna({"ot_plays": 0})
    out["ot_share"] = out["ot_plays"] / (out["plays_pg"] * out["games"])
    drives = reg.dropna(subset=["drive"]).groupby("posteam").apply(
        lambda d: d[["game_id", "drive"]].drop_duplicates().shape[0], include_groups=False)
    out["drives_pg"] = drives / games
    out["plays_per_drive"] = reg.groupby("posteam").size() / drives

    # Neutral-script tempo: seconds between consecutive snaps of the same drive.
    n = reg[reg["wp"].between(*NEUTRAL_WP)].sort_values(["game_id", "drive", "game_seconds_remaining"],
                                                        ascending=[True, True, False])
    gap = n.groupby(["game_id", "drive"])["game_seconds_remaining"].diff(-1).abs()
    n = n.assign(gap=gap)
    n = n[(n["gap"] > 0) & (n["gap"] <= 60)]         # drop timeouts, quarter breaks, reviews
    out["neutral_sec_per_play"] = n.groupby("posteam")["gap"].mean()
    out["neutral_pass_rate"] = reg[reg["wp"].between(*NEUTRAL_WP)].assign(
        x=lambda d: d["play_type"] == "pass").groupby("posteam")["x"].mean()
    return out.rename_axis("team")


def head_coaches(year: int) -> pd.Series:
    g = fetch.load_schedule()
    g = g[(g["season"] == year) & (g["game_type"] == "REG")].sort_values("week")
    long = pd.concat([g[["week", "home_team", "home_coach"]].set_axis(["week", "team", "coach"], axis=1),
                      g[["week", "away_team", "away_coach"]].set_axis(["week", "team", "coach"], axis=1)])
    return long.dropna().drop_duplicates("team", keep="last").set_index("team")["coach"]


def _driver(r) -> str:
    """Which part of the volume moved: number of drives or length of drives."""
    dd = r["drives_pg"] - r["drives_pg_prior"]
    pd_ = r["plays_per_drive"] - r["plays_per_drive_prior"]
    parts = []
    if abs(dd) >= 0.7:
        parts.append(f"{'more' if dd > 0 else 'fewer'} drives ({r['drives_pg']:.1f}/G vs {r['drives_pg_prior']:.1f})")
    if abs(pd_) >= 0.4:
        parts.append(f"{'longer' if pd_ > 0 else 'shorter'} drives ({r['plays_per_drive']:.1f} plays vs {r['plays_per_drive_prior']:.1f})")
    return " and ".join(parts)


def _verdict(r) -> tuple[str, str, str]:
    """(tone, verdict, reason). tone: real | script | stable | watch"""
    d, pace, script, lead = r["reg_delta"], r["pace_delta"], r["script_delta"], r["lead_delta"]
    notes = []
    if r["new_coach"]:
        notes.append(f"new HC {r['coach']}")
    if r["ot_share"] >= OT_SHARE:
        notes.append(f"{r['ot_plays']:.0f} OT snaps excluded")
    driver = _driver(r)
    tail = lambda *xs: "; ".join(x for x in (*xs, *notes) if x)

    if abs(d) < PLAYS_DELTA:
        return "stable", "Stable", tail(f"Within {PLAYS_DELTA:.0f} plays/G of last season")
    if d > 0:
        if pace >= PACE_DELTA:
            return "real", "Likely sustainable", tail(f"{pace:.1f}s faster per snap in neutral script", driver)
        if script >= SCRIPT_DELTA:
            return "script", "Game-script inflated", tail(
                f"trailing by 9+ on {r['trail_share']:.0%} of snaps (was {r['trail_share_prior']:.0%})", driver)
        if script <= -SCRIPT_DELTA:
            return "real", "Leaning sustainable", tail(
                f"more snaps while trailing less ({r['trail_share']:.0%} of snaps down 9+, was "
                f"{r['trail_share_prior']:.0%})", driver)
        if r["new_coach"]:
            return "watch", "Possibly real", tail("tempo not faster yet", driver)
        return "watch", "Expect some regression", tail("no tempo or script change behind it", driver)
    # fewer plays
    if pace <= -PACE_DELTA:
        return "real", "Likely a real slowdown", tail(f"{-pace:.1f}s slower per snap in neutral script", driver)
    if lead >= SCRIPT_DELTA:
        return "script", "Script-suppressed", tail(
            f"leading by 9+ on {r['lead_share']:.0%} of snaps (was {r['lead_share_prior']:.0%}), running clock", driver)
    tempo = (f"tempo is {pace:.1f}s faster, so the drop is" if pace >= PACE_DELTA
             else "tempo unchanged, so the drop is")
    return "watch", "Expect some rebound", tail(f"{tempo} {driver or 'fewer snaps per possession'}")


def volume_table() -> pd.DataFrame:
    cur, pri = team_volume(config.SEASON), team_volume(config.PRIOR_SEASON)
    df = cur.join(pri, rsuffix="_prior")
    coach, coach_prior = head_coaches(config.SEASON), head_coaches(config.PRIOR_SEASON)
    df["coach"] = coach.reindex(df.index)
    df["new_coach"] = (coach.reindex(df.index) != coach_prior.reindex(df.index)) & coach_prior.reindex(df.index).notna()
    df["reg_delta"] = df["reg_plays_pg"] - df["reg_plays_pg_prior"]
    df["pace_delta"] = df["neutral_sec_per_play_prior"] - df["neutral_sec_per_play"]   # + = faster now
    df["script_delta"] = df["trail_share"] - df["trail_share_prior"]
    df["lead_delta"] = df["lead_share"] - df["lead_share_prior"]
    v = df.apply(_verdict, axis=1, result_type="expand")
    df["tone"], df["verdict"], df["reason"] = v[0], v[1], v[2]
    # Rest-of-season expectation: trust this season more when a tempo change explains it.
    k = np.where(df["tone"] == "real", K_REAL, K_NOISE)
    g = df["games"]
    df["expected_pg"] = (g * df["reg_plays_pg"] + k * df["reg_plays_pg_prior"]) / (g + k)
    return df.reset_index()


def to_json(df: pd.DataFrame) -> list[dict]:
    cols = ["team", "games", "plays_pg", "reg_plays_pg", "reg_plays_pg_prior", "reg_delta", "ot_plays",
            "neutral_sec_per_play", "neutral_sec_per_play_prior", "pace_delta",
            "trail_share", "trail_share_prior", "lead_share", "lead_share_prior",
            "neutral_pass_rate", "neutral_pass_rate_prior", "coach", "new_coach",
            "drives_pg", "drives_pg_prior", "plays_per_drive", "plays_per_drive_prior",
            "tone", "verdict", "reason", "expected_pg"]
    out = df[cols].round(3)
    return out.astype(object).where(out.notna(), None).to_dict("records")
