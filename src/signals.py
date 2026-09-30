"""Per-player adjustments that feed the Top 200 from every other tab.

Each signal is small and bounded, and is recorded as the points it moves a
player's skill PPG so the table can show where a projection came from:

  Plays/G    the team's expected plays/G (the Plays/G verdict) vs. the plays
             the blend implicitly assumes (this season and last, weighted)
  Separation WR/TE separation change vs. last season (target-earning skill)
  CPOE       QB completion % over expected
  RYOE       RB rush yards over expected per carry

Matchup-level signals (this week only) adjust the matchup multiplier:

  Def vs Pos WRs face the opponent's WR1 or WR2 row when they hold that role;
             a Strong (behavior-confirmed) funnel raises matchup confidence
  Slot/Perim a Slot or Perimeter receiver facing a defense vulnerable (or not)
             in that area

xFP is part of skill PPG itself (see rankings.season_ppg), so it only adds
an evidence chip here. Red zone work is evidence only too: xFP already prices
touchdowns by field position, so adding it again would double count.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config
from src import defense, expected_points, pace, redzone, schedule, separation, skill_metrics, slot_perimeter

PACE_CLIP = (0.93, 1.07)
SEP_PER_YARD = 0.02        # +2% per yard of separation gained vs. last season ...
SEP_CLIP = 0.03            # ... capped at ±3%
CPOE_PER_PT = 0.004        # +0.4% per point of CPOE ...
CPOE_CLIP = 0.04           # ... capped at ±4%
RYOE_PER_YD = 0.02         # +2% per rush yard over expected per carry ...
RYOE_CLIP = 0.04
RYOE_MIN_ATT = 20
SEASON_CLIP = 0.15         # all season-level signals together move skill PPG at most ±15%
SLOT_FIT = 0.03            # ±3% matchup nudge for slot/perimeter fit
FUNNEL_CONF_BOOST = 15     # confidence points added for a Strong funnel


def _norm(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower().str.replace(r"[.'\-]", " ", regex=True)
    s = s.str.replace(r"\b(jr|sr|ii|iii|iv|v)\b", "", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def tables() -> dict[str, pd.DataFrame]:
    """Everything the signals read. build_dashboard computes these once and passes them in."""
    dvp = defense.defense_vs_position()
    return {
        "dvp": dvp,
        "xfp": expected_points.expected_vs_actual(),
        "rz": redzone.red_zone(),
        "cpoe": skill_metrics.cpoe(),
        "ryoe": skill_metrics.ryoe(),
        "sep": separation.separation_table(),
        "tags": slot_perimeter.receiver_tags(),
        "slot": slot_perimeter.slot_vs_perimeter(),
        "volume": pace.volume_table(),
        "sos": pd.DataFrame(schedule.sos(dvp, [])["summary"]),
    }


def _chip(text: str, tone: str, tab: str, week: bool = False) -> dict:
    """week=True marks evidence about this week's opponent only (hidden in ROS/playoff views)."""
    return {"t": text, "tone": tone, "tab": tab, **({"week": True} if week else {})}


def season_signals(df: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Adds fpoe_pg, sig_pace, sig_eff (points), skill_adj and a `chips` list."""
    df = df.copy()
    w = df["w_current"].fillna(0)
    base = df["skill_ppg"]
    chips = [[] for _ in range(len(df))]
    idx = {pid: i for i, pid in enumerate(df["player_id"])}

    # xFP is already blended into skill PPG; the chip shows how far this season ran from it.
    x = df[["player_id"]].merge(t["xfp"][["player_id", "xfp_pg", "actual_pg"]], on="player_id", how="left")
    luck = (x["actual_pg"] - x["xfp_pg"]).to_numpy()
    df["fpoe_pg"] = luck
    for i, l in enumerate(luck):
        if not np.isnan(l) and abs(l) >= 1.0:
            chips[i].append(_chip(f"{'Above' if l > 0 else 'Below'} xFP {abs(l):.1f}/G", "bad" if l > 0 else "good", "xfp"))

    # Team plays: expected ROS plays vs. what the blend assumes.
    v = t["volume"].set_index("team")
    cur, pri, exp = (df["team"].map(v[c]) for c in ("reg_plays_pg", "reg_plays_pg_prior", "expected_pg"))
    implied = w * cur + (1 - w) * pri
    factor = (exp / implied).clip(*PACE_CLIP).where(implied > 0).fillna(1.0)
    df["sig_pace"] = base * (factor - 1)
    verdict = df["team"].map(v["verdict"])
    for i, (fct, vd) in enumerate(zip(factor, verdict)):
        if abs(fct - 1) >= 0.02:
            chips[i].append(_chip(f"Plays {'+' if fct > 1 else '−'}{abs(fct - 1) * 100:.0f}% · {vd}",
                                  "good" if fct > 1 else "bad", "plays"))

    # Efficiency: separation (WR/TE), CPOE (QB), RYOE (RB).
    eff = np.zeros(len(df))
    sep = t["sep"].set_index("player_id")["sep_delta"]
    for i, (pid, pos) in enumerate(zip(df["player_id"], df["position"])):
        if pos in ("WR", "TE") and pid in sep.index and pd.notna(sep[pid]):
            d = float(sep[pid])
            eff[i] = np.clip(d * SEP_PER_YARD, -SEP_CLIP, SEP_CLIP)
            if abs(d) >= 0.5:
                chips[i].append(_chip(f"Sep {'+' if d > 0 else '−'}{abs(d):.1f} yd", "good" if d > 0 else "bad", "separation"))

    key = _norm(df["full_name"]) + "|" + df["team"].fillna("")
    cp = t["cpoe"].assign(k=_norm(t["cpoe"]["name"]) + "|" + t["cpoe"]["team"]).drop_duplicates("k").set_index("k")["cpoe"]
    ry = t["ryoe"][t["ryoe"]["attempts"] >= RYOE_MIN_ATT]
    ry = ry.assign(k=_norm(ry["name"]) + "|" + ry["team"]).drop_duplicates("k").set_index("k")["ryoe_per_att"]
    for i, (k, pos) in enumerate(zip(key, df["position"])):
        if pos == "QB" and k in cp.index:
            c = float(cp[k]); eff[i] = np.clip(c * CPOE_PER_PT, -CPOE_CLIP, CPOE_CLIP)
            if abs(c) >= 2:
                chips[i].append(_chip(f"CPOE {'+' if c > 0 else '−'}{abs(c):.1f}", "good" if c > 0 else "bad", "skill"))
        elif pos == "RB" and k in ry.index:
            r = float(ry[k]); eff[i] = np.clip(r * RYOE_PER_YD, -RYOE_CLIP, RYOE_CLIP)
            if abs(r) >= 0.3:
                chips[i].append(_chip(f"RYOE {'+' if r > 0 else '−'}{abs(r):.1f}/att", "good" if r > 0 else "bad", "skill"))
    df["sig_eff"] = base * eff

    total = df[["sig_pace", "sig_eff"]].sum(axis=1)
    cap = base.abs() * SEASON_CLIP
    df["sig_total"] = total.clip(-cap, cap)
    df["skill_adj"] = base + df["sig_total"]

    # Red zone: evidence only (already priced by xFP).
    rz = t["rz"].set_index("player_id")
    for pid, i in idx.items():
        if pid in rz.index:
            r = rz.loc[pid]
            if r["td_regression"]:
                chips[i].append(_chip(f"RZ {int(r['rz_opps'])} opps, 0 TD", "good", "redzone"))
            elif r["rz_opps"] >= 6:
                chips[i].append(_chip(f"RZ {int(r['rz_opps'])} opps", "neu", "redzone"))

    df["chips"] = chips
    return df


def schedule_signals(df: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ROS and fantasy-playoff projections from the SOS tab (WRs use the plain WR row)."""
    s = t["sos"].set_index(["team", "position"])
    keys = list(zip(df["team"], df["position"]))
    df["ros_mult"] = [s["ros"].get(k, np.nan) for k in keys]
    df["po_mult"] = [s["playoffs"].get(k, np.nan) for k in keys]
    df["ros_pts"] = df["skill_adj"] * df["ros_mult"].fillna(1.0)
    df["po_pts"] = df["skill_adj"] * df["po_mult"].fillna(1.0)
    for chips, r, p in zip(df["chips"], df["ros_mult"], df["po_mult"]):
        if pd.notna(p) and abs(p - 1) >= 0.05:
            chips.append(_chip(f"Playoff SOS {'easy' if p > 1 else 'hard'}", "good" if p > 1 else "bad", "sos"))
        elif pd.notna(r) and abs(r - 1) >= 0.05:
            chips.append(_chip(f"ROS SOS {'easy' if r > 1 else 'hard'}", "good" if r > 1 else "bad", "sos"))
    return df


def matchup_group(df: pd.DataFrame) -> pd.Series:
    """Def vs Pos row each player faces: WR1/WR2 for WRs holding that role."""
    roles = defense.wr_roles(config.SEASON).drop_duplicates("player_id", keep="last").set_index("player_id")["wr_role"]
    role = df["player_id"].map(roles)
    return pd.Series(np.where((df["position"] == "WR") & role.isin(defense.WR_ROLES), role, df["position"]),
                     index=df.index)


def matchup_signals(df: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Funnel confidence boost and slot/perimeter fit for this week's opponent.
    Expects opp, group, opp_confidence already merged; adds slot_fit and chips."""
    dvp = t["dvp"].set_index(["team", "position"])
    strong = [pd.notna(o) and (o, g) in dvp.index and dvp.loc[(o, g), "funnel_strength"] == "Strong"
              for o, g in zip(df["opp"], df["group"])]
    df["opp_confidence"] = np.where(strong, np.minimum(df["opp_confidence"] + FUNNEL_CONF_BOOST, 100),
                                    df["opp_confidence"])
    for chips, s, o, g in zip(df["chips"], strong, df["opp"], df["group"]):
        ins = dvp.loc[(o, g), "insight"] if pd.notna(o) and (o, g) in dvp.index else None
        if isinstance(ins, str) and ins:
            chips.append(_chip(f"{o}: {ins}", "good" if dvp.loc[(o, g), "insight_tone"] == "up"
                               else "bad" if dvp.loc[(o, g), "insight_tone"] == "down" else "neu", "defense", week=True))

    tag = df["player_id"].map(t["tags"].set_index("player_id")["tag"])
    tend = df["opp"].map(t["slot"].set_index("team")["tendency"])
    fit = np.select(
        [((tag == "Slot") & (tend == "Slot-vulnerable")) | ((tag == "Perimeter") & (tend == "Perimeter-vulnerable")),
         ((tag == "Slot") & (tend == "Perimeter-vulnerable")) | ((tag == "Perimeter") & (tend == "Slot-vulnerable"))],
        [1 + SLOT_FIT, 1 - SLOT_FIT], 1.0)
    df["slot_fit"] = fit
    for chips, f, tg in zip(df["chips"], fit, tag):
        if f != 1.0:
            chips.append(_chip(f"{tg} {'vs weak spot' if f > 1 else 'vs strength'}", "good" if f > 1 else "bad", "slot", week=True))
    return df
