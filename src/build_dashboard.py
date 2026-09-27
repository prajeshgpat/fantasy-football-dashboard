"""Compute every dashboard section and inject it into templates/dashboard.html.

Each {{TOKEN}} in the template is replaced with json.dumps(...) of a dataset.
The output is checked for balanced <script>/<table> tags before it is written,
because string-replacement injection can silently corrupt the file otherwise.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

import config
from src import (defense, expected_points, fetch, rankings, redzone, scoring,
                 separation, skill_metrics, slot_perimeter)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "templates" / "dashboard.html"

PLUS_RANK = 23     # opponent rank >= this (generous) -> Plus matchup
TOUGH_RANK = 10    # opponent rank <= this (stingy)   -> Tough matchup
LOW_CONFIDENCE = 45


def plays_per_game() -> list[dict]:
    def ppg(year):
        pbp = fetch.load_pbp(year)
        p = pbp[pbp["play_type"].isin(["pass", "run"])]
        return p.groupby(["posteam", "game_id"]).size().groupby("posteam").mean()
    df = pd.DataFrame({"current": ppg(config.SEASON), "prior": ppg(config.PRIOR_SEASON)})
    df["delta"] = df["current"] - df["prior"]
    return df.rename_axis("team").reset_index().round(2).to_dict("records")


def _name_key(name: str) -> str:
    """Case/punctuation/spacing-insensitive key, so "R J Harvey" matches "RJ Harvey"."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def defense_rows(dvp: pd.DataFrame) -> list[dict]:
    """Defense table rows plus who each defense faces next and that offense's key player."""
    rows = defense.to_json(dvp)
    opp = rankings.next_opponents().set_index("team")
    key = defense.key_players().set_index(["team", "position"])["player"]
    for r in rows:
        if r["team"] not in opp.index:
            continue
        o = opp.loc[r["team"]]
        r.update(faces=o["opp"], faces_week=int(o["week"]), faces_home=bool(o["home"]),
                 faces_bye=bool(o["bye"]), faces_player=key.get((o["opp"], r["position"])))
    return rows


def my_lineup(dvp: pd.DataFrame, slot: pd.DataFrame, ranks: pd.DataFrame) -> list[dict]:
    ros = fetch.load_roster(config.SEASON)
    ros = ros.assign(status_order=(ros["status"] != "ACT").astype(int),
                     name_key=ros["full_name"].fillna("").map(_name_key))
    ros = ros.sort_values(["status_order", "week"], ascending=[True, False])
    opp = rankings.next_opponents().set_index("team")
    tend = slot.set_index("team")["tendency"]
    # WRs are matched against the opponent's WR1 or WR2 row when the player holds that role.
    roles = defense.wr_roles(config.SEASON).set_index(["team", "player_id"])["wr_role"]
    dvp_ix = dvp.set_index(["team", "position"])
    proj = ranks.set_index("player_id")[["rank", "proj_pts"]]
    rows = []
    for name in config.MY_ROSTER:
        m = ros[ros["name_key"] == _name_key(name)]
        if m.empty:
            rows.append({"name": name, "read": "Not found in roster file"})
            continue
        p = m.iloc[0]
        pos = "RB" if p["position"] == "FB" else p["position"]
        row = {"name": p["full_name"], "position": pos, "team": p["team"], "status": p["status"]}
        if p["team"] in opp.index:
            o = opp.loc[p["team"]]
            row.update(opp=o["opp"], week=int(o["week"]), home=bool(o["home"]), bye=bool(o["bye"]))
            group = pos
            if pos == "WR":
                role = roles.get((p["team"], p["gsis_id"]))
                if role in defense.WR_ROLES and (o["opp"], role) in dvp_ix.index:
                    group = role
            row["matchup_group"] = group
            if (o["opp"], group) in dvp_ix.index:
                d = dvp_ix.loc[(o["opp"], group)]
                row.update(opp_rank=int(d["rank"]), opp_confidence=float(d["confidence"]),
                           opp_fp_pg=float(d["fp_pg"]), insight=d["insight"] or None,
                           insight_tone=d["insight_tone"] or None,
                           insight_effect=d["insight_effect"] or None,
                           insight_detail=d["insight_detail"] or None)
                if pos in ("WR", "TE"):
                    row["opp_slot"] = tend.get(o["opp"])
                read = ("Plus matchup" if d["rank"] >= PLUS_RANK
                        else "Tough matchup" if d["rank"] <= TOUGH_RANK else "Neutral")
                if d["confidence"] < LOW_CONFIDENCE:
                    read += " (low confidence)"
                row["read"] = read
        if p["gsis_id"] in proj.index:
            row.update(ovr_rank=int(proj.loc[p["gsis_id"], "rank"]),
                       proj_pts=float(proj.loc[p["gsis_id"], "proj_pts"]))
        rows.append(row)
    return rows


def _clean(obj):
    """NaN/inf -> null so the output is valid JSON as well as valid JS."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):          # numpy scalars
        return _clean(obj.item())
    return obj


def to_js(obj) -> str:
    # Escape "</" so a string value can never close the surrounding <script> tag.
    return json.dumps(_clean(obj), separators=(",", ":")).replace("</", "<\\/")


def check_balanced(html: str) -> None:
    for tag in ("script", "table"):
        opens = len(re.findall(rf"<{tag}[\s>]", html, re.I))
        closes = len(re.findall(rf"</{tag}>", html, re.I))
        if opens != closes:
            raise RuntimeError(f"unbalanced <{tag}>: {opens} open vs {closes} close")
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", html)
    if leftover:
        raise RuntimeError(f"unreplaced tokens: {sorted(set(leftover))}")


def compute() -> dict[str, object]:
    def step(label):
        print(f"  {label}", file=sys.stderr)

    step("defense vs position"); dvp = defense.defense_vs_position()
    step("slot vs perimeter"); slot = slot_perimeter.slot_vs_perimeter()
    step("rankings"); ranks, gaps = rankings.build_rankings(dvp)
    step("separation"); sep = separation.separation_table()
    step("expected points"); xfp = expected_points.expected_vs_actual()
    step("red zone"); rz = redzone.red_zone()
    step("skill metrics"); cp, ry = skill_metrics.cpoe(), skill_metrics.ryoe()
    step("plays per game"); plays = plays_per_game()
    step("lineup"); lineup = my_lineup(dvp, slot, ranks)

    pbp = fetch.load_pbp(config.SEASON)
    opp = rankings.next_opponents()
    meta = {
        "season": config.SEASON,
        "prior_season": config.PRIOR_SEASON,
        "history_seasons": config.HISTORY_SEASONS,
        "through_week": int(pbp["week"].max()) if len(pbp) else 0,
        "next_week": int(opp["week"].min()) if len(opp) else None,
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "league_size": config.LEAGUE_SIZE,
        "scoring": config.SCORING,
        "tprr_true_available": sep["tprr"].notna().any(),
        "min_routes": separation.MIN_ROUTES,
        "low_confidence": LOW_CONFIDENCE,
    }
    return {
        "META": meta,
        "TEAMS_DATA": {t: {"name": config.TEAM_NAMES[t], "color": config.TEAM_COLORS[t]}
                       for t in config.TEAM_NAMES},
        "PLAYS_DATA": plays,
        "SEPARATION_DATA": separation.to_json(sep),
        "DEFENSE_DATA": defense_rows(dvp),
        "SLOT_DATA": slot_perimeter.to_json(slot),
        "LINEUP_DATA": lineup,
        "XFP_DATA": expected_points.to_json(xfp),
        "REDZONE_DATA": redzone.to_json(rz),
        "CPOE_DATA": skill_metrics.to_json(cp),
        "RYOE_DATA": skill_metrics.to_json(ry),
        "RANKINGS_DATA": rankings.to_json(ranks),
        "GAPS_DATA": rankings.gaps_to_json(gaps),
    }


def to_fragment(html: str) -> str:
    """Strip the document wrapper (doctype/html/head/body) for hosts that supply
    their own skeleton, keeping <title>, <style> and <script> tags in order."""
    html = re.sub(r"<!doctype[^>]*>|</?html[^>]*>|</?head>|</?body>|<meta [^>]*>", "", html, flags=re.I)
    return html.strip() + "\n"


def build(output: Path | None = None, fragment: bool = False) -> Path:
    output = output or ROOT / config.OUTPUT_HTML
    html = TEMPLATE.read_text(encoding="utf-8")
    for token, data in compute().items():
        html = html.replace("{{" + token + "}}", to_js(data))
    check_balanced(html)
    if fragment:
        html = to_fragment(html)
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
    return output


if __name__ == "__main__":
    # usage: python -m src.build_dashboard [--fragment] [output.html]
    args = sys.argv[1:]
    frag = "--fragment" in args
    paths = [a for a in args if a != "--fragment"]
    build(Path(paths[0]) if paths else None, fragment=frag)
