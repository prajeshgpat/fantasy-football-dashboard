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
from src import (defense, expected_points, fetch, pace, rankings, rb_efficiency, redzone, schedule, scoring, signals,
                 separation, skill_metrics, slot_perimeter)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "templates" / "dashboard.html"

PLUS_RANK = 23     # opponent rank >= this (generous) -> Plus matchup
TOUGH_RANK = 10    # opponent rank <= this (stingy)   -> Tough matchup
LOW_CONFIDENCE = 45


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


class MatchupRows:
    """Builds My Lineup-style matchup rows for any rostered player, so My Lineup
    and Start/Sit show exactly the same columns and reads."""

    def __init__(self, dvp: pd.DataFrame, slot: pd.DataFrame, ranks: pd.DataFrame,
                 tags: pd.DataFrame | None = None, proj_all: pd.DataFrame | None = None):
        ros = fetch.load_roster(config.SEASON)
        ros = ros.assign(status_order=(ros["status"] != "ACT").astype(int),
                         name_key=ros["full_name"].fillna("").map(_name_key))
        self.ros = ros.sort_values(["status_order", "week"], ascending=[True, False])
        self.opp = rankings.next_opponents().set_index("team")
        self.tend = slot.set_index("team")["tendency"]
        self.tag_by_id = {} if tags is None else tags[tags["tag"] != ""].set_index("player_id")["tag"].to_dict()
        # WRs are matched against the opponent's WR1 or WR2 row when the player holds that role.
        self.roles = defense.wr_roles(config.SEASON).set_index(["team", "player_id"])["wr_role"]
        self.dvp_ix = dvp.set_index(["team", "position"])
        self.rank = ranks.set_index("player_id")["rank"]
        src = proj_all if proj_all is not None else ranks
        self.proj = src.set_index("player_id")["proj_pts"]

    def by_name(self, name: str) -> dict:
        m = self.ros[self.ros["name_key"] == _name_key(name)]
        if m.empty:
            return {"name": name, "read": "Not found in roster file"}
        return self.row(m.iloc[0])

    def by_id(self, gsis_id: str) -> dict | None:
        m = self.ros[self.ros["gsis_id"] == gsis_id]
        return None if m.empty else self.row(m.iloc[0])

    def row(self, p: pd.Series) -> dict:
        pos = "RB" if p["position"] == "FB" else p["position"]
        row = {"player_id": p["gsis_id"], "name": p["full_name"], "position": pos,
               "team": p["team"], "status": p["status"]}
        if p["team"] in self.opp.index:
            o = self.opp.loc[p["team"]]
            row.update(opp=o["opp"], week=int(o["week"]), home=bool(o["home"]), bye=bool(o["bye"]))
            group = pos
            if pos == "WR":
                role = self.roles.get((p["team"], p["gsis_id"]))
                if role in defense.WR_ROLES and (o["opp"], role) in self.dvp_ix.index:
                    group = role
            row["matchup_group"] = group
            if (o["opp"], group) in self.dvp_ix.index:
                d = self.dvp_ix.loc[(o["opp"], group)]
                row.update(opp_rank=int(d["rank"]), opp_confidence=float(d["confidence"]),
                           opp_fp_pg=float(d["fp_pg"]), insight=d["insight"] or None,
                           insight_tone=d["insight_tone"] or None,
                           insight_effect=d["insight_effect"] or None,
                           insight_detail=d["insight_detail"] or None)
                if pos in ("WR", "TE"):
                    row["opp_slot"] = self.tend.get(o["opp"])
                    tag = self.tag_by_id.get(p["gsis_id"])
                    if tag:
                        row["player_tag"] = tag
                        t = row["opp_slot"] or ""
                        row["slot_fit"] = ("Slot" in t and tag == "Slot") or ("Perimeter" in t and tag == "Perimeter")
                read = ("Plus matchup" if d["rank"] >= PLUS_RANK
                        else "Tough matchup" if d["rank"] <= TOUGH_RANK else "Neutral")
                if d["confidence"] < LOW_CONFIDENCE:
                    read += " (low confidence)"
                row["read"] = read
        if p["gsis_id"] in self.rank.index:
            row["ovr_rank"] = int(self.rank[p["gsis_id"]])
        if p["gsis_id"] in self.proj.index:
            row["proj_pts"] = float(self.proj[p["gsis_id"]])
        return row


def my_lineup(rows: MatchupRows) -> list[dict]:
    return [rows.by_name(n) for n in config.MY_ROSTER]


def start_sit_pool(rows: MatchupRows, proj_all: pd.DataFrame) -> list[dict]:
    """Matchup rows for every projectable player (active; QBs = team starters), plus
    your own roster even if they fall outside that pool (e.g. on IR)."""
    ids = list(proj_all["player_id"])
    mine = [r.get("player_id") for r in my_lineup(rows)]
    out, seen = [], set()
    for pid in ids + [m for m in mine if m]:
        if pid in seen:
            continue
        seen.add(pid)
        r = rows.by_id(pid)
        if r:
            out.append(r)
    return out


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

    step("tab tables"); t = signals.tables()
    dvp, slot, sep, xfp, rz = t["dvp"], t["slot"], t["sep"], t["xfp"], t["rz"]
    cp, ry, volume, tags = t["cpoe"], t["ryoe"], t["volume"], t["tags"]
    step("projections"); proj_all, gaps = rankings.project_players(dvp, t)
    ranks, _ = rankings.build_rankings(proj=(proj_all, gaps))
    step("lineup + start/sit"); mrows = MatchupRows(dvp, slot, ranks, tags, proj_all)
    lineup = my_lineup(mrows)
    start_sit = start_sit_pool(mrows, proj_all)
    step("strength of schedule"); sos = schedule.sos(dvp, lineup)

    pbp = fetch.load_pbp(config.SEASON)
    opp = rankings.next_opponents()
    slot_faces = slot_perimeter.opponent_receivers(slot, tags, opp)
    meta = {
        "season": config.SEASON,
        "prior_season": config.PRIOR_SEASON,
        "history_seasons": config.HISTORY_SEASONS,
        "ranking_weights": config.RANKING_WEIGHTS,
        "xfp_ppg_weight": config.XFP_PPG_WEIGHT,
        "through_week": int(pbp["week"].max()) if len(pbp) else 0,
        "next_week": int(opp["week"].min()) if len(opp) else None,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "league_size": config.LEAGUE_SIZE,
        "scoring": config.SCORING,
        "tprr_true_available": sep["tprr"].notna().any(),
        "min_routes": separation.MIN_ROUTES,
        "low_confidence": LOW_CONFIDENCE,
        "refresh_trigger_id": getattr(config, "REFRESH_TRIGGER_ID", None),
    }
    return {
        "META": meta,
        "TEAMS_DATA": {t: {"name": config.TEAM_NAMES[t], "color": config.TEAM_COLORS[t]}
                       for t in config.TEAM_NAMES},
        "VOLUME_DATA": pace.to_json(volume),
        "SOS_DATA": sos,
        "START_SIT_DATA": start_sit,
        "MY_ROSTER_IDS": [r.get("player_id") for r in lineup if r.get("player_id")],
        "RECEIVER_TAGS": tags[tags["tag"] != ""][["player_id", "name", "position", "team", "tgts", "tgts_cur",
                                                  "middle_share", "tag"]].round(3).to_dict("records"),
        "SLOT_FACES": slot_faces,
        "SEPARATION_DATA": separation.to_json(sep),
        "DEFENSE_DATA": defense_rows(dvp),
        "SLOT_DATA": slot_perimeter.to_json(slot),
        "LINEUP_DATA": lineup,
        "XFP_DATA": expected_points.to_json(xfp),
        "REDZONE_DATA": redzone.to_json(rz),
        "CPOE_DATA": skill_metrics.to_json(cp),
        "RYOE_DATA": skill_metrics.to_json(ry),
        "RB_EFF_DATA": rb_efficiency.to_json(),
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
