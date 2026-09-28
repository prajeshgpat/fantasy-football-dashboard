"""Strength of schedule: every team's opponent for weeks 1-18 and how hard each
matchup is at each position, plus the same view for the configured roster.

Future matchups use a defense strength that blends this season's FP allowed with
the 3-year baseline (the baseline counts as BLEND_GAMES games), so a three-game
sample doesn't dictate a week-16 projection.
"""
from __future__ import annotations

import pandas as pd

import config
from src import defense, fetch

BLEND_GAMES = 4


def season_grid(year: int = config.SEASON) -> pd.DataFrame:
    """team, week, opp (None on bye), home, played"""
    g = fetch.load_schedule()
    g = g[(g["season"] == year) & (g["game_type"] == "REG")]
    long = pd.concat([
        g.assign(team=g["home_team"], opp=g["away_team"], home=True),
        g.assign(team=g["away_team"], opp=g["home_team"], home=False),
    ])[["team", "week", "opp", "home", "result"]]
    long["played"] = long["result"].notna()
    weeks = range(1, int(g["week"].max()) + 1)
    full = pd.MultiIndex.from_product([sorted(long["team"].unique()), weeks], names=["team", "week"])
    out = long.drop(columns="result").set_index(["team", "week"]).reindex(full).reset_index()
    out["played"] = out["played"].fillna(False).astype(bool)   # bye rows
    return out


def defense_strength(dvp: pd.DataFrame) -> pd.DataFrame:
    """team, position -> blended FP/G allowed, multiplier vs league (>1 = easier), rank (1 = stingiest)."""
    d = dvp[["team", "position", "fp_pg", "baseline_fp_pg", "games"]].copy()
    d["blend_fp_pg"] = (d["games"] * d["fp_pg"] + BLEND_GAMES * d["baseline_fp_pg"]) / (d["games"] + BLEND_GAMES)
    d["mult"] = d["blend_fp_pg"] / d.groupby("position")["blend_fp_pg"].transform("mean")
    d["rank"] = d.groupby("position")["blend_fp_pg"].rank(method="min").astype(int)
    return d


def _summaries(cells: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average multiplier over played / remaining / playoff weeks (byes excluded)."""
    c = cells.dropna(subset=["mult"]).assign(played=lambda d: d["played"].astype(bool))
    po = c["week"].isin(config.FANTASY_PLAYOFF_WEEKS)
    agg = lambda mask, name: c[mask].groupby(keys)["mult"].mean().rename(name)
    return pd.concat([agg(c["played"], "past"), agg(~c["played"], "ros"), agg(po & ~c["played"], "playoffs"),
                      c[~c["played"]].groupby(keys).size().rename("ros_games")], axis=1).reset_index()


def sos(dvp: pd.DataFrame, lineup_groups: list[dict]) -> dict:
    grid = season_grid()
    st = defense_strength(dvp)
    cells = grid.merge(st.rename(columns={"team": "opp"})[["opp", "position", "mult", "rank"]], on="opp", how="left")
    summ = _summaries(cells, ["team", "position"])
    # Rank 1 = easiest schedule
    for col in ("past", "ros", "playoffs"):
        summ[col + "_rank"] = summ.groupby("position")[col].rank(ascending=False, method="min")

    roster_rows = []
    for p in lineup_groups:
        if not p.get("team") or not p.get("matchup_group"):
            continue
        pc = cells[(cells["team"] == p["team"]) & (cells["position"] == p["matchup_group"])]
        by_week = grid[grid["team"] == p["team"]].merge(
            pc[["week", "mult", "rank"]], on="week", how="left").sort_values("week")
        s = _summaries(pc.assign(k=1), ["k"])
        s = s.iloc[0].to_dict() if len(s) else {}
        roster_rows.append({
            "name": p["name"], "position": p["position"], "group": p["matchup_group"], "team": p["team"],
            "past": s.get("past"), "ros": s.get("ros"), "playoffs": s.get("playoffs"),
            "weeks": [{"week": int(r.week), "opp": r.opp if pd.notna(r.opp) else None,
                       "home": bool(r.home) if pd.notna(r.home) else None,
                       "played": bool(r.played) if pd.notna(r.played) else False,
                       "rank": int(r["rank"]) if pd.notna(r["rank"]) else None,
                       "mult": round(float(r.mult), 3) if pd.notna(r.mult) else None}
                      for _, r in by_week.iterrows()],
        })

    sched = {}
    for team, g in grid.groupby("team"):
        sched[team] = [{"week": int(r.week), "opp": r.opp if pd.notna(r.opp) else None,
                        "home": bool(r.home) if pd.notna(r.home) else None,
                        "played": bool(r.played) if pd.notna(r.played) else False} for _, r in g.iterrows()]
    strength = {pos: {r.team: {"rank": int(r["rank"]), "mult": round(float(r.mult), 3)}
                      for _, r in grp.iterrows()} for pos, grp in st.groupby("position")}
    summary = summ.round(3)
    return {
        "weeks": sorted(int(w) for w in grid["week"].unique()),
        "playoff_weeks": config.FANTASY_PLAYOFF_WEEKS,
        "blend_games": BLEND_GAMES,
        "schedule": sched,
        "strength": strength,
        "summary": summary.astype(object).where(summary.notna(), None).to_dict("records"),
        "roster": roster_rows,
    }
