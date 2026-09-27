"""CPOE and RYOE: straight passthrough of NGS season-to-date (week == 0) rows."""
from __future__ import annotations

import pandas as pd

import config
from src import fetch


def _season(kind: str, year: int) -> pd.DataFrame:
    n = fetch.load_ngs(kind)
    n = n[(n["season"] == year) & (n["week"] == 0) & (n["season_type"] == "REG")]
    return n.assign(team_abbr=n["team_abbr"].replace(fetch.TEAM_ALIASES))


def cpoe(year: int = config.SEASON) -> pd.DataFrame:
    cols = {"player_display_name": "name", "team_abbr": "team", "attempts": "attempts",
            "completion_percentage_above_expectation": "cpoe",
            "avg_intended_air_yards": "adot", "avg_time_to_throw": "ttt",
            "completion_percentage": "comp_pct", "expected_completion_percentage": "xcomp_pct"}
    return _season("passing", year)[list(cols)].rename(columns=cols).sort_values("cpoe", ascending=False)


def ryoe(year: int = config.SEASON) -> pd.DataFrame:
    cols = {"player_display_name": "name", "team_abbr": "team", "rush_attempts": "attempts",
            "rush_yards_over_expected": "ryoe", "rush_yards_over_expected_per_att": "ryoe_per_att",
            "percent_attempts_gte_eight_defenders": "stacked_box_pct", "avg_rush_yards": "ypc",
            "efficiency": "efficiency"}
    return _season("rushing", year)[list(cols)].rename(columns=cols).sort_values("ryoe_per_att", ascending=False)


def to_json(df: pd.DataFrame) -> list[dict]:
    return df.round(2).to_dict("records")
