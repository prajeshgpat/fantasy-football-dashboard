"""RB rushing efficiency vs. fantasy output: yards per carry against PPR points per game."""
from __future__ import annotations

import pandas as pd

import config
from src import scoring

MIN_CARRIES = 10           # season total ...
MIN_CARRIES_PG = 5         # ... and per game, so one-off cameos stay off the chart


def rb_efficiency(year: int) -> pd.DataFrame:
    s = scoring.player_seasons(year)
    s = s[(s["position"] == "RB") & (s["rush_att"] >= MIN_CARRIES)
          & (s["rush_att"] / s["games"] >= MIN_CARRIES_PG)].copy()
    s["ypc"] = s["rush_yds"] / s["rush_att"]
    s["carries_pg"] = s["rush_att"] / s["games"]
    s["targets_pg"] = s["targets"] / s["games"]
    s["rush_fp_pg"] = s["rush_fp"] / s["games"]
    s["rec_fp_pg"] = s["rec_fp"] / s["games"]
    return s.sort_values("ppg", ascending=False).reset_index(drop=True)


def to_json(years: list[int] | None = None) -> dict:
    years = years or [config.SEASON, config.PRIOR_SEASON]
    cols = ["player_id", "full_name", "team", "games", "rush_att", "rush_yds", "rush_td", "ypc",
            "carries_pg", "targets_pg", "rush_fp_pg", "rec_fp_pg", "ppg"]
    out = {}
    for y in years:
        df = rb_efficiency(y)[cols].round(2)
        out[str(y)] = df.astype(object).where(df.notna(), None).to_dict("records")
    return {"seasons": [str(y) for y in years], "min_carries": MIN_CARRIES,
            "min_carries_pg": MIN_CARRIES_PG, "rows": out}
