"""Download and cache nflverse release assets under data/.

Run directly (`python -m src.fetch`) to warm the cache; every other module goes
through the `load_*` helpers, which download on a cache miss.
"""
from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

import config

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
    "play_type", "yardline_100", "pass_location", "air_yards",
    "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name",
    "pass_attempt", "rush_attempt", "complete_pass", "sack", "qb_dropback", "qb_scramble",
    "passing_yards", "rushing_yards", "receiving_yards",
    "pass_touchdown", "rush_touchdown", "interception",
    "fumble_lost", "fumbled_1_player_id", "wp",
    "qtr", "drive", "game_seconds_remaining", "score_differential",
]


# NGS uses a few legacy abbreviations; pbp/rosters/schedules use these.
TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS"}


class NotPublished(Exception):
    """The asset does not exist upstream (e.g. participation for an in-progress season)."""


def _download(tag: str, filename: str, refresh: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / filename
    if dest.exists() and not refresh:
        return dest
    url = f"{BASE_URL}/{tag}/{filename}"
    print(f"  downloading {url}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404:
            raise NotPublished(f"{tag}/{filename} is not published") from e
        raise
    tmp.replace(dest)
    return dest


def _cached(name: str, build, required: list[str] | None = None) -> pd.DataFrame:
    """Pickle-cache a parsed frame so repeated builds skip the slow CSV parse.
    A cache missing any `required` column (written by an older version) is rebuilt."""
    pkl = DATA_DIR / f"{name}.pkl"
    if pkl.exists():
        df = pd.read_pickle(pkl)
        if not required or set(required) <= set(df.columns):
            return df
    df = build()
    df.to_pickle(pkl)
    return df


def load_pbp(year: int) -> pd.DataFrame:
    """Regular-season play-by-play, trimmed to the columns the pipeline uses."""
    def build():
        path = _download("pbp", f"play_by_play_{year}.csv.gz")
        df = pd.read_csv(path, usecols=lambda c: c in PBP_COLUMNS, low_memory=False)
        return df[df["season_type"] == "REG"].reset_index(drop=True)
    return _cached(f"pbp_{year}", build, required=PBP_COLUMNS)


def load_roster(year: int) -> pd.DataFrame:
    return pd.read_parquet(_download("rosters", f"roster_{year}.parquet"))


def load_schedule() -> pd.DataFrame:
    return pd.read_csv(_download("schedules", "games.csv"), low_memory=False)


def load_ngs(kind: str) -> pd.DataFrame:
    """kind: receiving | rushing | passing"""
    return pd.read_csv(_download("nextgen_stats", f"ngs_{kind}.csv.gz"), low_memory=False)


def load_snap_counts(year: int) -> pd.DataFrame:
    return pd.read_parquet(_download("snap_counts", f"snap_counts_{year}.parquet"))


def load_participation(year: int) -> pd.DataFrame | None:
    """Returns None when nflverse has not published it (always the case in-season)."""
    try:
        path = _download("pbp_participation", f"pbp_participation_{year}.parquet")
    except NotPublished:
        return None
    return pd.read_parquet(path, columns=["nflverse_game_id", "play_id", "offense_players"])


def refresh_current(year: int) -> None:
    """Drop cached current-season files so the weekly refresh picks up new games."""
    for pattern in (f"play_by_play_{year}.csv.gz", f"pbp_{year}.pkl", f"roster_{year}.parquet",
                    f"snap_counts_{year}.parquet", "games.csv", "ngs_*.csv.gz"):
        for p in DATA_DIR.glob(pattern):
            p.unlink()


def main() -> None:
    refresh_current(config.SEASON)
    years = sorted({config.SEASON, *config.HISTORY_SEASONS})
    for y in years:
        print(f"pbp {y}: {len(load_pbp(y)):,} plays", file=sys.stderr)
        print(f"roster {y}: {len(load_roster(y)):,} rows", file=sys.stderr)
    for y in (config.SEASON, config.PRIOR_SEASON):
        print(f"snap counts {y}: {len(load_snap_counts(y)):,} rows", file=sys.stderr)
    for y in (config.SEASON, config.PRIOR_SEASON):
        part = load_participation(y)
        print(f"participation {y}: " + ("not published (in-season)" if part is None else f"{len(part):,} plays"),
              file=sys.stderr)
    print(f"schedule: {len(load_schedule()):,} games", file=sys.stderr)
    for kind in ("receiving", "rushing", "passing"):
        print(f"ngs {kind}: {len(load_ngs(kind)):,} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
