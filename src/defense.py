"""Defense vs. position: fantasy points allowed, ranks, and confidence scores.

Rank 1 = stingiest, 32 = most generous. Confidence (0-100) is a *weighted
average* of three factors; the original multiplicative version crushed every
small-sample score under ~33 and should not come back.
"""
from __future__ import annotations

import pandas as pd

import config
from src import fetch, scoring

POSITIONS = scoring.SKILL_POSITIONS


def _team_games(year: int) -> pd.DataFrame:
    pbp = fetch.load_pbp(year)
    return (pbp.dropna(subset=["defteam"]).groupby("defteam")["game_id"].nunique()
               .rename("games").reset_index().rename(columns={"defteam": "team"}))


def fp_allowed_by_game(year: int) -> pd.DataFrame:
    """FP allowed per defense per game per position (zeros filled in)."""
    c = scoring.credits_with_position(year)
    c = c[c["position"].isin(POSITIONS)]
    g = c.groupby(["defteam", "game_id", "position"])["fp"].sum()
    games = fetch.load_pbp(year).dropna(subset=["defteam"])[["defteam", "game_id"]].drop_duplicates()
    idx = pd.MultiIndex.from_frame(
        games.merge(pd.DataFrame({"position": POSITIONS}), how="cross")[["defteam", "game_id", "position"]])
    return g.reindex(idx, fill_value=0.0).rename("fp").reset_index()


def fp_allowed_per_game(year: int) -> pd.DataFrame:
    """team, position, fp_pg, games"""
    g = fp_allowed_by_game(year)
    out = (g.groupby(["defteam", "position"])
            .agg(fp_pg=("fp", "mean"), games=("game_id", "nunique"), fp_total=("fp", "sum"))
            .reset_index().rename(columns={"defteam": "team"}))
    return out


def _baseline() -> pd.DataFrame:
    """3-year recency-weighted FP allowed per game (50/30/20), renormalised if a season is missing."""
    frames = []
    for yr, w in zip(config.HISTORY_SEASONS, config.HISTORY_WEIGHTS):
        f = fp_allowed_per_game(yr)[["team", "position", "fp_pg"]]
        frames.append(f.assign(w=w))
    h = pd.concat(frames)
    h["wx"] = h["fp_pg"] * h["w"]
    b = h.groupby(["team", "position"]).agg(wx=("wx", "sum"), w=("w", "sum")).reset_index()
    b["baseline_fp_pg"] = b["wx"] / b["w"]
    return b[["team", "position", "baseline_fp_pg"]]


def _elite_ratio(year: int) -> pd.DataFrame:
    """Share of each defense's FP allowed that went to top-12 players at the position."""
    c = scoring.credits_with_position(year)
    c = c[c["position"].isin(POSITIONS)]
    totals = c.groupby(["position", "player_id"])["fp"].sum().reset_index()
    top12 = (totals.sort_values("fp", ascending=False).groupby("position").head(12)["player_id"])
    c = c.assign(elite=c["player_id"].isin(set(top12)))
    agg = c.groupby(["defteam", "position"]).apply(
        lambda d: pd.Series({"elite_fp": d.loc[d.elite, "fp"].sum(), "total_fp": d["fp"].sum()}),
        include_groups=False).reset_index()
    agg["elite_ratio"] = (agg["elite_fp"] / agg["total_fp"].where(agg["total_fp"] > 0)).fillna(0).clip(0, 1)
    return agg.rename(columns={"defteam": "team"})[["team", "position", "elite_ratio"]]


def confidence(games: float, team_baseline: float, league_baseline: float, elite_ratio: float) -> float:
    sample_factor = min(0.4 + games * 0.25, 1.0)
    gap = abs(team_baseline - league_baseline) / league_baseline if league_baseline else 0.0
    gap_factor = min(0.4 + gap * 1.5, 1.0)
    conc_factor = max(1.0 - elite_ratio * 0.7, 0.3)
    return 100 * (0.25 * sample_factor + 0.45 * gap_factor + 0.30 * conc_factor)


def defense_vs_position(year: int = config.SEASON) -> pd.DataFrame:
    """One row per (team, position): current FP/G allowed, rank, baseline, confidence."""
    cur = fp_allowed_per_game(year)
    base = _baseline()
    df = base.merge(cur, on=["team", "position"], how="left")
    df["games"] = df["games"].fillna(0).astype(int)
    # Preseason / no games yet: fall back to the baseline so ranks still exist.
    df["fp_pg"] = df["fp_pg"].fillna(df["baseline_fp_pg"])
    df = df.merge(_elite_ratio(year), on=["team", "position"], how="left")
    df["elite_ratio"] = df["elite_ratio"].fillna(0)

    league_base = df.groupby("position")["baseline_fp_pg"].transform("mean")
    df["league_fp_pg"] = df.groupby("position")["fp_pg"].transform("mean")
    df["confidence"] = [
        round(confidence(g, tb, lb, er), 1)
        for g, tb, lb, er in zip(df["games"], df["baseline_fp_pg"], league_base, df["elite_ratio"])
    ]
    df["rank"] = df.groupby("position")["fp_pg"].rank(method="min").astype(int)
    return df.sort_values(["position", "rank"]).reset_index(drop=True)


def to_json(df: pd.DataFrame) -> list[dict]:
    cols = ["team", "position", "fp_pg", "rank", "games", "baseline_fp_pg", "league_fp_pg",
            "elite_ratio", "confidence"]
    return df[cols].round(2).to_dict("records")
