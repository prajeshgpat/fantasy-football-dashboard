"""Defense vs. position: fantasy points allowed, ranks, confidence scores, and
funnel insights.

Rank 1 = stingiest, 32 = most generous. Confidence (0-100) is a *weighted
average* of three factors; the original multiplicative version crushed every
small-sample score under ~33 and should not come back.

WRs are also split into WR1 / WR2 by each offense's season target ranking, so a
defense that erases No. 1 receivers but leaks to No. 2s shows up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import fetch, scoring

POSITIONS = scoring.SKILL_POSITIONS
WR_ROLES = ["WR1", "WR2"]
GROUPS = ["QB", "RB", "WR", "WR1", "WR2", "TE"]

NEUTRAL_WP = (0.20, 0.80)   # "neutral script" plays for pass-rate comparisons
FUNNEL_GAP = 0.25           # tough side must allow >=25% fewer FP (vs league) than the soft side
TOUGH_RANK = 12             # ...and rank in the stingiest 12 at that position
SOFT_RANK = 17              # soft side must rank in the more generous half
BEHAVIOR_SIGNAL = 0.03      # 3-point shift in pass rate / TE target share counts as confirmation


def wr_roles(year: int) -> pd.DataFrame:
    """player_id, team -> WR1 / WR2 / WR3+ by the player's targets for that team that season."""
    c = scoring.credits_with_position(year)
    t = c[(c["position"] == "WR") & (c["role"] == "target")]
    t = t.groupby(["posteam", "player_id"])["targets"].sum().reset_index()
    t = t.sort_values(["posteam", "targets"], ascending=[True, False])
    t["n"] = t.groupby("posteam").cumcount() + 1
    t["wr_role"] = np.where(t["n"] <= 2, "WR" + t["n"].astype(str), "WR3+")
    return t.rename(columns={"posteam": "team"})[["team", "player_id", "wr_role", "targets"]]


def key_players(year: int = config.SEASON) -> pd.DataFrame:
    """team, position group -> the offense's main player there this season (by volume):
    QB by dropbacks, RB by carries + targets, TE by targets, WR1/WR2 by target rank
    (the combined WR row shows the WR1)."""
    c = scoring.credits_with_position(year)
    c = c[c["position"].isin(POSITIONS)]
    vol = c.groupby(["posteam", "player_id", "full_name", "position"])[["dropbacks", "rush_att", "targets"]].sum()
    vol = vol.reset_index()
    vol["volume"] = np.select([vol["position"] == "QB", vol["position"] == "RB"],
                              [vol["dropbacks"], vol["rush_att"] + vol["targets"]], vol["targets"])
    top = (vol[vol["position"] != "WR"].sort_values("volume", ascending=False)
              .drop_duplicates(["posteam", "position"])[["posteam", "position", "full_name"]])
    wr = wr_roles(year).rename(columns={"team": "posteam"})
    wr = wr[wr["wr_role"].isin(WR_ROLES)].merge(
        vol[["posteam", "player_id", "full_name"]], on=["posteam", "player_id"])
    wr = wr.rename(columns={"wr_role": "position"})[["posteam", "position", "full_name"]]
    combined = wr[wr["position"] == "WR1"].assign(position="WR")
    out = pd.concat([top, wr, combined], ignore_index=True)
    return out.rename(columns={"posteam": "team", "full_name": "player"})


def _grouped_credits(year: int) -> pd.DataFrame:
    """Credit rows tagged with `group`: the base position, plus WR1/WR2 copies of WR rows."""
    c = scoring.credits_with_position(year)
    c = c[c["position"].isin(POSITIONS)]
    base = c.assign(group=c["position"])
    roles = wr_roles(year).rename(columns={"team": "posteam"})[["posteam", "player_id", "wr_role"]]
    wr = c[c["position"] == "WR"].merge(roles, on=["posteam", "player_id"], how="left")
    wr = wr[wr["wr_role"].isin(WR_ROLES)].assign(group=lambda d: d["wr_role"]).drop(columns="wr_role")
    return pd.concat([base, wr], ignore_index=True)


def fp_allowed_by_game(year: int) -> pd.DataFrame:
    """FP allowed per defense per game per group (zeros filled in)."""
    c = _grouped_credits(year)
    g = c.groupby(["defteam", "game_id", "group"])["fp"].sum()
    games = fetch.load_pbp(year).dropna(subset=["defteam"])[["defteam", "game_id"]].drop_duplicates()
    idx = pd.MultiIndex.from_frame(
        games.merge(pd.DataFrame({"group": GROUPS}), how="cross")[["defteam", "game_id", "group"]])
    return g.reindex(idx, fill_value=0.0).rename("fp").reset_index()


def fp_allowed_per_game(year: int) -> pd.DataFrame:
    """team, position (group), fp_pg, games"""
    g = fp_allowed_by_game(year)
    return (g.groupby(["defteam", "group"])
             .agg(fp_pg=("fp", "mean"), games=("game_id", "nunique"), fp_total=("fp", "sum"))
             .reset_index().rename(columns={"defteam": "team", "group": "position"}))


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
    """Share of each defense's FP allowed that went to top-12 players at the base position."""
    c = _grouped_credits(year)
    totals = c[c["group"] == c["position"]].groupby(["position", "player_id"])["fp"].sum().reset_index()
    top12 = set(totals.sort_values("fp", ascending=False).groupby("position").head(12)["player_id"])
    c = c.assign(elite_fp=np.where(c["player_id"].isin(top12), c["fp"], 0.0))
    agg = c.groupby(["defteam", "group"]).agg(elite_fp=("elite_fp", "sum"), total_fp=("fp", "sum")).reset_index()
    agg["elite_ratio"] = (agg["elite_fp"] / agg["total_fp"].where(agg["total_fp"] > 0)).fillna(0).clip(0, 1)
    return agg.rename(columns={"defteam": "team", "group": "position"})[["team", "position", "elite_ratio"]]


def confidence(games: float, team_baseline: float, league_baseline: float, elite_ratio: float) -> float:
    sample_factor = min(0.4 + games * 0.25, 1.0)
    gap = abs(team_baseline - league_baseline) / league_baseline if league_baseline else 0.0
    gap_factor = min(0.4 + gap * 1.5, 1.0)
    conc_factor = max(1.0 - elite_ratio * 0.7, 0.3)
    return 100 * (0.25 * sample_factor + 0.45 * gap_factor + 0.30 * conc_factor)


# ---------- funnel effect ----------

def _behavior_vs_norm(year: int) -> pd.DataFrame:
    """How offenses behave against each defense relative to their own norm, in neutral script.

    pass_rate_oe: opponent pass rate vs that offense's pass rate in its other games.
    te_share_oe:  opponent TE share of targets vs that offense's TE share in its other games.
    Positive pass_rate_oe = offenses throw more than usual against this defense.
    """
    pbp = fetch.load_pbp(year)
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["wp"].between(*NEUTRAL_WP)]
    p = p.assign(is_pass=(p["play_type"] == "pass").astype(float))
    g = p.groupby(["posteam", "defteam", "game_id"]).agg(n=("is_pass", "size"), passes=("is_pass", "sum")).reset_index()
    tot = g.groupby("posteam")[["n", "passes"]].transform("sum")
    other_n, other_pass = tot["n"] - g["n"], tot["passes"] - g["passes"]
    g["pass_oe"] = g["passes"] / g["n"] - (other_pass / other_n.where(other_n > 0))

    c = scoring.credits_with_position(year)
    t = c[(c["role"] == "target") & c["position"].isin(["RB", "WR", "TE"])]
    t = t.assign(te=(t["position"] == "TE").astype(float))
    tg = t.groupby(["posteam", "defteam", "game_id"]).agg(tn=("te", "size"), te=("te", "sum")).reset_index()
    ttot = tg.groupby("posteam")[["tn", "te"]].transform("sum")
    o_n, o_te = ttot["tn"] - tg["tn"], ttot["te"] - tg["te"]
    tg["te_oe"] = tg["te"] / tg["tn"] - (o_te / o_n.where(o_n > 0))

    def wavg(d, col, w):
        d = d.dropna(subset=[col])
        return np.average(d[col], weights=d[w]) if len(d) else np.nan
    out = pd.DataFrame({
        "pass_rate_oe": g.groupby("defteam").apply(lambda d: wavg(d, "pass_oe", "n"), include_groups=False),
        "te_share_oe": tg.groupby("defteam").apply(lambda d: wavg(d, "te_oe", "tn"), include_groups=False),
    })
    return out.rename_axis("team").reset_index()


def funnel_insights(dvp: pd.DataFrame, year: int = config.SEASON) -> pd.DataFrame:
    """Per defense: which way it funnels offenses, with FP-split and behavioral evidence.

    funnel values:
      pass  - stout vs RBs, soft vs WR/TE: volume should shift to the passing game
      run   - stout vs WRs, soft vs RBs: expect a run-heavier script
      te    - stout vs WRs, soft vs TEs: expect short/underneath targets to TEs
    """
    wide = dvp.pivot(index="team", columns="position", values="fp_pg")
    league = dvp.groupby("position")["fp_pg"].mean()
    rel = wide / league - 1                                     # -0.3 = allows 30% below league avg
    rank = dvp.pivot(index="team", columns="position", values="rank")
    beh = _behavior_vs_norm(year).set_index("team")

    rows = []
    for team in wide.index:
        r, k = rel.loc[team], rank.loc[team]
        pass_oe = beh["pass_rate_oe"].get(team, np.nan)
        te_oe = beh["te_share_oe"].get(team, np.nan)
        soft_pass = "WR" if r["WR"] >= r["TE"] else "TE"
        cands = []  # (funnel, fp gap, behavior: +1 confirms, 0 neutral, -1 contradicts)

        def signal(x, sign):
            if pd.isna(x):
                return 0
            return 1 if x * sign > BEHAVIOR_SIGNAL else -1 if x * sign < -BEHAVIOR_SIGNAL else 0
        # Run-stout -> pass funnel
        if k["RB"] <= TOUGH_RANK and k[soft_pass] >= SOFT_RANK and r[soft_pass] - r["RB"] >= FUNNEL_GAP:
            cands.append(("pass", r[soft_pass] - r["RB"], signal(pass_oe, +1)))
        # Pass-stout vs WRs -> run funnel or TE funnel
        if k["WR"] <= TOUGH_RANK:
            if k["RB"] >= SOFT_RANK and r["RB"] - r["WR"] >= FUNNEL_GAP:
                cands.append(("run", r["RB"] - r["WR"], signal(pass_oe, -1)))
            if k["TE"] >= SOFT_RANK and r["TE"] - r["WR"] >= FUNNEL_GAP:
                cands.append(("te", r["TE"] - r["WR"], signal(te_oe, +1)))
        # Offenses visibly doing the opposite means the FP split is likely noise: no call.
        cands = [c for c in cands if c[2] >= 0]
        if cands:
            funnel, gap, sig = max(cands, key=lambda x: (x[2], x[1]))
            strength = "Strong" if sig > 0 else "Lean"
        else:
            funnel, gap, strength = None, np.nan, None
        rows.append({"team": team, "funnel": funnel, "funnel_strength": strength, "funnel_gap": gap,
                     "pass_rate_oe": pass_oe, "te_share_oe": te_oe,
                     "rb_rank": int(k["RB"]), "wr_rank": int(k["WR"]), "te_rank": int(k["TE"]),
                     "soft_pass": soft_pass})
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:+.0f}%"


def insight_for(f: pd.Series, position: str) -> tuple[str, str, str, str]:
    """(tone, label, effect, evidence) for one defense viewed from one position.
    tone: up (funnel sends volume toward this position) | down (away) | '' (indirect)."""
    if pd.isna(f["funnel"]):
        return "", "", "", ""
    base = "WR" if position in WR_ROLES else position
    label = f["funnel_strength"] + " " + {"pass": "pass funnel", "run": "run funnel", "te": "TE funnel"}[f["funnel"]]
    ranks = f"Allows RB #{f['rb_rank']} · WR #{f['wr_rank']} · TE #{f['te_rank']}"
    if f["funnel"] == "te":
        evid = f"{ranks} · opp TE target share {_pct(f['te_share_oe'])} vs norm"
    else:
        evid = f"{ranks} · opp pass rate {_pct(f['pass_rate_oe'])} vs norm"
    tgt = f["soft_pass"]
    table = {
        "pass": {"RB": ("down", f"Run game stalls; volume shifts to {tgt}s"),
                 "WR": ("up" if tgt == "WR" else "", "Extra dropbacks"),
                 "TE": ("up" if tgt == "TE" else "", "Extra dropbacks"),
                 "QB": ("up", "Extra dropbacks")},
        "run": {"RB": ("up", "Extra carries likely"),
                "WR": ("down", "Run-heavy script"),
                "QB": ("down", "Run-heavy script"),
                "TE": ("", "Fewer dropbacks")},
        "te": {"TE": ("up", "Short targets go to TEs"),
               "WR": ("down", "Targets move underneath to TEs"),
               "RB": ("", "Underneath passing favored"),
               "QB": ("", "Shorter throws")},
    }
    tone, effect = table[f["funnel"]].get(base, ("", ""))
    return tone, label, effect, evid


def defense_vs_position(year: int = config.SEASON) -> pd.DataFrame:
    """One row per (team, position group): FP/G allowed, rank, baseline, confidence, funnel insight."""
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

    fun = funnel_insights(df, year).set_index("team")
    ins = [insight_for(fun.loc[t], p) if t in fun.index else ("", "", "", "")
           for t, p in zip(df["team"], df["position"])]
    df["insight_tone"], df["insight"], df["insight_effect"], df["insight_detail"] = map(list, zip(*ins))
    df = df.join(fun[["funnel", "funnel_strength", "pass_rate_oe", "te_share_oe"]], on="team")
    return df.sort_values(["position", "rank"]).reset_index(drop=True)


def to_json(df: pd.DataFrame) -> list[dict]:
    cols = ["team", "position", "fp_pg", "rank", "games", "baseline_fp_pg", "league_fp_pg",
            "elite_ratio", "confidence", "funnel", "funnel_strength", "pass_rate_oe", "te_share_oe",
            "insight_tone", "insight", "insight_effect", "insight_detail"]
    out = df[cols].round(3)
    return out.astype(object).where(out.notna(), None).to_dict("records")
