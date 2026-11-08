"""Cluster all 32 defenses into groups of similar units (k-means), PFF-style.

PFF's version uses its own grades and coverage charting, which are paywalled.
This one uses what nflverse publishes for free: play-by-play (EPA, success,
first downs, 3rd/4th-down conversions, sacks, QB hits, explosive plays,
takeaways) and FTN charting (blitzers, pass rushers, men in the box).

Each feature is standardized across the 32 teams and capped at ±Z_CAP, k-means (k-means++ seeds,
many restarts, fixed seed) groups them into K clusters, and the first two
principal components give the 2-D map. Clusters are named from their average
profile, and every team gets a note with its standout ranks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import dst, fetch

K = 4
RESTARTS = 200
SEED = 7
EXPLOSIVE_PASS, EXPLOSIVE_RUN = 20, 10      # yards
Z_CAP = 2.0                # standardized values are capped at ±Z_CAP so one extreme team can't take a cluster alone

# key: (label, higher is better for the defense?, format, side)
FEATURES = {
    "epa_pp":        ("EPA/play allowed", False, "epa", "overall"),
    "success_rate":  ("Success rate allowed", False, "pct", "overall"),
    "first_down_rate": ("1st-down rate allowed", False, "pct", "overall"),
    "third_down_rate": ("3rd-down conv. allowed", False, "pct", "overall"),
    "pass_epa_pp":   ("Dropback EPA allowed", False, "epa", "pass"),
    "sack_rate":     ("Sack rate", True, "pct", "pass"),
    "hit_rate":      ("QB hit rate (pressure)", True, "pct", "pass"),
    "explosive_pass": (f"{EXPLOSIVE_PASS}+ yd pass rate allowed", False, "pct", "pass"),
    "takeaway_rate": ("Takeaways per 100 plays", True, "num", "pass"),
    "rush_epa_pp":   ("Rush EPA allowed", False, "epa", "run"),
    "stuff_rate":    ("Run stuff rate (≤0 yds)", True, "pct", "run"),
    "explosive_run": (f"{EXPLOSIVE_RUN}+ yd run rate allowed", False, "pct", "run"),
    "blitz_rate":    ("Blitz rate", None, "pct", "scheme"),
    "rushers":       ("Avg pass rushers", None, "num", "scheme"),
    "box":           ("Avg men in box vs run", None, "num", "scheme"),
}
EXTRA = {   # shown, not clustered (small samples)
    "fourth_down_rate": ("4th-down conv. allowed", False, "pct", "overall"),
}


def team_features(year: int = config.SEASON) -> pd.DataFrame:
    pbp = fetch.load_pbp(year)
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["defteam"].notna() & pbp["epa"].notna()].copy()
    ftn = fetch.load_ftn(year)
    if ftn is not None:
        p = p.merge(ftn.astype({"play_id": float}), on=["game_id", "play_id"], how="left")
    else:
        p[["n_defense_box", "n_blitzers", "n_pass_rushers"]] = np.nan
    db, ru = p[p["pass"] == 1], p[p["rush"] == 1]
    # FTN codes uncharted plays as zeros: use only dropbacks with a charted rush.
    charted = db[db["n_pass_rushers"] > 0]
    g = lambda d: d.groupby("defteam")
    third = p[(p["third_down_converted"] == 1) | (p["third_down_failed"] == 1)]
    fourth = p[(p["fourth_down_converted"] == 1) | (p["fourth_down_failed"] == 1)]
    out = pd.DataFrame({
        "games": g(p)["game_id"].nunique(),
        "plays": g(p).size(),
        "epa_pp": g(p)["epa"].mean(),
        "success_rate": g(p)["success"].mean(),
        "first_down_rate": g(p)["first_down"].mean(),
        "third_down_rate": g(third)["third_down_converted"].mean(),
        "fourth_down_rate": g(fourth)["fourth_down_converted"].mean(),
        "fourth_downs": g(fourth).size(),
        "pass_epa_pp": g(db)["epa"].mean(),
        "sack_rate": g(db)["sack"].mean(),
        "hit_rate": g(db.assign(h=(db["qb_hit"].fillna(0) + db["sack"].fillna(0)) > 0))["h"].mean(),
        "explosive_pass": g(db.assign(x=db["yards_gained"] >= EXPLOSIVE_PASS))["x"].mean(),
        "takeaway_rate": g(p.assign(t=p["interception"].fillna(0) + p["fumble_lost"].fillna(0)))["t"].mean() * 100,
        "rush_epa_pp": g(ru)["epa"].mean(),
        "stuff_rate": g(ru.assign(x=ru["yards_gained"] <= 0))["x"].mean(),
        "explosive_run": g(ru.assign(x=ru["yards_gained"] >= EXPLOSIVE_RUN))["x"].mean(),
        "blitz_rate": g(charted.assign(b=charted["n_blitzers"] > 0))["b"].mean(),
        "rushers": g(charted)["n_pass_rushers"].mean(),
        "box": g(ru[ru["n_defense_box"] > 0])["n_defense_box"].mean(),
    })
    tg = dst.team_games(year)
    out["dst_pg"] = tg.groupby("defense")["dst_fp"].mean()
    return out.rename_axis("team")


def kmeans(Z: np.ndarray, k: int, restarts: int = RESTARTS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(restarts):
        c = [Z[rng.integers(len(Z))]]
        for _ in range(k - 1):                                 # k-means++ seeding
            d2 = np.min([((Z - ci) ** 2).sum(1) for ci in c], axis=0)
            c.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
        C = np.array(c)
        for _ in range(100):
            lab = np.argmin(((Z[:, None] - C[None]) ** 2).sum(2), axis=1)
            newC = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((Z - C[lab]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, lab, C)
    return best[1], best[2]


def _good(df: pd.DataFrame) -> pd.DataFrame:
    """Each clustered feature's z-score, signed so positive = better for the defense
    (scheme features keep their natural sign)."""
    z = (df - df.mean()) / df.std(ddof=0)
    for f, (_, higher_better, _, _) in FEATURES.items():
        if higher_better is False:
            z[f] = -z[f]
    return z


STRONG = {"epa_pp": "stingy overall", "success_rate": "wins downs", "first_down_rate": "few first downs",
          "third_down_rate": "gets off the field on 3rd down", "pass_epa_pp": "strong vs the pass",
          "sack_rate": "sack-heavy", "hit_rate": "heavy pressure", "explosive_pass": "limits big plays",
          "takeaway_rate": "ball-hawking", "rush_epa_pp": "stout vs the run", "stuff_rate": "stuffs runs",
          "explosive_run": "limits big runs"}
WEAK = {"epa_pp": "leaky overall", "success_rate": "loses downs", "first_down_rate": "gives up first downs",
        "third_down_rate": "struggles on 3rd down", "pass_epa_pp": "beatable through the air",
        "sack_rate": "few sacks", "hit_rate": "little pressure", "explosive_pass": "gives up big plays",
        "takeaway_rate": "few takeaways", "rush_epa_pp": "soft vs the run", "stuff_rate": "rarely stuffs runs",
        "explosive_run": "gives up big runs"}


def _name(profile: pd.Series, overall_rank: int, k: int) -> str:
    """Name a cluster from its average signed z-scores: its best trait, its weakest, and blitz style."""
    graded = profile[[f for f in profile.index if f in STRONG]]
    top, bottom = graded.idxmax(), graded.idxmin()
    if overall_rank == 0:
        base = f"Elite: {STRONG[top]}"
    elif overall_rank == k - 1:
        base = f"Struggling: {WEAK[bottom]}"
    else:
        base = f"{STRONG[top].capitalize()}, {WEAK[bottom]}"
    if profile.get("blitz_rate", 0) >= 0.5:
        base += " (blitz-heavy)"
    elif profile.get("blitz_rate", 0) <= -0.5:
        base += " (rushes four)"
    return base


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _fmt(key: str, v: float) -> str:
    kind = (FEATURES.get(key) or EXTRA[key])[2]
    return f"{v * 100:.1f}%" if kind == "pct" else f"{v:+.2f}" if kind == "epa" else f"{v:.2f}"


def clusters(year: int = config.SEASON) -> dict:
    df = team_features(year)
    feats = [f for f in FEATURES if df[f].notna().all()]       # drop FTN features if charting is missing
    X = df[feats]
    Z = np.clip(((X - X.mean()) / X.std(ddof=0)).to_numpy(), -Z_CAP, Z_CAP)
    lab, _ = kmeans(Z, K)

    # Order clusters best to worst by overall EPA allowed, so "Cluster 1" is the best group.
    order = pd.Series(df["epa_pp"].to_numpy()).groupby(lab).mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    df["cluster"] = [remap[l] for l in lab]

    # 2-D map: first two principal components of the standardized features.
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs = Z @ Vt[:2].T
    if np.corrcoef(pcs[:, 0], -df["epa_pp"])[0, 1] < 0:          # right = better defense
        pcs[:, 0] *= -1
    if np.corrcoef(pcs[:, 1], df["hit_rate"])[0, 1] < 0:         # up = more pressure
        pcs[:, 1] *= -1
    df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]
    explained = (S ** 2 / (S ** 2).sum())[:2]
    # What each axis means: correlation of each feature (signed so + = better for the defense) with it.
    gz = _good(df[feats])
    loadings = pd.DataFrame({pc: [np.corrcoef(gz[f], df[pc])[0, 1] for f in feats] for pc in ("pc1", "pc2")}, index=feats)

    good = _good(df[feats])
    ranks = pd.DataFrame(index=df.index)
    for f in list(FEATURES) + list(EXTRA):
        if f not in df or df[f].isna().all():
            continue
        hb = (FEATURES.get(f) or EXTRA[f])[1]
        ranks[f] = df[f].rank(ascending=not hb if hb is not None else False, method="min")

    def note(team: str) -> str:
        r = ranks.loc[team, [f for f in feats if FEATURES[f][1] is not None]]
        best, worst = r.nsmallest(2), r.nlargest(1)
        parts = [f"{_ordinal(int(v))} in {FEATURES[f][0]} ({_fmt(f, df.at[team, f])})" for f, v in best.items()]
        w = worst.index[0]
        tail = f"; {_ordinal(int(worst.iloc[0]))} in {FEATURES[w][0]} ({_fmt(w, df.at[team, w])})"
        scheme = f" Blitzes {df.at[team, 'blitz_rate'] * 100:.0f}% of dropbacks." if "blitz_rate" in feats else ""
        return "Ranks " + ", ".join(parts) + tail + "." + scheme

    out_clusters = []
    for c in range(K):
        members = df[df["cluster"] == c]
        prof = good.loc[members.index].mean()
        out_clusters.append({
            "id": c + 1,
            "name": _name(prof, c, K),
            "teams": members.sort_values("epa_pp").index.tolist(),
            "avg": {f: round(float(members[f].mean()), 4) for f in feats + ["fourth_down_rate", "dst_pg"]},
            "strengths": [FEATURES[f][0] for f in prof[[f for f in feats if FEATURES[f][1] is not None]].nlargest(2).index],
            "weaknesses": [FEATURES[f][0] for f in prof[[f for f in feats if FEATURES[f][1] is not None]].nsmallest(2).index],
        })
    teams = []
    for t, r in df.iterrows():
        teams.append({"team": t, "cluster": int(r["cluster"]) + 1, "pc1": round(float(r["pc1"]), 3),
                      "pc2": round(float(r["pc2"]), 3), "note": note(t),
                      **{f: (None if pd.isna(r[f]) else round(float(r[f]), 4)) for f in list(FEATURES) + list(EXTRA) + ["dst_pg", "games", "fourth_downs"] if f in r},
                      **{f"{f}_rank": int(ranks.at[t, f]) for f in ranks.columns}})
    league = {f: round(float(df[f].mean()), 4) for f in feats + ["fourth_down_rate", "dst_pg"]}
    pc_text = []
    for pc in ("pc1", "pc2"):
        top = loadings[pc].abs().nlargest(3).index
        pc_text.append([{"feature": FEATURES[f][0], "weight": round(float(loadings.at[f, pc]), 2)} for f in top])
    return {
        "season": year, "k": K, "features": [{"key": f, "label": FEATURES[f][0], "higher_better": FEATURES[f][1],
                                              "fmt": FEATURES[f][2], "side": FEATURES[f][3]} for f in feats]
                                            + [{"key": f, "label": v[0], "higher_better": v[1], "fmt": v[2], "side": v[3], "extra": True}
                                               for f, v in EXTRA.items()],
        "ftn": "blitz_rate" in feats,
        "explained": [round(float(e), 3) for e in explained],
        "pc_loadings": pc_text,
        "clusters": out_clusters,
        "teams": teams,
        "league": league,
    }
