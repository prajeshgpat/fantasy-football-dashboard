# Fantasy Football Dashboard

A self-contained NFL analytics dashboard built entirely from public
[nflverse](https://github.com/nflverse/nflverse-data) data. The output is a
single static HTML file with the computed data embedded as JSON and rendered
with Chart.js.

No API keys needed — everything comes from nflverse GitHub release assets.

## Setup

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pyarrow` is required: several nflverse files are parquet-only, and pandas
fails with "Unable to find a usable engine" without it.

## Run

```bash
python -m src.fetch              # download + cache into data/ (slow first run)
python -m src.build_dashboard    # compute every section, write dashboard.html
```

`build_dashboard` accepts an optional output path
(`python -m src.build_dashboard site/index.html`).

**Weekly refresh:** re-run both. `src.fetch` drops the cached current-season
files (pbp, roster, snap counts, schedule, NGS) before re-downloading; prior
seasons stay cached. nflverse usually posts a completed week within ~24h of
Monday night's game.

## Configure

Everything league-specific lives in `config.py`: season, league size, roster
slots, `MY_ROSTER` (names only — teams are resolved from the roster file so
in-season trades are picked up), team names and colors.

## Layout

```
config.py                  league settings, roster, team colors/names
src/fetch.py               download + cache nflverse data
src/scoring.py             PPR fantasy points from play-by-play
src/defense.py             defense-vs-position ranks + confidence scores
src/slot_perimeter.py      slot/middle vs perimeter FP allowed (proxy)
src/separation.py          NGS separation + TPRR
src/expected_points.py     xFP model (calibrate + apply)
src/redzone.py             red zone opportunity
src/skill_metrics.py       CPOE / RYOE passthrough
src/rankings.py            Top-200 tiered rankings
src/build_dashboard.py     inject JSON into the HTML template
templates/dashboard.html   HTML/CSS/JS template with {{TOKEN}} placeholders
data/                      gitignored cache (all re-downloadable)
```

## Dashboard sections

| Section | What it shows |
|---|---|
| My Lineup | Each rostered player's next opponent, that defense's rank/confidence at the position, slot tendency, and a Plus / Tough / Neutral read ("low confidence" when confidence < 45). |
| Offensive plays per game | Pass + run plays per game, current vs prior season. |
| Separation vs targets per route | NGS `avg_separation` vs true TPRR (prior season, from `pbp_participation`, min 50 routes) or a targets ÷ snaps proxy (current season). Flags ±1.0 yd separation swings and +50% / −30% target-rate swings. |
| Defense vs position | FP allowed per game, rank (1 = stingiest), 3-year baseline, confidence score. |
| Slot vs perimeter | WR+TE points allowed by `pass_location` (middle vs left/right) — a proxy, read relatively. |
| Expected vs actual FP | xFP calibrated on the last completed season; negative delta = buy-low, positive = regression risk. |
| Red zone | Targets/carries inside the 20 and 10, TDs, and TD-regression candidates. |
| CPOE & RYOE | NGS season aggregates. |
| Top 200 | `proj_pts = skill_ppg × matchup_mult`, tiered 1–200. |

### Method notes

- **Scoring (PPR):** 0.04/pass yd, 4/pass TD, −2/INT, 0.1/rush or rec yd,
  6/rush or rec TD, 1/reception, −2/fumble lost.
- **Confidence** is a weighted average (25% sample, 45% baseline gap, 30%
  concentration), not a product — the multiplicative version crushes every
  small-sample score under ~33.
- **Baselines** are 3-year recency-weighted (50/30/20) over the prior three
  seasons, renormalised when a season is missing.
- **Rankings** include only roster `status == "ACT"` players. QBs are limited
  to each team's starter (most current-season pass attempts). Zero-game ACT
  players are kept at `ppg_3yr × 0.92`. Active players with no play-by-play
  history at all are listed as data gaps under the table rather than dropped
  silently. Rankings are by raw projected points — no positional-scarcity
  adjustment, so a QB-heavy top 12 is expected and is not draft advice.
- **Separation:** NGS `avg_separation` is tracking-derived (yards from nearest
  defender). It is *not* Fantasy Points' proprietary "Average Separation
  Score," which is human-charted and not derivable from public data.

### Known data constraints

- `pbp_participation` is only published for completed seasons, so true TPRR is
  prior-season only; the current season uses targets ÷ offensive snaps.
- pbp has no position column; positions come from the matching season's roster.
- Snap counts are keyed by PFR id / name, not `gsis_id`; they're joined through
  the roster's `pfr_id`.
- Charts load Chart.js, Hammer.js and chartjs-plugin-zoom from cdnjs. If the
  CDN is unreachable the tables still render and the charts show a notice.

## Data hygiene

Never commit `data/`. `play_by_play_*.csv.gz` expands to ~150 MB pickled,
which exceeds GitHub's 100 MB per-file limit and gets the push rejected.
Everything there is re-downloadable.

## Publishing

Keep the repo **private** — it contains your roster and league context. The
generated `dashboard.html` is a single file you can open locally. GitHub Pages
from a private repo requires a paid plan; on the free tier Pages needs a
public repo.
