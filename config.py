"""League-specific settings. Everything the pipeline needs to know about *your*
league lives here so the rest of the code stays reusable."""

SEASON = 2026                 # current (possibly in-progress) season
PRIOR_SEASON = SEASON - 1     # most recent completed season (xFP calibration, true TPRR)
HISTORY_SEASONS = [SEASON - 1, SEASON - 2, SEASON - 3]
HISTORY_WEIGHTS = [0.50, 0.30, 0.20]   # recency weights for 3-year baselines

LEAGUE_SIZE = 8
SCORING = "PPR"               # 1.0 per reception
ROSTER_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DST": 1}

# Names only — teams are resolved from the nflverse roster file at build time,
# so in-season trades are picked up automatically. Replace with your roster.
MY_ROSTER = [
    "Joe Burrow",
    "Jonathan Taylor",
    "Saquon Barkley",
    "James Cook",
    "Tee Higgins",
    "George Pickens",
    "Davante Adams",
    "Brock Bowers",
    "Jalen Coker",
    "Stefon Diggs",
    "RJ Harvey",
    "Isaiah Likely",
    "Jeremiyah Love",
    "Jordan Mason",
]

# Fantasy playoff weeks, highlighted in strength of schedule.
FANTASY_PLAYOFF_WEEKS = [15, 16, 17]

# Waiver targets. There's no league API, so the page assumes the top N at each
# position (by projection, by 3-year reputation, and early-round rookies) are
# rostered. List players you know are taken to exclude them too.
WAIVER_ROSTERED_DEPTH = {"QB": 12, "RB": 32, "WR": 40, "TE": 12}
ROSTERED_ELSEWHERE: list[str] = []

OUTPUT_HTML = "dashboard.html"

# Claude Code routine that re-runs fetch + build and republishes the dashboard.
# The page's "Refresh data" button fires it (published artifact only). Set to
# None to hide the button.
REFRESH_TRIGGER_ID = "trig_01S5RJqFPoxZAyHDp9U5VtJp"

TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LA": "#003594", "LAC": "#0080C6", "LV": "#000000", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#002244",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414",
}
