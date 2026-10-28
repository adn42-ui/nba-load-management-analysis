"""Pull regular-season player/team game logs and playoff player logs from stats.nba.com.

Saves one CSV per season per log type to data/raw/, e.g.
data/raw/player_logs_2023-24.csv, data/raw/team_logs_2023-24.csv and
data/raw/playoff_player_logs_2023-24.csv.

Run from the project root:
    python src/pull_data.py
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Six seasons before the 2023-24 Player Participation Policy, three after.
# 2016-17 is only there so 2017-18 stars can be judged on the prior season.
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
    "2023-24", "2024-25", "2025-26",
]
# Playoffs only, so 2017-18 stars can be judged on 3 prior playoff runs.
EXTRA_PLAYOFF_SEASONS = ["2014-15", "2015-16"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_log(season: str, kind: str, season_type: str, retries: int = 3) -> pd.DataFrame:
    """Fetch one season of game logs. kind is 'P' (players) or 'T' (teams)."""
    for attempt in range(1, retries + 1):
        try:
            resp = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation=kind,
                timeout=60,
            )
            return resp.get_data_frames()[0]
        except Exception as exc:  # stats.nba.com times out often; back off and retry
            print(f"  attempt {attempt} failed: {exc}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch {kind} logs for {season}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(s, "P", "playoff_player", "Playoffs") for s in EXTRA_PLAYOFF_SEASONS]
    for s in SEASONS:
        jobs += [
            (s, "P", "player", "Regular Season"),
            (s, "T", "team", "Regular Season"),
            (s, "P", "playoff_player", "Playoffs"),
        ]
    for season, kind, name, season_type in jobs:
        out = RAW_DIR / f"{name}_logs_{season}.csv"
        if out.exists():
            print(f"skip {out.name} (already downloaded)")
            continue
        print(f"fetching {name} logs {season}...")
        df = fetch_log(season, kind, season_type)
        df.to_csv(out, index=False)
        print(f"  saved {len(df):,} rows")
        time.sleep(1.5)  # be polite to the API


if __name__ == "__main__":
    main()
