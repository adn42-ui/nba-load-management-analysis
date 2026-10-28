"""Pull season-level defensive stats used by the Defensive Star Rating.

Five tables per season (player tables are season totals):
  hustle   - contested shots, deflections, charges drawn, loose balls
  ptdefend - opponent FG% when this player is the closest defender (tracking)
  defense  - steals, blocks, defensive rating, defensive win shares
  team     - each team's defensive rating, to judge players against teammates
  starts   - stats in games started only (its GP column is games started)

Saves data/raw/defense_<table>_<season>.csv. Run from the project root:
    python src/pull_defense.py
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashptdefend,
    leaguedashteamstats,
    leaguehustlestatsplayer,
)

# 2014-15 and 2015-16 give 2017-18 a full 3-season lookback where the data exists.
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TABLES = {
    "hustle": lambda s: leaguehustlestatsplayer.LeagueHustleStatsPlayer(
        season=s, per_mode_time="Totals", timeout=60),
    "ptdefend": lambda s: leaguedashptdefend.LeagueDashPtDefend(
        season=s, defense_category="Overall", per_mode_simple="Totals", timeout=60),
    "defense": lambda s: leaguedashplayerstats.LeagueDashPlayerStats(
        season=s, measure_type_detailed_defense="Defense",
        per_mode_detailed="Totals", timeout=60),
    "team": lambda s: leaguedashteamstats.LeagueDashTeamStats(
        season=s, measure_type_detailed_defense="Advanced", timeout=60),
    # Stats in games a player started only, so GP here is games started.
    "starts": lambda s: leaguedashplayerstats.LeagueDashPlayerStats(
        season=s, starter_bench_nullable="Starters", per_mode_detailed="Totals", timeout=60),
}


def fetch(table: str, season: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            return TABLES[table](season).get_data_frames()[0]
        except Exception as exc:  # stats.nba.com times out often; back off and retry
            print(f"  attempt {attempt} failed: {exc}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch {table} for {season}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        for table in TABLES:
            out = RAW_DIR / f"defense_{table}_{season}.csv"
            if out.exists():
                continue
            df = fetch(table, season)
            if df.empty:
                print(f"{table} {season}: no data (not tracked that season)")
                continue
            df.to_csv(out, index=False)
            print(f"{table} {season}: {len(df):,} players")
            time.sleep(1.5)  # be polite to the API


if __name__ == "__main__":
    main()
