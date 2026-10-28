"""Build the player-game panel: one row per team game a player could have played.

For each player-season, the player is "eligible" for every game his team plays
from his first appearance of the season to the end of the regular season. If he
is traded, games count toward whichever team he last appeared for. played = 1
if he appeared in that game.

Sample (decided from LAST season, so it can't be affected by the policy):
  - averaged 20+ minutes per game last season (rotation players)
  - played 25%+ of his team's games last season
  - appeared at least once this season

Saves data/processed/player_games.csv. Run from the project root, after stars.py:
    python src/availability.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FIRST_SEASON = 2017     # 2017-18; earlier seasons only feed the star definitions
POLICY_SEASON = 2023    # Player Participation Policy took effect in 2023-24
MIN_PREV_MPG = 20
MIN_PREV_AVAILABILITY = 0.25


def load_logs(prefix: str) -> pd.DataFrame:
    df = pd.concat(pd.read_csv(f) for f in RAW_DIR.glob(f"{prefix}_logs_*.csv"))
    df["season"] = df.SEASON_ID.astype(str).str[1:].astype(int)
    df["GAME_DATE"] = pd.to_datetime(df.GAME_DATE)
    return df


def team_schedule() -> pd.DataFrame:
    """Every team game, flagged if it's the second night of a back-to-back."""
    teams = load_logs("team")[["season", "TEAM_ID", "GAME_ID", "GAME_DATE"]]
    teams = teams.sort_values(["TEAM_ID", "GAME_DATE"])
    gap = teams.groupby(["season", "TEAM_ID"]).GAME_DATE.diff().dt.days
    teams["b2b"] = (gap == 1).astype(int)
    return teams


def build() -> pd.DataFrame:
    players = load_logs("player")
    schedule = team_schedule()
    stars = pd.read_csv(PROCESSED / "stars.csv")

    # Sample: rotation players last season (prev_availability comes from stars.py).
    mpg = players.groupby(["PLAYER_ID", "season"]).MIN.mean().rename("prev_mpg").reset_index()
    mpg["season"] += 1
    sample = stars.merge(mpg, on=["PLAYER_ID", "season"], how="left")
    sample = sample[(sample.season >= FIRST_SEASON)
                    & (sample.prev_mpg >= MIN_PREV_MPG)
                    & (sample.prev_availability >= MIN_PREV_AVAILABILITY)]

    apps = players.merge(sample[["PLAYER_ID", "season"]], on=["PLAYER_ID", "season"])
    apps = apps[["PLAYER_ID", "season", "TEAM_ID", "GAME_ID", "GAME_DATE"]]

    rows = []
    for (pid, season), a in apps.groupby(["PLAYER_ID", "season"]):
        a = a.sort_values("GAME_DATE")
        sched = schedule[schedule.season == season]
        # Team on each date = team of his most recent appearance on or before it.
        stints = a[["GAME_DATE", "TEAM_ID"]].drop_duplicates("TEAM_ID", keep="first")
        for i, (start, team) in enumerate(zip(stints.GAME_DATE, stints.TEAM_ID)):
            end = stints.GAME_DATE.iloc[i + 1] if i + 1 < len(stints) else pd.Timestamp.max
            games = sched[(sched.TEAM_ID == team)
                          & (sched.GAME_DATE >= start) & (sched.GAME_DATE < end)]
            rows.append(games.assign(PLAYER_ID=pid))

    panel = pd.concat(rows, ignore_index=True)
    played = set(zip(apps.PLAYER_ID, apps.GAME_ID))
    panel["played"] = [int((p, g) in played) for p, g in zip(panel.PLAYER_ID, panel.GAME_ID)]
    panel = panel.merge(
        sample[["PLAYER_ID", "season", "PLAYER_NAME", "official_star", "broad_star"]],
        on=["PLAYER_ID", "season"])
    panel["post"] = (panel.season >= POLICY_SEASON).astype(int)
    return panel


def main() -> None:
    panel = build()
    panel.to_csv(PROCESSED / "player_games.csv", index=False)
    by = panel.groupby(["season", "official_star"]).played.mean().unstack()
    by.columns = ["non-stars", "official stars"]
    print(f"{len(panel):,} player-games, {panel.groupby(['PLAYER_ID', 'season']).ngroups:,} player-seasons")
    print("\nShare of eligible games played:\n", by.round(3).to_string())


if __name__ == "__main__":
    main()
