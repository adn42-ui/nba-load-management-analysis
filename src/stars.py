"""Decide who counts as a "star" in each season, using only earlier seasons.

Two definitions:
  official_star - All-Star or All-NBA in any of the 3 prior seasons. This is the
                  NBA's own definition in the 2023-24 Player Participation Policy.
  broad_star    - official_star, OR a defensive star (top-15 Defensive Star
                  Rating among players with 27+ minutes and 50+ starts, in at
                  least 2 of the 3 prior seasons; see defense_rating.py), OR a
                  top-30 playoff Game Score per game in EACH of the 3 prior
                  playoffs. Catches playmakers and defenders (Haliburton, Gobert
                  types) and proven playoff performers, while one good year or
                  one deep run is not enough. On top of that, the player must have
                  played a quarter of his team's games last season, so former
                  All-Stars who missed most of a year don't count. The bar is
                  low on purpose: load-managed stars (Kawhi, Embiid) are who the
                  policy targeted and must stay in.

official_star stays exactly the NBA's definition, because those are the players
the policy actually applied to.

Status only uses past seasons on purpose: a player who stays healthy is more
likely to make the All-Star team that same year, which would bias the results.

Saves data/processed/stars.csv. Run from the project root:
    python src/stars.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "stars.csv"

LOOKBACK = 3            # seasons of awards that count, matching the NBA policy
PLAYOFF_TOP_N = 30      # must rank this high in playoff Game Score per game...
PLAYOFF_SEASONS = 3     # ...in each of this many straight prior playoffs
PLAYOFF_MIN_GAMES = 4   # a first-round sweep is the smallest real sample
DEF_TOP_N = 15          # top Defensive Star Rating among...
DEF_MIN_MPG = 27        # ...players with starter minutes...
DEF_MIN_STARTS = 50     # ...and starter role (per 82 games), so bench bigs don't qualify
DEF_SEASONS_NEEDED = 2  # in at least this many of the LOOKBACK prior seasons
AVAIL_MIN = 0.25        # broad stars must have played a quarter of their team's games last season

AWARD_FLAGS = {
    "NBA All-Star": "all_star",
    "All-NBA": "all_nba",
    "All-Defensive Team": "all_defense",
}


def season_start(season: str) -> int:
    """'2023-24' -> 2023."""
    return int(str(season)[:4])


def game_score(df: pd.DataFrame) -> pd.Series:
    """John Hollinger's Game Score: one-number summary of a box score line."""
    return (
        df.PTS + 0.4 * df.FGM - 0.7 * df.FGA - 0.4 * (df.FTA - df.FTM)
        + 0.7 * df.OREB + 0.3 * df.DREB + df.STL + 0.7 * df.AST
        + 0.7 * df.BLK - 0.4 * df.PF - df.TOV
    )


def load_awards() -> pd.DataFrame:
    """One row per (player, season) with a 0/1 column per award we care about."""
    awards = pd.read_csv(RAW_DIR / "awards.csv")
    awards = awards[awards.DESCRIPTION.isin(AWARD_FLAGS)].copy()
    awards["season"] = awards.SEASON.map(season_start)
    awards["award"] = awards.DESCRIPTION.map(AWARD_FLAGS)
    wide = (
        awards.assign(flag=1)
        .pivot_table(index=["PERSON_ID", "season"], columns="award",
                     values="flag", aggfunc="max", fill_value=0)
        .reset_index()
        .rename(columns={"PERSON_ID": "PLAYER_ID"})
    )
    for col in AWARD_FLAGS.values():
        if col not in wide:
            wide[col] = 0
    return wide


def playoff_impact() -> pd.DataFrame:
    """Playoff Game Score per game and its league rank per (player, season).

    Per game rather than total, so playing more rounds alone doesn't help.
    """
    logs = pd.concat(pd.read_csv(f) for f in RAW_DIR.glob("playoff_player_logs_*.csv"))
    logs["gmsc"] = game_score(logs)
    logs["season"] = logs.SEASON_ID.astype(str).str[1:].astype(int)
    impact = (logs.groupby(["PLAYER_ID", "season"]).gmsc
              .agg(playoff_games="count", playoff_impact="mean").reset_index())
    impact = impact[impact.playoff_games >= PLAYOFF_MIN_GAMES]
    impact["playoff_rank"] = impact.groupby("season").playoff_impact.rank(ascending=False)
    return impact


def season_games() -> dict[int, int]:
    """Games per team each season (fewer in the 2019-20 and 2020-21 COVID seasons)."""
    teams = pd.concat(pd.read_csv(f) for f in RAW_DIR.glob("team_logs_*.csv"))
    teams["season"] = teams.SEASON_ID.astype(str).str[1:].astype(int)
    return teams.groupby(["season", "TEAM_ID"]).size().groupby("season").max().to_dict()


def defensive_top(games: dict[int, int]) -> pd.DataFrame:
    """(player, season) pairs in the top DEF_TOP_N DSR among full-time starters."""
    ratings = pd.read_csv(ROOT / "data" / "processed" / "defensive_ratings.csv")
    min_starts = ratings.season.map(lambda s: DEF_MIN_STARTS * games.get(s, 82) / 82)
    starters = ratings[(ratings.mpg >= DEF_MIN_MPG) & (ratings.GS >= min_starts)].copy()
    starters["rank"] = starters.groupby("season").dsr.rank(ascending=False)
    return starters[starters["rank"] <= DEF_TOP_N][["PLAYER_ID", "season"]]


def build() -> pd.DataFrame:
    logs = pd.concat(pd.read_csv(f) for f in RAW_DIR.glob("player_logs_*.csv"))
    logs["season"] = logs.SEASON_ID.astype(str).str[1:].astype(int)
    players = logs[["PLAYER_ID", "PLAYER_NAME", "season"]].drop_duplicates(["PLAYER_ID", "season"])

    games = season_games()
    awards = load_awards()
    impact = playoff_impact()
    def_top = defensive_top(games)

    # Share of team games played LAST season. Never this season: availability
    # this season is the outcome being studied.
    gp = logs.groupby(["PLAYER_ID", "season"]).GAME_ID.nunique().rename("gp").reset_index()
    gp["season"] += 1
    gp["prev_availability"] = gp.gp / gp.season.map(lambda s: games.get(s - 1, 82))
    players = players.merge(gp[["PLAYER_ID", "season", "prev_availability"]],
                            on=["PLAYER_ID", "season"], how="left")
    players["prev_availability"] = players.prev_availability.fillna(0)

    rows = []
    for (pid, season), _ in players.groupby(["PLAYER_ID", "season"]):
        past = awards[(awards.PLAYER_ID == pid)
                      & awards.season.between(season - LOOKBACK, season - 1)]
        prev_playoffs = impact[(impact.PLAYER_ID == pid)
                               & impact.season.between(season - PLAYOFF_SEASONS, season - 1)
                               & (impact.playoff_rank <= PLAYOFF_TOP_N)]
        rows.append({
            "PLAYER_ID": pid,
            "season": season,
            "all_star_3y": int(past.all_star.any()),
            "all_nba_3y": int(past.all_nba.any()),
            "all_defense_3y": int(past.all_defense.any()),
            "top_playoff_seasons": len(prev_playoffs),
            "top_defense_seasons": int(((def_top.PLAYER_ID == pid)
                                        & def_top.season.between(season - LOOKBACK, season - 1)).sum()),
        })

    stars = players.merge(pd.DataFrame(rows), on=["PLAYER_ID", "season"])
    stars["official_star"] = (stars.all_star_3y | stars.all_nba_3y).astype(int)
    stars["sustained_playoff_star"] = (stars.top_playoff_seasons >= PLAYOFF_SEASONS).astype(int)
    stars["defensive_star"] = (stars.top_defense_seasons >= DEF_SEASONS_NEEDED).astype(int)
    stars["broad_star"] = ((stars.official_star.astype(bool)
                            | stars.defensive_star.astype(bool)
                            | stars.sustained_playoff_star.astype(bool))
                           & (stars.prev_availability >= AVAIL_MIN)).astype(int)
    return stars.sort_values(["season", "PLAYER_NAME"])


def main() -> None:
    stars = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stars.to_csv(OUT, index=False)
    summary = stars.groupby("season")[["official_star", "broad_star"]].sum()
    print("stars per season:\n", summary.to_string())


if __name__ == "__main__":
    main()
