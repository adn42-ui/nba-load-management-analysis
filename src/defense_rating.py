"""Defensive Star Rating (DSR): one number for how good a player's defense was.

Three equally weighted pillars, each built from per-36-minute rates so that
playing more games doesn't raise the rating (availability is what we study):

  Shot defense  - makes prevented: (opponents' usual FG% - FG% when this player
                  is the closest defender) x shots defended; contested shots
  Disruption    - steals, blocks, deflections, charges drawn, defensive loose
                  balls recovered
  Team impact   - defensive win shares; defensive rating compared with his
                  own team's (how much better the defense is with him on)

Every stat is turned into a z-score among qualified players that season
(0 = average qualified player, 1 = one standard deviation better), averaged
within its pillar, and the pillars are averaged into the DSR. Seasons without
hustle tracking (before 2016-17) use the stats that exist.

Saves data/processed/defensive_ratings.csv. Run from the project root:
    python src/defense_rating.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "defensive_ratings.csv"

MIN_MPG = 20              # rotation players only
MIN_GAMES_SHARE = 0.5     # must play half the season (handles shortened seasons)
MIN_HUSTLE_PLAYERS = 300  # 2015-16 hustle data only covers a partial sample

PILLARS = {
    "shot_defense": ["makes_prevented", "contests"],
    "disruption": ["steals", "blocks", "deflections", "charges", "loose_balls"],
    "team_impact": ["def_ws", "def_rating_vs_team"],
}


def load_season(season: str) -> pd.DataFrame:
    d = pd.read_csv(RAW_DIR / f"defense_defense_{season}.csv")
    df = d[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GP", "MIN", "STL", "BLK", "DEF_WS", "DEF_RATING"]]
    team = pd.read_csv(RAW_DIR / f"defense_team_{season}.csv")
    df = df.merge(team[["TEAM_ID", "DEF_RATING"]].rename(columns={"DEF_RATING": "TEAM_DEF_RATING"}),
                  on="TEAM_ID", how="left")
    starts = pd.read_csv(RAW_DIR / f"defense_starts_{season}.csv")
    df = df.merge(starts[["PLAYER_ID", "GP"]].rename(columns={"GP": "GS"}), on="PLAYER_ID", how="left")
    df["GS"] = df.GS.fillna(0).astype(int)

    pt = pd.read_csv(RAW_DIR / f"defense_ptdefend_{season}.csv")
    pt["makes_prevented_total"] = (pt.NORMAL_FG_PCT - pt.D_FG_PCT) * pt.D_FGA
    df = df.merge(pt[["CLOSE_DEF_PERSON_ID", "makes_prevented_total"]]
                  .rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"}),
                  on="PLAYER_ID", how="left")

    hustle_file = RAW_DIR / f"defense_hustle_{season}.csv"
    hustle = pd.read_csv(hustle_file) if hustle_file.exists() else pd.DataFrame()
    hustle_cols = ["CONTESTED_SHOTS", "DEFLECTIONS", "CHARGES_DRAWN", "DEF_LOOSE_BALLS_RECOVERED"]
    if len(hustle) >= MIN_HUSTLE_PLAYERS:
        df = df.merge(hustle[["PLAYER_ID"] + hustle_cols], on="PLAYER_ID", how="left")
    else:
        for col in hustle_cols:
            df[col] = float("nan")

    per36 = 36 / df.MIN
    df = df.assign(
        season=int(season[:4]),
        mpg=df.MIN / df.GP,
        makes_prevented=df.makes_prevented_total * per36,
        contests=df.CONTESTED_SHOTS * per36,
        steals=df.STL * per36,
        blocks=df.BLK * per36,
        deflections=df.DEFLECTIONS * per36,
        charges=df.CHARGES_DRAWN * per36,
        loose_balls=df.DEF_LOOSE_BALLS_RECOVERED * per36,
        def_ws=df.DEF_WS * per36,
        # Points per 100 the team allows overall minus with him on the floor, so a
        # great team defense doesn't make all of its players look great.
        def_rating_vs_team=df.TEAM_DEF_RATING - df.DEF_RATING,
    )
    qualified = (df.mpg >= MIN_MPG) & (df.GP >= MIN_GAMES_SHARE * df.GP.max())
    return df[qualified].copy()


def rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add pillar scores and the DSR to one season of qualified players."""
    for pillar, stats in PILLARS.items():
        z = [(df[s] - df[s].mean()) / df[s].std() for s in stats if df[s].notna().any()]
        df[pillar] = pd.concat(z, axis=1).mean(axis=1)
    df["dsr"] = df[list(PILLARS)].mean(axis=1)
    df["dsr_rank"] = df.dsr.rank(ascending=False)
    return df


def build() -> pd.DataFrame:
    seasons = sorted(f.stem.removeprefix("defense_defense_")
                     for f in RAW_DIR.glob("defense_defense_*.csv"))
    rated = pd.concat(rate(load_season(s)) for s in seasons)
    cols = ["PLAYER_ID", "PLAYER_NAME", "season", "GP", "GS", "mpg", *PILLARS, "dsr", "dsr_rank"]
    return rated[cols].sort_values(["season", "dsr_rank"])


def validate(ratings: pd.DataFrame) -> float:
    """AUC against All-Defensive Team picks: the chance a random All-Defense
    player is rated above a random non-pick in the same season (0.5 = coin flip).
    """
    awards = pd.read_csv(RAW_DIR / "awards.csv")
    picks = awards[awards.DESCRIPTION == "All-Defensive Team"]
    picks = set(zip(picks.PERSON_ID, picks.SEASON.str[:4].astype(int)))
    df = ratings.assign(all_defense=[(p, s) in picks for p, s in zip(ratings.PLAYER_ID, ratings.season)])
    aucs = []
    for _, g in df.groupby("season"):
        pos, neg = g[g.all_defense].dsr, g[~g.all_defense].dsr
        if len(pos):
            aucs.append((pos.values[:, None] > neg.values[None, :]).mean())
    return sum(aucs) / len(aucs)


def main() -> None:
    ratings = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(OUT, index=False)
    print(f"AUC vs All-Defensive Teams: {validate(ratings):.3f}")
    for season in (2016, 2020, 2024, 2025):
        top = ratings[ratings.season == season].head(10)
        print(f"\nTop 10 DSR {season}-{(season + 1) % 100:02d}: " + ", ".join(top.PLAYER_NAME))


if __name__ == "__main__":
    main()
