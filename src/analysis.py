"""Did the 2023-24 Player Participation Policy make stars play more often?

Difference-in-differences on the player-game panel: compare how much stars'
availability changed after the policy with how much non-stars' changed. Non-
stars weren't covered by the policy, so their change captures everything else
(league-wide injury trends, schedule changes, etc.).

    played = b * star x post + star + season fixed effects

b is the policy effect in share of games; x 82 gives games per season.
Standard errors are clustered by player (the same player shows up in many games).

Also runs:
  - the same model with the broad star definition and without the COVID seasons
  - back-to-backs: did the effect concentrate on second nights, where rest
    was most common before the policy? (triple difference)
  - an event study: the star vs non-star gap season by season, which checks that
    the two groups were on similar paths before the policy (parallel trends)

Saves results/did_results.csv and charts in figures/. Run from the project root,
after availability.py:
    python src/analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

POLICY_SEASON = 2023
BASE_SEASON = 2022      # last season before the policy; event study compares to it
COVID_SEASONS = [2019, 2020]
GAMES = 82

STAR_COLOR, OTHER_COLOR = "#C8102E", "#1D428A"


def season_label(s: int) -> str:
    return f"{s}-{(s + 1) % 100:02d}"


def sample_for(panel: pd.DataFrame, star_col: str) -> pd.DataFrame:
    """Stars by star_col vs players who are stars under neither definition."""
    clean_control = (panel.official_star == 0) & (panel.broad_star == 0)
    df = panel[(panel[star_col] == 1) | clean_control].copy()
    df["star"] = df[star_col]
    return df


def fit(formula: str, df: pd.DataFrame):
    return smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df.PLAYER_ID})


def did(df: pd.DataFrame, label: str) -> dict:
    m = fit("played ~ star + star:post + C(season)", df)
    b, se = m.params["star:post"], m.bse["star:post"]
    return {"model": label, "effect_share": b, "se": se, "p_value": m.pvalues["star:post"],
            "games_per_season": b * GAMES, "ci_low_games": (b - 1.96 * se) * GAMES,
            "ci_high_games": (b + 1.96 * se) * GAMES, "n_player_games": int(m.nobs)}


def b2b(df: pd.DataFrame, label: str) -> dict:
    # post on its own is absorbed by the season effects, so only its interactions go in.
    m = fit("played ~ star * b2b + star:post + b2b:post + star:b2b:post + C(season)", df)
    term = "star:b2b:post"
    b, se = m.params[term], m.bse[term]
    return {"model": label, "effect_share": b, "se": se, "p_value": m.pvalues[term],
            "games_per_season": None, "ci_low_games": None, "ci_high_games": None,
            "n_player_games": int(m.nobs)}


def event_study(df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(s for s in df.season.unique() if s != BASE_SEASON)
    for s in seasons:
        df[f"star_{s}"] = ((df.season == s) & (df.star == 1)).astype(int)
    m = fit("played ~ star + C(season) + " + " + ".join(f"star_{s}" for s in seasons), df)
    rows = [{"season": s, "gap": m.params[f"star_{s}"], "se": m.bse[f"star_{s}"]} for s in seasons]
    rows.append({"season": BASE_SEASON, "gap": 0.0, "se": 0.0})
    return pd.DataFrame(rows).sort_values("season")


def plot_trend(df: pd.DataFrame) -> None:
    rates = df.groupby(["season", "star"]).played.mean().unstack() * GAMES
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rates.index, rates[1], "o-", color=STAR_COLOR, lw=2.5, label="Stars (All-Star / All-NBA, last 3 yrs)")
    ax.plot(rates.index, rates[0], "o-", color=OTHER_COLOR, lw=2.5, label="Other rotation players")
    ax.axvspan(POLICY_SEASON - 0.5, rates.index.max() + 0.5, color="grey", alpha=0.12)
    ax.text(POLICY_SEASON - 0.4, ax.get_ylim()[1] - 0.6, "Player Participation Policy", fontsize=10, va="top")
    ax.axvspan(min(COVID_SEASONS) - 0.5, max(COVID_SEASONS) + 0.5, color="orange", alpha=0.08)
    ax.text(min(COVID_SEASONS) - 0.4, ax.get_ylim()[1] - 0.6, "COVID seasons", fontsize=10,
            color="darkorange", va="top")
    ax.set_xticks(rates.index, [season_label(s) for s in rates.index], rotation=30)
    ax.set_ylabel("Games played per 82 available")
    ax.set_title("Games played per 82 available: stars vs. other rotation players", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES / "availability_trend.png", dpi=200)
    plt.close(fig)


def plot_event_study(es: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    games, err = es.gap * GAMES, 1.96 * es.se * GAMES
    colors = [STAR_COLOR if s >= POLICY_SEASON else OTHER_COLOR for s in es.season]
    ax.errorbar(es.season, games, yerr=err, fmt="none", ecolor="grey", capsize=4)
    ax.scatter(es.season, games, c=colors, s=70, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(POLICY_SEASON - 0.5, color="grey", ls="--")
    ax.set_xticks(es.season, [season_label(s) for s in es.season], rotation=30)
    ax.set_ylabel(f"Star minus non-star gap vs. {season_label(BASE_SEASON)}\n(games per 82, 95% CI)")
    ax.set_title("Star availability gap by season, relative to the last pre-policy season", loc="left", fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES / "event_study.png", dpi=200)
    plt.close(fig)


def plot_b2b(df: pd.DataFrame) -> None:
    stars = df[df.star == 1].assign(period=lambda d: d.post.map({0: "Before policy", 1: "After policy"}))
    stars["night"] = stars.b2b.map({0: "Normal rest", 1: "2nd night of back-to-back"})
    rates = stars.groupby(["period", "night"]).played.mean().mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=rates, x="night", y="played", hue="period", ax=ax,
                hue_order=["Before policy", "After policy"], palette=[OTHER_COLOR, STAR_COLOR])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", padding=3)
    ax.set_xlabel("")
    ax.set_ylabel("Stars: % of available games played")
    ax.set_ylim(60, 90)
    ax.set_title("Did stars stop sitting out back-to-backs?", loc="left", fontweight="bold")
    ax.legend(frameon=False, title="")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES / "back_to_backs.png", dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    panel = pd.read_csv(PROCESSED / "player_games.csv")

    official = sample_for(panel, "official_star")
    broad = sample_for(panel, "broad_star")
    no_covid = lambda d: d[~d.season.isin(COVID_SEASONS)]

    results = pd.DataFrame([
        did(official, "Official stars (main)"),
        did(broad, "Broad stars"),
        did(no_covid(official), "Official stars, no COVID seasons"),
        did(no_covid(broad), "Broad stars, no COVID seasons"),
        b2b(official, "Back-to-back 2nd nights, official stars (triple diff)"),
        b2b(broad, "Back-to-back 2nd nights, broad stars (triple diff)"),
    ])
    results.to_csv(RESULTS / "did_results.csv", index=False)

    es = event_study(official)
    es.to_csv(RESULTS / "event_study.csv", index=False)
    plot_trend(official)
    plot_event_study(es)
    plot_b2b(official)

    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    print("\nEvent study (games per 82 vs 2022-23):")
    print(es.assign(games=es.gap * GAMES, ci=1.96 * es.se * GAMES)[["season", "games", "ci"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
