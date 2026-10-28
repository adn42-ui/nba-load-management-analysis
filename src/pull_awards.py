"""Pull career awards (All-Star, All-NBA, All-Defense, ...) for candidate players.

Candidates are anyone who averaged 20+ minutes over 20+ games in any season of
data/raw/player_logs_*.csv, which covers every All-Star with room to spare.
Run src/pull_data.py first.

Saves data/raw/awards.csv. Run from the project root:
    python src/pull_awards.py
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playerawards

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = RAW_DIR / "awards.csv"


def candidate_ids() -> list[int]:
    logs = pd.concat(pd.read_csv(f) for f in RAW_DIR.glob("player_logs_*.csv"))
    per_season = logs.groupby(["PLAYER_ID", "SEASON_ID"]).MIN.agg(["count", "mean"])
    good = per_season[(per_season["count"] >= 20) & (per_season["mean"] >= 20)]
    return sorted(good.index.get_level_values("PLAYER_ID").unique())


def fetch_awards(player_id: int, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            resp = playerawards.PlayerAwards(player_id=player_id, timeout=60)
            return resp.get_data_frames()[0]
        except Exception as exc:  # stats.nba.com times out often; back off and retry
            print(f"  {player_id} attempt {attempt} failed: {exc}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not fetch awards for {player_id}")


def main() -> None:
    ids = candidate_ids()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["PERSON_ID"])
    todo = [p for p in ids if p not in set(done.PERSON_ID)]
    print(f"{len(ids)} candidates, {len(todo)} left to fetch")

    frames = [done]
    for i, pid in enumerate(todo, 1):
        df = fetch_awards(pid)
        # Players with no awards return no rows; keep a marker so we don't refetch them.
        frames.append(df if len(df) else pd.DataFrame({"PERSON_ID": [pid]}))
        if i % 50 == 0 or i == len(todo):
            pd.concat(frames).to_csv(OUT, index=False)  # checkpoint
            print(f"  {i}/{len(todo)} fetched")
        time.sleep(0.6)


if __name__ == "__main__":
    main()
