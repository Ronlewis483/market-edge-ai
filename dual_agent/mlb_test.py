
"""
MARKET EDGE AI V5
MLB RESEARCH ENGINE TEST
"""

from dual_agent.mlb_research import (
    fetch_mlb_games,
    build_mlb_pregame_features,
    summarize_mlb_dataset,
)


def run_mlb_test():

    print("Starting MLB research engine test...")

    # Retrieve a small historical sample.
    games = fetch_mlb_games(
        "2025-04-01",
        "2025-04-15",
    )

    print(f"Historical games retrieved: {len(games)}")

    if games.empty:
        print("ERROR: No historical MLB games returned.")
        return

    # Generate pregame team statistics.
    features = build_mlb_pregame_features(games)

    print(f"Pregame feature rows: {len(features)}")

    # Display the historical dataset summary.
    summary = summarize_mlb_dataset(features)

    print("MLB DATASET SUMMARY")

    for key, value in summary.items():
        print(f"{key}: {value}")

    # Display a few example matchups.
    print("SAMPLE PREGAME FEATURES")

    columns = [
        "home_team",
        "away_team",
        "home_win",
        "home_win_pct",
        "away_win_pct",
        "win_pct_diff",
    ]

    print(
        features[columns].head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    run_mlb_test()
