
"""
MARKET EDGE AI V5
MLB HISTORICAL RESEARCH ENGINE

Phase 1:
- Historical game retrieval
- Pregame team statistics
- Recent performance
- Leakage-aware feature construction

No live betting recommendations are generated here.
"""

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


MLB_API_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
)


# ==========================================
# HISTORICAL MLB GAME RETRIEVAL
# ==========================================

def fetch_mlb_games(
    start_date,
    end_date,
):
    """
    Retrieve completed MLB regular-season games.

    Only games with a final result are included.

    Returns one row per completed game.
    """

    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "gameTypes": "R",
    }

    response = requests.get(
        MLB_API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    games = []

    for date_entry in data.get("dates", []):

        for game in date_entry.get("games", []):

            status = game.get("status", {})

            if status.get("abstractGameState") != "Final":
                continue

            teams = game.get("teams", {})

            home = teams.get("home", {})
            away = teams.get("away", {})

            home_score = home.get("score")
            away_score = away.get("score")

            if home_score is None or away_score is None:
                continue

            home_team = home.get("team", {})
            away_team = away.get("team", {})

            if not home_team.get("id") or not away_team.get("id"):
                continue

            games.append(
                {
                    "game_id": game.get("gamePk"),
                    "season_id": game.get("season"),
                    "start_time": game.get("gameDate"),

                    "home_team": home_team.get("name"),
                    "away_team": away_team.get("name"),

                    "home_team_id": home_team.get("id"),
                    "away_team_id": away_team.get("id"),

                    "home_score": home_score,
                    "away_score": away_score,
                }
            )

    df = pd.DataFrame(games)

    if df.empty:
        return df

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["start_time"]
    )

    df = df.drop_duplicates(
        subset=["game_id"]
    )

    return df.sort_values(
        ["start_time", "game_id"]
    ).reset_index(drop=True)


# ==========================================
# TEAM HISTORICAL STATISTICS
# ==========================================

def calculate_team_features(
    history,
):
    """
    Calculate team performance from
    previously available game results.
    """

    if not history:

        return {
            "games_played": 0,
            "win_pct": 0.50,
            "avg_runs_for": 0.0,
            "avg_runs_against": 0.0,
            "avg_run_diff": 0.0,
            "recent_5_win_pct": 0.50,
            "recent_10_win_pct": 0.50,
            "recent_5_run_diff": 0.0,
        }

    df = pd.DataFrame(history)

    recent_5 = df.tail(5)
    recent_10 = df.tail(10)

    return {
        "games_played": int(len(df)),

        "win_pct": float(
            df["win"].mean()
        ),

        "avg_runs_for": float(
            df["runs_for"].mean()
        ),

        "avg_runs_against": float(
            df["runs_against"].mean()
        ),

        "avg_run_diff": float(
            df["run_diff"].mean()
        ),

        "recent_5_win_pct": float(
            recent_5["win"].mean()
        ),

        "recent_10_win_pct": float(
            recent_10["win"].mean()
        ),

        "recent_5_run_diff": float(
            recent_5["run_diff"].mean()
        ),
    }


# ==========================================
# MLB PREGAME FEATURE BUILDER
# ==========================================

def build_mlb_pregame_features(
    games,
):
    """
    Build historical MLB matchup features.

    Team histories are updated only after
    all games with the same start timestamp
    have received their features.

    IMPORTANT:
    Historical result availability must be
    verified before these features are used
    for production-grade backtesting.
    """

    if games is None or games.empty:
        return pd.DataFrame()

    df = games.copy()

    required_columns = [
        "game_id",
        "start_time",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing MLB columns: {missing}"
        )

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    df["home_score"] = pd.to_numeric(
        df["home_score"],
        errors="coerce",
    )

    df["away_score"] = pd.to_numeric(
        df["away_score"],
        errors="coerce",
    )

    df = df.dropna(
        subset=required_columns
    )

    df = df.drop_duplicates(
        subset=["game_id"]
    )

    df = df.sort_values(
        ["start_time", "game_id"]
    ).reset_index(drop=True)

    team_history = defaultdict(list)

    feature_rows = []

    feature_names = [
        "games_played",
        "win_pct",
        "avg_runs_for",
        "avg_runs_against",
        "avg_run_diff",
        "recent_5_win_pct",
        "recent_10_win_pct",
        "recent_5_run_diff",
    ]

    # Process games in chronological groups.
    for game_time, group in df.groupby(
        "start_time",
        sort=True,
    ):

        pending_updates = []

        for _, game in group.iterrows():

            home_id = int(
                game["home_team_id"]
            )

            away_id = int(
                game["away_team_id"]
            )

            home_features = calculate_team_features(
                team_history[home_id]
            )

            away_features = calculate_team_features(
                team_history[away_id]
            )

            home_score = float(
                game["home_score"]
            )

            away_score = float(
                game["away_score"]
            )

            # A tie is not a binary win/loss.
            home_win = (
                np.nan
                if home_score == away_score
                else int(home_score > away_score)
            )

            row = {
                "game_id": game["game_id"],
                "season_id": game.get("season_id"),
                "start_time": game_time,

                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),

                "home_team_id": home_id,
                "away_team_id": away_id,

                "home_win": home_win,
            }

            # Store pregame team features.
            for name in feature_names:

                row[f"home_{name}"] = (
                    home_features[name]
                )

                row[f"away_{name}"] = (
                    away_features[name]
                )

                row[f"{name}_diff"] = (
                    home_features[name]
                    - away_features[name]
                )

            feature_rows.append(row)

            # Store results for later updates.
            if home_score > away_score:
                home_result = 1.0
                away_result = 0.0

            elif away_score > home_score:
                home_result = 0.0
                away_result = 1.0

            else:
                home_result = 0.5
                away_result = 0.5

            pending_updates.append(
                (
                    home_id,
                    {
                        "win": home_result,
                        "runs_for": home_score,
                        "runs_against": away_score,
                        "run_diff":
                            home_score - away_score,
                    },
                )
            )

            pending_updates.append(
                (
                    away_id,
                    {
                        "win": away_result,
                        "runs_for": away_score,
                        "runs_against": home_score,
                        "run_diff":
                            away_score - home_score,
                    },
                )
            )

        # Update histories only after the
        # entire timestamp group is processed.
        for team_id, result in pending_updates:

            team_history[team_id].append(
                result
            )

    return pd.DataFrame(feature_rows)


# ==========================================
# RESEARCH DATASET SUMMARY
# ==========================================

def summarize_mlb_dataset(
    features,
):
    """
    Return basic historical dataset diagnostics.
    """

    if features is None or features.empty:

        return {
            "success": False,
            "message": "No MLB features available.",
        }

    valid = features.dropna(
        subset=["home_win"]
    )

    return {
        "success": True,

        "total_games": int(
            len(features)
        ),

        "labeled_games": int(
            len(valid)
        ),

        "home_win_rate": float(
            valid["home_win"].mean()
        ) if not valid.empty else None,

        "first_game": str(
            features["start_time"].min()
        ),

        "last_game": str(
            features["start_time"].max()
        ),

        "feature_count": int(
            len(features.columns)
        ),
    }
