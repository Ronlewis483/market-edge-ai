import pandas as pd
import numpy as np


def build_nfl_pregame_features(games):
    """
    Build leakage-safe NFL pregame features.

    Every feature for a game is calculated ONLY from games
    played before that game's kickoff.
    """

    if games is None or games.empty:
        return pd.DataFrame()

    df = games.copy()

    # ---------------------------------------------------------
    # CLEAN / SORT DATA
    # ---------------------------------------------------------

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
        subset=[
            "start_time",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    )

    df = df.sort_values("start_time").reset_index(drop=True)

    # ---------------------------------------------------------
    # TEAM HISTORY STORAGE
    # ---------------------------------------------------------

    team_history = {}

    feature_rows = []

    def get_team_history(team):
        if team not in team_history:
            team_history[team] = []

        return team_history[team]

    # ---------------------------------------------------------
    # HELPER: CALCULATE TEAM FEATURES
    # ---------------------------------------------------------

    def calculate_team_features(history, game_time):
        if not history:
            return {
                "games_played": 0,
                "win_pct": 0.50,
                "avg_points_for": 0.0,
                "avg_points_against": 0.0,
                "avg_point_diff": 0.0,
                "recent_3_win_pct": 0.50,
                "recent_5_win_pct": 0.50,
                "recent_5_point_diff": 0.0,
                "days_rest": 7.0,
            }

        history_df = pd.DataFrame(history)

        games_played = len(history_df)

        win_pct = history_df["win"].mean()

        avg_points_for = history_df["points_for"].mean()

        avg_points_against = history_df["points_against"].mean()

        avg_point_diff = history_df["point_diff"].mean()

        recent_3 = history_df.tail(3)

        recent_5 = history_df.tail(5)

        recent_3_win_pct = recent_3["win"].mean()

        recent_5_win_pct = recent_5["win"].mean()

        recent_5_point_diff = recent_5["point_diff"].mean()

        last_game_time = history_df.iloc[-1]["game_time"]

        days_rest = (
            game_time - last_game_time
        ).total_seconds() / 86400.0

        return {
            "games_played": games_played,
            "win_pct": win_pct,
            "avg_points_for": avg_points_for,
            "avg_points_against": avg_points_against,
            "avg_point_diff": avg_point_diff,
            "recent_3_win_pct": recent_3_win_pct,
            "recent_5_win_pct": recent_5_win_pct,
            "recent_5_point_diff": recent_5_point_diff,
            "days_rest": days_rest,
        }

    # ---------------------------------------------------------
    # WALK FORWARD THROUGH EVERY GAME
    # ---------------------------------------------------------

    for _, game in df.iterrows():

        game_time = game["start_time"]

        home_team = game["home_team"]
        away_team = game["away_team"]

        home_history = get_team_history(home_team)
        away_history = get_team_history(away_team)

        # Calculate features BEFORE adding current result.
        home_features = calculate_team_features(
            home_history,
            game_time,
        )

        away_features = calculate_team_features(
            away_history,
            game_time,
        )

        home_score = game["home_score"]
        away_score = game["away_score"]

        # Tie = no binary target.
        if home_score == away_score:
            home_win = np.nan
        else:
            home_win = int(home_score > away_score)

        feature_row = {
            "game_id": game.get("game_id"),
            "season_id": game.get("season_id"),
            "start_time": game_time,
            "home_team": home_team,
            "away_team": away_team,

            # Target
            "home_win": home_win,

            # Home pregame information
            "home_games_played":
                home_features["games_played"],

            "home_win_pct":
                home_features["win_pct"],

            "home_avg_points_for":
                home_features["avg_points_for"],

            "home_avg_points_against":
                home_features["avg_points_against"],

            "home_avg_point_diff":
                home_features["avg_point_diff"],

            "home_recent_3_win_pct":
                home_features["recent_3_win_pct"],

            "home_recent_5_win_pct":
                home_features["recent_5_win_pct"],

            "home_recent_5_point_diff":
                home_features["recent_5_point_diff"],

            "home_days_rest":
                home_features["days_rest"],

            # Away pregame information
            "away_games_played":
                away_features["games_played"],

            "away_win_pct":
                away_features["win_pct"],

            "away_avg_points_for":
                away_features["avg_points_for"],

            "away_avg_points_against":
                away_features["avg_points_against"],

            "away_avg_point_diff":
                away_features["avg_point_diff"],

            "away_recent_3_win_pct":
                away_features["recent_3_win_pct"],

            "away_recent_5_win_pct":
                away_features["recent_5_win_pct"],

            "away_recent_5_point_diff":
                away_features["recent_5_point_diff"],

            "away_days_rest":
                away_features["days_rest"],
        }

        # -----------------------------------------------------
        # DIFFERENTIAL FEATURES
        # -----------------------------------------------------

        feature_row["win_pct_diff"] = (
            home_features["win_pct"]
            - away_features["win_pct"]
        )

        feature_row["avg_point_diff_diff"] = (
            home_features["avg_point_diff"]
            - away_features["avg_point_diff"]
        )

        feature_row["recent_5_win_pct_diff"] = (
            home_features["recent_5_win_pct"]
            - away_features["recent_5_win_pct"]
        )

        feature_row["recent_5_point_diff_diff"] = (
            home_features["recent_5_point_diff"]
            - away_features["recent_5_point_diff"]
        )

        feature_row["rest_diff"] = (
            home_features["days_rest"]
            - away_features["days_rest"]
        )

        feature_rows.append(feature_row)

        # -----------------------------------------------------
        # UPDATE HISTORY ONLY AFTER FEATURES ARE CREATED
        # -----------------------------------------------------

        if home_score > away_score:
            home_result = 1.0
            away_result = 0.0

        elif away_score > home_score:
            home_result = 0.0
            away_result = 1.0

        else:
            # Tie counts as half a win for historical form.
            home_result = 0.5
            away_result = 0.5

        home_history.append(
            {
                "game_time": game_time,
                "win": home_result,
                "points_for": home_score,
                "points_against": away_score,
                "point_diff": home_score - away_score,
            }
        )

        away_history.append(
            {
                "game_time": game_time,
                "win": away_result,
                "points_for": away_score,
                "points_against": home_score,
                "point_diff": away_score - home_score,
            }
        )

    features = pd.DataFrame(feature_rows)

    return features
