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

def run_nfl_walkforward_model(
    feature_games,
    min_train_games=100,
    retrain_every=25,
):
    """
    Run leakage-safe walk-forward NFL validation.

    The model trains only on games that occurred BEFORE
    the game being predicted.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        brier_score_loss,
        log_loss,
    )

    if feature_games is None or feature_games.empty:
        raise ValueError("NFL feature dataset is empty.")

    df = feature_games.copy()

    # Remove ties because home_win is NaN for tied games.
    df = df.dropna(subset=["home_win"]).copy()

    df = df.sort_values("start_time").reset_index(drop=True)

    feature_columns = [
        "win_pct_diff",
        "avg_point_diff_diff",
        "recent_5_win_pct_diff",
        "recent_5_point_diff_diff",
        "rest_diff",
    ]

    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing NFL model features: {missing_columns}"
        )

    if len(df) <= min_train_games:
        raise ValueError(
            f"Not enough NFL games for walk-forward validation. "
            f"Found {len(df)}, need more than {min_train_games}."
        )

    predictions = []

    model = None

    for i in range(min_train_games, len(df)):

        train_df = df.iloc[:i].copy()
        test_row = df.iloc[[i]].copy()

        X_train = train_df[feature_columns]
        y_train = train_df["home_win"].astype(int)

        X_test = test_row[feature_columns]

        # LogisticRegression needs both classes in training data.
        if y_train.nunique() < 2:
            continue

        # Retrain periodically instead of fitting on every single game.
        if (
            model is None
            or (i - min_train_games) % retrain_every == 0
        ):
            model = LogisticRegression(
                max_iter=2000
            )

            model.fit(
                X_train,
                y_train,
            )

        home_probability = float(
            model.predict_proba(X_test)[0][1]
        )

        prediction = int(
            home_probability >= 0.50
        )

        actual = int(
            test_row["home_win"].iloc[0]
        )

        predictions.append(
            {
                "game_id": test_row["game_id"].iloc[0],
                "season_id": test_row["season_id"].iloc[0],
                "start_time": test_row["start_time"].iloc[0],
                "home_team": test_row["home_team"].iloc[0],
                "away_team": test_row["away_team"].iloc[0],
                "probability": home_probability,
                "prediction": prediction,
                "actual": actual,
                "correct": int(prediction == actual),
            }
        )

    predictions_df = pd.DataFrame(predictions)

    if predictions_df.empty:
        raise ValueError(
            "NFL walk-forward model produced no predictions."
        )

    y_true = predictions_df["actual"]
    y_prob = predictions_df["probability"]
    y_pred = predictions_df["prediction"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    brier = brier_score_loss(
        y_true,
        y_prob,
    )

    logloss = log_loss(
        y_true,
        y_prob,
        labels=[0, 1],
    )

    if y_true.nunique() > 1:
        auc = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        auc = np.nan

    baseline_home_prediction = np.ones(
        len(y_true),
        dtype=int,
    )

    baseline_accuracy = accuracy_score(
        y_true,
        baseline_home_prediction,
    )

    return {
        "success": True,
        "model_name": "NFL Logistic Regression Walk-Forward",
        "feature_columns": feature_columns,
        "total_feature_games": len(df),
        "training_start_games": min_train_games,
        "prediction_count": len(predictions_df),
        "accuracy": float(accuracy),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "brier": float(brier),
        "log_loss": float(logloss),
        "baseline_home_accuracy": float(baseline_accuracy),
        "accuracy_vs_baseline": float(
            accuracy - baseline_accuracy
        ),
        
        "predictions": predictions_df,
        "probability_bands": analyze_nfl_probability_bands(
            predictions_df
        ),
    }

def analyze_nfl_probability_bands(predictions_df):
    """
    Evaluate historical NFL prediction accuracy
    across different model-confidence ranges.

    Uses walk-forward predictions rather than
    training-set predictions.
    """

    import pandas as pd
    import numpy as np

    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame()

    df = predictions_df.copy()

    required_columns = {
        "probability",
        "actual",
        "correct",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    # Convert home-team probability into the
    # probability of the model's selected winner.

    df["model_confidence"] = np.maximum(
        df["probability"],
        1.0 - df["probability"],
    )

    # Assign each prediction to a confidence band.

    bins = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.000001,
    ]

    labels = [
        "50-59%",
        "60-69%",
        "70-79%",
        "80-89%",
        "90-100%",
    ]

    df["confidence_band"] = pd.cut(
        df["model_confidence"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    # Calculate historical performance.

    results = (
        df.groupby(
            "confidence_band",
            observed=False,
        )
        .agg(
            total_predictions=("correct", "count"),
            correct_predictions=("correct", "sum"),
            average_confidence=("model_confidence", "mean"),
            actual_win_rate=("correct", "mean"),
        )
        .reset_index()
    )

    # Compare predicted confidence with actual results.

    results["calibration_gap"] = (
        results["actual_win_rate"]
        - results["average_confidence"]
    )

    # Calculate uncertainty around the observed win rate.
    # Wilson 95% confidence interval.

    n = results["total_predictions"].astype(float)
    p = results["actual_win_rate"]
    z = 1.96

    denominator = 1 + z**2 / n.replace(0, np.nan)

    center = (
        p + z**2 / (2 * n.replace(0, np.nan))
    ) / denominator

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / n.replace(0, np.nan)
            + z**2 / (4 * n.replace(0, np.nan)**2)
        )
        / denominator
    )

    results["win_rate_lower_95"] = center - margin
    results["win_rate_upper_95"] = center + margin

    # Remove bands without historical predictions.

    results = results[
        results["total_predictions"] > 0
    ].copy()

    return results




def build_nfl_future_matchup_features(
    feature_games,
    home_team,
    away_team,
    game_time,
):
    """
    Build leakage-safe features for a future NFL matchup using only
    historical games that occurred before the future game's start time.
    """

    if feature_games is None or feature_games.empty:
        raise ValueError("NFL feature dataset is empty.")

    df = feature_games.copy()

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    game_time = pd.to_datetime(
        game_time,
        utc=True,
        errors="coerce",
    )

    if pd.isna(game_time):
        raise ValueError("Future NFL game time is invalid.")

    history = df[df["start_time"] < game_time].copy()

    if history.empty:
        raise ValueError(
            "No historical NFL games exist before this matchup."
        )

    def get_team_history(team):
        team_games = history[
            (history["home_team"] == team)
            | (history["away_team"] == team)
        ].copy()

        return team_games.sort_values("start_time")

    def summarize_team(team):
        games = get_team_history(team)

        if games.empty:
            raise ValueError(
                f"No historical NFL games found for {team}."
            )

        wins = []
        point_diffs = []

        for _, game in games.iterrows():

            is_home = game["home_team"] == team

            if is_home:
                team_score = game["home_score"]
                opponent_score = game["away_score"]
            else:
                team_score = game["away_score"]
                opponent_score = game["home_score"]

            if pd.isna(team_score) or pd.isna(opponent_score):
                continue

            wins.append(
                1 if team_score > opponent_score else 0
            )

            point_diffs.append(
                team_score - opponent_score
            )

        if not wins:
            raise ValueError(
                f"No completed historical NFL games found for {team}."
            )

        recent_wins = wins[-5:]
        recent_point_diffs = point_diffs[-5:]

        last_game_time = games["start_time"].max()

        rest_days = (
            game_time - last_game_time
        ).total_seconds() / 86400.0

        return {
            "win_pct": sum(wins) / len(wins),
            "avg_point_diff": sum(point_diffs) / len(point_diffs),
            "recent_5_win_pct": (
                sum(recent_wins) / len(recent_wins)
            ),
            "recent_5_point_diff": (
                sum(recent_point_diffs)
                / len(recent_point_diffs)
            ),
            "rest_days": rest_days,
        }

    home = summarize_team(home_team)
    away = summarize_team(away_team)

    return pd.DataFrame(
        [
            {
                "home_team": home_team,
                "away_team": away_team,
                "start_time": game_time,
                "win_pct_diff": (
                    home["win_pct"] - away["win_pct"]
                ),
                "avg_point_diff_diff": (
                    home["avg_point_diff"]
                    - away["avg_point_diff"]
                ),
                "recent_5_win_pct_diff": (
                    home["recent_5_win_pct"]
                    - away["recent_5_win_pct"]
                ),
                "recent_5_point_diff_diff": (
                    home["recent_5_point_diff"]
                    - away["recent_5_point_diff"]
                ),
                "rest_diff": (
                    home["rest_days"]
                    - away["rest_days"]
                ),
            }
        ]
    )


def predict_nfl_matchup(
    feature_games,
    future_features,
):
    """
    Train the NFL model on historical data and predict one future matchup.
    """

    from sklearn.linear_model import LogisticRegression

    feature_columns = [
        "win_pct_diff",
        "avg_point_diff_diff",
        "recent_5_win_pct_diff",
        "recent_5_point_diff_diff",
        "rest_diff",
    ]

    if feature_games is None or feature_games.empty:
        raise ValueError("NFL feature dataset is empty.")

    if future_features is None or future_features.empty:
        raise ValueError("Future NFL matchup features are empty.")

    df = feature_games.copy()

    df = df.dropna(
        subset=feature_columns + ["home_win"]
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid NFL training rows are available."
        )

    X_train = df[feature_columns]
    y_train = df["home_win"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "NFL training data needs both home wins and home losses."
        )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train,
    )

    X_future = future_features[
        feature_columns
    ]

    home_probability = float(
        model.predict_proba(X_future)[0][1]
    )

    away_probability = 1.0 - home_probability

    if home_probability >= away_probability:
        predicted_team = future_features.iloc[0][
            "home_team"
        ]
        confidence = home_probability
    else:
        predicted_team = future_features.iloc[0][
            "away_team"
        ]
        confidence = away_probability

    return {
        "home_team": future_features.iloc[0]["home_team"],
        "away_team": future_features.iloc[0]["away_team"],
        "start_time": future_features.iloc[0]["start_time"],
        "home_win_probability": home_probability,
        "away_win_probability": away_probability,
        "predicted_team": predicted_team,
        "confidence": confidence,
    }

def evaluate_nfl_prediction_reliability(
    historical_predictions,
    min_samples=30,
):
    """
    Evaluate historical NFL prediction accuracy and calibration.

    historical_predictions must contain:
        probability: predicted probability of a home win
        actual: actual game result (1 = home win, 0 = away win)

    Use only completed, out-of-sample predictions.
    """

    import pandas as pd
    import numpy as np

    required_columns = ["probability", "actual"]

    if historical_predictions is None:
        raise ValueError(
            "Historical NFL predictions are unavailable."
        )

    missing = [
        col for col in required_columns
        if col not in historical_predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing historical prediction columns: {missing}"
        )

    df = historical_predictions[
        required_columns
    ].copy()

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="coerce",
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df = df.dropna()

    df = df[
        df["probability"].between(0, 1)
        & df["actual"].isin([0, 1])
    ].copy()

    sample_size = len(df)

    if sample_size == 0:
        return {
            "historical_accuracy": None,
            "brier_score": None,
            "sample_size": 0,
            "reliability_passed": False,
            "reason": "No valid historical predictions.",
        }

    df["predicted"] = (
        df["probability"] >= 0.50
    ).astype(int)

    df["correct"] = (
        df["predicted"] == df["actual"]
    ).astype(int)

    historical_accuracy = float(
        df["correct"].mean()
    )

    brier_score = float(
        np.mean(
            (
                df["probability"]
                - df["actual"]
            ) ** 2
        )
    )

    reliability_passed = (
        sample_size >= min_samples
        and historical_accuracy >= 0.60
        and brier_score < 0.25
    )

    return {
        "historical_accuracy": historical_accuracy,
        "brier_score": brier_score,
        "sample_size": sample_size,
        "reliability_passed": bool(reliability_passed),
        "reason": (
            "Historical validation passed."
            if reliability_passed
            else "Historical validation requirements not met."
        ),
    }
