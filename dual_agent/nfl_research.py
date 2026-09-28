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
        
            # Final result data
            # Used only AFTER a game is completed when constructing
            # features for future games.
            "home_score": home_score,
            "away_score": away_score,
        
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
    if retrain_every < 1 or min_train_games < 1:
        raise ValueError("Training and retraining intervals must be positive.")

    df = feature_games.copy()
    df = df.dropna(subset=["home_win"]).copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
    feature_columns = [
        "win_pct_diff", "avg_point_diff_diff", "recent_5_win_pct_diff",
        "recent_5_point_diff_diff", "rest_diff",
    ]
    missing_columns = [col for col in feature_columns if col not in df.columns]
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

        test_row = df.iloc[[i]].copy()
        train_df = df[df["start_time"] < test_row["start_time"].iloc[0]].copy()
        if len(train_df) < min_train_games:
            continue

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
            or model_training_end >= test_row["start_time"].iloc[0]
        ):
            model = LogisticRegression(
                max_iter=2000
            )

            model.fit(
                X_train,
                y_train,
            )
            model_training_end = train_df["start_time"].max()

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

# ============================================================
# NFL MODEL V2
# Expanded leakage-safe winner prediction model
#
# IMPORTANT:
# - V1 remains untouched.
# - V2 uses only information available before kickoff.
# - V2 can be benchmarked against V1 before becoming live.
# ============================================================


def build_nfl_v2_model_features(feature_games):
    """
    Build the expanded NFL V2 model feature set from the
    leakage-safe pregame information already created by
    build_nfl_pregame_features().

    This function does NOT use final scores from the game
    being predicted.

    Final scores stored in feature_games are used only by
    the historical feature builder to construct FUTURE
    team state.
    """

    if feature_games is None or feature_games.empty:
        raise ValueError(
            "NFL feature dataset is empty."
        )

    df = feature_games.copy()

    # --------------------------------------------------------
    # REQUIRED PREGAME COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "start_time",
        "home_win",

        "home_games_played",
        "away_games_played",

        "home_win_pct",
        "away_win_pct",

        "home_avg_points_for",
        "away_avg_points_for",

        "home_avg_points_against",
        "away_avg_points_against",

        "home_avg_point_diff",
        "away_avg_point_diff",

        "home_recent_3_win_pct",
        "away_recent_3_win_pct",

        "home_recent_5_win_pct",
        "away_recent_5_win_pct",

        "home_recent_5_point_diff",
        "away_recent_5_point_diff",

        "home_days_rest",
        "away_days_rest",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "NFL V2 missing required pregame columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # CLEAN TIME
    # --------------------------------------------------------

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    # ========================================================
    # V2 DIFFERENTIAL FEATURES
    # ========================================================

    # --------------------------------------------------------
    # OVERALL TEAM QUALITY
    # --------------------------------------------------------

    df["v2_win_pct_diff"] = (
        df["home_win_pct"]
        - df["away_win_pct"]
    )

    df["v2_point_diff_diff"] = (
        df["home_avg_point_diff"]
        - df["away_avg_point_diff"]
    )

    # --------------------------------------------------------
    # OFFENSIVE STRENGTH
    #
    # Positive =
    # home team historically scores more points.
    # --------------------------------------------------------

    df["v2_offense_diff"] = (
        df["home_avg_points_for"]
        - df["away_avg_points_for"]
    )

    # --------------------------------------------------------
    # DEFENSIVE STRENGTH
    #
    # Positive =
    # away team allows more points than home team.
    #
    # This orientation means positive values generally favor
    # the home defense.
    # --------------------------------------------------------

    df["v2_defense_diff"] = (
        df["away_avg_points_against"]
        - df["home_avg_points_against"]
    )

    # --------------------------------------------------------
    # OFFENSE VS OPPOSING DEFENSE
    #
    # Home offense scoring ability compared with how many
    # points the away defense typically allows.
    # --------------------------------------------------------

    df["v2_home_offense_matchup"] = (
        df["home_avg_points_for"]
        - df["away_avg_points_against"]
    )

    # --------------------------------------------------------
    # AWAY OFFENSE VS HOME DEFENSE
    #
    # Converted so positive values favor the HOME team.
    # --------------------------------------------------------

    df["v2_away_offense_matchup"] = (
        df["home_avg_points_against"]
        - df["away_avg_points_for"]
    )

    # --------------------------------------------------------
    # COMBINED MATCHUP ADVANTAGE
    # --------------------------------------------------------

    df["v2_matchup_advantage"] = (
        df["v2_home_offense_matchup"]
        + df["v2_away_offense_matchup"]
    )

    # --------------------------------------------------------
    # SHORT-TERM FORM
    # --------------------------------------------------------

    df["v2_recent_3_win_diff"] = (
        df["home_recent_3_win_pct"]
        - df["away_recent_3_win_pct"]
    )

    df["v2_recent_5_win_diff"] = (
        df["home_recent_5_win_pct"]
        - df["away_recent_5_win_pct"]
    )

    df["v2_recent_5_point_diff"] = (
        df["home_recent_5_point_diff"]
        - df["away_recent_5_point_diff"]
    )

    # --------------------------------------------------------
    # FORM TREND
    #
    # Measures whether recent performance is stronger or
    # weaker than the team's longer-term performance.
    # --------------------------------------------------------

    df["v2_home_form_trend"] = (
        df["home_recent_5_win_pct"]
        - df["home_win_pct"]
    )

    df["v2_away_form_trend"] = (
        df["away_recent_5_win_pct"]
        - df["away_win_pct"]
    )

    df["v2_form_trend_diff"] = (
        df["v2_home_form_trend"]
        - df["v2_away_form_trend"]
    )

    # --------------------------------------------------------
    # REST / FATIGUE
    # --------------------------------------------------------

    df["v2_rest_diff"] = (
        df["home_days_rest"]
        - df["away_days_rest"]
    )

    # Cap extreme values.
    df["v2_rest_diff"] = (
        df["v2_rest_diff"]
        .clip(
            lower=-14.0,
            upper=14.0,
        )
    )

    # --------------------------------------------------------
    # EXPERIENCE / SAMPLE DEPTH
    #
    # Helps the model distinguish early-season estimates
    # from teams with larger current-season samples.
    # --------------------------------------------------------

    df["v2_games_played_diff"] = (
        df["home_games_played"]
        - df["away_games_played"]
    )

    df["v2_home_sample_size"] = (
        df["home_games_played"]
    )

    df["v2_away_sample_size"] = (
        df["away_games_played"]
    )

    # --------------------------------------------------------
    # EARLY-SEASON FLAG
    #
    # 1 when either team has fewer than four historical
    # games feeding its pregame statistics.
    # --------------------------------------------------------

    df["v2_early_season"] = (
        (
            (df["home_games_played"] < 4)
            | (df["away_games_played"] < 4)
        )
        .astype(int)
    )

    return df


# ============================================================
# V2 FEATURE LIST
# ============================================================


def get_nfl_v2_feature_columns():
    """
    Return the exact feature set used by NFL V2.

    Keeping this centralized prevents the historical
    validator and future prediction engine from silently
    using different feature sets.
    """

    return [
        "v2_win_pct_diff",
        "v2_point_diff_diff",

        "v2_offense_diff",
        "v2_defense_diff",

        "v2_home_offense_matchup",
        "v2_away_offense_matchup",
        "v2_matchup_advantage",

        "v2_recent_3_win_diff",
        "v2_recent_5_win_diff",
        "v2_recent_5_point_diff",

        "v2_home_form_trend",
        "v2_away_form_trend",
        "v2_form_trend_diff",

        "v2_rest_diff",

        "v2_games_played_diff",
        "v2_home_sample_size",
        "v2_away_sample_size",

        "v2_early_season",
    ]


# ============================================================
# NFL V2 WALK-FORWARD VALIDATION
# ============================================================


def run_nfl_walkforward_model_v2(
    feature_games,
    min_train_games=100,
    retrain_every=25,
):
    """
    Leakage-safe walk-forward validation for NFL V2.

    V2 remains separate from V1.

    Every prediction is generated using ONLY games whose
    kickoff occurred before the game being predicted.
    """

    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        brier_score_loss,
        log_loss,
    )

    from sklearn.pipeline import Pipeline

    from sklearn.preprocessing import StandardScaler

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if feature_games is None or feature_games.empty:
        raise ValueError(
            "NFL V2 feature dataset is empty."
        )

    if min_train_games < 1:
        raise ValueError(
            "NFL V2 min_train_games must be positive."
        )

    if retrain_every < 1:
        raise ValueError(
            "NFL V2 retrain_every must be positive."
        )

    # --------------------------------------------------------
    # BUILD V2 FEATURES
    # --------------------------------------------------------

    df = build_nfl_v2_model_features(
        feature_games
    )

    feature_columns = (
        get_nfl_v2_feature_columns()
    )

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "start_time",
            "home_win",
        ]
    ).copy()

    df = df.sort_values(
        "start_time"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # NUMERIC CLEANING
    # --------------------------------------------------------

    for column in feature_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        df[column] = df[column].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

    df = df.dropna(
        subset=feature_columns
    ).copy()

    if len(df) <= min_train_games:
        raise ValueError(
            "Not enough NFL games for V2 "
            "walk-forward validation. "
            f"Found {len(df)}, "
            f"need more than {min_train_games}."
        )

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    predictions = []

    model = None
    model_training_end = None

    for i in range(
        min_train_games,
        len(df),
    ):

        test_row = df.iloc[
            [i]
        ].copy()

        prediction_time = (
            test_row[
                "start_time"
            ].iloc[0]
        )

        # ----------------------------------------------------
        # STRICT LEAKAGE PROTECTION
        # ----------------------------------------------------

        train_df = df[
            df["start_time"]
            < prediction_time
        ].copy()

        if len(train_df) < min_train_games:
            continue

        X_train = train_df[
            feature_columns
        ]

        y_train = train_df[
            "home_win"
        ].astype(int)

        X_test = test_row[
            feature_columns
        ]

        # Logistic regression requires both outcomes.
        if y_train.nunique() < 2:
            continue

        # ----------------------------------------------------
        # RETRAIN MODEL
        # ----------------------------------------------------

        should_retrain = (
            model is None
            or (
                (i - min_train_games)
                % retrain_every
                == 0
            )
        )

        if (
            model_training_end
            is not None
            and model_training_end
            >= prediction_time
        ):
            should_retrain = True

        if should_retrain:

            model = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            C=1.0,
                        ),
                    ),
                ]
            )

            model.fit(
                X_train,
                y_train,
            )

            model_training_end = (
                train_df[
                    "start_time"
                ].max()
            )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        home_probability = float(
            model.predict_proba(
                X_test
            )[0][1]
        )

        away_probability = (
            1.0
            - home_probability
        )

        prediction = int(
            home_probability
            >= 0.50
        )

        actual = int(
            test_row[
                "home_win"
            ].iloc[0]
        )

        confidence = max(
            home_probability,
            away_probability,
        )

        # ----------------------------------------------------
        # SAVE OUT-OF-SAMPLE PREDICTION
        # ----------------------------------------------------

        predictions.append(
            {
                "game_id":
                    test_row[
                        "game_id"
                    ].iloc[0],

                "season_id":
                    test_row[
                        "season_id"
                    ].iloc[0],

                "start_time":
                    prediction_time,

                "home_team":
                    test_row[
                        "home_team"
                    ].iloc[0],

                "away_team":
                    test_row[
                        "away_team"
                    ].iloc[0],

                "probability":
                    home_probability,

                "home_probability":
                    home_probability,

                "away_probability":
                    away_probability,

                "confidence":
                    confidence,

                "prediction":
                    prediction,

                "actual":
                    actual,

                "correct":
                    int(
                        prediction
                        == actual
                    ),
            }
        )

    # ========================================================
    # RESULTS
    # ========================================================

    predictions_df = pd.DataFrame(
        predictions
    )

    if predictions_df.empty:
        raise ValueError(
            "NFL V2 walk-forward model "
            "produced no predictions."
        )

    y_true = predictions_df[
        "actual"
    ]

    y_prob = predictions_df[
        "probability"
    ]

    y_pred = predictions_df[
        "prediction"
    ]

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # BRIER SCORE
    #
    # Lower is better.
    # --------------------------------------------------------

    brier = brier_score_loss(
        y_true,
        y_prob,
    )

    # --------------------------------------------------------
    # LOG LOSS
    #
    # Lower is better.
    # --------------------------------------------------------

    logloss = log_loss(
        y_true,
        y_prob,
        labels=[
            0,
            1,
        ],
    )

    # --------------------------------------------------------
    # AUC
    #
    # Higher is better.
    # --------------------------------------------------------

    if y_true.nunique() > 1:

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

    else:

        auc = np.nan

    # --------------------------------------------------------
    # HOME-TEAM BASELINE
    # --------------------------------------------------------

    baseline_home_prediction = np.ones(
        len(y_true),
        dtype=int,
    )

    baseline_accuracy = accuracy_score(
        y_true,
        baseline_home_prediction,
    )

    # --------------------------------------------------------
    # CONFIDENCE PERFORMANCE
    # --------------------------------------------------------

    high_confidence_df = (
        predictions_df[
            predictions_df[
                "confidence"
            ]
            >= 0.70
        ]
    )

    if high_confidence_df.empty:

        high_confidence_accuracy = (
            np.nan
        )

        high_confidence_count = 0

    else:

        high_confidence_accuracy = float(
            high_confidence_df[
                "correct"
            ].mean()
        )

        high_confidence_count = len(
            high_confidence_df
        )

    # ========================================================
    # RETURN V2 VALIDATION
    # ========================================================

    return {
        "success": True,

        "model_version":
            "V2",

        "model_name":
            "NFL V2 Expanded Logistic Regression",

        "feature_columns":
            feature_columns,

        "feature_count":
            len(feature_columns),

        "total_feature_games":
            len(df),

        "training_start_games":
            min_train_games,

        "prediction_count":
            len(predictions_df),

        "accuracy":
            float(accuracy),

        "auc":
            (
                float(auc)
                if not np.isnan(auc)
                else np.nan
            ),

        "brier":
            float(brier),

        "log_loss":
            float(logloss),

        "baseline_home_accuracy":
            float(baseline_accuracy),

        "accuracy_vs_baseline":
            float(
                accuracy
                - baseline_accuracy
            ),

        "high_confidence_count":
            int(
                high_confidence_count
            ),

        "high_confidence_accuracy":
            (
                float(
                    high_confidence_accuracy
                )
                if not np.isnan(
                    high_confidence_accuracy
                )
                else np.nan
            ),

        "predictions":
            predictions_df,

        "probability_bands":
            analyze_nfl_probability_bands(
                predictions_df
            ),
    }


# ============================================================
# V1 VS V2 BENCHMARK
# ============================================================


def compare_nfl_v1_v2(
    feature_games,
    min_train_games=100,
    retrain_every=25,
):
    """
    Run V1 and V2 against the same historical dataset.

    This function does NOT promote V2.

    It only returns an apples-to-apples benchmark so we can
    determine whether V2 actually improved the NFL model.
    """

    v1 = run_nfl_walkforward_model(
        feature_games=feature_games,
        min_train_games=min_train_games,
        retrain_every=retrain_every,
    )

    v2 = run_nfl_walkforward_model_v2(
        feature_games=feature_games,
        min_train_games=min_train_games,
        retrain_every=retrain_every,
    )

    comparison = pd.DataFrame(
        [
            {
                "model":
                    "V1",

                "features":
                    len(
                        v1[
                            "feature_columns"
                        ]
                    ),

                "predictions":
                    v1[
                        "prediction_count"
                    ],

                "accuracy":
                    v1[
                        "accuracy"
                    ],

                "auc":
                    v1[
                        "auc"
                    ],

                "brier":
                    v1[
                        "brier"
                    ],

                "log_loss":
                    v1[
                        "log_loss"
                    ],

                "baseline_accuracy":
                    v1[
                        "baseline_home_accuracy"
                    ],

                "accuracy_vs_baseline":
                    v1[
                        "accuracy_vs_baseline"
                    ],
            },

            {
                "model":
                    "V2",

                "features":
                    len(
                        v2[
                            "feature_columns"
                        ]
                    ),

                "predictions":
                    v2[
                        "prediction_count"
                    ],

                "accuracy":
                    v2[
                        "accuracy"
                    ],

                "auc":
                    v2[
                        "auc"
                    ],

                "brier":
                    v2[
                        "brier"
                    ],

                "log_loss":
                    v2[
                        "log_loss"
                    ],

                "baseline_accuracy":
                    v2[
                        "baseline_home_accuracy"
                    ],

                "accuracy_vs_baseline":
                    v2[
                        "accuracy_vs_baseline"
                    ],
            },
        ]
    )

    return {
        "success": True,

        "v1":
            v1,

        "v2":
            v2,

        "comparison":
            comparison,

        "accuracy_change":
            float(
                v2["accuracy"]
                - v1["accuracy"]
            ),

        "auc_change":
            (
                float(
                    v2["auc"]
                    - v1["auc"]
                )
                if (
                    not np.isnan(
                        v1["auc"]
                    )
                    and not np.isnan(
                        v2["auc"]
                    )
                )
                else np.nan
            ),

        "brier_change":
            float(
                v2["brier"]
                - v1["brier"]
            ),

        "log_loss_change":
            float(
                v2["log_loss"]
                - v1["log_loss"]
            ),
    }

def predict_nfl_matchup(feature_games, future_features):
    """Predict one future matchup from completed games before its kickoff."""
    from sklearn.linear_model import LogisticRegression

    if feature_games is None or feature_games.empty:
        raise ValueError("NFL feature dataset is empty.")
    if future_features is None or future_features.empty:
        raise ValueError("Future NFL matchup features are empty.")
    features = [
        "win_pct_diff", "avg_point_diff_diff", "recent_5_win_pct_diff",
        "recent_5_point_diff_diff", "rest_diff",
    ]
    missing = [c for c in features if c not in feature_games.columns or c not in future_features.columns]
    if missing:
        raise ValueError(f"Missing NFL model features: {missing}")
    future = future_features.iloc[[0]].copy()
    game_time = pd.to_datetime(future.iloc[0]["start_time"], utc=True, errors="coerce")
    if pd.isna(game_time):
        raise ValueError("Upcoming NFL game time is invalid.")
    df = feature_games.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df = df[(df["start_time"] < game_time) & df["home_win"].notna()].copy()
    df = df.dropna(subset=features).sort_values("start_time")
    if df.empty or df["home_win"].nunique() < 2:
        raise ValueError("Not enough completed historical NFL games from both outcome classes before kickoff.")
    if future[features].isna().any().any():
        raise ValueError("Future NFL matchup contains missing model features.")
    model = LogisticRegression(max_iter=2000)
    model.fit(df[features], df["home_win"].astype(int))
    home_probability = float(model.predict_proba(future[features])[0, 1])
    away_probability = 1.0 - home_probability
    home_team = future.iloc[0]["home_team"]
    away_team = future.iloc[0]["away_team"]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "start_time": game_time,
        "home_win_probability": home_probability,
        "away_win_probability": away_probability,
        "predicted_team": home_team if home_probability >= 0.5 else away_team,
        "confidence": max(home_probability, away_probability),
        "training_games": len(df),
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
    Build leakage-safe features for a future NFL matchup.
    """

    if feature_games is None or feature_games.empty:
        raise ValueError(
            "NFL feature dataframe is empty."
        )

    print(
        "NFL FUTURE DEBUG | columns:",
        list(feature_games.columns),
    )

    df = feature_games.copy()

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    prediction_time = pd.to_datetime(
        game_time,
        utc=True,
        errors="coerce",
    )

    if pd.isna(prediction_time):
        raise ValueError(
            f"Invalid future game time: {game_time}"
        )

    # --------------------------------------------
    # ONLY games completed before prediction time
    # --------------------------------------------

    history = df[
        df["start_time"] < prediction_time
    ].copy()

    history = history.sort_values(
        "start_time"
    ).reset_index(drop=True)

    if history.empty:
        raise ValueError(
            "No historical NFL games exist before "
            "this matchup."
        )

    # ============================================
    # REBUILD CURRENT TEAM STATE
    # ============================================

    def build_team_state(team):

        team_games = history[
            (history["home_team"] == team)
            | (history["away_team"] == team)
        ].copy()

        team_games = team_games.sort_values(
            "start_time"
        )

        if team_games.empty:
            return {
                "games": 0,
                "wins": 0,
                "win_pct": 0.5,
                "avg_point_diff": 0.0,
                "recent_5_win_pct": 0.5,
                "recent_5_point_diff": 0.0,
                "last_game_time": None,
            }

        results = []
        point_diffs = []

        for _, game in team_games.iterrows():

            home_score = game.get("home_score")
            away_score = game.get("away_score")

            if pd.isna(home_score) or pd.isna(away_score):
                continue

            home_score = float(home_score)
            away_score = float(away_score)

            if game["home_team"] == team:

                team_score = home_score
                opponent_score = away_score

            else:

                team_score = away_score
                opponent_score = home_score

            point_diff = (
                team_score - opponent_score
            )

            point_diffs.append(point_diff)

            if team_score > opponent_score:
                results.append(1.0)

            elif team_score < opponent_score:
                results.append(0.0)

            else:
                # NFL ties count as half a win for
                # win-percentage purposes.
                results.append(0.5)

        games_played = len(results)

        if games_played == 0:
            return {
                "games": 0,
                "wins": 0,
                "win_pct": 0.5,
                "avg_point_diff": 0.0,
                "recent_5_win_pct": 0.5,
                "recent_5_point_diff": 0.0,
                "last_game_time": None,
            }

        wins = sum(results)

        win_pct = (
            wins / games_played
        )

        avg_point_diff = (
            sum(point_diffs)
            / len(point_diffs)
        )

        recent_results = results[-5:]
        recent_point_diffs = point_diffs[-5:]

        recent_5_win_pct = (
            sum(recent_results)
            / len(recent_results)
        )

        recent_5_point_diff = (
            sum(recent_point_diffs)
            / len(recent_point_diffs)
        )

        last_game_time = (
            team_games.iloc[-1]["start_time"]
        )

        return {
            "games": games_played,
            "wins": wins,
            "win_pct": win_pct,
            "avg_point_diff": avg_point_diff,
            "recent_5_win_pct": recent_5_win_pct,
            "recent_5_point_diff": recent_5_point_diff,
            "last_game_time": last_game_time,
        }

    # --------------------------------------------
    # BUILD HOME / AWAY STATE
    # --------------------------------------------

    home_state = build_team_state(
        home_team
    )

    away_state = build_team_state(
        away_team
    )

    # ============================================
    # REST
    # ============================================

    def calculate_rest_days(
        last_game_time,
    ):

        if last_game_time is None:
            return 7.0

        rest = (
            prediction_time
            - last_game_time
        ).total_seconds() / 86400.0

        # Keep extreme offseason gaps from
        # dominating the model.
        return min(
            max(rest, 0.0),
            14.0,
        )

    home_rest = calculate_rest_days(
        home_state["last_game_time"]
    )

    away_rest = calculate_rest_days(
        away_state["last_game_time"]
    )

    # ============================================
    # MATCHUP DIFFERENTIALS
    # ============================================

    future_features = {
        "start_time": prediction_time,
        "home_team": home_team,
        "away_team": away_team,
    
        "win_pct_diff":
            home_state["win_pct"]
            - away_state["win_pct"],
            
        "avg_point_diff_diff":
            home_state["avg_point_diff"]
            - away_state["avg_point_diff"],

        "recent_5_win_pct_diff":
            home_state["recent_5_win_pct"]
            - away_state["recent_5_win_pct"],

        "recent_5_point_diff_diff":
            home_state["recent_5_point_diff"]
            - away_state["recent_5_point_diff"],

        "rest_diff":
            home_rest
            - away_rest,
    }

    return pd.DataFrame(
        [future_features]
    )



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
