
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

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

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

    Long historical ranges are requested one calendar
    year at a time and then combined. This avoids relying
    on one large multi-season MLB Stats API response.

    Only games with a final result are included.

    Returns one row per completed game.
    """

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if start_ts > end_ts:
        raise ValueError(
            "MLB start_date must be before end_date."
        )

    games = []

    # --------------------------------------
    # REQUEST ONE CALENDAR YEAR AT A TIME
    # --------------------------------------

    for year in range(
        start_ts.year,
        end_ts.year + 1,
    ):

        chunk_start = max(
            start_ts,
            pd.Timestamp(
                year=year,
                month=1,
                day=1,
            ),
        )

        chunk_end = min(
            end_ts,
            pd.Timestamp(
                year=year,
                month=12,
                day=31,
            ),
        )

        params = {
            "sportId": 1,
            "startDate":
                chunk_start.strftime("%Y-%m-%d"),
            "endDate":
                chunk_end.strftime("%Y-%m-%d"),
            "gameTypes": "R",
            "hydrate": "probablePitcher",
        }

        response = requests.get(
            MLB_API_URL,
            params=params,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        for date_entry in data.get(
            "dates",
            [],
        ):

            for game in date_entry.get(
                "games",
                [],
            ):

                status = game.get(
                    "status",
                    {},
                )

                if (
                    status.get(
                        "abstractGameState"
                    )
                    != "Final"
                ):
                    continue

                teams = game.get(
                    "teams",
                    {},
                )

                home = teams.get(
                    "home",
                    {},
                )

                away = teams.get(
                    "away",
                    {},
                )

                home_score = home.get(
                    "score"
                )

                away_score = away.get(
                    "score"
                )

                if (
                    home_score is None
                    or away_score is None
                ):
                    continue

                home_team = home.get(
                    "team",
                    {},
                )

                away_team = away.get(
                    "team",
                    {},
                )

                if (
                    not home_team.get("id")
                    or not away_team.get("id")
                ):
                    continue

                games.append(
                    {
                        "game_id":
                            game.get("gamePk"),

                        "season_id":
                            game.get("season"),

                        "start_time":
                            game.get("gameDate"),

                        "home_team":
                            home_team.get("name"),

                        "away_team":
                            away_team.get("name"),

                        "home_team_id":
                            home_team.get("id"),

                        "away_team_id":
                            away_team.get("id"),

                        "home_score":
                            home_score,

                        "away_score":
                            away_score,
                    }
                )

    # --------------------------------------
    # BUILD COMBINED HISTORICAL DATASET
    # --------------------------------------

    df = pd.DataFrame(games)

    if df.empty:
        return df

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "game_id",
            "start_time",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
        ]
    )

    df = df.drop_duplicates(
        subset=["game_id"],
        keep="first",
    )

    df = df.sort_values(
        [
            "start_time",
            "game_id",
        ]
    ).reset_index(drop=True)

    return df

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


# ==========================================
# MLB GAME-WINNER PREDICTION MODEL
# ==========================================

MLB_MODEL_FEATURES = [
    "games_played_diff",
    "win_pct_diff",
    "avg_runs_for_diff",
    "avg_runs_against_diff",
    "avg_run_diff_diff",
    "recent_5_win_pct_diff",
    "recent_10_win_pct_diff",
    "recent_5_run_diff_diff",
]


def train_mlb_prediction_model(features):
    """
    Train and evaluate an MLB game-winner model.

    Uses chronological training and testing.

    IMPORTANT:
    This function is for research only.
    Historical feature availability must be
    verified before production use.
    """

    if features is None or features.empty:
        raise ValueError(
            "No MLB training features available."
        )

    df = features.copy()

    df = df.dropna(
        subset=["home_win"]
    )

    df = df.sort_values(
        ["start_time", "game_id"]
    ).reset_index(drop=True)

    if len(df) < 500:
        raise ValueError(
            "At least 500 historical MLB games "
            "are required for this initial test."
        )

    # --------------------------------------
    # CHRONOLOGICAL TRAIN / TEST SPLIT
    # --------------------------------------

    split_index = int(
        len(df) * 0.80
    )

    # Keep games sharing the boundary
    # timestamp on the same side.
    boundary_time = df.iloc[
        split_index
    ]["start_time"]

    train = df[
        df["start_time"] < boundary_time
    ].copy()

    test = df[
        df["start_time"] >= boundary_time
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Invalid chronological train/test split."
        )

    X_train = train[
        MLB_MODEL_FEATURES
    ].fillna(0)

    y_train = train[
        "home_win"
    ].astype(int)

    X_test = test[
        MLB_MODEL_FEATURES
    ].fillna(0)

    y_test = test[
        "home_win"
    ].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain both "
            "home wins and home losses."
        )

    # --------------------------------------
    # TRAIN THE MODEL
    # --------------------------------------

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------
    # GENERATE HISTORICAL PREDICTIONS
    # --------------------------------------

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------
    # HISTORICAL BASELINE
    # --------------------------------------

    baseline_probability = float(
        y_train.mean()
    )

    baseline_predictions = np.full(
        len(y_test),
        int(baseline_probability >= 0.50),
    )

    baseline_probabilities = np.full(
        len(y_test),
        baseline_probability,
    )

    # --------------------------------------
    # MODEL EVALUATION
    # --------------------------------------

    model_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    model_brier = brier_score_loss(
        y_test,
        probabilities,
    )

    baseline_brier = brier_score_loss(
        y_test,
        baseline_probabilities,
    )

    model_log_loss = log_loss(
        y_test,
        probabilities,
        labels=[0, 1],
    )

    baseline_log_loss = log_loss(
        y_test,
        baseline_probabilities,
        labels=[0, 1],
    )

    auc = None

    if y_test.nunique() == 2:
        auc = float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        )

    # --------------------------------------
    # VALIDATION REPORT
    # --------------------------------------

    metrics = {
        "training_games": int(
            len(train)
        ),

        "test_games": int(
            len(test)
        ),

        "training_start": str(
            train["start_time"].min()
        ),

        "training_end": str(
            train["start_time"].max()
        ),

        "test_start": str(
            test["start_time"].min()
        ),

        "test_end": str(
            test["start_time"].max()
        ),

        "model_accuracy": float(
            model_accuracy
        ),

        "baseline_accuracy": float(
            baseline_accuracy
        ),

        "model_brier": float(
            model_brier
        ),

        "baseline_brier": float(
            baseline_brier
        ),

        "model_log_loss": float(
            model_log_loss
        ),

        "baseline_log_loss": float(
            baseline_log_loss
        ),

        "auc": auc,

        "test_home_win_rate": float(
            y_test.mean()
        ),

        "passes_benchmarks": bool(
            model_brier < baseline_brier
            and model_log_loss < baseline_log_loss
            and model_accuracy > baseline_accuracy
            and auc is not None
            and auc > 0.50
        ),
    }

    return {
        "model": model,
        "metrics": metrics,
    }

# ==========================================
# MLB V1 WALK-FORWARD BENCHMARK
# ==========================================

def run_mlb_walkforward_v1(
    features,
    min_train_games=500,
    retrain_every=100,
):
    """
    Leakage-safe expanding-window walk-forward benchmark
    for the existing MLB V1 feature set.

    Each prediction is generated using only games that
    occurred before that game's start_time.

    Games sharing the same start timestamp are predicted
    together so results from simultaneous games cannot
    leak into one another.
    """

    if features is None or features.empty:
        raise ValueError(
            "No MLB features available for walk-forward testing."
        )

    df = features.copy()

    required = (
        [
            "game_id",
            "start_time",
            "home_win",
        ]
        + MLB_MODEL_FEATURES
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"MLB walk-forward data is missing columns: {missing}"
        )

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "game_id",
            "start_time",
            "home_win",
        ]
    )

    df = df.sort_values(
        [
            "start_time",
            "game_id",
        ]
    ).reset_index(drop=True)

    if len(df) <= min_train_games:
        raise ValueError(
            "Not enough MLB games for walk-forward testing."
        )

    rows = []

    model = None
    trained_through_time = None
    predictions_since_fit = retrain_every

    # Process identical timestamps as a single prediction batch.
    time_groups = list(
        df.groupby(
            "start_time",
            sort=True,
        )
    )

    for game_time, current_group in time_groups:

        train = df[
            df["start_time"] < game_time
        ].copy()

        if len(train) < min_train_games:
            continue

        y_train = (
            train["home_win"]
            .astype(int)
        )

        if y_train.nunique() < 2:
            continue

        needs_retrain = (
            model is None
            or predictions_since_fit >= retrain_every
        )

        if needs_retrain:

            X_train = (
                train[MLB_MODEL_FEATURES]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .fillna(0.0)
            )

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            )

            model.fit(
                X_train,
                y_train,
            )

            trained_through_time = (
                train["start_time"].max()
            )

            predictions_since_fit = 0

        X_current = (
            current_group[MLB_MODEL_FEATURES]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(0.0)
        )

        probabilities = model.predict_proba(
            X_current
        )[:, 1]

        for position, (_, game) in enumerate(
            current_group.iterrows()
        ):

            probability = float(
                probabilities[position]
            )

            actual = int(
                game["home_win"]
            )

            predicted = int(
                probability >= 0.50
            )

            confidence = float(
                max(
                    probability,
                    1.0 - probability,
                )
            )

            rows.append(
                {
                    "game_id":
                        game["game_id"],

                    "season_id":
                        game.get("season_id"),

                    "start_time":
                        game_time,

                    "home_team":
                        game.get("home_team"),

                    "away_team":
                        game.get("away_team"),

                    "actual_home_win":
                        actual,

                    "home_win_probability":
                        probability,

                    "predicted_home_win":
                        predicted,

                    "confidence":
                        confidence,

                    "correct":
                        int(
                            predicted == actual
                        ),

                    "trained_through":
                        trained_through_time,

                    "model_version":
                        "MLB_V1",
                }
            )

        predictions_since_fit += len(
            current_group
        )

    predictions = pd.DataFrame(rows)

    if predictions.empty:
        raise ValueError(
            "MLB V1 walk-forward produced no predictions."
        )

    return (
        predictions
        .sort_values(
            [
                "start_time",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_mlb_walkforward_v1(
    predictions,
):
    """
    Summarize out-of-sample MLB V1 walk-forward performance.
    """

    if predictions is None or predictions.empty:
        raise ValueError(
            "No MLB walk-forward predictions to summarize."
        )

    df = predictions.copy()

    y_true = (
        df["actual_home_win"]
        .astype(int)
    )

    probabilities = (
        pd.to_numeric(
            df["home_win_probability"],
            errors="coerce",
        )
        .clip(
            1e-6,
            1.0 - 1e-6,
        )
    )

    predicted = (
        df["predicted_home_win"]
        .astype(int)
    )

    valid = (
        probabilities.notna()
        & y_true.notna()
    )

    df = df.loc[valid].copy()
    y_true = y_true.loc[valid]
    probabilities = probabilities.loc[valid]
    predicted = predicted.loc[valid]

    if df.empty:
        raise ValueError(
            "No valid MLB walk-forward predictions."
        )

    accuracy = float(
        accuracy_score(
            y_true,
            predicted,
        )
    )

    brier = float(
        brier_score_loss(
            y_true,
            probabilities,
        )
    )

    model_log_loss = float(
        log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        )
    )

    auc = None

    if y_true.nunique() == 2:
        auc = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    # --------------------------------------
    # SIMPLE HOME-RATE BASELINE
    # --------------------------------------
    #
    # This is intentionally simple.
    # The more important comparison later
    # will be V2A against this exact V1.
    # --------------------------------------

    baseline_probability = float(
        y_true.mean()
    )

    baseline_probabilities = np.full(
        len(y_true),
        baseline_probability,
        dtype=float,
    )

    baseline_prediction = int(
        baseline_probability >= 0.50
    )

    baseline_predictions = np.full(
        len(y_true),
        baseline_prediction,
        dtype=int,
    )

    baseline_accuracy = float(
        accuracy_score(
            y_true,
            baseline_predictions,
        )
    )

    baseline_brier = float(
        brier_score_loss(
            y_true,
            baseline_probabilities,
        )
    )

    baseline_log_loss = float(
        log_loss(
            y_true,
            baseline_probabilities,
            labels=[0, 1],
        )
    )

    # --------------------------------------
    # CONFIDENCE BUCKETS
    # --------------------------------------

    def confidence_bucket(value):

        if value >= 0.70:
            return "70%+"

        if value >= 0.60:
            return "60-69.9%"

        if value >= 0.55:
            return "55-59.9%"

        return "50-54.9%"

    df["confidence_bucket"] = (
        df["confidence"]
        .apply(confidence_bucket)
    )

    confidence_summary = (
        df.groupby(
            "confidence_bucket",
            as_index=False,
        )
        .agg(
            predictions=(
                "correct",
                "size",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
            average_confidence=(
                "confidence",
                "mean",
            ),
        )
    )

    bucket_order = {
        "50-54.9%": 0,
        "55-59.9%": 1,
        "60-69.9%": 2,
        "70%+": 3,
    }

    confidence_summary[
        "_order"
    ] = confidence_summary[
        "confidence_bucket"
    ].map(bucket_order)

    confidence_summary = (
        confidence_summary
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    # --------------------------------------
    # SEASON-BY-SEASON PERFORMANCE
    # --------------------------------------

    season_summary = (
        df.groupby(
            "season_id",
            dropna=False,
            as_index=False,
        )
        .agg(
            predictions=(
                "correct",
                "size",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
        )
    )

    return {
        "prediction_count":
            int(len(df)),

        "accuracy":
            accuracy,

        "auc":
            auc,

        "brier":
            brier,

        "log_loss":
            model_log_loss,

        "baseline_accuracy":
            baseline_accuracy,

        "baseline_brier":
            baseline_brier,

        "baseline_log_loss":
            baseline_log_loss,

        "first_prediction":
            str(
                df["start_time"].min()
            ),

        "last_prediction":
            str(
                df["start_time"].max()
            ),

        "confidence_summary":
            confidence_summary,

        "season_summary":
            season_summary,

        "predictions":
            df,
    }
