
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

                home_probable_pitcher = (
                    home.get(
                        "probablePitcher",
                        {},
                    )
                )
                
                away_probable_pitcher = (
                    away.get(
                        "probablePitcher",
                        {},
                    )
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
            
                        "home_starting_pitcher_id":
                            home_probable_pitcher.get("id"),
            
                        "home_starting_pitcher":
                            home_probable_pitcher.get("fullName"),
            
                        "away_starting_pitcher_id":
                            away_probable_pitcher.get("id"),
            
                        "away_starting_pitcher":
                            away_probable_pitcher.get("fullName"),
            
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

# ==========================================
# MLB V2A — STARTING PITCHER ENGINE
# ==========================================

MLB_BOXSCORE_URL = (
    "https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
)


def _innings_to_outs(innings):
    """
    Convert MLB innings notation to outs.

    Examples:
        5.0 -> 15 outs
        5.1 -> 16 outs
        5.2 -> 17 outs
    """

    if innings is None:
        return 0

    value = str(innings).strip()

    if not value:
        return 0

    try:
        if "." in value:
            whole, partial = value.split(".", 1)

            whole = int(whole)
            partial = int(partial[:1] or 0)

            partial = max(
                0,
                min(partial, 2),
            )

            return (
                whole * 3
                + partial
            )

        return int(float(value)) * 3

    except (TypeError, ValueError):
        return 0


def _safe_number(value, default=0.0):
    """
    Convert an MLB API statistic to float safely.
    """

    try:
        if value is None or value == "":
            return float(default)

        return float(value)

    except (TypeError, ValueError):
        return float(default)


def fetch_mlb_starting_pitcher_game_logs(
    games,
    timeout=30,
):
    """
    Retrieve the completed-game pitching line for the
    listed home and away starters.

    This produces POSTGAME pitcher records.

    These records are NOT themselves pregame features.
    They are later accumulated chronologically so that
    each game's features use only earlier starts.
    """

    if games is None or games.empty:
        return pd.DataFrame()

    required = [
        "game_id",
        "start_time",
        "season_id",
        "home_starting_pitcher_id",
        "away_starting_pitcher_id",
    ]

    missing = [
        column
        for column in required
        if column not in games.columns
    ]

    if missing:
        raise ValueError(
            "MLB pitcher log retrieval is missing "
            f"columns: {missing}"
        )

    df = games.copy()

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[
                "game_id",
                "start_time",
            ]
        )
        .drop_duplicates(
            subset=["game_id"]
        )
        .sort_values(
            [
                "start_time",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    rows = []

    session = requests.Session()

    for _, game in df.iterrows():

        game_id = int(
            game["game_id"]
        )

        url = MLB_BOXSCORE_URL.format(
            game_id=game_id
        )

        try:

            response = session.get(
                url,
                timeout=timeout,
            )

            response.raise_for_status()

            payload = response.json()

        except (
            requests.RequestException,
            ValueError,
        ):
            continue

        teams = payload.get(
            "teams",
            {},
        )

        starter_targets = [
            (
                "home",
                game.get(
                    "home_starting_pitcher_id"
                ),
                game.get(
                    "home_starting_pitcher"
                ),
            ),
            (
                "away",
                game.get(
                    "away_starting_pitcher_id"
                ),
                game.get(
                    "away_starting_pitcher"
                ),
            ),
        ]

        for side, pitcher_id, pitcher_name in starter_targets:

            if pd.isna(pitcher_id):
                continue

            pitcher_id = int(
                pitcher_id
            )

            side_data = teams.get(
                side,
                {},
            )

            players = side_data.get(
                "players",
                {},
            )

            player = players.get(
                f"ID{pitcher_id}",
                {},
            )

            if not player:
                continue

            pitching = (
                player.get(
                    "stats",
                    {},
                )
                .get(
                    "pitching",
                    {},
                )
            )

            if not pitching:
                continue

            outs = _innings_to_outs(
                pitching.get(
                    "inningsPitched"
                )
            )

            innings = (
                outs / 3.0
            )

            hits = _safe_number(
                pitching.get("hits")
            )

            runs = _safe_number(
                pitching.get("runs")
            )

            earned_runs = _safe_number(
                pitching.get(
                    "earnedRuns"
                )
            )

            walks = _safe_number(
                pitching.get(
                    "baseOnBalls"
                )
            )

            strikeouts = _safe_number(
                pitching.get(
                    "strikeOuts"
                )
            )

            home_runs = _safe_number(
                pitching.get(
                    "homeRuns"
                )
            )

            batters_faced = _safe_number(
                pitching.get(
                    "battersFaced"
                )
            )

            pitches = _safe_number(
                pitching.get(
                    "numberOfPitches"
                )
            )

            strikes = _safe_number(
                pitching.get(
                    "strikes"
                )
            )

            rows.append(
                {
                    "game_id":
                        game_id,

                    "season_id":
                        game.get(
                            "season_id"
                        ),

                    "start_time":
                        game["start_time"],

                    "side":
                        side,

                    "pitcher_id":
                        pitcher_id,

                    "pitcher_name":
                        (
                            player.get(
                                "person",
                                {},
                            ).get(
                                "fullName"
                            )
                            or pitcher_name
                        ),

                    "outs":
                        int(outs),

                    "innings":
                        float(innings),

                    "hits":
                        float(hits),

                    "runs":
                        float(runs),

                    "earned_runs":
                        float(earned_runs),

                    "walks":
                        float(walks),

                    "strikeouts":
                        float(strikeouts),

                    "home_runs":
                        float(home_runs),

                    "batters_faced":
                        float(batters_faced),

                    "pitches":
                        float(pitches),

                    "strikes":
                        float(strikes),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "start_time",
                "game_id",
                "side",
            ]
        )
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


# ==========================================
# PREGAME PITCHER FEATURES
# ==========================================

def calculate_pitcher_features(
    history,
):
    """
    Calculate pitcher quality using only starts
    completed before the current game.
    """

    if not history:

        return {
            "starter_prior_starts": 0.0,
            "starter_prior_innings": 0.0,
            "starter_era": 4.50,
            "starter_whip": 1.30,
            "starter_k_per_9": 8.00,
            "starter_bb_per_9": 3.00,
            "starter_hr_per_9": 1.20,
            "starter_k_bb_ratio": 2.50,
            "starter_recent_3_era": 4.50,
            "starter_recent_3_whip": 1.30,
            "starter_recent_3_k_per_9": 8.00,
        }

    df = pd.DataFrame(history)

    outs = float(
        df["outs"].sum()
    )

    innings = (
        outs / 3.0
    )

    hits = float(
        df["hits"].sum()
    )

    earned_runs = float(
        df["earned_runs"].sum()
    )

    walks = float(
        df["walks"].sum()
    )

    strikeouts = float(
        df["strikeouts"].sum()
    )

    home_runs = float(
        df["home_runs"].sum()
    )

    era = (
        earned_runs * 9.0 / innings
        if innings > 0
        else 4.50
    )

    whip = (
        (walks + hits) / innings
        if innings > 0
        else 1.30
    )

    k_per_9 = (
        strikeouts * 9.0 / innings
        if innings > 0
        else 8.00
    )

    bb_per_9 = (
        walks * 9.0 / innings
        if innings > 0
        else 3.00
    )

    hr_per_9 = (
        home_runs * 9.0 / innings
        if innings > 0
        else 1.20
    )

    k_bb_ratio = (
        strikeouts / walks
        if walks > 0
        else strikeouts
    )

    recent = df.tail(3)

    recent_outs = float(
        recent["outs"].sum()
    )

    recent_innings = (
        recent_outs / 3.0
    )

    recent_hits = float(
        recent["hits"].sum()
    )

    recent_walks = float(
        recent["walks"].sum()
    )

    recent_earned_runs = float(
        recent["earned_runs"].sum()
    )

    recent_strikeouts = float(
        recent["strikeouts"].sum()
    )

    recent_era = (
        recent_earned_runs
        * 9.0
        / recent_innings
        if recent_innings > 0
        else 4.50
    )

    recent_whip = (
        (
            recent_walks
            + recent_hits
        )
        / recent_innings
        if recent_innings > 0
        else 1.30
    )

    recent_k_per_9 = (
        recent_strikeouts
        * 9.0
        / recent_innings
        if recent_innings > 0
        else 8.00
    )

    return {
        "starter_prior_starts":
            float(len(df)),

        "starter_prior_innings":
            float(innings),

        "starter_era":
            float(era),

        "starter_whip":
            float(whip),

        "starter_k_per_9":
            float(k_per_9),

        "starter_bb_per_9":
            float(bb_per_9),

        "starter_hr_per_9":
            float(hr_per_9),

        "starter_k_bb_ratio":
            float(k_bb_ratio),

        "starter_recent_3_era":
            float(recent_era),

        "starter_recent_3_whip":
            float(recent_whip),

        "starter_recent_3_k_per_9":
            float(recent_k_per_9),
    }


MLB_V2A_PITCHER_FEATURES = [
    "starter_prior_starts_diff",
    "starter_prior_innings_diff",
    "starter_era_diff",
    "starter_whip_diff",
    "starter_k_per_9_diff",
    "starter_bb_per_9_diff",
    "starter_hr_per_9_diff",
    "starter_k_bb_ratio_diff",
    "starter_recent_3_era_diff",
    "starter_recent_3_whip_diff",
    "starter_recent_3_k_per_9_diff",
]


MLB_V2A_MODEL_FEATURES = (
    MLB_MODEL_FEATURES
    + MLB_V2A_PITCHER_FEATURES
)


def build_mlb_v2a_features(
    games,
    team_features,
    pitcher_logs,
):
    """
    Add leakage-safe starting-pitcher features to the
    existing V1 team feature dataset.

    Pitcher histories are updated only AFTER all games
    at the current timestamp receive their pregame
    pitcher features.
    """

    if games is None or games.empty:
        raise ValueError(
            "No MLB games available for V2A."
        )

    if team_features is None or team_features.empty:
        raise ValueError(
            "No MLB V1 team features available for V2A."
        )

    if pitcher_logs is None or pitcher_logs.empty:
        raise ValueError(
            "No MLB pitcher logs available for V2A."
        )

    required_game_columns = [
        "game_id",
        "start_time",
        "home_starting_pitcher_id",
        "away_starting_pitcher_id",
    ]

    missing = [
        column
        for column in required_game_columns
        if column not in games.columns
    ]

    if missing:
        raise ValueError(
            f"MLB V2A games are missing columns: {missing}"
        )

    game_df = games.copy()

    game_df["start_time"] = pd.to_datetime(
        game_df["start_time"],
        utc=True,
        errors="coerce",
    )

    game_df = (
        game_df.dropna(
            subset=[
                "game_id",
                "start_time",
            ]
        )
        .drop_duplicates(
            subset=["game_id"]
        )
        .sort_values(
            [
                "start_time",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    logs = pitcher_logs.copy()

    logs["start_time"] = pd.to_datetime(
        logs["start_time"],
        utc=True,
        errors="coerce",
    )

    logs = logs.dropna(
        subset=[
            "game_id",
            "start_time",
            "pitcher_id",
        ]
    )

    logs_by_game = defaultdict(list)

    for _, log_row in logs.iterrows():

        logs_by_game[
            int(log_row["game_id"])
        ].append(
            log_row.to_dict()
        )

    pitcher_history = defaultdict(list)

    pitcher_rows = []

    for game_time, group in game_df.groupby(
        "start_time",
        sort=True,
    ):

        pending_updates = []

        for _, game in group.iterrows():

            game_id = int(
                game["game_id"]
            )

            home_pitcher_id = (
                game.get(
                    "home_starting_pitcher_id"
                )
            )

            away_pitcher_id = (
                game.get(
                    "away_starting_pitcher_id"
                )
            )

            if pd.isna(home_pitcher_id):
                home_pitcher_id = None
            else:
                home_pitcher_id = int(
                    home_pitcher_id
                )

            if pd.isna(away_pitcher_id):
                away_pitcher_id = None
            else:
                away_pitcher_id = int(
                    away_pitcher_id
                )

            home_history = (
                pitcher_history[
                    home_pitcher_id
                ]
                if home_pitcher_id is not None
                else []
            )

            away_history = (
                pitcher_history[
                    away_pitcher_id
                ]
                if away_pitcher_id is not None
                else []
            )

            home_features = (
                calculate_pitcher_features(
                    home_history
                )
            )

            away_features = (
                calculate_pitcher_features(
                    away_history
                )
            )

            row = {
                "game_id":
                    game_id,

                "home_starting_pitcher_id":
                    home_pitcher_id,

                "away_starting_pitcher_id":
                    away_pitcher_id,

                "home_starting_pitcher":
                    game.get(
                        "home_starting_pitcher"
                    ),

                "away_starting_pitcher":
                    game.get(
                        "away_starting_pitcher"
                    ),
            }

            feature_names = [
                name.replace(
                    "_diff",
                    "",
                )
                for name
                in MLB_V2A_PITCHER_FEATURES
            ]

            for name in feature_names:

                home_value = (
                    home_features[name]
                )

                away_value = (
                    away_features[name]
                )

                row[
                    f"home_{name}"
                ] = home_value

                row[
                    f"away_{name}"
                ] = away_value

                row[
                    f"{name}_diff"
                ] = (
                    home_value
                    - away_value
                )

            pitcher_rows.append(
                row
            )

            # Only update histories after the
            # current timestamp group is scored.
            for log_record in logs_by_game.get(
                game_id,
                [],
            ):

                pitcher_id = int(
                    log_record[
                        "pitcher_id"
                    ]
                )

                pending_updates.append(
                    (
                        pitcher_id,
                        {
                            "outs":
                                log_record[
                                    "outs"
                                ],

                            "hits":
                                log_record[
                                    "hits"
                                ],

                            "earned_runs":
                                log_record[
                                    "earned_runs"
                                ],

                            "walks":
                                log_record[
                                    "walks"
                                ],

                            "strikeouts":
                                log_record[
                                    "strikeouts"
                                ],

                            "home_runs":
                                log_record[
                                    "home_runs"
                                ],
                        },
                    )
                )

        for pitcher_id, result in pending_updates:

            pitcher_history[
                pitcher_id
            ].append(
                result
            )

    pitcher_features = pd.DataFrame(
        pitcher_rows
    )

    combined = team_features.merge(
        pitcher_features,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    return (
        combined
        .sort_values(
            [
                "start_time",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )


# ==========================================
# MLB V2A WALK-FORWARD CHALLENGER
# ==========================================

def run_mlb_walkforward_v2a(
    features,
    min_train_games=500,
    retrain_every=100,
):
    """
    Walk-forward evaluation for MLB V2A.

    Uses:
        V1 team features
        +
        leakage-safe starting-pitcher features
    """

    if features is None or features.empty:
        raise ValueError(
            "No MLB V2A features available."
        )

    df = features.copy()

    required = (
        [
            "game_id",
            "start_time",
            "home_win",
        ]
        + MLB_V2A_MODEL_FEATURES
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"MLB V2A data is missing columns: {missing}"
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

    df = (
        df.sort_values(
            [
                "start_time",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    rows = []

    model = None
    trained_through_time = None
    predictions_since_fit = retrain_every

    for game_time, current_group in df.groupby(
        "start_time",
        sort=True,
    ):

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
            or predictions_since_fit
            >= retrain_every
        )

        if needs_retrain:

            X_train = (
                train[
                    MLB_V2A_MODEL_FEATURES
                ]
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
            current_group[
                MLB_V2A_MODEL_FEATURES
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(0.0)
        )

        probabilities = (
            model.predict_proba(
                X_current
            )[:, 1]
        )

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
                        game.get(
                            "season_id"
                        ),

                    "start_time":
                        game_time,

                    "home_team":
                        game.get(
                            "home_team"
                        ),

                    "away_team":
                        game.get(
                            "away_team"
                        ),

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
                            predicted
                            == actual
                        ),

                    "trained_through":
                        trained_through_time,

                    "model_version":
                        "MLB_V2A",
                }
            )

        predictions_since_fit += len(
            current_group
        )

    predictions = pd.DataFrame(
        rows
    )

    if predictions.empty:
        raise ValueError(
            "MLB V2A walk-forward produced no predictions."
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


def summarize_mlb_walkforward_v2a(
    predictions,
):
    """
    Use the same scoring methodology as V1 so
    V1 and V2A remain directly comparable.
    """

    results = (
        summarize_mlb_walkforward_v1(
            predictions
        )
    )

    results[
        "model_version"
    ] = "MLB_V2A"

    return results

# ==========================================
# MLB V2A — BATCHED PITCHER LOG COLLECTOR
# ==========================================

def get_missing_mlb_pitcher_log_games(
    games,
    existing_logs=None,
):
    """
    Determine which MLB games still need starting-pitcher
    boxscore data.

    A game is considered complete when both expected
    starting pitchers have matching pitcher-log records.
    """

    if games is None or games.empty:
        return pd.DataFrame()

    df = games.copy()

    required = [
        "game_id",
        "home_starting_pitcher_id",
        "away_starting_pitcher_id",
    ]

    missing_columns = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "MLB pitcher cache check is missing columns: "
            f"{missing_columns}"
        )

    # No existing cache yet.
    if existing_logs is None or existing_logs.empty:
        return df.reset_index(drop=True)

    logs = existing_logs.copy()

    if (
        "game_id" not in logs.columns
        or "pitcher_id" not in logs.columns
    ):
        return df.reset_index(drop=True)

    logs["game_id"] = pd.to_numeric(
        logs["game_id"],
        errors="coerce",
    )

    logs["pitcher_id"] = pd.to_numeric(
        logs["pitcher_id"],
        errors="coerce",
    )

    logs = logs.dropna(
        subset=[
            "game_id",
            "pitcher_id",
        ]
    )

    cached_pairs = set(
        zip(
            logs["game_id"].astype(int),
            logs["pitcher_id"].astype(int),
        )
    )

    missing_indexes = []

    for index, game in df.iterrows():

        game_id = int(
            game["game_id"]
        )

        expected_pitchers = []

        home_pitcher = game.get(
            "home_starting_pitcher_id"
        )

        away_pitcher = game.get(
            "away_starting_pitcher_id"
        )

        if pd.notna(home_pitcher):
            expected_pitchers.append(
                int(home_pitcher)
            )

        if pd.notna(away_pitcher):
            expected_pitchers.append(
                int(away_pitcher)
            )

        # If pitcher identities are unavailable,
        # this game cannot be completed by V2A.
        if len(expected_pitchers) < 2:
            continue

        complete = all(
            (
                game_id,
                pitcher_id,
            )
            in cached_pairs
            for pitcher_id
            in expected_pitchers
        )

        if not complete:
            missing_indexes.append(
                index
            )

    return (
        df.loc[missing_indexes]
        .copy()
        .reset_index(drop=True)
    )


def collect_mlb_pitcher_logs_batch(
    games,
    existing_logs=None,
    batch_size=250,
):
    """
    Collect one batch of missing MLB starting-pitcher logs.

    This deliberately does NOT attempt the entire historical
    archive in one request cycle.

    Returns:
        logs
        requested_games
        remaining_games
        complete
    """

    if games is None or games.empty:
        raise ValueError(
            "No MLB games supplied to pitcher-log collector."
        )

    if existing_logs is None:
        existing_logs = pd.DataFrame()

    missing_games = (
        get_missing_mlb_pitcher_log_games(
            games,
            existing_logs,
        )
    )

    if missing_games.empty:

        return {
            "logs":
                existing_logs.copy(),

            "requested_games":
                0,

            "remaining_games":
                0,

            "complete":
                True,
        }

    batch_size = max(
        1,
        int(batch_size),
    )

    batch = (
        missing_games
        .head(batch_size)
        .copy()
    )

    new_logs = (
        fetch_mlb_starting_pitcher_game_logs(
            batch
        )
    )

    if (
        existing_logs.empty
        and new_logs.empty
    ):

        combined = pd.DataFrame()

    elif existing_logs.empty:

        combined = new_logs.copy()

    elif new_logs.empty:

        combined = existing_logs.copy()

    else:

        combined = pd.concat(
            [
                existing_logs,
                new_logs,
            ],
            ignore_index=True,
        )

    if not combined.empty:

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "game_id",
                    "pitcher_id",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "start_time",
                    "game_id",
                    "side",
                ]
            )
            .reset_index(drop=True)
        )

    remaining = (
        get_missing_mlb_pitcher_log_games(
            games,
            combined,
        )
    )

    return {
        "logs":
            combined,

        "requested_games":
            int(len(batch)),

        "new_pitcher_rows":
            int(len(new_logs)),

        "cached_pitcher_rows":
            int(len(combined)),

        "remaining_games":
            int(len(remaining)),

        "complete":
            bool(remaining.empty),
    }

# ==========================================
# MLB DAILY PREGAME INTELLIGENCE LAYER
# ==========================================

MLB_LIVE_FEED_URL = (
    "https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
)


def fetch_mlb_daily_slate(game_date=None, timeout=30):
    """Return every MLB game scheduled for a calendar date.

    Unlike fetch_mlb_games(), this includes pregame/live/final games because
    production predictions begin with the complete daily slate.
    """
    target = pd.Timestamp(game_date or datetime.now(timezone.utc).date())
    date_text = target.strftime("%Y-%m-%d")
    response = requests.get(
        MLB_API_URL,
        params={
            "sportId": 1,
            "date": date_text,
            "hydrate": "probablePitcher,venue",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_team = home.get("team", {})
            away_team = away.get("team", {})
            venue = game.get("venue", {})
            status = game.get("status", {})
            rows.append({
                "game_id": game.get("gamePk"),
                "season_id": game.get("season"),
                "game_date": date_text,
                "start_time": game.get("gameDate"),
                "game_type": game.get("gameType"),
                "status": status.get("detailedState"),
                "abstract_status": status.get("abstractGameState"),
                "home_team_id": home_team.get("id"),
                "home_team": home_team.get("name"),
                "away_team_id": away_team.get("id"),
                "away_team": away_team.get("name"),
                "home_starting_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                "home_starting_pitcher": (home.get("probablePitcher") or {}).get("fullName"),
                "away_starting_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                "away_starting_pitcher": (away.get("probablePitcher") or {}).get("fullName"),
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    return df.drop_duplicates(subset=["game_id"]).sort_values(["start_time", "game_id"]).reset_index(drop=True)


def _mlb_player_snapshot(player, side, batting_order_lookup, bullpen_ids):
    person = player.get("person", {}) or {}
    position = player.get("position", {}) or {}
    batting = (player.get("stats", {}) or {}).get("batting", {}) or {}
    pitching = (player.get("stats", {}) or {}).get("pitching", {}) or {}
    season = player.get("seasonStats", {}) or {}
    season_batting = season.get("batting", {}) or {}
    season_pitching = season.get("pitching", {}) or {}
    player_id = person.get("id")
    return {
        "side": side,
        "player_id": player_id,
        "player_name": person.get("fullName"),
        "position": position.get("abbreviation") or position.get("name"),
        "batting_order": batting_order_lookup.get(player_id),
        "is_bullpen": player_id in bullpen_ids,
        "game_batting": batting,
        "game_pitching": pitching,
        "season_batting": season_batting,
        "season_pitching": season_pitching,
        "person": person,
    }


def fetch_mlb_pregame_game_snapshot(game_id, timeout=30):
    """Build the current information snapshot for one scheduled MLB game.

    The raw feed is retained so future feature engineering can use fields that
    are not flattened today. No postgame result is converted into a feature here.
    """
    url = MLB_LIVE_FEED_URL.format(game_id=int(game_id))
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    feed = response.json()

    game_data = feed.get("gameData", {}) or {}
    live_data = feed.get("liveData", {}) or {}
    boxscore = live_data.get("boxscore", {}) or {}
    teams = boxscore.get("teams", {}) or {}
    gd_teams = game_data.get("teams", {}) or {}
    venue = game_data.get("venue", {}) or {}
    weather = game_data.get("weather", {}) or {}
    datetime_data = game_data.get("datetime", {}) or {}
    status = game_data.get("status", {}) or {}
    probable = game_data.get("probablePitchers", {}) or {}

    player_rows = []
    team_context = {}
    for side in ("home", "away"):
        side_box = teams.get(side, {}) or {}
        batting_order = side_box.get("battingOrder", []) or []
        batting_order_lookup = {pid: index + 1 for index, pid in enumerate(batting_order)}
        bullpen_ids = set(side_box.get("bullpen", []) or [])
        side_players = side_box.get("players", {}) or {}
        for player in side_players.values():
            player_rows.append(_mlb_player_snapshot(player, side, batting_order_lookup, bullpen_ids))
        team_context[side] = {
            "team": gd_teams.get(side, {}),
            "team_stats": side_box.get("teamStats", {}) or {},
            "batting_order": batting_order,
            "bullpen": list(bullpen_ids),
            "bench": side_box.get("bench", []) or [],
            "pitchers": side_box.get("pitchers", []) or [],
        }

    return {
        "game_id": int(game_id),
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "start_time": datetime_data.get("dateTime"),
        "status": status,
        "home_team": gd_teams.get("home", {}),
        "away_team": gd_teams.get("away", {}),
        "probable_pitchers": probable,
        "venue": venue,
        "weather": weather,
        "game_info": game_data.get("gameInfo", {}) or {},
        "officials": boxscore.get("officials", []) or [],
        "team_context": team_context,
        "players": player_rows,
        "raw_feed": feed,
    }


def build_mlb_daily_pregame_intelligence(game_date=None, timeout=30):
    """Build one current pregame intelligence package for the day's MLB slate.

    This is the shared upstream data object for future winner and player-prop
    models. It intentionally gathers game context before deciding which model
    features to calculate.
    """
    slate = fetch_mlb_daily_slate(game_date=game_date, timeout=timeout)
    snapshots = []
    errors = []
    if slate.empty:
        return {"slate": slate, "games": [], "errors": []}

    for _, game in slate.iterrows():
        game_id = int(game["game_id"])
        try:
            snapshot = fetch_mlb_pregame_game_snapshot(game_id, timeout=timeout)
            snapshot["schedule"] = game.to_dict()
            snapshots.append(snapshot)
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append({"game_id": game_id, "error": str(exc)})

    return {
        "snapshot_date": str(pd.Timestamp(game_date or datetime.now(timezone.utc).date()).date()),
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "slate": slate,
        "games": snapshots,
        "game_count": int(len(slate)),
        "snapshots_built": int(len(snapshots)),
        "errors": errors,
    }

# ==========================================
# MLB DAILY PREGAME ENRICHMENT LAYER
# ==========================================

MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people/{player_id}"
MLB_PERSON_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
MLB_TEAM_STATS_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
MLB_ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"


def _mlb_api_get_json(url, params=None, timeout=30, session=None):
    """Small shared MLB request helper used by the enrichment layer."""
    client = session or requests
    response = client.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _mlb_first_split(payload):
    """Return the first MLB Stats API stat split, or an empty dict."""
    stats = (payload or {}).get("stats", []) or []
    for block in stats:
        splits = block.get("splits", []) or []
        if splits:
            return splits[0].get("stat", {}) or {}
    return {}


def fetch_mlb_player_profile(player_id, timeout=30, session=None):
    """Public MLB player profile facts useful to pregame modeling."""
    if player_id is None:
        return {}

    payload = _mlb_api_get_json(
        MLB_PEOPLE_URL.format(player_id=int(player_id)),
        params={"hydrate": "currentTeam"},
        timeout=timeout,
        session=session,
    )

    people = payload.get("people", []) or []
    if not people:
        return {}

    person = people[0] or {}

    return {
        "player_id": person.get("id"),
        "full_name": person.get("fullName"),
        "birth_date": person.get("birthDate"),
        "current_age": person.get("currentAge"),
        "height": person.get("height"),
        "weight": person.get("weight"),
        "active": person.get("active"),
        "primary_position": person.get("primaryPosition", {}) or {},
        "bat_side": person.get("batSide", {}) or {},
        "pitch_hand": person.get("pitchHand", {}) or {},
        "mlb_debut_date": person.get("mlbDebutDate"),
        "current_team": person.get("currentTeam", {}) or {},
    }


def fetch_mlb_player_season_stats(
    player_id, season, group, timeout=30, session=None,
):
    if player_id is None or season is None:
        return {}

    payload = _mlb_api_get_json(
        MLB_PERSON_STATS_URL.format(player_id=int(player_id)),
        params={
            "stats": "season",
            "group": str(group),
            "season": int(season),
        },
        timeout=timeout,
        session=session,
    )
    return _mlb_first_split(payload)


def fetch_mlb_player_recent_stats(
    player_id, group, end_date, days=14, timeout=30, session=None,
):
    """Rolling stats ending the day before the game to avoid same-game leakage."""
    if player_id is None or end_date is None:
        return {}

    end_ts = pd.Timestamp(end_date)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert("UTC").tz_localize(None)

    safe_end = end_ts.normalize() - pd.Timedelta(days=1)
    safe_start = safe_end - pd.Timedelta(days=max(int(days) - 1, 0))

    payload = _mlb_api_get_json(
        MLB_PERSON_STATS_URL.format(player_id=int(player_id)),
        params={
            "stats": "byDateRange",
            "group": str(group),
            "startDate": safe_start.strftime("%m/%d/%Y"),
            "endDate": safe_end.strftime("%m/%d/%Y"),
        },
        timeout=timeout,
        session=session,
    )
    return _mlb_first_split(payload)


def fetch_mlb_team_season_stats(
    team_id, season, group, timeout=30, session=None,
):
    if team_id is None or season is None:
        return {}

    payload = _mlb_api_get_json(
        MLB_TEAM_STATS_URL.format(team_id=int(team_id)),
        params={
            "stats": "season",
            "group": str(group),
            "season": int(season),
        },
        timeout=timeout,
        session=session,
    )
    return _mlb_first_split(payload)


def fetch_mlb_active_roster(team_id, roster_date, timeout=30, session=None):
    if team_id is None:
        return []

    payload = _mlb_api_get_json(
        MLB_ROSTER_URL.format(team_id=int(team_id)),
        params={
            "rosterType": "active",
            "date": pd.Timestamp(roster_date).strftime("%Y-%m-%d"),
        },
        timeout=timeout,
        session=session,
    )

    rows = []
    for entry in payload.get("roster", []) or []:
        person = entry.get("person", {}) or {}
        rows.append({
            "player_id": person.get("id"),
            "player_name": person.get("fullName"),
            "jersey_number": entry.get("jerseyNumber"),
            "status": entry.get("status", {}) or {},
            "position": entry.get("position", {}) or {},
        })
    return rows


def _mlb_pitcher_history_before_game(pitcher_logs, pitcher_id, game_time):
    if (
        pitcher_logs is None
        or pitcher_logs.empty
        or pitcher_id is None
        or game_time is None
    ):
        return []

    logs = pitcher_logs.copy()
    if "pitcher_id" not in logs.columns or "start_time" not in logs.columns:
        return []

    logs["pitcher_id"] = pd.to_numeric(logs["pitcher_id"], errors="coerce")
    logs["start_time"] = pd.to_datetime(
        logs["start_time"], utc=True, errors="coerce"
    )
    cutoff = pd.to_datetime(game_time, utc=True, errors="coerce")

    if pd.isna(cutoff):
        return []

    selected = logs[
        (logs["pitcher_id"] == int(pitcher_id))
        & (logs["start_time"] < cutoff)
    ].sort_values("start_time")

    columns = [
        c for c in (
            "outs", "hits", "earned_runs", "walks", "strikeouts", "home_runs"
        )
        if c in selected.columns
    ]
    return selected[columns].to_dict("records") if not selected.empty else []


def _mlb_team_history_before_game(historical_games, team_id, game_time):
    if (
        historical_games is None
        or historical_games.empty
        or team_id is None
        or game_time is None
    ):
        return []

    games = historical_games.copy()
    required = {
        "start_time", "home_team_id", "away_team_id",
        "home_score", "away_score",
    }
    if not required.issubset(games.columns):
        return []

    games["start_time"] = pd.to_datetime(
        games["start_time"], utc=True, errors="coerce"
    )
    cutoff = pd.to_datetime(game_time, utc=True, errors="coerce")
    if pd.isna(cutoff):
        return []

    team_id = int(team_id)
    home_ids = pd.to_numeric(games["home_team_id"], errors="coerce")
    away_ids = pd.to_numeric(games["away_team_id"], errors="coerce")

    prior = games[
        (games["start_time"] < cutoff)
        & ((home_ids == team_id) | (away_ids == team_id))
    ].sort_values(["start_time", "game_id"])

    history = []
    for _, game in prior.iterrows():
        home_id = int(game["home_team_id"])
        home_score = _safe_number(game["home_score"])
        away_score = _safe_number(game["away_score"])

        if home_id == team_id:
            runs_for, runs_against = home_score, away_score
        else:
            runs_for, runs_against = away_score, home_score

        win = 1.0 if runs_for > runs_against else 0.0 if runs_for < runs_against else 0.5
        history.append({
            "win": win,
            "runs_for": runs_for,
            "runs_against": runs_against,
            "run_diff": runs_for - runs_against,
        })
    return history


def enrich_mlb_pregame_game_snapshot(
    snapshot,
    historical_games=None,
    pitcher_logs=None,
    timeout=30,
    recent_days=14,
    session=None,
):
    """Create the shared game/player knowledge object for winner + prop models."""
    if not snapshot:
        return {}

    enriched = dict(snapshot)
    schedule = snapshot.get("schedule", {}) or {}
    game_time = snapshot.get("start_time") or schedule.get("start_time")
    season = schedule.get("season_id")

    if season is None and game_time is not None:
        season = pd.Timestamp(game_time).year

    game_date = schedule.get("game_date")
    if game_date is None and game_time is not None:
        game_date = pd.Timestamp(game_time).date()

    team_enrichment = {}
    player_enrichment = {}
    errors = []
    client = session or requests.Session()

    for side in ("home", "away"):
        team = snapshot.get(f"{side}_team", {}) or {}
        team_id = team.get("id") or schedule.get(f"{side}_team_id")

        history = _mlb_team_history_before_game(
            historical_games, team_id, game_time
        )

        context = {
            "team_id": team_id,
            "team_name": team.get("name") or schedule.get(f"{side}_team"),
            "historical_features": calculate_team_features(history),
            "active_roster": [],
            "season_hitting": {},
            "season_pitching": {},
        }

        try:
            context["active_roster"] = fetch_mlb_active_roster(
                team_id, game_date, timeout=timeout, session=client
            )
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append({
                "scope": "team_roster", "side": side,
                "team_id": team_id, "error": str(exc),
            })

        for group, key in (
            ("hitting", "season_hitting"),
            ("pitching", "season_pitching"),
        ):
            try:
                context[key] = fetch_mlb_team_season_stats(
                    team_id, season, group, timeout=timeout, session=client
                )
            except (requests.RequestException, ValueError, TypeError) as exc:
                errors.append({
                    "scope": f"team_{group}", "side": side,
                    "team_id": team_id, "error": str(exc),
                })

        team_enrichment[side] = context

    seen_players = set()

    for player in snapshot.get("players", []) or []:
        player_id = player.get("player_id")
        if player_id is None or player_id in seen_players:
            continue
        seen_players.add(player_id)

        position = str(player.get("position") or "").upper()
        probable = snapshot.get("probable_pitchers", {}) or {}
        probable_ids = {
            (probable.get("home", {}) or {}).get("id"),
            (probable.get("away", {}) or {}).get("id"),
        }

        is_pitcher = (
            position == "P"
            or player.get("is_bullpen")
            or player_id in probable_ids
        )
        group = "pitching" if is_pitcher else "hitting"

        item = {
            "player_id": player_id,
            "player_name": player.get("player_name"),
            "side": player.get("side"),
            "position": player.get("position"),
            "batting_order": player.get("batting_order"),
            "is_bullpen": bool(player.get("is_bullpen")),
            "stat_group": group,
            "profile": {},
            "season_stats": {},
            "recent_stats": {},
        }

        try:
            item["profile"] = fetch_mlb_player_profile(
                player_id, timeout=timeout, session=client
            )
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append({
                "scope": "player_profile",
                "player_id": player_id, "error": str(exc),
            })

        try:
            item["season_stats"] = fetch_mlb_player_season_stats(
                player_id, season, group, timeout=timeout, session=client
            )
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append({
                "scope": "player_season",
                "player_id": player_id, "error": str(exc),
            })

        try:
            item["recent_stats"] = fetch_mlb_player_recent_stats(
                player_id, group, game_date, days=recent_days,
                timeout=timeout, session=client
            )
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors.append({
                "scope": "player_recent",
                "player_id": player_id, "error": str(exc),
            })

        if is_pitcher:
            history = _mlb_pitcher_history_before_game(
                pitcher_logs, player_id, game_time
            )
            item["historical_pitching_features"] = (
                calculate_pitcher_features(history)
            )

        player_enrichment[str(player_id)] = item

    enriched["enrichment"] = {
        "season": season,
        "game_date": str(game_date) if game_date is not None else None,
        "recent_window_days": int(recent_days),
        "teams": team_enrichment,
        "players": player_enrichment,
        "errors": errors,
    }
    return enriched


def build_mlb_enriched_daily_pregame_intelligence(
    game_date=None,
    historical_games=None,
    pitcher_logs=None,
    timeout=30,
    recent_days=14,
):
    """Build one enriched daily package shared by winner and player-prop models."""
    package = build_mlb_daily_pregame_intelligence(
        game_date=game_date,
        timeout=timeout,
    )

    games = package.get("games", []) or []
    if not games:
        package["enriched"] = True
        package["enrichment_errors"] = []
        return package

    session = requests.Session()
    enriched_games = []
    all_errors = list(package.get("errors", []) or [])

    for snapshot in games:
        try:
            enriched = enrich_mlb_pregame_game_snapshot(
                snapshot,
                historical_games=historical_games,
                pitcher_logs=pitcher_logs,
                timeout=timeout,
                recent_days=recent_days,
                session=session,
            )
            enriched_games.append(enriched)
            all_errors.extend(
                (enriched.get("enrichment", {}) or {}).get("errors", []) or []
            )
        except (requests.RequestException, ValueError, TypeError) as exc:
            all_errors.append({
                "scope": "game_enrichment",
                "game_id": snapshot.get("game_id"),
                "error": str(exc),
            })
            enriched_games.append(snapshot)

    package["games"] = enriched_games
    package["enriched"] = True
    package["enriched_games"] = int(len(enriched_games))
    package["enrichment_errors"] = all_errors
    package["enrichment_time"] = datetime.now(timezone.utc).isoformat()
    return package


# ==========================================
# MLB SHARED PREGAME FEATURE ENGINE
# ==========================================

MLB_WINNER_LIVE_FEATURES = [
    "games_played_diff",
    "win_pct_diff",
    "avg_runs_for_diff",
    "avg_runs_against_diff",
    "avg_run_diff_diff",
    "recent_5_win_pct_diff",
    "recent_10_win_pct_diff",
    "recent_5_run_diff_diff",
    "starter_prior_starts_diff",
    "starter_prior_innings_diff",
    "starter_era_diff",
    "starter_whip_diff",
    "starter_k_per_9_diff",
    "starter_bb_per_9_diff",
    "starter_hr_per_9_diff",
    "starter_k_bb_ratio_diff",
    "starter_recent_3_era_diff",
    "starter_recent_3_whip_diff",
    "starter_recent_3_k_per_9_diff",
    "season_runs_per_game_diff",
    "season_ops_diff",
    "season_era_diff",
    "season_whip_diff",
    "lineup_ops_diff",
    "lineup_recent_ops_diff",
    "lineup_k_rate_diff",
    "lineup_recent_k_rate_diff",
    "bullpen_era_diff",
    "bullpen_whip_diff",
]

MLB_PROP_BASE_FEATURES = [
    "season_games",
    "season_plate_appearances",
    "season_at_bats",
    "season_hits",
    "season_home_runs",
    "season_walks",
    "season_strikeouts",
    "season_avg",
    "season_obp",
    "season_slg",
    "season_ops",
    "recent_games",
    "recent_plate_appearances",
    "recent_at_bats",
    "recent_hits",
    "recent_home_runs",
    "recent_walks",
    "recent_strikeouts",
    "recent_avg",
    "recent_obp",
    "recent_slg",
    "recent_ops",
    "season_innings",
    "season_batters_faced",
    "season_pitcher_strikeouts",
    "season_pitcher_walks",
    "season_pitcher_hits",
    "season_pitcher_home_runs",
    "season_era",
    "season_whip",
    "season_k_per_9",
    "season_bb_per_9",
    "recent_innings",
    "recent_batters_faced",
    "recent_pitcher_strikeouts",
    "recent_pitcher_walks",
    "recent_pitcher_hits",
    "recent_pitcher_home_runs",
    "recent_era",
    "recent_whip",
    "recent_k_per_9",
    "recent_bb_per_9",
    "batting_order",
    "is_home",
    "opponent_season_era",
    "opponent_season_whip",
    "opponent_season_ops",
    "opponent_starter_era",
    "opponent_starter_whip",
    "opponent_starter_k_per_9",
    "temperature_f",
    "wind_speed_mph",
]


def _mlb_num(mapping, *keys, default=0.0):
    """Read the first present numeric MLB stat from a mapping."""
    mapping = mapping or {}
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", "-.--"):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float(default)


def _mlb_rate(numerator, denominator, scale=1.0):
    denominator = _safe_number(denominator, 0.0)
    if denominator <= 0:
        return 0.0
    return float(_safe_number(numerator, 0.0) / denominator * scale)


def _mlb_pitching_innings(stat):
    return float(_innings_to_outs((stat or {}).get("inningsPitched")) / 3.0)


def _mlb_team_rate_features(team_context):
    team_context = team_context or {}
    hitting = team_context.get("season_hitting", {}) or {}
    pitching = team_context.get("season_pitching", {}) or {}

    games = max(
        _mlb_num(hitting, "gamesPlayed"),
        _mlb_num(pitching, "gamesPlayed"),
        0.0,
    )
    runs = _mlb_num(hitting, "runs")

    return {
        "season_games": games,
        "season_runs_per_game": runs / games if games > 0 else 0.0,
        "season_avg": _mlb_num(hitting, "avg"),
        "season_obp": _mlb_num(hitting, "obp"),
        "season_slg": _mlb_num(hitting, "slg"),
        "season_ops": _mlb_num(hitting, "ops"),
        "season_k_rate": _mlb_rate(
            _mlb_num(hitting, "strikeOuts"),
            _mlb_num(hitting, "plateAppearances", "atBats"),
        ),
        "season_bb_rate": _mlb_rate(
            _mlb_num(hitting, "baseOnBalls", "walks"),
            _mlb_num(hitting, "plateAppearances", "atBats"),
        ),
        "season_era": _mlb_num(pitching, "era", default=4.50),
        "season_whip": _mlb_num(pitching, "whip", default=1.30),
        "season_k_per_9": _mlb_num(
            pitching, "strikeoutsPer9Inn", default=8.0
        ),
        "season_bb_per_9": _mlb_num(
            pitching, "walksPer9Inn", default=3.0
        ),
    }


def _mlb_player_hitting_features(stat, prefix):
    stat = stat or {}
    pa = _mlb_num(stat, "plateAppearances", "atBats")
    return {
        f"{prefix}_games": _mlb_num(stat, "gamesPlayed", "games"),
        f"{prefix}_plate_appearances": pa,
        f"{prefix}_at_bats": _mlb_num(stat, "atBats"),
        f"{prefix}_hits": _mlb_num(stat, "hits"),
        f"{prefix}_doubles": _mlb_num(stat, "doubles"),
        f"{prefix}_triples": _mlb_num(stat, "triples"),
        f"{prefix}_home_runs": _mlb_num(stat, "homeRuns"),
        f"{prefix}_runs": _mlb_num(stat, "runs"),
        f"{prefix}_rbi": _mlb_num(stat, "rbi"),
        f"{prefix}_walks": _mlb_num(stat, "baseOnBalls", "walks"),
        f"{prefix}_strikeouts": _mlb_num(stat, "strikeOuts"),
        f"{prefix}_stolen_bases": _mlb_num(stat, "stolenBases"),
        f"{prefix}_avg": _mlb_num(stat, "avg"),
        f"{prefix}_obp": _mlb_num(stat, "obp"),
        f"{prefix}_slg": _mlb_num(stat, "slg"),
        f"{prefix}_ops": _mlb_num(stat, "ops"),
        f"{prefix}_k_rate": _mlb_rate(_mlb_num(stat, "strikeOuts"), pa),
        f"{prefix}_bb_rate": _mlb_rate(
            _mlb_num(stat, "baseOnBalls", "walks"), pa
        ),
    }


def _mlb_player_pitching_features(stat, prefix):
    stat = stat or {}
    innings = _mlb_pitching_innings(stat)
    strikeouts = _mlb_num(stat, "strikeOuts")
    walks = _mlb_num(stat, "baseOnBalls", "walks")
    hits = _mlb_num(stat, "hits")
    home_runs = _mlb_num(stat, "homeRuns")
    batters_faced = _mlb_num(stat, "battersFaced")

    return {
        f"{prefix}_games": _mlb_num(stat, "gamesPlayed", "games"),
        f"{prefix}_games_started": _mlb_num(stat, "gamesStarted"),
        f"{prefix}_innings": innings,
        f"{prefix}_batters_faced": batters_faced,
        f"{prefix}_pitcher_strikeouts": strikeouts,
        f"{prefix}_pitcher_walks": walks,
        f"{prefix}_pitcher_hits": hits,
        f"{prefix}_pitcher_home_runs": home_runs,
        f"{prefix}_earned_runs": _mlb_num(stat, "earnedRuns"),
        f"{prefix}_era": _mlb_num(stat, "era", default=4.50),
        f"{prefix}_whip": _mlb_num(stat, "whip", default=1.30),
        f"{prefix}_k_per_9": _mlb_num(
            stat, "strikeoutsPer9Inn",
            default=(strikeouts * 9.0 / innings if innings > 0 else 8.0),
        ),
        f"{prefix}_bb_per_9": _mlb_num(
            stat, "walksPer9Inn",
            default=(walks * 9.0 / innings if innings > 0 else 3.0),
        ),
        f"{prefix}_hr_per_9": (
            home_runs * 9.0 / innings if innings > 0 else 1.20
        ),
    }


def _mlb_lineup_aggregate(player_map, side):
    hitters = []
    for player in (player_map or {}).values():
        if player.get("side") != side or player.get("stat_group") != "hitting":
            continue
        order = player.get("batting_order")
        if order in (None, "", 0):
            continue
        hitters.append(player)

    if not hitters:
        return {
            "lineup_size": 0.0,
            "lineup_ops": 0.0,
            "lineup_recent_ops": 0.0,
            "lineup_k_rate": 0.0,
            "lineup_recent_k_rate": 0.0,
        }

    season_ops = []
    recent_ops = []
    season_ks = season_pa = recent_ks = recent_pa = 0.0

    for player in hitters:
        season = player.get("season_stats", {}) or {}
        recent = player.get("recent_stats", {}) or {}
        season_ops.append(_mlb_num(season, "ops"))
        recent_ops.append(_mlb_num(recent, "ops"))
        season_ks += _mlb_num(season, "strikeOuts")
        season_pa += _mlb_num(season, "plateAppearances", "atBats")
        recent_ks += _mlb_num(recent, "strikeOuts")
        recent_pa += _mlb_num(recent, "plateAppearances", "atBats")

    return {
        "lineup_size": float(len(hitters)),
        "lineup_ops": float(np.mean(season_ops)) if season_ops else 0.0,
        "lineup_recent_ops": float(np.mean(recent_ops)) if recent_ops else 0.0,
        "lineup_k_rate": _mlb_rate(season_ks, season_pa),
        "lineup_recent_k_rate": _mlb_rate(recent_ks, recent_pa),
    }


def _mlb_bullpen_aggregate(player_map, side):
    relievers = []
    for player in (player_map or {}).values():
        if player.get("side") != side or not player.get("is_bullpen"):
            continue
        if player.get("stat_group") != "pitching":
            continue
        relievers.append(player)

    innings = strikeouts = walks = hits = earned_runs = 0.0
    for player in relievers:
        stat = player.get("season_stats", {}) or {}
        ip = _mlb_pitching_innings(stat)
        innings += ip
        strikeouts += _mlb_num(stat, "strikeOuts")
        walks += _mlb_num(stat, "baseOnBalls", "walks")
        hits += _mlb_num(stat, "hits")
        earned_runs += _mlb_num(stat, "earnedRuns")

    return {
        "bullpen_size": float(len(relievers)),
        "bullpen_innings": innings,
        "bullpen_era": earned_runs * 9.0 / innings if innings > 0 else 4.50,
        "bullpen_whip": (walks + hits) / innings if innings > 0 else 1.30,
        "bullpen_k_per_9": strikeouts * 9.0 / innings if innings > 0 else 8.0,
        "bullpen_bb_per_9": walks * 9.0 / innings if innings > 0 else 3.0,
    }


def _mlb_probable_pitcher_id(game, side):
    probable = game.get("probable_pitchers", {}) or {}
    pitcher = probable.get(side, {}) or {}
    return pitcher.get("id") or pitcher.get("player_id")


def _mlb_starter_features(player_map, game, side):
    pitcher_id = _mlb_probable_pitcher_id(game, side)
    player = (player_map or {}).get(str(pitcher_id), {}) if pitcher_id else {}
    history = player.get("historical_pitching_features", {}) or {}

    defaults = calculate_pitcher_features([])
    return {
        key: _safe_number(history.get(key), defaults[key])
        for key in defaults
    }


def _mlb_weather_features(game):
    weather = game.get("weather", {}) or {}
    temp = weather.get("temp")
    wind = str(weather.get("wind") or "")

    wind_speed = 0.0
    for token in wind.replace(",", " ").split():
        try:
            wind_speed = float(token)
            break
        except ValueError:
            continue

    return {
        "temperature_f": _safe_number(temp, 0.0),
        "wind_speed_mph": wind_speed,
        "condition": weather.get("condition"),
    }


def build_mlb_winner_feature_row(enriched_game):
    """Convert one enriched pregame snapshot into one game-winner feature row."""
    enrichment = enriched_game.get("enrichment", {}) or {}
    teams = enrichment.get("teams", {}) or {}
    players = enrichment.get("players", {}) or {}

    home_team = teams.get("home", {}) or {}
    away_team = teams.get("away", {}) or {}
    home_history = home_team.get("historical_features", {}) or {}
    away_history = away_team.get("historical_features", {}) or {}

    row = {
        "game_id": enriched_game.get("game_id"),
        "start_time": enriched_game.get("start_time"),
        "home_team": home_team.get("team_name"),
        "away_team": away_team.get("team_name"),
        "home_team_id": home_team.get("team_id"),
        "away_team_id": away_team.get("team_id"),
    }

    team_history_names = [
        "games_played", "win_pct", "avg_runs_for", "avg_runs_against",
        "avg_run_diff", "recent_5_win_pct", "recent_10_win_pct",
        "recent_5_run_diff",
    ]
    for name in team_history_names:
        home_value = _safe_number(home_history.get(name), 0.0)
        away_value = _safe_number(away_history.get(name), 0.0)
        row[f"home_{name}"] = home_value
        row[f"away_{name}"] = away_value
        row[f"{name}_diff"] = home_value - away_value

    home_starter = _mlb_starter_features(players, enriched_game, "home")
    away_starter = _mlb_starter_features(players, enriched_game, "away")
    for name in home_starter:
        home_value = _safe_number(home_starter[name], 0.0)
        away_value = _safe_number(away_starter[name], 0.0)
        row[f"home_{name}"] = home_value
        row[f"away_{name}"] = away_value
        row[f"{name}_diff"] = home_value - away_value

    home_rates = _mlb_team_rate_features(home_team)
    away_rates = _mlb_team_rate_features(away_team)
    for name in home_rates:
        row[f"home_{name}"] = home_rates[name]
        row[f"away_{name}"] = away_rates[name]
        row[f"{name}_diff"] = home_rates[name] - away_rates[name]

    home_lineup = _mlb_lineup_aggregate(players, "home")
    away_lineup = _mlb_lineup_aggregate(players, "away")
    for name in home_lineup:
        row[f"home_{name}"] = home_lineup[name]
        row[f"away_{name}"] = away_lineup[name]
        row[f"{name}_diff"] = home_lineup[name] - away_lineup[name]

    home_bullpen = _mlb_bullpen_aggregate(players, "home")
    away_bullpen = _mlb_bullpen_aggregate(players, "away")
    for name in home_bullpen:
        row[f"home_{name}"] = home_bullpen[name]
        row[f"away_{name}"] = away_bullpen[name]
        row[f"{name}_diff"] = home_bullpen[name] - away_bullpen[name]

    row.update(_mlb_weather_features(enriched_game))
    row["venue_id"] = (enriched_game.get("venue", {}) or {}).get("id")
    row["venue_name"] = (enriched_game.get("venue", {}) or {}).get("name")
    return row


def build_mlb_winner_feature_matrix(enriched_package):
    rows = [
        build_mlb_winner_feature_row(game)
        for game in (enriched_package or {}).get("games", []) or []
        if game.get("enrichment")
    ]
    return pd.DataFrame(rows)


def build_mlb_player_prop_feature_rows(enriched_game):
    """Create one pregame row per active player for future prop-specific models."""
    enrichment = enriched_game.get("enrichment", {}) or {}
    teams = enrichment.get("teams", {}) or {}
    players = enrichment.get("players", {}) or {}
    weather = _mlb_weather_features(enriched_game)
    rows = []

    for player_id, player in players.items():
        side = player.get("side")
        if side not in ("home", "away"):
            continue
        opponent = "away" if side == "home" else "home"
        opponent_rates = _mlb_team_rate_features(teams.get(opponent, {}))
        opponent_starter = _mlb_starter_features(players, enriched_game, opponent)

        row = {
            "game_id": enriched_game.get("game_id"),
            "start_time": enriched_game.get("start_time"),
            "player_id": int(player_id),
            "player_name": player.get("player_name"),
            "side": side,
            "is_home": float(side == "home"),
            "position": player.get("position"),
            "stat_group": player.get("stat_group"),
            "batting_order": _safe_number(player.get("batting_order"), 0.0),
            "opponent_team_id": (teams.get(opponent, {}) or {}).get("team_id"),
            "opponent_team": (teams.get(opponent, {}) or {}).get("team_name"),
            "opponent_season_era": opponent_rates.get("season_era", 4.50),
            "opponent_season_whip": opponent_rates.get("season_whip", 1.30),
            "opponent_season_ops": opponent_rates.get("season_ops", 0.0),
            "opponent_starter_era": opponent_starter.get("starter_era", 4.50),
            "opponent_starter_whip": opponent_starter.get("starter_whip", 1.30),
            "opponent_starter_k_per_9": opponent_starter.get("starter_k_per_9", 8.0),
            **weather,
        }

        group = player.get("stat_group")
        season = player.get("season_stats", {}) or {}
        recent = player.get("recent_stats", {}) or {}

        if group == "hitting":
            row.update(_mlb_player_hitting_features(season, "season"))
            row.update(_mlb_player_hitting_features(recent, "recent"))
        elif group == "pitching":
            row.update(_mlb_player_pitching_features(season, "season"))
            row.update(_mlb_player_pitching_features(recent, "recent"))
            history = player.get("historical_pitching_features", {}) or {}
            for key, value in history.items():
                row[f"history_{key}"] = _safe_number(value, 0.0)

        rows.append(row)

    return rows


def build_mlb_player_prop_feature_matrix(enriched_package):
    rows = []
    for game in (enriched_package or {}).get("games", []) or []:
        if game.get("enrichment"):
            rows.extend(build_mlb_player_prop_feature_rows(game))
    return pd.DataFrame(rows)


def build_mlb_shared_pregame_feature_package(enriched_package):
    """One shared feature package for the winner engine and all prop engines."""
    winner_features = build_mlb_winner_feature_matrix(enriched_package)
    player_features = build_mlb_player_prop_feature_matrix(enriched_package)

    return {
        "snapshot_time": (enriched_package or {}).get("snapshot_time"),
        "game_date": (enriched_package or {}).get("game_date"),
        "winner_features": winner_features,
        "player_features": player_features,
        "winner_game_count": int(len(winner_features)),
        "player_row_count": int(len(player_features)),
        "winner_model_feature_names": list(MLB_WINNER_LIVE_FEATURES),
        "prop_base_feature_names": list(MLB_PROP_BASE_FEATURES),
    }

# ==========================================
# MLB V3 — RICH PREGAME CHALLENGER FRAMEWORK
# ==========================================

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)


MLB_V3_MODEL_FEATURES = list(MLB_WINNER_LIVE_FEATURES)

MLB_V3_REQUIRED_COLUMNS = (
    [
        "game_id",
        "start_time",
        "home_win",
    ]
    + MLB_V3_MODEL_FEATURES
)


def validate_mlb_v3_historical_features(features):
    """
    Validate that V3 receives a true historical pregame feature table.

    IMPORTANT:
    The live feature engine creates rich features for today's slate.
    That is not enough for a historical benchmark. Every historical row
    passed here must represent information that was available before that
    game's start_time.
    """
    if features is None or features.empty:
        raise ValueError(
            "MLB V3 requires a historical pregame feature dataset. "
            "Today's live enriched slate cannot be used as a backtest."
        )

    missing = [
        column
        for column in MLB_V3_REQUIRED_COLUMNS
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            "MLB V3 historical data is not ready. Missing columns: "
            f"{missing}. Build these features historically at each game's "
            "pregame timestamp before running V3."
        )

    df = features.copy()

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce",
    )

    df["home_win"] = pd.to_numeric(
        df["home_win"],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[
                "game_id",
                "start_time",
                "home_win",
            ]
        )
        .drop_duplicates(
            subset=["game_id"],
            keep="first",
        )
        .sort_values(
            ["start_time", "game_id"]
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "MLB V3 historical feature validation produced no usable games."
        )

    invalid_labels = set(
        df["home_win"].dropna().unique()
    ) - {0, 1, 0.0, 1.0}

    if invalid_labels:
        raise ValueError(
            "MLB V3 home_win must contain only binary 0/1 labels."
        )

    numeric = (
        df[MLB_V3_MODEL_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
    )

    completely_missing = [
        column
        for column in MLB_V3_MODEL_FEATURES
        if numeric[column].notna().sum() == 0
    ]

    if completely_missing:
        raise ValueError(
            "MLB V3 contains feature columns with no historical values: "
            f"{completely_missing}"
        )

    df[MLB_V3_MODEL_FEATURES] = numeric

    return df


def _make_mlb_v3_model(model_name):
    """
    Create one V3 challenger model.

    Logistic remains the interpretable linear challenger.
    Random forest and histogram gradient boosting allow nonlinear
    interactions among starter, lineup, bullpen, team, and environment
    variables without adding external ML dependencies.
    """
    name = str(model_name).strip().lower()

    if name in {
        "logistic",
        "logistic_regression",
        "lr",
    }:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                random_state=42,
            ),
        )

    if name in {
        "random_forest",
        "rf",
    }:
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )

    if name in {
        "hist_gradient_boosting",
        "hist_gb",
        "hgb",
    }:
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )

    raise ValueError(
        "Unknown MLB V3 model. Use logistic, random_forest, "
        "or hist_gradient_boosting."
    )


def run_mlb_walkforward_v3(
    features,
    model_name="hist_gradient_boosting",
    min_train_games=500,
    retrain_every=100,
):
    """
    Leakage-safe expanding-window V3 walk-forward test.

    Each game is predicted only from rows with start_time earlier than
    the game being scored. Games sharing a timestamp are predicted as
    one batch, matching the V1/V2A leakage protection.

    This function intentionally refuses today's live-only feature table.
    It requires historical V3 features reconstructed as-of each game.
    """
    df = validate_mlb_v3_historical_features(features)

    if len(df) <= int(min_train_games):
        raise ValueError(
            "Not enough historical MLB V3 games for walk-forward testing."
        )

    rows = []
    model = None
    trained_through_time = None
    predictions_since_fit = int(retrain_every)

    for game_time, current_group in df.groupby(
        "start_time",
        sort=True,
    ):
        train = df[
            df["start_time"] < game_time
        ].copy()

        if len(train) < int(min_train_games):
            continue

        y_train = train["home_win"].astype(int)

        if y_train.nunique() < 2:
            continue

        needs_retrain = (
            model is None
            or predictions_since_fit >= int(retrain_every)
        )

        if needs_retrain:
            X_train = (
                train[MLB_V3_MODEL_FEATURES]
                .apply(pd.to_numeric, errors="coerce")
            )

            # Use training-only medians. This avoids looking forward
            # when filling missing values.
            train_medians = X_train.median(numeric_only=True)
            X_train = X_train.fillna(train_medians).fillna(0.0)

            model = _make_mlb_v3_model(model_name)
            model.fit(X_train, y_train)

            trained_through_time = train["start_time"].max()
            predictions_since_fit = 0

        X_current = (
            current_group[MLB_V3_MODEL_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(train_medians)
            .fillna(0.0)
        )

        probabilities = model.predict_proba(
            X_current
        )[:, 1]

        for position, (_, game) in enumerate(
            current_group.iterrows()
        ):
            probability = float(probabilities[position])
            actual = int(game["home_win"])
            predicted = int(probability >= 0.50)
            confidence = float(
                max(probability, 1.0 - probability)
            )

            rows.append(
                {
                    "game_id": game["game_id"],
                    "season_id": game.get("season_id"),
                    "start_time": game_time,
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "actual_home_win": actual,
                    "home_win_probability": probability,
                    "predicted_home_win": predicted,
                    "confidence": confidence,
                    "correct": int(predicted == actual),
                    "trained_through": trained_through_time,
                    "model_version": (
                        f"MLB_V3_{str(model_name).upper()}"
                    ),
                }
            )

        predictions_since_fit += len(current_group)

    predictions = pd.DataFrame(rows)

    if predictions.empty:
        raise ValueError(
            "MLB V3 walk-forward produced no predictions."
        )

    return (
        predictions
        .sort_values(["start_time", "game_id"])
        .reset_index(drop=True)
    )


def summarize_mlb_walkforward_v3(predictions):
    """
    Score V3 with the exact core metrics used for V1:
    accuracy, AUC, Brier score, log loss, confidence buckets,
    and season-by-season accuracy.
    """
    results = summarize_mlb_walkforward_v1(predictions)

    versions = (
        predictions["model_version"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if "model_version" in predictions.columns
        else []
    )

    results["model_version"] = (
        versions[0]
        if len(versions) == 1
        else "MLB_V3"
    )

    return results


def run_mlb_v3_model_bakeoff(
    features,
    min_train_games=500,
    retrain_every=100,
    model_names=None,
):
    """
    Run the same historical V3 rows through multiple model families.

    No winner is automatically promoted. The returned comparison exposes
    accuracy AND probability-quality metrics so a model cannot be promoted
    merely because it became more confident.
    """
    if model_names is None:
        model_names = [
            "logistic",
            "random_forest",
            "hist_gradient_boosting",
        ]

    prediction_sets = {}
    summary_rows = []

    for model_name in model_names:
        predictions = run_mlb_walkforward_v3(
            features,
            model_name=model_name,
            min_train_games=min_train_games,
            retrain_every=retrain_every,
        )

        summary = summarize_mlb_walkforward_v3(
            predictions
        )

        prediction_sets[model_name] = predictions

        summary_rows.append(
            {
                "model_name": model_name,
                "prediction_count": summary["prediction_count"],
                "accuracy": summary["accuracy"],
                "auc": summary["auc"],
                "brier": summary["brier"],
                "log_loss": summary["log_loss"],
                "baseline_accuracy": summary["baseline_accuracy"],
                "baseline_brier": summary["baseline_brier"],
                "baseline_log_loss": summary["baseline_log_loss"],
            }
        )

    comparison = pd.DataFrame(summary_rows)

    return {
        "comparison": comparison,
        "predictions": prediction_sets,
        "feature_names": list(MLB_V3_MODEL_FEATURES),
    }


def compare_mlb_v3_to_v1(
    v1_predictions,
    v3_predictions,
):
    """
    Compare V1 and one V3 challenger on their overlapping game IDs only.

    This prevents a challenger from looking better simply because it was
    evaluated on a different or easier subset of games.
    """
    if v1_predictions is None or v1_predictions.empty:
        raise ValueError("V1 predictions are required.")

    if v3_predictions is None or v3_predictions.empty:
        raise ValueError("V3 predictions are required.")

    v1 = v1_predictions.copy()
    v3 = v3_predictions.copy()

    common_ids = set(
        pd.to_numeric(v1["game_id"], errors="coerce")
        .dropna()
        .astype(int)
    ).intersection(
        set(
            pd.to_numeric(v3["game_id"], errors="coerce")
            .dropna()
            .astype(int)
        )
    )

    if not common_ids:
        raise ValueError(
            "V1 and V3 have no overlapping game IDs."
        )

    v1_common = v1[
        pd.to_numeric(v1["game_id"], errors="coerce")
        .isin(common_ids)
    ].copy()

    v3_common = v3[
        pd.to_numeric(v3["game_id"], errors="coerce")
        .isin(common_ids)
    ].copy()

    v1_summary = summarize_mlb_walkforward_v1(
        v1_common
    )
    v3_summary = summarize_mlb_walkforward_v3(
        v3_common
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "MLB_V1",
                "games": v1_summary["prediction_count"],
                "accuracy": v1_summary["accuracy"],
                "auc": v1_summary["auc"],
                "brier": v1_summary["brier"],
                "log_loss": v1_summary["log_loss"],
            },
            {
                "model": v3_summary.get(
                    "model_version",
                    "MLB_V3",
                ),
                "games": v3_summary["prediction_count"],
                "accuracy": v3_summary["accuracy"],
                "auc": v3_summary["auc"],
                "brier": v3_summary["brier"],
                "log_loss": v3_summary["log_loss"],
            },
        ]
    )

    return {
        "overlap_games": int(len(common_ids)),
        "comparison": comparison,
        "v1_summary": v1_summary,
        "v3_summary": v3_summary,
    }

# ==========================================
# MLB V3 — HISTORICAL RICH FEATURE BRIDGE
# ==========================================

def _mlb_boxscore_player_postgame_row(
    game,
    side,
    player,
    batting_order_lookup,
    bullpen_ids,
):
    """
    Convert one completed-game boxscore player record into a compact
    POSTGAME history row.

    These rows are never used for the same game. The historical builder
    adds them to player history only after all games sharing the current
    start timestamp have been scored.
    """
    person = player.get("person", {}) or {}
    position = player.get("position", {}) or {}
    stats = player.get("stats", {}) or {}
    batting = stats.get("batting", {}) or {}
    pitching = stats.get("pitching", {}) or {}
    player_id = person.get("id")

    if player_id is None:
        return None

    batting_order = batting_order_lookup.get(int(player_id))
    is_bullpen = int(player_id) in bullpen_ids

    return {
        "game_id": int(game["game_id"]),
        "season_id": game.get("season_id"),
        "start_time": game["start_time"],
        "side": side,
        "team_id": int(game[f"{side}_team_id"]),
        "opponent_team_id": int(
            game["away_team_id"] if side == "home"
            else game["home_team_id"]
        ),
        "player_id": int(player_id),
        "player_name": person.get("fullName"),
        "position": position.get("abbreviation") or position.get("name"),
        "batting_order": batting_order,
        "is_bullpen": bool(is_bullpen),

        # Hitting line
        "batting_games": 1.0 if batting else 0.0,
        "plate_appearances": _safe_number(
            batting.get("plateAppearances"),
            _safe_number(batting.get("atBats"))
            + _safe_number(batting.get("baseOnBalls"))
            + _safe_number(batting.get("hitByPitch"))
            + _safe_number(batting.get("sacFlies")),
        ),
        "at_bats": _safe_number(batting.get("atBats")),
        "hits": _safe_number(batting.get("hits")),
        "doubles": _safe_number(batting.get("doubles")),
        "triples": _safe_number(batting.get("triples")),
        "batting_home_runs": _safe_number(batting.get("homeRuns")),
        "batting_runs": _safe_number(batting.get("runs")),
        "rbi": _safe_number(batting.get("rbi")),
        "batting_walks": _safe_number(
            batting.get("baseOnBalls"),
            _safe_number(batting.get("walks")),
        ),
        "batting_strikeouts": _safe_number(batting.get("strikeOuts")),
        "stolen_bases": _safe_number(batting.get("stolenBases")),

        # Pitching line
        "pitching_games": 1.0 if pitching else 0.0,
        "games_started": _safe_number(pitching.get("gamesStarted")),
        "pitching_outs": float(
            _innings_to_outs(pitching.get("inningsPitched"))
        ),
        "batters_faced": _safe_number(pitching.get("battersFaced")),
        "pitcher_strikeouts": _safe_number(pitching.get("strikeOuts")),
        "pitcher_walks": _safe_number(
            pitching.get("baseOnBalls"),
            _safe_number(pitching.get("walks")),
        ),
        "pitcher_hits": _safe_number(pitching.get("hits")),
        "pitcher_home_runs": _safe_number(pitching.get("homeRuns")),
        "earned_runs": _safe_number(pitching.get("earnedRuns")),
        "pitches": _safe_number(pitching.get("numberOfPitches")),
        "strikes": _safe_number(pitching.get("strikes")),
    }


def fetch_mlb_historical_player_game_logs(
    games,
    timeout=30,
):
    """
    Collect compact postgame player lines from completed MLB boxscores.

    One boxscore request supplies the players needed for both the historical
    lineup reconstruction and bullpen/pitcher reconstruction.
    """
    if games is None or games.empty:
        return pd.DataFrame()

    required = [
        "game_id",
        "start_time",
        "season_id",
        "home_team_id",
        "away_team_id",
    ]
    missing = [c for c in required if c not in games.columns]
    if missing:
        raise ValueError(
            f"Historical MLB player-log collection is missing: {missing}"
        )

    df = games.copy()
    df["start_time"] = pd.to_datetime(
        df["start_time"], utc=True, errors="coerce"
    )
    df = (
        df.dropna(subset=["game_id", "start_time"])
        .drop_duplicates(subset=["game_id"])
        .sort_values(["start_time", "game_id"])
        .reset_index(drop=True)
    )

    rows = []
    session = requests.Session()

    for _, game in df.iterrows():
        game_id = int(game["game_id"])
        try:
            payload = _mlb_api_get_json(
                MLB_BOXSCORE_URL.format(game_id=game_id),
                timeout=timeout,
                session=session,
            )
        except (requests.RequestException, ValueError, TypeError):
            continue

        teams = payload.get("teams", {}) or {}

        for side in ("home", "away"):
            side_box = teams.get(side, {}) or {}
            batting_order = side_box.get("battingOrder", []) or []
            batting_order_lookup = {
                int(pid): index + 1
                for index, pid in enumerate(batting_order)
                if pid is not None
            }
            bullpen_ids = {
                int(pid)
                for pid in (side_box.get("bullpen", []) or [])
                if pid is not None
            }

            for player in (side_box.get("players", {}) or {}).values():
                row = _mlb_boxscore_player_postgame_row(
                    game,
                    side,
                    player,
                    batting_order_lookup,
                    bullpen_ids,
                )
                if row is not None:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return (
        result
        .drop_duplicates(
            subset=["game_id", "player_id"],
            keep="first",
        )
        .sort_values(["start_time", "game_id", "side", "player_id"])
        .reset_index(drop=True)
    )


def get_missing_mlb_player_log_games(
    games,
    existing_logs=None,
):
    """
    Return games whose boxscore player history has not been cached yet.

    At least one cached player row marks the game as collected because one
    boxscore request retrieves the full game's player data.
    """
    if games is None or games.empty:
        return pd.DataFrame()

    if existing_logs is None or existing_logs.empty:
        return games.copy().reset_index(drop=True)

    if "game_id" not in existing_logs.columns:
        return games.copy().reset_index(drop=True)

    cached_ids = set(
        pd.to_numeric(
            existing_logs["game_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    game_ids = pd.to_numeric(
        games["game_id"],
        errors="coerce",
    )

    return (
        games.loc[
            ~game_ids.isin(cached_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )


def collect_mlb_player_logs_batch(
    games,
    existing_logs=None,
    batch_size=250,
    timeout=30,
):
    """
    Collect one manageable historical player-log batch.

    Designed to work like the existing pitcher batch collector so Streamlit
    can persist each batch to Supabase instead of attempting ~10k requests
    in one run.
    """
    missing = get_missing_mlb_player_log_games(
        games,
        existing_logs=existing_logs,
    )

    if missing.empty:
        combined = (
            existing_logs.copy()
            if existing_logs is not None
            else pd.DataFrame()
        )
        return {
            "new_logs": pd.DataFrame(),
            "combined_logs": combined,
            "games_requested": 0,
            "games_remaining": 0,
            "complete": True,
        }

    batch_size = max(int(batch_size), 1)
    batch = missing.head(batch_size).copy()

    new_logs = fetch_mlb_historical_player_game_logs(
        batch,
        timeout=timeout,
    )

    frames = []
    if existing_logs is not None and not existing_logs.empty:
        frames.append(existing_logs.copy())
    if new_logs is not None and not new_logs.empty:
        frames.append(new_logs.copy())

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )

    if not combined.empty:
        combined = (
            combined
            .drop_duplicates(
                subset=["game_id", "player_id"],
                keep="last",
            )
            .sort_values(
                ["start_time", "game_id", "side", "player_id"]
            )
            .reset_index(drop=True)
        )

    remaining = get_missing_mlb_player_log_games(
        games,
        existing_logs=combined,
    )

    return {
        "new_logs": new_logs,
        "combined_logs": combined,
        "games_requested": int(len(batch)),
        "games_remaining": int(len(remaining)),
        "complete": bool(remaining.empty),
    }


def _mlb_sum_history(history, column):
    if not history:
        return 0.0
    return float(sum(
        _safe_number(row.get(column), 0.0)
        for row in history
    ))


def _mlb_historical_hitting_features(history, recent_games=None):
    rows = list(history or [])
    if recent_games is not None:
        rows = rows[-int(recent_games):]

    pa = _mlb_sum_history(rows, "plate_appearances")
    ab = _mlb_sum_history(rows, "at_bats")
    hits = _mlb_sum_history(rows, "hits")
    doubles = _mlb_sum_history(rows, "doubles")
    triples = _mlb_sum_history(rows, "triples")
    home_runs = _mlb_sum_history(rows, "batting_home_runs")
    walks = _mlb_sum_history(rows, "batting_walks")
    strikeouts = _mlb_sum_history(rows, "batting_strikeouts")
    total_bases = hits + doubles + (2.0 * triples) + (3.0 * home_runs)

    avg = hits / ab if ab > 0 else 0.0
    obp_denom = ab + walks
    obp = (hits + walks) / obp_denom if obp_denom > 0 else 0.0
    slg = total_bases / ab if ab > 0 else 0.0

    return {
        "games": float(sum(
            1 for row in rows
            if _safe_number(row.get("plate_appearances"), 0.0) > 0
        )),
        "plate_appearances": pa,
        "at_bats": ab,
        "hits": hits,
        "home_runs": home_runs,
        "walks": walks,
        "strikeouts": strikeouts,
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": obp + slg,
        "k_rate": strikeouts / pa if pa > 0 else 0.0,
        "bb_rate": walks / pa if pa > 0 else 0.0,
    }


def _mlb_historical_pitching_features(history, recent_games=None):
    rows = list(history or [])
    if recent_games is not None:
        rows = rows[-int(recent_games):]

    outs = _mlb_sum_history(rows, "pitching_outs")
    innings = outs / 3.0
    strikeouts = _mlb_sum_history(rows, "pitcher_strikeouts")
    walks = _mlb_sum_history(rows, "pitcher_walks")
    hits = _mlb_sum_history(rows, "pitcher_hits")
    home_runs = _mlb_sum_history(rows, "pitcher_home_runs")
    earned_runs = _mlb_sum_history(rows, "earned_runs")

    return {
        "games": float(sum(
            1 for row in rows
            if _safe_number(row.get("pitching_outs"), 0.0) > 0
        )),
        "innings": innings,
        "strikeouts": strikeouts,
        "walks": walks,
        "hits": hits,
        "home_runs": home_runs,
        "earned_runs": earned_runs,
        "era": earned_runs * 9.0 / innings if innings > 0 else 4.50,
        "whip": (walks + hits) / innings if innings > 0 else 1.30,
        "k_per_9": strikeouts * 9.0 / innings if innings > 0 else 8.0,
        "bb_per_9": walks * 9.0 / innings if innings > 0 else 3.0,
    }


def _mlb_historical_lineup_features(
    player_history,
    lineup_ids,
):
    hitters = []
    for player_id in lineup_ids:
        history = player_history.get(int(player_id), [])
        if not history:
            continue
        hitters.append({
            "season": _mlb_historical_hitting_features(history),
            "recent": _mlb_historical_hitting_features(
                history,
                recent_games=14,
            ),
        })

    if not hitters:
        return {
            "lineup_size": 0.0,
            "lineup_ops": 0.0,
            "lineup_recent_ops": 0.0,
            "lineup_k_rate": 0.0,
            "lineup_recent_k_rate": 0.0,
        }

    return {
        "lineup_size": float(len(hitters)),
        "lineup_ops": float(np.mean([
            h["season"]["ops"] for h in hitters
        ])),
        "lineup_recent_ops": float(np.mean([
            h["recent"]["ops"] for h in hitters
        ])),
        "lineup_k_rate": float(np.mean([
            h["season"]["k_rate"] for h in hitters
        ])),
        "lineup_recent_k_rate": float(np.mean([
            h["recent"]["k_rate"] for h in hitters
        ])),
    }


def _mlb_historical_bullpen_features(
    player_history,
    bullpen_ids,
):
    combined = []
    for player_id in bullpen_ids:
        combined.extend(
            player_history.get(int(player_id), [])
        )

    pitching = _mlb_historical_pitching_features(combined)

    return {
        "bullpen_size": float(len(set(
            int(pid) for pid in bullpen_ids
        ))),
        "bullpen_innings": pitching["innings"],
        "bullpen_era": pitching["era"],
        "bullpen_whip": pitching["whip"],
        "bullpen_k_per_9": pitching["k_per_9"],
        "bullpen_bb_per_9": pitching["bb_per_9"],
    }


def _mlb_historical_team_season_features(team_history):
    rows = list(team_history or [])
    games = float(len(rows))
    runs_for = _mlb_sum_history(rows, "runs_for")
    runs_against = _mlb_sum_history(rows, "runs_against")

    # Historical game scores provide clean season run-rate context.
    # OPS/ERA/WHIP below are reconstructed from player histories elsewhere.
    return {
        "season_games": games,
        "season_runs_per_game": (
            runs_for / games if games > 0 else 0.0
        ),
        "runs_against_per_game": (
            runs_against / games if games > 0 else 0.0
        ),
    }


def build_mlb_v3_historical_features(
    games,
    player_logs,
    pitcher_logs=None,
):
    """
    Reconstruct V3 features chronologically.

    Critical leakage rule:
    Features for a timestamp are calculated first. Only after every game at
    that timestamp has been converted into a feature row are the completed
    game/player records added to history.

    The historical boxscore identifies the lineup/bullpen participants for
    that game, but their SAME-GAME statistics are not used in that row.
    Only their earlier accumulated history is used.
    """
    if games is None or games.empty:
        raise ValueError("No MLB historical games supplied for V3.")

    if player_logs is None or player_logs.empty:
        raise ValueError(
            "MLB V3 historical reconstruction requires cached player logs."
        )

    game_df = games.copy()
    game_df["start_time"] = pd.to_datetime(
        game_df["start_time"], utc=True, errors="coerce"
    )
    game_df = (
        game_df.dropna(
            subset=[
                "game_id", "start_time",
                "home_team_id", "away_team_id",
                "home_score", "away_score",
            ]
        )
        .drop_duplicates(subset=["game_id"])
        .sort_values(["start_time", "game_id"])
        .reset_index(drop=True)
    )

    logs = player_logs.copy()
    logs["start_time"] = pd.to_datetime(
        logs["start_time"], utc=True, errors="coerce"
    )
    logs = logs.dropna(
        subset=["game_id", "start_time", "player_id"]
    )

    logs_by_game = defaultdict(list)
    for _, log in logs.iterrows():
        logs_by_game[int(log["game_id"])].append(
            log.to_dict()
        )

    team_history = defaultdict(list)
    player_history = defaultdict(list)
    pitcher_history = defaultdict(list)

    # If the permanent starter cache is supplied, use it as an independent
    # starter-history source. It is rolled forward chronologically below.
    pitcher_logs_by_game = defaultdict(list)
    if pitcher_logs is not None and not pitcher_logs.empty:
        p_logs = pitcher_logs.copy()
        p_logs["start_time"] = pd.to_datetime(
            p_logs["start_time"], utc=True, errors="coerce"
        )
        p_logs = p_logs.dropna(
            subset=["game_id", "start_time", "pitcher_id"]
        )
        for _, log in p_logs.iterrows():
            pitcher_logs_by_game[int(log["game_id"])].append(
                log.to_dict()
            )

    rows = []

    for game_time, group in game_df.groupby(
        "start_time",
        sort=True,
    ):
        pending_team = []
        pending_player = []
        pending_pitcher = []

        for _, game in group.iterrows():
            game_id = int(game["game_id"])
            home_id = int(game["home_team_id"])
            away_id = int(game["away_team_id"])

            home_hist = calculate_team_features(
                team_history[home_id]
            )
            away_hist = calculate_team_features(
                team_history[away_id]
            )

            row = {
                "game_id": game_id,
                "season_id": game.get("season_id"),
                "start_time": game_time,
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_win": int(
                    _safe_number(game["home_score"])
                    > _safe_number(game["away_score"])
                ),
            }

            history_names = [
                "games_played",
                "win_pct",
                "avg_runs_for",
                "avg_runs_against",
                "avg_run_diff",
                "recent_5_win_pct",
                "recent_10_win_pct",
                "recent_5_run_diff",
            ]
            for name in history_names:
                hv = _safe_number(home_hist.get(name))
                av = _safe_number(away_hist.get(name))
                row[f"home_{name}"] = hv
                row[f"away_{name}"] = av
                row[f"{name}_diff"] = hv - av

            game_player_logs = logs_by_game.get(game_id, [])

            side_logs = {
                "home": [
                    r for r in game_player_logs
                    if r.get("side") == "home"
                ],
                "away": [
                    r for r in game_player_logs
                    if r.get("side") == "away"
                ],
            }

            side_lineups = {}
            side_bullpens = {}

            for side in ("home", "away"):
                side_lineups[side] = [
                    int(r["player_id"])
                    for r in sorted(
                        side_logs[side],
                        key=lambda x: (
                            999 if x.get("batting_order") in (None, "")
                            else _safe_number(x.get("batting_order"), 999)
                        ),
                    )
                    if x.get("batting_order") not in (None, "", 0)
                ]

                side_bullpens[side] = [
                    int(r["player_id"])
                    for r in side_logs[side]
                    if bool(r.get("is_bullpen"))
                ]

            # Starting-pitcher history.
            for side in ("home", "away"):
                pitcher_id = game.get(
                    f"{side}_starting_pitcher_id"
                )
                if pd.isna(pitcher_id):
                    pitcher_id = None
                else:
                    pitcher_id = int(pitcher_id)

                starter = calculate_pitcher_features(
                    pitcher_history[pitcher_id]
                    if pitcher_id is not None
                    else []
                )

                for name, value in starter.items():
                    row[f"{side}_{name}"] = value

            starter_names = [
                name.replace("_diff", "")
                for name in MLB_V2A_PITCHER_FEATURES
            ]
            for name in starter_names:
                hv = _safe_number(row.get(f"home_{name}"))
                av = _safe_number(row.get(f"away_{name}"))
                row[f"{name}_diff"] = hv - av

            # Team season run-rate context reconstructed from prior games.
            home_team_season = _mlb_historical_team_season_features(
                team_history[home_id]
            )
            away_team_season = _mlb_historical_team_season_features(
                team_history[away_id]
            )
            row["home_season_runs_per_game"] = (
                home_team_season["season_runs_per_game"]
            )
            row["away_season_runs_per_game"] = (
                away_team_season["season_runs_per_game"]
            )
            row["season_runs_per_game_diff"] = (
                row["home_season_runs_per_game"]
                - row["away_season_runs_per_game"]
            )

            # Reconstruct team season hitting/pitching from all players who
            # have appeared for that team before this game.
            team_player_ids = {}
            for side, team_id in (
                ("home", home_id),
                ("away", away_id),
            ):
                ids = set()
                for pid, history in player_history.items():
                    if any(
                        int(h.get("team_id", -1)) == team_id
                        for h in history
                    ):
                        ids.add(int(pid))
                team_player_ids[side] = ids

            team_rates = {}
            for side in ("home", "away"):
                team_rows = []
                for pid in team_player_ids[side]:
                    team_rows.extend(player_history.get(pid, []))

                hit = _mlb_historical_hitting_features(team_rows)
                pitch = _mlb_historical_pitching_features(team_rows)

                team_rates[side] = {
                    "season_ops": hit["ops"],
                    "season_era": pitch["era"],
                    "season_whip": pitch["whip"],
                }

            for name in (
                "season_ops",
                "season_era",
                "season_whip",
            ):
                hv = team_rates["home"][name]
                av = team_rates["away"][name]
                row[f"home_{name}"] = hv
                row[f"away_{name}"] = av
                row[f"{name}_diff"] = hv - av

            home_lineup = _mlb_historical_lineup_features(
                player_history,
                side_lineups["home"],
            )
            away_lineup = _mlb_historical_lineup_features(
                player_history,
                side_lineups["away"],
            )
            for name in (
                "lineup_ops",
                "lineup_recent_ops",
                "lineup_k_rate",
                "lineup_recent_k_rate",
            ):
                hv = home_lineup[name]
                av = away_lineup[name]
                row[f"home_{name}"] = hv
                row[f"away_{name}"] = av
                row[f"{name}_diff"] = hv - av

            home_bullpen = _mlb_historical_bullpen_features(
                player_history,
                side_bullpens["home"],
            )
            away_bullpen = _mlb_historical_bullpen_features(
                player_history,
                side_bullpens["away"],
            )
            for name in (
                "bullpen_era",
                "bullpen_whip",
            ):
                hv = home_bullpen[name]
                av = away_bullpen[name]
                row[f"home_{name}"] = hv
                row[f"away_{name}"] = av
                row[f"{name}_diff"] = hv - av

            rows.append(row)

            home_score = _safe_number(game["home_score"])
            away_score = _safe_number(game["away_score"])

            pending_team.extend([
                (
                    home_id,
                    {
                        "win": 1.0 if home_score > away_score else 0.0,
                        "runs_for": home_score,
                        "runs_against": away_score,
                        "run_diff": home_score - away_score,
                    },
                ),
                (
                    away_id,
                    {
                        "win": 1.0 if away_score > home_score else 0.0,
                        "runs_for": away_score,
                        "runs_against": home_score,
                        "run_diff": away_score - home_score,
                    },
                ),
            ])

            pending_player.extend(game_player_logs)

            if pitcher_logs_by_game:
                pending_pitcher.extend(
                    pitcher_logs_by_game.get(game_id, [])
                )
            else:
                # Fall back to the richer player boxscore cache for starters.
                starter_ids = {
                    int(pid)
                    for pid in (
                        game.get("home_starting_pitcher_id"),
                        game.get("away_starting_pitcher_id"),
                    )
                    if not pd.isna(pid)
                }
                for record in game_player_logs:
                    if int(record["player_id"]) not in starter_ids:
                        continue
                    pending_pitcher.append({
                        "pitcher_id": int(record["player_id"]),
                        "outs": record.get("pitching_outs", 0.0),
                        "hits": record.get("pitcher_hits", 0.0),
                        "earned_runs": record.get("earned_runs", 0.0),
                        "walks": record.get("pitcher_walks", 0.0),
                        "strikeouts": record.get("pitcher_strikeouts", 0.0),
                        "home_runs": record.get("pitcher_home_runs", 0.0),
                    })

        # Only now may this timestamp's results enter future histories.
        for team_id, result in pending_team:
            team_history[int(team_id)].append(result)

        for record in pending_player:
            player_history[int(record["player_id"])].append(record)

        for record in pending_pitcher:
            pitcher_history[int(record["pitcher_id"])].append({
                "outs": _safe_number(
                    record.get("outs"),
                    _safe_number(record.get("pitching_outs")),
                ),
                "hits": _safe_number(
                    record.get("hits"),
                    _safe_number(record.get("pitcher_hits")),
                ),
                "earned_runs": _safe_number(record.get("earned_runs")),
                "walks": _safe_number(
                    record.get("walks"),
                    _safe_number(record.get("pitcher_walks")),
                ),
                "strikeouts": _safe_number(
                    record.get("strikeouts"),
                    _safe_number(record.get("pitcher_strikeouts")),
                ),
                "home_runs": _safe_number(
                    record.get("home_runs"),
                    _safe_number(record.get("pitcher_home_runs")),
                ),
            })

    result = pd.DataFrame(rows)

    # These are part of the live feature schema but are not safely recoverable
    # from the compact boxscore cache alone. Keep them explicit rather than
    # fabricating historical weather values.
    for column in MLB_V3_MODEL_FEATURES:
        if column not in result.columns:
            result[column] = np.nan

    return (
        result
        .sort_values(["start_time", "game_id"])
        .reset_index(drop=True)
    )


def summarize_mlb_v3_historical_readiness(features):
    """
    Show exactly how much of the V3 historical matrix is populated before
    a bakeoff is attempted.
    """
    if features is None or features.empty:
        return {
            "ready": False,
            "games": 0,
            "feature_coverage": pd.DataFrame(),
        }

    coverage_rows = []
    for column in MLB_V3_MODEL_FEATURES:
        if column not in features.columns:
            coverage = 0.0
        else:
            coverage = float(
                pd.to_numeric(
                    features[column],
                    errors="coerce",
                )
                .notna()
                .mean()
            )

        coverage_rows.append({
            "feature": column,
            "coverage": coverage,
            "coverage_pct": coverage * 100.0,
        })

    coverage_df = pd.DataFrame(coverage_rows)

    return {
        "ready": bool(
            not coverage_df.empty
            and (coverage_df["coverage"] > 0).all()
        ),
        "games": int(len(features)),
        "feature_coverage": coverage_df,
        "fully_populated_features": int(
            (coverage_df["coverage"] >= 0.999).sum()
        ),
        "missing_features": coverage_df.loc[
            coverage_df["coverage"] <= 0,
            "feature",
        ].tolist(),
    }
