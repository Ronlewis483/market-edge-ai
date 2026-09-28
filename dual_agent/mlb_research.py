
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
