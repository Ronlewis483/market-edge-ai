import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from dual_agent.nba_data import get_historical_games


# ============================================================
# MARKET EDGE AI - NBA RESEARCH ENGINE V1
#
# Purpose:
#   1. Download historical NBA games
#   2. Build PRE-GAME team features
#   3. Prevent future-data leakage
#   4. Walk-forward validate a moneyline model
#   5. Compare model against simple baselines
#
# This file does NOT make today's picks.
# It is historical research only.
# ============================================================


MIN_TEAM_HISTORY = 5

FEATURE_COLUMNS = [
    "home_win_pct_5",
    "away_win_pct_5",
    "home_margin_5",
    "away_margin_5",
    "home_points_5",
    "away_points_5",
    "home_allowed_5",
    "away_allowed_5",
    "home_win_pct_10",
    "away_win_pct_10",
    "home_margin_10",
    "away_margin_10",
    "home_rest_days",
    "away_rest_days",
]


# ============================================================
# HELPERS
# ============================================================

def _first_existing(df, names):
    """
    Find the first column that exists from a list of
    possible SportsDataIO field names.
    """

    for name in names:
        if name in df.columns:
            return name

    return None


def _safe_float(value):
    try:
        if pd.isna(value):
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def _safe_date(value):
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


def _mean(values):
    if not values:
        return np.nan

    return float(np.mean(values))


# ============================================================
# NORMALIZE SPORTSDATAIO GAMES
# ============================================================

def normalize_games(raw):
    """
    Convert SportsDataIO game records into a stable format.

    This intentionally checks multiple possible field names
    so the research engine is more tolerant of feed variants.
    """

    if raw is None or len(raw) == 0:
        raise ValueError(
            "SportsDataIO returned no historical NBA games."
        )

    df = raw.copy()

    date_col = _first_existing(
        df,
        [
            "DateTime",
            "Day",
            "Date",
            "GameEndDateTime",
        ],
    )

    home_col = _first_existing(
        df,
        [
            "HomeTeam",
            "HomeTeamKey",
        ],
    )

    away_col = _first_existing(
        df,
        [
            "AwayTeam",
            "AwayTeamKey",
        ],
    )

    home_score_col = _first_existing(
        df,
        [
            "HomeTeamScore",
            "HomeTeamScore2",
            "HomeScore",
        ],
    )

    away_score_col = _first_existing(
        df,
        [
            "AwayTeamScore",
            "AwayTeamScore2",
            "AwayScore",
        ],
    )

    status_col = _first_existing(
        df,
        [
            "Status",
            "GameStatus",
        ],
    )

    game_id_col = _first_existing(
        df,
        [
            "GameID",
            "GlobalGameID",
        ],
    )

    required = {
        "date": date_col,
        "home_team": home_col,
        "away_team": away_col,
        "home_score": home_score_col,
        "away_score": away_score_col,
    }

    missing = [
        name
        for name, col in required.items()
        if col is None
    ]

    if missing:
        raise ValueError(
            "Historical NBA feed is missing required fields: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(map(str, df.columns))
        )

    out = pd.DataFrame()

    if game_id_col:
        out["game_id"] = df[game_id_col]
    else:
        out["game_id"] = np.arange(len(df))

    out["date"] = df[date_col].apply(_safe_date)
    out["home_team"] = df[home_col].astype(str)
    out["away_team"] = df[away_col].astype(str)

    out["home_score"] = (
        pd.to_numeric(df[home_score_col], errors="coerce")
    )

    out["away_score"] = (
        pd.to_numeric(df[away_score_col], errors="coerce")
    )

    if status_col:
        out["status"] = df[status_col].astype(str)
    else:
        out["status"] = "Final"

    # Keep only games with actual final scores.
    out = out.dropna(
        subset=[
            "date",
            "home_score",
            "away_score",
        ]
    )

    # Ties should not normally occur in completed NBA games.
    out = out[
        out["home_score"] != out["away_score"]
    ]

    out["home_win"] = (
        out["home_score"] > out["away_score"]
    ).astype(int)

    out["home_margin"] = (
        out["home_score"] - out["away_score"]
    )

    out = (
        out
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    return out


# ============================================================
# TEAM HISTORY
# ============================================================

def _team_record(
    team,
    game_date,
    points_for,
    points_against,
    won,
):
    return {
        "team": team,
        "date": game_date,
        "points_for": float(points_for),
        "points_against": float(points_against),
        "margin": float(points_for - points_against),
        "win": int(won),
    }


def _rolling_features(history, team, current_date):
    """
    Calculate features using ONLY games played before
    the current game.

    This is the critical anti-leakage step.
    """

    games = history.get(team, [])

    if len(games) < MIN_TEAM_HISTORY:
        return None

    last5 = games[-5:]
    last10 = games[-10:]

    last_game_date = games[-1]["date"]

    rest_days = (
        current_date.normalize()
        - last_game_date.normalize()
    ).days - 1

    # Cap unusual gaps so offseason/long breaks do not
    # dominate the model.
    rest_days = max(0, min(rest_days, 7))

    return {
        "win_pct_5":
            _mean([g["win"] for g in last5]),

        "margin_5":
            _mean([g["margin"] for g in last5]),

        "points_5":
            _mean([g["points_for"] for g in last5]),

        "allowed_5":
            _mean([g["points_against"] for g in last5]),

        "win_pct_10":
            _mean([g["win"] for g in last10]),

        "margin_10":
            _mean([g["margin"] for g in last10]),

        "rest_days":
            float(rest_days),
    }


# ============================================================
# BUILD PRE-GAME FEATURE TABLE
# ============================================================

def build_feature_table(games):
    """
    Build one row per historical game.

    IMPORTANT:
    Features are calculated BEFORE the current game's
    result is added to team history.
    """

    history = {}
    rows = []

    for _, game in games.iterrows():

        home = game["home_team"]
        away = game["away_team"]
        game_date = game["date"]

        home_features = _rolling_features(
            history,
            home,
            game_date,
        )

        away_features = _rolling_features(
            history,
            away,
            game_date,
        )

        if (
            home_features is not None
            and away_features is not None
        ):
            rows.append(
                {
                    "game_id": game["game_id"],
                    "date": game_date,
                    "home_team": home,
                    "away_team": away,

                    "home_win_pct_5":
                        home_features["win_pct_5"],

                    "away_win_pct_5":
                        away_features["win_pct_5"],

                    "home_margin_5":
                        home_features["margin_5"],

                    "away_margin_5":
                        away_features["margin_5"],

                    "home_points_5":
                        home_features["points_5"],

                    "away_points_5":
                        away_features["points_5"],

                    "home_allowed_5":
                        home_features["allowed_5"],

                    "away_allowed_5":
                        away_features["allowed_5"],

                    "home_win_pct_10":
                        home_features["win_pct_10"],

                    "away_win_pct_10":
                        away_features["win_pct_10"],

                    "home_margin_10":
                        home_features["margin_10"],

                    "away_margin_10":
                        away_features["margin_10"],

                    "home_rest_days":
                        home_features["rest_days"],

                    "away_rest_days":
                        away_features["rest_days"],

                    "target":
                        int(game["home_win"]),
                }
            )

        # ----------------------------------------------------
        # Update history ONLY AFTER features were generated.
        # ----------------------------------------------------

        history.setdefault(home, []).append(
            _team_record(
                team=home,
                game_date=game_date,
                points_for=game["home_score"],
                points_against=game["away_score"],
                won=game["home_win"] == 1,
            )
        )

        history.setdefault(away, []).append(
            _team_record(
                team=away,
                game_date=game_date,
                points_for=game["away_score"],
                points_against=game["home_score"],
                won=game["home_win"] == 0,
            )
        )

    features = pd.DataFrame(rows)

    if features.empty:
        raise ValueError(
            "Not enough historical games were available "
            "to create NBA pre-game features."
        )

    features = features.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    features = (
        features
        .sort_values("date")
        .reset_index(drop=True)
    )

    return features


# ============================================================
# MODEL
# ============================================================

def make_model():
    """
    Logistic regression is intentionally used as the
    first NBA benchmark model.

    It gives us interpretable probabilities and a strong
    baseline before testing more complex models.
    """

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                ),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def _metrics(y_true, probability):
    prediction = (
        np.asarray(probability) >= 0.50
    ).astype(int)

    result = {
        "games": int(len(y_true)),
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    prediction,
                )
            ),
        "brier":
            float(
                brier_score_loss(
                    y_true,
                    probability,
                )
            ),
        "logloss":
            float(
                log_loss(
                    y_true,
                    probability,
                    labels=[0, 1],
                )
            ),
    }

    if len(np.unique(y_true)) == 2:
        result["auc"] = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )
    else:
        result["auc"] = np.nan

    return result


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_validate(
    features,
    minimum_training_games=250,
    test_block_size=100,
):
    """
    Expanding-window chronological validation.

    Example:

    Train: games 1-250
    Test:  251-350

    Train: games 1-350
    Test:  351-450

    etc.

    Future games are never used to train past predictions.
    """

    df = (
        features
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(df) <= minimum_training_games:
        raise ValueError(
            f"Need more than {minimum_training_games} "
            f"feature rows for walk-forward validation. "
            f"Only {len(df)} are available."
        )

    predictions = []
    fold_results = []

    start = minimum_training_games
    fold = 1

    while start < len(df):

        stop = min(
            start + test_block_size,
            len(df),
        )

        train = df.iloc[:start].copy()
        test = df.iloc[start:stop].copy()

        if len(test) == 0:
            break

        X_train = train[FEATURE_COLUMNS]
        y_train = train["target"].astype(int)

        X_test = test[FEATURE_COLUMNS]
        y_test = test["target"].astype(int)

        if y_train.nunique() < 2:
            start = stop
            continue

        model = make_model()

        model.fit(
            X_train,
            y_train,
        )

        probability = model.predict_proba(
            X_test
        )[:, 1]

        fold_metric = _metrics(
            y_test,
            probability,
        )

        fold_metric["fold"] = fold
        fold_metric["train_games"] = len(train)
        fold_metric["test_start"] = test["date"].min()
        fold_metric["test_end"] = test["date"].max()

        fold_results.append(fold_metric)

        for i, (_, row) in enumerate(
            test.iterrows()
        ):
            predictions.append(
                {
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "actual_home_win":
                        int(row["target"]),
                    "model_probability":
                        float(probability[i]),
                    "fold": fold,
                }
            )

        start = stop
        fold += 1

    pred = pd.DataFrame(predictions)

    if pred.empty:
        raise ValueError(
            "Walk-forward validation produced "
            "no predictions."
        )

    overall = _metrics(
        pred["actual_home_win"],
        pred["model_probability"],
    )

    # --------------------------------------------------------
    # BASELINE
    #
    # Historical home-team win rate from the training-style
    # feature dataset.
    # --------------------------------------------------------

    base_rate = float(
        features["target"].mean()
    )

    baseline_probability = np.full(
        len(pred),
        base_rate,
    )

    baseline = _metrics(
        pred["actual_home_win"],
        baseline_probability,
    )

    return {
        "overall": overall,
        "baseline": baseline,
        "base_rate": base_rate,
        "folds": pd.DataFrame(fold_results),
        "predictions": pred,
    }


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def confidence_report(predictions):
    """
    Measure whether higher-confidence model predictions
    actually performed better historically.
    """

    df = predictions.copy()

    df["pick_home"] = (
        df["model_probability"] >= 0.50
    )

    df["confidence"] = np.where(
        df["pick_home"],
        df["model_probability"],
        1.0 - df["model_probability"],
    )

    df["correct"] = np.where(
        df["pick_home"],
        df["actual_home_win"] == 1,
        df["actual_home_win"] == 0,
    ).astype(int)

    bands = [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 1.01),
    ]

    rows = []

    for low, high in bands:

        sample = df[
            (df["confidence"] >= low)
            & (df["confidence"] < high)
        ]

        if sample.empty:
            continue

        rows.append(
            {
                "confidence_band":
                    (
                        f"{low:.0%}–"
                        f"{min(high, 1.0):.0%}"
                    ),
                "games": len(sample),
                "accuracy":
                    float(sample["correct"].mean()),
                "avg_confidence":
                    float(sample["confidence"].mean()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FULL NBA RESEARCH RUN
# ============================================================

def run_nba_research(
    season,
    minimum_training_games=250,
    test_block_size=100,
):
    """
    Complete historical NBA research pipeline.
    """

    raw = get_historical_games(season)

    games = normalize_games(raw)

    features = build_feature_table(games)

    validation = walk_forward_validate(
        features,
        minimum_training_games=
            minimum_training_games,
        test_block_size=test_block_size,
    )

    confidence = confidence_report(
        validation["predictions"]
    )

    return {
        "season": season,
        "raw_games": len(raw),
        "completed_games": len(games),
        "feature_rows": len(features),
        "home_win_rate":
            float(games["home_win"].mean()),
        "features": features,
        "overall": validation["overall"],
        "baseline": validation["baseline"],
        "base_rate": validation["base_rate"],
        "folds": validation["folds"],
        "predictions":
            validation["predictions"],
        "confidence": confidence,
    }

# ============================================================
# MULTI-SEASON NBA RESEARCH
# ============================================================

def run_multi_season_nba_research(
    seasons,
    minimum_training_games=500,
    test_block_size=150,
):
    """
    Combine multiple NBA seasons and run chronological
    walk-forward validation across the full dataset.
    """

    if not seasons:
        raise ValueError(
            "At least one NBA season is required."
        )

    all_games = []
    season_summary = []

    for season in seasons:
        season = str(season).strip()

        raw = get_historical_games(season)
        games = normalize_games(raw)

        if games.empty:
            continue

        games = games.copy()
        games["season"] = season

        all_games.append(games)

        season_summary.append(
            {
                "season": season,
                "raw_games": int(len(raw)),
                "completed_games": int(len(games)),
                "home_win_rate": float(
                    games["home_win"].mean()
                ),
            }
        )

    if not all_games:
        raise ValueError(
            "No historical NBA games were returned "
            "for the requested seasons."
        )

    combined_games = pd.concat(
        all_games,
        ignore_index=True,
    )

    combined_games = (
        combined_games
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    features = build_feature_table(
        combined_games
    )

    validation = walk_forward_validate(
        features,
        minimum_training_games=minimum_training_games,
        test_block_size=test_block_size,
    )

    confidence = confidence_report(
        validation["predictions"]
    )

    return {
        "seasons": [
            str(x).strip()
            for x in seasons
        ],
        "season_summary": pd.DataFrame(
            season_summary
        ),
        "raw_games": int(
            sum(
                x["raw_games"]
                for x in season_summary
            )
        ),
        "completed_games": int(
            len(combined_games)
        ),
        "feature_rows": int(
            len(features)
        ),
        "home_win_rate": float(
            combined_games["home_win"].mean()
        ),
        "features": features,
        "overall": validation["overall"],
        "baseline": validation["baseline"],
        "base_rate": validation["base_rate"],
        "folds": validation["folds"],
        "predictions": validation["predictions"],
        "confidence": confidence,
    }


# ============================================================
# MARKET EDGE AI - NBA CALIBRATION ANALYSIS
# ============================================================

def nba_calibration_report(predictions, bins=None):
    """
    Evaluate whether the NBA model's predicted probabilities
    match the actual observed win rates.

    This is research/validation only.
    """

    if predictions is None or len(predictions) == 0:
        return pd.DataFrame()

    df = predictions.copy()

    # Support the probability/target column names used by
    # different versions of the NBA research engine.
    probability_candidates = [
        "probability",
        "predicted_probability",
        "pred_prob",
        "home_win_probability",
        "p",
        "P",
    ]

    target_candidates = [
        "actual",
        "target",
        "home_win",
        "y_true",
        "result",
    ]

    probability_col = next(
        (c for c in probability_candidates if c in df.columns),
        None,
    )

    target_col = next(
        (c for c in target_candidates if c in df.columns),
        None,
    )

    if probability_col is None:
        raise ValueError(
            "Calibration could not find the model probability column. "
            f"Available columns: {list(df.columns)}"
        )

    if target_col is None:
        raise ValueError(
            "Calibration could not find the actual-result column. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[probability_col, target_col]].copy()

    df[probability_col] = pd.to_numeric(
        df[probability_col],
        errors="coerce",
    )

    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    df = df.dropna()

    if len(df) == 0:
        return pd.DataFrame()

    if bins is None:
        bins = [
            0.00,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            1.000001,
        ]

    labels = [
        "Below 50%",
        "50%-55%",
        "55%-60%",
        "60%-65%",
        "65%-70%",
        "70%-75%",
        "75%-80%",
        "80%-85%",
        "85%-90%",
        "90%-100%",
    ]

    df["probability_band"] = pd.cut(
        df[probability_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    rows = []

    for band, group in df.groupby(
        "probability_band",
        observed=True,
    ):
        if len(group) == 0:
            continue

        avg_probability = float(group[probability_col].mean())
        actual_win_rate = float(group[target_col].mean())

        rows.append(
            {
                "probability_band": str(band),
                "games": int(len(group)),
                "avg_predicted_probability": avg_probability,
                "actual_win_rate": actual_win_rate,
                "calibration_gap": actual_win_rate - avg_probability,
                "absolute_calibration_error": abs(
                    actual_win_rate - avg_probability
                ),
            }
        )

    return pd.DataFrame(rows)


def nba_high_confidence_report(predictions):
    """
    Measure performance at increasingly strict model-confidence
    thresholds.

    This helps determine an evidence-based qualification gate
    instead of choosing an arbitrary probability threshold.
    """

    if predictions is None or len(predictions) == 0:
        return pd.DataFrame()

    df = predictions.copy()

    probability_candidates = [
        "probability",
        "predicted_probability",
        "pred_prob",
        "home_win_probability",
        "p",
        "P",
    ]

    target_candidates = [
        "actual",
        "target",
        "home_win",
        "y_true",
        "result",
    ]

    probability_col = next(
        (c for c in probability_candidates if c in df.columns),
        None,
    )

    target_col = next(
        (c for c in target_candidates if c in df.columns),
        None,
    )

    if probability_col is None or target_col is None:
        return pd.DataFrame()

    df[probability_col] = pd.to_numeric(
        df[probability_col],
        errors="coerce",
    )

    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[probability_col, target_col]
    )

    thresholds = [
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    rows = []

    for threshold in thresholds:

        subset = df[
            df[probability_col] >= threshold
        ].copy()

        if len(subset) == 0:
            continue

        accuracy = float(
            subset[target_col].mean()
        )

        average_probability = float(
            subset[probability_col].mean()
        )

        rows.append(
            {
                "minimum_probability": threshold,
                "games": int(len(subset)),
                "accuracy": accuracy,
                "avg_model_probability": average_probability,
                "calibration_gap": accuracy - average_probability,
            }
        )

    return pd.DataFrame(rows)


def nba_calibration_summary(predictions):
    """
    Produce the complete calibration package used by the
    Research Lab.
    """

    calibration = nba_calibration_report(predictions)

    high_confidence = nba_high_confidence_report(predictions)

    if len(calibration):
        weighted_error = (
            calibration["absolute_calibration_error"]
            * calibration["games"]
        ).sum() / calibration["games"].sum()
    else:
        weighted_error = None

    return {
        "calibration": calibration,
        "high_confidence": high_confidence,
        "weighted_calibration_error": weighted_error,
    }
# ============================================================
# BALLDONTLIE -> NBA RESEARCH BRIDGE
# ============================================================

def prepare_balldontlie_games_for_research(games):
    """
    Convert the clean BALLDONTLIE historical dataset into
    a standardized chronological game dataset for NBA research.

    This function does NOT train the model.
    """

    if games is None or len(games) == 0:
        raise ValueError("No BALLDONTLIE games were supplied.")

    df = games.copy()

    required_columns = [
        "game_id",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
        "home_win",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "BALLDONTLIE dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Clean core fields
    # --------------------------------------------------------

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    df["home_points"] = pd.to_numeric(
        df["home_points"],
        errors="coerce",
    )

    df["away_points"] = pd.to_numeric(
        df["away_points"],
        errors="coerce",
    )

    df["home_win"] = pd.to_numeric(
        df["home_win"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "game_date",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "home_win",
        ]
    ).copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Basic outcome fields
    # --------------------------------------------------------

    df["point_margin"] = (
        df["home_points"] - df["away_points"]
    )

    df["total_points"] = (
        df["home_points"] + df["away_points"]
    )

    # Verify target
    calculated_home_win = (
        df["home_points"] > df["away_points"]
    ).astype(int)

    target_mismatches = int(
        (calculated_home_win != df["home_win"]).sum()
    )

    if target_mismatches:
        raise ValueError(
            f"{target_mismatches} home_win values do not match "
            "the actual game scores."
        )

    df["home_win"] = calculated_home_win

    return df
# ============================================================
# BALLDONTLIE CHRONOLOGICAL PRE-GAME FEATURE ENGINE
# ============================================================

def build_balldontlie_pregame_features(games, min_games=5):
    """
    Build pre-game NBA features using ONLY information that
    existed before each game.

    Team histories are updated only AFTER the current game's
    features have been created. This prevents target leakage.
    """

    df = prepare_balldontlie_games_for_research(games)

    team_history = {}
    feature_rows = []

    def get_team_history(team):
        if team not in team_history:
            team_history[team] = []
        return team_history[team]

    def summarize_history(history, current_date, venue=None):
        """
        Calculate team statistics using games played strictly
        before the current game.
        """

        if not history:
            return {
                "games_played": 0,
                "win_pct": 0.5,
                "win_pct_5": 0.5,
                "win_pct_10": 0.5,
                "avg_points": 0.0,
                "avg_points_allowed": 0.0,
                "avg_margin": 0.0,
                "venue_win_pct": 0.5,
                "rest_days": 7.0,
            }

        recent_5 = history[-5:]
        recent_10 = history[-10:]

        games_played = len(history)

        win_pct = sum(g["win"] for g in history) / games_played

        win_pct_5 = (
            sum(g["win"] for g in recent_5) / len(recent_5)
        )

        win_pct_10 = (
            sum(g["win"] for g in recent_10) / len(recent_10)
        )

        avg_points = (
            sum(g["points_for"] for g in recent_10)
            / len(recent_10)
        )

        avg_points_allowed = (
            sum(g["points_against"] for g in recent_10)
            / len(recent_10)
        )

        avg_margin = (
            sum(g["margin"] for g in recent_10)
            / len(recent_10)
        )

        venue_games = [
            g for g in history
            if g["venue"] == venue
        ]

        if venue_games:
            venue_recent = venue_games[-10:]

            venue_win_pct = (
                sum(g["win"] for g in venue_recent)
                / len(venue_recent)
            )
        else:
            venue_win_pct = 0.5

        last_game_date = history[-1]["game_date"]

        rest_days = (
            current_date - last_game_date
        ).days

        rest_days = max(0, min(rest_days, 10))

        return {
            "games_played": games_played,
            "win_pct": win_pct,
            "win_pct_5": win_pct_5,
            "win_pct_10": win_pct_10,
            "avg_points": avg_points,
            "avg_points_allowed": avg_points_allowed,
            "avg_margin": avg_margin,
            "venue_win_pct": venue_win_pct,
            "rest_days": rest_days,
        }

    for _, game in df.iterrows():

        game_date = game["game_date"]

        home_team = game["home_team"]
        away_team = game["away_team"]

        home_history = get_team_history(home_team)
        away_history = get_team_history(away_team)

        # ----------------------------------------------------
        # CREATE FEATURES BEFORE ADDING CURRENT GAME RESULT
        # ----------------------------------------------------

        home_stats = summarize_history(
            home_history,
            game_date,
            venue="home",
        )

        away_stats = summarize_history(
            away_history,
            game_date,
            venue="away",
        )

        # Require both teams to have enough prior games.
        if (
            home_stats["games_played"] >= min_games
            and away_stats["games_played"] >= min_games
        ):

            feature_rows.append({
                "game_id": game["game_id"],
                "game_date": game_date,
                "season": game["season"],
                "home_team": home_team,
                "away_team": away_team,

                "home_games_played":
                    home_stats["games_played"],

                "away_games_played":
                    away_stats["games_played"],

                "home_win_pct":
                    home_stats["win_pct"],

                "away_win_pct":
                    away_stats["win_pct"],

                "home_win_pct_5":
                    home_stats["win_pct_5"],

                "away_win_pct_5":
                    away_stats["win_pct_5"],

                "home_win_pct_10":
                    home_stats["win_pct_10"],

                "away_win_pct_10":
                    away_stats["win_pct_10"],

                "home_avg_points":
                    home_stats["avg_points"],

                "away_avg_points":
                    away_stats["avg_points"],

                "home_avg_points_allowed":
                    home_stats["avg_points_allowed"],

                "away_avg_points_allowed":
                    away_stats["avg_points_allowed"],

                "home_avg_margin":
                    home_stats["avg_margin"],

                "away_avg_margin":
                    away_stats["avg_margin"],

                "home_venue_win_pct":
                    home_stats["venue_win_pct"],

                "away_venue_win_pct":
                    away_stats["venue_win_pct"],

                "home_rest_days":
                    home_stats["rest_days"],

                "away_rest_days":
                    away_stats["rest_days"],

                # Difference features
                "win_pct_diff":
                    home_stats["win_pct"]
                    - away_stats["win_pct"],

                "recent_5_diff":
                    home_stats["win_pct_5"]
                    - away_stats["win_pct_5"],

                "recent_10_diff":
                    home_stats["win_pct_10"]
                    - away_stats["win_pct_10"],

                "margin_diff":
                    home_stats["avg_margin"]
                    - away_stats["avg_margin"],

                "venue_win_pct_diff":
                    home_stats["venue_win_pct"]
                    - away_stats["venue_win_pct"],

                "rest_days_diff":
                    home_stats["rest_days"]
                    - away_stats["rest_days"],

                # Target
                "home_win": int(game["home_win"]),
            })

        # ----------------------------------------------------
        # ONLY NOW UPDATE TEAM HISTORY WITH CURRENT GAME
        # ----------------------------------------------------

        home_points = float(game["home_points"])
        away_points = float(game["away_points"])

        home_history.append({
            "game_date": game_date,
            "venue": "home",
            "win": int(home_points > away_points),
            "points_for": home_points,
            "points_against": away_points,
            "margin": home_points - away_points,
        })

        away_history.append({
            "game_date": game_date,
            "venue": "away",
            "win": int(away_points > home_points),
            "points_for": away_points,
            "points_against": home_points,
            "margin": away_points - home_points,
        })

    features = pd.DataFrame(feature_rows)

    if features.empty:
        raise ValueError(
            "No model-ready feature rows were created."
        )

    features = features.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    return features
    # ============================================================
# PRE-GAME FEATURE LEAKAGE AUDIT
# ============================================================

def audit_balldontlie_pregame_features(feature_games):
    """
    Audit the pre-game feature dataset for obvious target leakage.

    Historical statistics calculated strictly before the current
    game are valid model features.
    """

    if feature_games is None or len(feature_games) == 0:
        raise ValueError("No pre-game feature data was supplied.")

    df = feature_games.copy()
    columns = list(df.columns)

    # These columns may remain in the dataset for identification
    # or evaluation, but must never be model predictors.
    protected_columns = {
        "game_id",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
        "point_margin",
        "total_points",
        "home_win",
    }

    # Numeric columns available to the prediction model.
    model_features = [
        column
        for column in columns
        if column not in protected_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    # Exact current-game outcome fields that would represent leakage.
    # We intentionally use exact names rather than substring matching.
    # Historical fields such as home_avg_margin are valid pre-game data.
    forbidden_model_features = {
        "home_points",
        "away_points",
        "point_margin",
        "total_points",
        "home_win",
        "final",
        "result",
        "winner",
        "score",
    }

    suspicious_columns = [
        column
        for column in model_features
        if column.lower() in forbidden_model_features
    ]

    missing_values = (
        df[model_features]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_values = missing_values[
        missing_values > 0
    ]

    return {
        "total_rows": len(df),
        "total_columns": len(columns),
        "model_feature_count": len(model_features),
        "model_features": model_features,
        "protected_columns_present": [
            column
            for column in protected_columns
            if column in columns
        ],
        "suspicious_model_features": suspicious_columns,
        "missing_feature_values": missing_values,
    }
# ============================================================
# BALLDONTLIE WALK-FORWARD NBA MODEL
# ============================================================

def run_balldontlie_walkforward_model(feature_games):
    """
    Chronological walk-forward validation for the BALLDONTLIE
    pre-game NBA feature dataset.

    The model is trained only on games that occurred before
    the games being predicted.
    """

    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        brier_score_loss,
        log_loss,
    )
    from sklearn.preprocessing import StandardScaler

    if feature_games is None or len(feature_games) == 0:
        raise ValueError("No feature games supplied.")

    df = feature_games.copy()

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    df = df.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Explicit pre-game feature list
    # --------------------------------------------------------

    feature_columns = [
        "home_games_played",
        "away_games_played",
        "home_win_pct",
        "away_win_pct",
        "home_win_pct_5",
        "away_win_pct_5",
        "home_win_pct_10",
        "away_win_pct_10",
        "home_avg_points",
        "away_avg_points",
        "home_avg_points_allowed",
        "away_avg_points_allowed",
        "home_avg_margin",
        "away_avg_margin",
        "home_venue_win_pct",
        "away_venue_win_pct",
        "home_rest_days",
        "away_rest_days",
        "win_pct_diff",
        "recent_5_diff",
        "recent_10_diff",
        "margin_diff",
        "venue_win_pct_diff",
        "rest_days_diff",
    ]

    missing_columns = [
        column
        for column in feature_columns + ["home_win"]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Walk-forward model is missing columns: "
            + ", ".join(missing_columns)
        )

    df = df.dropna(
        subset=feature_columns + ["home_win", "game_date"]
    ).copy()

    if len(df) < 500:
        raise ValueError(
            "Not enough games for reliable walk-forward testing."
        )

    # --------------------------------------------------------
    # Expanding chronological test windows
    # --------------------------------------------------------

    initial_train_size = 800
    test_window_size = 200

    prediction_rows = []

    train_end = initial_train_size

    while train_end < len(df):

        test_end = min(
            train_end + test_window_size,
            len(df),
        )

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        if test_df.empty:
            break

        X_train = train_df[feature_columns].astype(float)
        y_train = train_df["home_win"].astype(int)

        X_test = test_df[feature_columns].astype(float)
        y_test = test_df["home_win"].astype(int)

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=2000,
            random_state=42,
        )

        model.fit(
            X_train_scaled,
            y_train,
        )

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        for position, (_, game) in enumerate(
            test_df.iterrows()
        ):
            prediction_rows.append(
                {
                    "game_id": game["game_id"],
                    "game_date": game["game_date"],
                    "season": game["season"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "actual": int(y_test.iloc[position]),
                    "probability": float(
                        probabilities[position]
                    ),
                    "prediction": int(
                        predictions[position]
                    ),
                    "train_games": int(train_end),
                }
            )

        train_end = test_end

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    if predictions_df.empty:
        raise ValueError(
            "Walk-forward model produced no predictions."
        )

    y_true = predictions_df["actual"].astype(int)

    probabilities = predictions_df[
        "probability"
    ].astype(float)

    predicted_classes = predictions_df[
        "prediction"
    ].astype(int)

    accuracy = accuracy_score(
        y_true,
        predicted_classes,
    )

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    logloss = log_loss(
        y_true,
        probabilities,
    )

    # --------------------------------------------------------
    # Naive home-team baseline
    # --------------------------------------------------------

    baseline_probability = float(
        df.iloc[:initial_train_size]["home_win"].mean()
    )

    baseline_probabilities = np.full(
        len(y_true),
        baseline_probability,
    )

    baseline_predictions = (
        baseline_probabilities >= 0.50
    ).astype(int)

    baseline_accuracy = accuracy_score(
        y_true,
        baseline_predictions,
    )

    baseline_brier = brier_score_loss(
        y_true,
        baseline_probabilities,
    )

    baseline_logloss = log_loss(
        y_true,
        baseline_probabilities,
    )

    return {
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "total_feature_games": len(df),
        "initial_train_games": initial_train_size,
        "test_window_size": test_window_size,
        "predictions": predictions_df,
        "games_predicted": len(predictions_df),

        "accuracy": float(accuracy),
        "auc": float(auc),
        "brier": float(brier),
        "log_loss": float(logloss),

        "baseline_probability": baseline_probability,
        "baseline_accuracy": float(
            baseline_accuracy
        ),
        "baseline_brier": float(
            baseline_brier
        ),
        "baseline_log_loss": float(
            baseline_logloss
        ),
    }
    # ============================================================
# BALLDONTLIE LIVE / FUTURE MATCHUP FEATURE BUILDER
# ============================================================

def build_balldontlie_future_matchup_features(
    historical_games,
    home_team,
    away_team,
    game_date,
):
    """
    Build pre-game features for a future NBA matchup using ONLY
    games played before game_date.

    No current/future game result is required.
    """

    if historical_games is None or len(historical_games) == 0:
        raise ValueError("No historical games were supplied.")

    history = historical_games.copy()

    history["game_date"] = pd.to_datetime(
        history["game_date"],
        errors="coerce",
    )

    target_date = pd.to_datetime(game_date)

    # Absolutely no games from the target date or future may be used.
    history = history[
        history["game_date"] < target_date
    ].copy()

    history = history.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    def get_team_history(team):
        team_games = history[
            (history["home_team"] == team)
            | (history["away_team"] == team)
        ].copy()

        team_games = team_games.sort_values(
            ["game_date", "game_id"]
        )

        records = []

        for _, game in team_games.iterrows():

            if game["home_team"] == team:
                points = game["home_points"]
                points_allowed = game["away_points"]
                win = int(
                    game["home_points"] > game["away_points"]
                )
                venue = "home"

            else:
                points = game["away_points"]
                points_allowed = game["home_points"]
                win = int(
                    game["away_points"] > game["home_points"]
                )
                venue = "away"

            records.append(
                {
                    "game_date": game["game_date"],
                    "win": win,
                    "points": points,
                    "points_allowed": points_allowed,
                    "margin": points - points_allowed,
                    "venue": venue,
                }
            )

        return pd.DataFrame(records)

    home_history = get_team_history(home_team)
    away_history = get_team_history(away_team)

    if len(home_history) < 5:
        raise ValueError(
            f"{home_team} does not have at least 5 prior games."
        )

    if len(away_history) < 5:
        raise ValueError(
            f"{away_team} does not have at least 5 prior games."
        )

    def summarize_team(team_history, venue):

        last_5 = team_history.tail(5)
        last_10 = team_history.tail(10)

        venue_history = team_history[
            team_history["venue"] == venue
        ]

        last_game_date = team_history[
            "game_date"
        ].max()

        rest_days = (
            target_date - last_game_date
        ).days

        return {
            "games_played": len(team_history),

            "win_pct":
                team_history["win"].mean(),

            "win_pct_5":
                last_5["win"].mean(),

            "win_pct_10":
                last_10["win"].mean(),

            "avg_points":
                team_history["points"].mean(),

            "avg_points_allowed":
                team_history["points_allowed"].mean(),

            "avg_margin":
                team_history["margin"].mean(),

            "venue_win_pct":
                venue_history["win"].mean()
                if len(venue_history) > 0
                else 0.5,

            "rest_days":
                rest_days,
        }

    home = summarize_team(
        home_history,
        "home",
    )

    away = summarize_team(
        away_history,
        "away",
    )

    features = {
        "home_games_played": home["games_played"],
        "away_games_played": away["games_played"],

        "home_win_pct": home["win_pct"],
        "away_win_pct": away["win_pct"],

        "home_win_pct_5": home["win_pct_5"],
        "away_win_pct_5": away["win_pct_5"],

        "home_win_pct_10": home["win_pct_10"],
        "away_win_pct_10": away["win_pct_10"],

        "home_avg_points": home["avg_points"],
        "away_avg_points": away["avg_points"],

        "home_avg_points_allowed":
            home["avg_points_allowed"],

        "away_avg_points_allowed":
            away["avg_points_allowed"],

        "home_avg_margin": home["avg_margin"],
        "away_avg_margin": away["avg_margin"],

        "home_venue_win_pct":
            home["venue_win_pct"],

        "away_venue_win_pct":
            away["venue_win_pct"],

        "home_rest_days": home["rest_days"],
        "away_rest_days": away["rest_days"],
    }

    features["win_pct_diff"] = (
        features["home_win_pct"]
        - features["away_win_pct"]
    )

    features["recent_5_diff"] = (
        features["home_win_pct_5"]
        - features["away_win_pct_5"]
    )

    features["recent_10_diff"] = (
        features["home_win_pct_10"]
        - features["away_win_pct_10"]
    )

    features["margin_diff"] = (
        features["home_avg_margin"]
        - features["away_avg_margin"]
    )

    features["venue_win_pct_diff"] = (
        features["home_venue_win_pct"]
        - features["away_venue_win_pct"]
    )

    features["rest_days_diff"] = (
        features["home_rest_days"]
        - features["away_rest_days"]
    )

    features["game_date"] = target_date

    return pd.DataFrame([features])

# ============================================================
# BALLDONTLIE LIVE NBA PREDICTION MODEL
# ============================================================

def predict_balldontlie_matchup(feature_games, matchup_features):
    """
    Train the NBA model on historical pre-game features and
    generate probabilities for one future matchup.
    """

    if feature_games is None or len(feature_games) == 0:
        raise ValueError("No historical feature games supplied.")

    if matchup_features is None or len(matchup_features) == 0:
        raise ValueError("No future matchup features supplied.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    df = feature_games.copy()
    future = matchup_features.copy()

    # Prevent look-ahead leakage when testing historical matchups.
    # Only games strictly before the matchup date may be used
    # to train the prediction model.
    if "game_date" in future.columns:
        prediction_date = pd.to_datetime(
            future["game_date"].iloc[0]
        )
    
        df["game_date"] = pd.to_datetime(
            df["game_date"]
        )
       
    
        df = df[
            df["game_date"] < prediction_date
        ].copy()
    
        
        if len(df) == 0:
            raise ValueError(
                "No historical games exist before the prediction date."
            )
        
        protected_columns = {
            "game_id",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "point_margin",
            "total_points",
            "home_win",
        }
    
        feature_columns = [
            column
            for column in df.columns
            if column not in protected_columns
            and pd.api.types.is_numeric_dtype(df[column])
        ]
    
        if not feature_columns:
            raise ValueError("No model feature columns were found.")
    
        missing_future_columns = [
            column
            for column in feature_columns
            if column not in future.columns
        ]
    
        if missing_future_columns:
            raise ValueError(
                "Future matchup is missing model features: "
                + ", ".join(missing_future_columns)
            )
    
        train_df = df.dropna(
            subset=feature_columns + ["home_win"]
        ).copy()
    
        X_train = train_df[feature_columns]
        y_train = train_df["home_win"].astype(int)
    
        X_future = future[feature_columns]
    
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
    
        model.fit(X_train, y_train)
    
        home_probability = float(
            model.predict_proba(X_future)[0][1]
        )
    
        away_probability = 1.0 - home_probability
    
        predicted_side = (
            "HOME"
            if home_probability >= 0.50
            else "AWAY"
        )
    
        confidence = max(
            home_probability,
            away_probability,
        )
    
        return {
            "home_win_probability": home_probability,
            "away_win_probability": away_probability,
            "predicted_side": predicted_side,
            "confidence": confidence,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "training_games": len(train_df),
        }
   
# ============================================================
# SPORTSBOOK ODDS UTILITIES
# ============================================================

def american_odds_to_implied_probability(odds):
    """
    Convert American sportsbook odds into implied probability.

    Examples:
        -110 -> 52.38%
        +150 -> 40.00%
        -200 -> 66.67%
    """

    odds = float(odds)

    if odds == 0:
        raise ValueError("American odds cannot be zero.")

    if odds > 0:
        probability = 100.0 / (odds + 100.0)
    else:
        probability = abs(odds) / (abs(odds) + 100.0)

    return float(probability)

# ============================================================
# MODEL VS SPORTSBOOK EDGE
# ============================================================

def calculate_model_edge(model_probability, american_odds):
    """
    Compare Market Edge AI probability against sportsbook
    implied probability.

    Returns probabilities as decimals.

    Example:
        Model probability: 0.71
        Sportsbook odds: -150
        Sportsbook implied probability: 0.60
        Model edge: +0.11
    """

    model_probability = float(model_probability)

    if not 0.0 <= model_probability <= 1.0:
        raise ValueError(
            "Model probability must be between 0 and 1."
        )

    sportsbook_probability = (
        american_odds_to_implied_probability(
            american_odds
        )
    )

    model_edge = (
        model_probability - sportsbook_probability
    )

    return {
        "model_probability": model_probability,
        "american_odds": float(american_odds),
        "sportsbook_probability": float(
            sportsbook_probability
        ),
        "model_edge": float(model_edge),
    }

# ============================================================
# NO-VIG SPORTSBOOK PROBABILITIES
# ============================================================

def calculate_no_vig_probabilities(
    home_american_odds,
    away_american_odds,
):
    """
    Convert both sides of an American moneyline market into
    normalized no-vig probabilities.

    Example:
        Home: -110
        Away: -110

        Raw implied probabilities:
            Home = 52.38%
            Away = 52.38%

        No-vig probabilities:
            Home = 50.00%
            Away = 50.00%
    """

    home_raw_probability = (
        american_odds_to_implied_probability(
            home_american_odds
        )
    )

    away_raw_probability = (
        american_odds_to_implied_probability(
            away_american_odds
        )
    )

    total_raw_probability = (
        home_raw_probability
        + away_raw_probability
    )

    if total_raw_probability <= 0:
        raise ValueError(
            "Sportsbook probabilities must total more than zero."
        )

    home_no_vig_probability = (
        home_raw_probability
        / total_raw_probability
    )

    away_no_vig_probability = (
        away_raw_probability
        / total_raw_probability
    )

    sportsbook_hold = (
        total_raw_probability - 1.0
    )

    return {
        "home_raw_probability": float(
            home_raw_probability
        ),
        "away_raw_probability": float(
            away_raw_probability
        ),
        "home_no_vig_probability": float(
            home_no_vig_probability
        ),
        "away_no_vig_probability": float(
            away_no_vig_probability
        ),
        "sportsbook_hold": float(
            sportsbook_hold
        ),
    }

# ============================================================
# MODEL VS NO-VIG MARKET EDGE
# ============================================================

def calculate_no_vig_model_edge(
    home_model_probability,
    home_american_odds,
    away_american_odds,
):
    """
    Compare the NBA model probability against the sportsbook's
    normalized no-vig market probability for both teams.
    """

    home_model_probability = float(
        home_model_probability
    )

    if not 0.0 <= home_model_probability <= 1.0:
        raise ValueError(
            "Home model probability must be between 0 and 1."
        )

    away_model_probability = (
        1.0 - home_model_probability
    )

    market = calculate_no_vig_probabilities(
        home_american_odds,
        away_american_odds,
    )

    home_market_probability = (
        market["home_no_vig_probability"]
    )

    away_market_probability = (
        market["away_no_vig_probability"]
    )

    home_edge = (
        home_model_probability
        - home_market_probability
    )

    away_edge = (
        away_model_probability
        - away_market_probability
    )

    if home_edge >= away_edge:
        best_side = "HOME"
        best_edge = home_edge
        best_model_probability = (
            home_model_probability
        )
        best_market_probability = (
            home_market_probability
        )
    else:
        best_side = "AWAY"
        best_edge = away_edge
        best_model_probability = (
            away_model_probability
        )
        best_market_probability = (
            away_market_probability
        )

    return {
        "home_model_probability": float(
            home_model_probability
        ),
        "away_model_probability": float(
            away_model_probability
        ),
        "home_market_probability": float(
            home_market_probability
        ),
        "away_market_probability": float(
            away_market_probability
        ),
        "home_edge": float(home_edge),
        "away_edge": float(away_edge),
        "best_side": best_side,
        "best_edge": float(best_edge),
        "best_model_probability": float(
            best_model_probability
        ),
        "best_market_probability": float(
            best_market_probability
        ),
        "sportsbook_hold": float(
            market["sportsbook_hold"]
        ),
    }
def get_live_nba_moneylines():
    """
    Retrieve current NBA moneyline odds from The Odds API.
    """

    import requests
    import streamlit as st

    api_key = st.secrets["ODDS_API_KEY"]

    url = (
        "https://api.the-odds-api.com/v4/"
        "sports/basketball_nba/odds"
    )

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    games = response.json()

    results = []

    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                home_odds = None
                away_odds = None

                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        home_odds = outcome.get("price")

                    elif outcome.get("name") == away_team:
                        away_odds = outcome.get("price")

                if home_odds is not None and away_odds is not None:
                    results.append(
                        {
                            "event_id": game.get("id"),
                            "commence_time": game.get(
                                "commence_time"
                            ),
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker.get("title"),
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                        }
                    )

    return results
