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

    st.markdown("### 🏀 Multi-Season NBA Validation")

    nba_seasons_text = st.text_input(
        "NBA seasons to validate",
        value="2022,2023,2024,2025",
        help=(
            "Enter SportsDataIO NBA seasons separated "
            "by commas."
        ),
    )

    if st.button("Run Multi-Season NBA Validation"):

        seasons = [
            x.strip()
            for x in nba_seasons_text.split(",")
            if x.strip()
        ]

        try:
            with st.spinner(
                "Downloading multiple NBA seasons and "
                "running chronological walk-forward validation..."
            ):

                multi_result = (
                    run_multi_season_nba_research(
                        seasons,
                        minimum_training_games=500,
                        test_block_size=150,
                    )
                )

                st.session_state[
                    "nba_multi_research_result"
                ] = multi_result

            st.success(
                "Multi-season NBA validation complete."
            )

        except Exception as e:
            st.error(
                f"Multi-season validation error: {e}"
            )

    if (
        "nba_multi_research_result"
        in st.session_state
    ):

        mr = st.session_state[
            "nba_multi_research_result"
        ]

        st.markdown("#### Multi-Season Dataset")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Seasons",
            len(mr["seasons"]),
        )

        s2.metric(
            "Completed games",
            mr["completed_games"],
        )

        s3.metric(
            "Feature rows",
            mr["feature_rows"],
        )

        s4.metric(
            "Home win rate",
            f"{mr['home_win_rate']:.1%}",
        )

        st.markdown("#### Season Coverage")

        st.dataframe(
            mr["season_summary"],
            use_container_width=True,
            hide_index=True,
        )

        model = mr["overall"]
        baseline = mr["baseline"]

        st.markdown(
            "#### Multi-Season Walk-Forward Performance"
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "AUC",
            f"{model['auc']:.3f}",
        )

        p2.metric(
            "Accuracy",
            f"{model['accuracy']:.1%}",
        )

        p3.metric(
            "Brier Score",
            f"{model['brier']:.4f}",
        )

        p4.metric(
            "Log Loss",
            f"{model['logloss']:.4f}",
        )

        st.markdown(
            "#### Multi-Season Model vs Baseline"
        )

        multi_comparison = pd.DataFrame(
            [
                {
                    "Model": "NBA Model",
                    "Accuracy":
                        model["accuracy"],
                    "Brier":
                        model["brier"],
                    "Log Loss":
                        model["logloss"],
                    "AUC":
                        model["auc"],
                },
                {
                    "Model":
                        "Home Win Base Rate",
                    "Accuracy":
                        baseline["accuracy"],
                    "Brier":
                        baseline["brier"],
                    "Log Loss":
                        baseline["logloss"],
                    "AUC":
                        baseline["auc"],
                },
            ]
        )

        st.dataframe(
            multi_comparison,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### Multi-Season Confidence Analysis"
        )

        if len(mr["confidence"]):

            st.dataframe(
                mr["confidence"],
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No confidence-band results "
                "were generated."
            )

        with st.expander(
            "Multi-Season Walk-Forward Folds"
        ):

            st.dataframe(
                mr["folds"],
                use_container_width=True,
                hide_index=True,
            )
