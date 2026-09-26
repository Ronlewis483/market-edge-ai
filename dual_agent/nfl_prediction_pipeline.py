"""
MARKET EDGE AI V5
NFL AUTOMATED PREDICTION PIPELINE

Purpose:
Run the complete NFL live prediction workflow from one function.

Workflow:
1. Validate historical feature data
2. Retrieve live NFL moneylines
3. Find best available sportsbook lines
4. Build future matchup features
5. Generate model predictions
6. Analyze model-vs-market opportunities
7. Return all results
"""

import pandas as pd

from dual_agent.nfl_odds import (
    get_live_nfl_moneylines,
    get_best_nfl_moneylines,
)

from dual_agent.nfl_research import (
    build_nfl_future_matchup_features,
    predict_nfl_matchup,
)

from dual_agent.nfl_live_engine import (
    build_live_nfl_opportunities,
)


def run_nfl_prediction_pipeline(
    feature_games,
    historical_accuracy=None,
    historical_sample=None,
):
    """
    Run the complete NFL live prediction pipeline.

    Parameters
    ----------
    feature_games:
        Historical leakage-safe NFL pregame feature dataset.

    historical_accuracy:
        Historical out-of-sample model accuracy.

    historical_sample:
        Number of historical out-of-sample predictions.

    Returns
    -------
    dict
        Live odds, best lines, upcoming predictions,
        market opportunities, and any per-game errors.
    """

    # ==========================================
    # 1. VALIDATE HISTORICAL FEATURES
    # ==========================================

    if feature_games is None:
        raise ValueError(
            "NFL historical feature data was not supplied."
        )

    if not isinstance(feature_games, pd.DataFrame):
        feature_games = pd.DataFrame(feature_games)

    if feature_games.empty:
        raise ValueError(
            "NFL historical feature data is empty."
        )

    # ==========================================
    # 2. LOAD LIVE NFL MONEYLINES
    # ==========================================

    live_odds = get_live_nfl_moneylines()

    if live_odds is None:
        raise ValueError(
            "NFL live odds retrieval returned no data."
        )

    if not isinstance(live_odds, pd.DataFrame):
        live_odds = pd.DataFrame(live_odds)

    if live_odds.empty:
        raise ValueError(
            "No live NFL moneylines are available."
        )

    # ==========================================
    # 3. FIND BEST AVAILABLE MONEYLINES
    # ==========================================

    best_lines = get_best_nfl_moneylines(
    live_odds.to_dict("records")
)

    if best_lines is None:
        raise ValueError(
            "NFL best-line calculation returned no data."
        )

    if not isinstance(best_lines, pd.DataFrame):
        best_lines = pd.DataFrame(best_lines)

    if best_lines.empty:
        raise ValueError(
            "No usable NFL best lines were found."
        )

    required_line_columns = {
        "home_team",
        "away_team",
        "commence_time",
    }

    missing_line_columns = (
        required_line_columns
        - set(best_lines.columns)
    )

    if missing_line_columns:
        raise ValueError(
            "NFL best lines are missing required columns: "
            + ", ".join(
                sorted(missing_line_columns)
            )
        )

    # ==========================================
    # 4. GENERATE UPCOMING NFL PREDICTIONS
    # ==========================================

    prediction_rows = []
    prediction_errors = []

    current_time = pd.Timestamp.now(
        tz="UTC"
    )

    for _, game in best_lines.iterrows():

        home_team = game["home_team"]
        away_team = game["away_team"]

        game_time = pd.to_datetime(
            game["commence_time"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(game_time):
            prediction_errors.append(
                f"{away_team} at {home_team}: "
                "invalid commence time."
            )
            continue

        # Never predict a game that has already started.
        if game_time <= current_time:
            continue

        try:

            future_features = (
                build_nfl_future_matchup_features(
                    feature_games=feature_games,
                    home_team=home_team,
                    away_team=away_team,
                    game_time=game_time,
                )
            )

            prediction = predict_nfl_matchup(
                feature_games=feature_games,
                future_features=future_features,
            )

            prediction_rows.append(
                {
                    "commence_time": game_time,
                    "away_team": away_team,
                    "home_team": home_team,
                    "predicted_team":
                        prediction["predicted_team"],
                    "confidence":
                        prediction["confidence"],
                    "home_win_probability":
                        prediction[
                            "home_win_probability"
                        ],
                    "away_win_probability":
                        prediction[
                            "away_win_probability"
                        ],
                    "training_games":
                        prediction.get(
                            "training_games"
                        ),
                }
            )

        except Exception as error:

            prediction_errors.append(
                f"{away_team} at {home_team}: {error}"
            )

    predictions = pd.DataFrame(
        prediction_rows
    )

    if predictions.empty:

        error_preview = "; ".join(
            prediction_errors[:5]
        )

        message = (
            "No upcoming NFL predictions "
            "could be generated."
        )

        if error_preview:
            message += (
                f" Prediction errors: {error_preview}"
            )

        raise ValueError(message)

    # ==========================================
    # 5. BUILD LIVE NFL OPPORTUNITIES
    # ==========================================

    opportunities = build_live_nfl_opportunities(
        model_predictions=predictions,
        best_lines=best_lines,
        historical_accuracy=historical_accuracy,
        historical_sample=historical_sample,
    )

    if opportunities is None:
        opportunities = pd.DataFrame()

    # ==========================================
    # 6. RETURN COMPLETE PIPELINE
    # ==========================================

    return {
        "live_odds": live_odds,
        "best_lines": best_lines,
        "predictions": predictions,
        "opportunities": opportunities,
        "prediction_errors": prediction_errors,
    }
