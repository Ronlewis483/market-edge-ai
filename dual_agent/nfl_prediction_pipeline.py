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
import time

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


def nfl_week_rows(rows, now=None):
    """Keep unstarted games in the next seven days, one row per matchup."""
    if rows is None:
        return None
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "commence_time" not in frame:
        return frame.iloc[:0].copy()
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    kickoff = pd.to_datetime(frame["commence_time"], utc=True, errors="coerce")
    frame = frame.loc[(kickoff > current) & (kickoff <= current + pd.Timedelta(days=7))].copy()
    frame["commence_time"] = kickoff.loc[frame.index]
    if "confidence" in frame:
        frame = frame.sort_values("confidence", ascending=False, kind="stable")
    identity = [c for c in ("home_team", "away_team", "commence_time") if c in frame]
    if len(identity) == 3:
        frame = frame.drop_duplicates(identity, keep="first")
    return frame.reset_index(drop=True)


def run_nfl_prediction_pipeline(
    feature_games,
    historical_accuracy=None,
    historical_sample=None,
    *,
    live_odds=None,
    on_progress=None,
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

    pipeline_clock = time.perf_counter()
    model_fits = 0

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

    if live_odds is None:
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

    best_lines = nfl_week_rows(best_lines, now=current_time)

    for matchup_number, (_, game) in enumerate(best_lines.iterrows(), 1):
        if on_progress is not None:
            on_progress(f"Calculating NFL matchup {matchup_number} of {len(best_lines)}")

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

        # Do not predict games that already started.
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

            from dual_agent.nfl_connected_engine import predict_original_cached
            prediction = predict_original_cached(
                feature_games=feature_games,
                future_features=future_features,
            )

            model_fits += int(not prediction.get('original_model_cache_hit'))

            prediction_rows.append(
                {
                    "commence_time": game_time,
                    "original_home_win_probability": prediction['home_win_probability'],
                    "connected_features": {k: float(future_features.iloc[0][k]) for k in
                        ('win_pct_diff','avg_point_diff_diff','recent_5_win_pct_diff','recent_5_point_diff_diff','rest_diff')},
                    "connected_status": "Initial team forecast · current context loading in background",
                    "input_status": "Initial team forecast · current context loading in background",
                    "connected_applied": False,
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
            "No NFL predictions for the next seven days "
            "could be generated."
        )

        if error_preview:
            message += (
                f" Prediction errors: {error_preview}"
            )

        raise ValueError(message)

    predictions = nfl_week_rows(predictions, now=current_time)

    # Context, correction model loads and forecast persistence run in the
    # existing app worker, not on the first-card path.
    import hashlib
    import inspect
    connected_base_id = 'nfl_live:' + hashlib.sha256(inspect.getsource(predict_nfl_matchup).encode()).hexdigest()[:20]
    connected_issue = None

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
        "connected_learning_issue": connected_issue,
        "connected_base_id": connected_base_id,
        "connected_historical_accuracy": historical_accuracy,
        "connected_historical_sample": historical_sample,
        "connected_context_pending": True,
        "nfl_original_model_fits": model_fits,
        "nfl_stage_timings_seconds": {"initial_predictions": round(time.perf_counter()-pipeline_clock, 3)},
    }
