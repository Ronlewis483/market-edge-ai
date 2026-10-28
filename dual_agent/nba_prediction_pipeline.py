import pandas as pd
import streamlit as st
import time

from dual_agent.balldontlie_data import (
    get_multiple_historical_seasons,
)

from dual_agent.nba_research import (
    prepare_balldontlie_games_for_research,
    build_balldontlie_pregame_features,
    build_balldontlie_future_matchup_features,
    train_balldontlie_live_model,
    predict_balldontlie_with_trained_model,
    get_live_nba_moneylines,
    normalize_nba_team_name,
)


NBA_PIPELINE_VERSION = 3

# ============================================================
# MARKET EDGE AI
# NBA ONE-CLICK PREDICTION PIPELINE
# ============================================================


def _american_odds_metrics(
    probability,
    american_odds,
):
    """
    Calculate sportsbook implied probability,
    model edge, and expected ROI.
    """

    probability = float(probability)
    american_odds = float(american_odds)

    if abs(american_odds) < 100:
        return None

    if american_odds > 0:
        implied_probability = (
            100 / (american_odds + 100)
        )
        profit_per_dollar = (
            american_odds / 100
        )

    else:
        implied_probability = (
            abs(american_odds)
            / (abs(american_odds) + 100)
        )
        profit_per_dollar = (
            100 / abs(american_odds)
        )

    model_edge = (
        probability
        - implied_probability
    )

    expected_roi = (
        probability * profit_per_dollar
        - (1 - probability)
    )

    return {
        "implied_probability":
            float(implied_probability),

        "model_edge":
            float(model_edge),

        "expected_roi":
            float(expected_roi),
    }

def _load_nba_historical_features(seasons, cutoff_date, progress=None):
    """Check incremental disk history; reuse unchanged prepared feature tables."""
    historical_games, season_summary, requests_used = get_multiple_historical_seasons(
        seasons, progress=progress)
    prepared, features = _build_cached_nba_historical_features(historical_games, cutoff_date)
    return prepared, features, season_summary, requests_used

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def _build_cached_nba_historical_features(
    historical_games,
    cutoff_date,
):
    """
    Prepare historical NBA games and build leakage-safe
    pregame features once, then reuse them for 24 hours.
    Feature schema 2 uses shared training/live feature definitions.
    """

    if historical_games is None or len(historical_games) == 0:
        raise ValueError(
            "No historical NBA games were supplied."
        )

    historical_games = historical_games.copy()

    historical_dates = pd.to_datetime(
        historical_games["game_date"],
        utc=True,
        errors="coerce",
    )

    historical_games = historical_games[
        historical_dates.dt.date < cutoff_date
    ].copy()

    if historical_games.empty:
        raise ValueError(
            "No historical NBA games are available "
            "before the upcoming matchups."
        )

    prepared_games = (
        prepare_balldontlie_games_for_research(
            historical_games
        )
    )

    feature_games = (
        build_balldontlie_pregame_features(
            prepared_games
        )
    )

    if feature_games is None or len(feature_games) == 0:
        raise ValueError(
            "NBA historical feature generation "
            "returned no usable games."
        )

    return prepared_games, feature_games


@st.cache_resource(ttl=86400, show_spinner=False)
def _train_cached_nba_model(feature_games, prediction_date):
    """Version 3: reuse chronological calibration and holdout assessment."""
    trained = train_balldontlie_live_model(feature_games, prediction_date=prediction_date)
    if trained.get('probability_version') != NBA_PIPELINE_VERSION:
        raise RuntimeError('NBA research/pipeline version mismatch. Replace the entire dual_agent/nba_research.py with the version 3 probability implementation.')
    return trained


def _nba_box_setting(name, default=None):
    import os
    value = os.environ.get(name)
    if value is not None:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def _load_nba_box_context():
    """One compact saved artifact per cache interval; no sports API requests."""
    if str(_nba_box_setting('NBA_WINNER_BOX_CONTEXT_ENABLED', 'false')).lower() != 'true':
        return None, 'Box context awaiting review and activation'
    try:
        from dual_agent.nba_pregame_snapshots import Client
        from dual_agent.nba_winner_box_context import ARTIFACT
        settings = {name: _nba_box_setting(name) for name in
                    ('SUPABASE_URL', 'SUPABASE_KEY', 'BALLDONTLIE_API_KEY')}
        settings['NBA_SNAPSHOT_BUCKET'] = _nba_box_setting('NBA_SNAPSHOT_BUCKET', 'market-edge-data')
        artifact = Client(env=settings).read(ARTIFACT, compressed=True)
        if not artifact or not artifact.get('checks_passed'):
            return None, 'Box context not ready or comparison checks did not pass'
        return artifact, 'Saved box context loaded'
    except Exception as exc:
        return None, 'Box context unavailable (' + type(exc).__name__ + '); team-form model retained'


def _nba_unified_client():
    from dual_agent.nba_pregame_snapshots import Client
    settings = {name: _nba_box_setting(name) for name in
                ('SUPABASE_URL', 'SUPABASE_KEY', 'BALLDONTLIE_API_KEY')}
    settings['NBA_SNAPSHOT_BUCKET'] = _nba_box_setting('NBA_SNAPSHOT_BUCKET', 'market-edge-data')
    return Client(env=settings)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_nba_unified_bundle():
    """Cache successful loads; missing first-run artifacts must not be cached."""
    from dual_agent.nba_unified_winner import ARTIFACT
    artifact = _nba_unified_client().read(ARTIFACT, compressed=True)
    if artifact is None:
        raise FileNotFoundError('Train NBA unified winner first')
    return artifact


def _load_nba_unified_winner():
    """Reuse one compact bundle, including while predictions are compared."""
    try:
        return _cached_nba_unified_bundle(), 'ready'
    except FileNotFoundError:
        return None, 'Train NBA unified winner first'
    except Exception as exc:
        return None, 'Unified winner unavailable (' + type(exc).__name__ + ')'


def run_nba_prediction_pipeline(
    progress_callback=None,
):
    """
    Generate predictions for all upcoming NBA games.

    Returns:
        live_odds:
            Raw upcoming sportsbook moneylines.

        predictions:
            One model prediction per NBA matchup.

        opportunities:
            Model-vs-market analysis for available
            sportsbook moneylines.

        prediction_errors:
            Games that could not be modeled.
    """

    pipeline_start = time.perf_counter()
    
    def report(stage):
        elapsed = (
            time.perf_counter()
            - pipeline_start
        )
    
        if progress_callback:
            progress_callback(
                stage,
                elapsed,
            )

    # ========================================================
    # 1. LOAD LIVE NBA MONEYLINES
    # ========================================================

    report("Loading live NBA odds...")

    live_games = get_live_nba_moneylines()

    report("Live NBA odds loaded")

    if not live_games:
        raise ValueError(
            "No upcoming NBA moneylines are "
            "currently available."
        )

    odds_df = pd.DataFrame(live_games)

    required_columns = [
        "event_id",
        "commence_time",
        "home_team",
        "away_team",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in odds_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "NBA live odds are missing required "
            "columns: "
            + ", ".join(missing_columns)
        )

    odds_df["game_time"] = pd.to_datetime(
        odds_df["commence_time"],
        utc=True,
        errors="coerce",
    )

    odds_df = odds_df.dropna(
        subset=["game_time"]
    ).copy()

    current_time = pd.Timestamp.now(
        tz="UTC"
    )

    odds_df = odds_df[
        (odds_df["game_time"] > current_time)
        & (odds_df["game_time"] <= current_time + pd.Timedelta(days=7))
    ].copy()

    if odds_df.empty:
        raise ValueError(
            "No upcoming NBA games were found."
        )

    # Each sportsbook may return the same event.
    # Model each unique NBA game only once.

    unique_games = (
        odds_df
        .sort_values("game_time")
        .drop_duplicates(
            subset=["event_id"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 2. DETERMINE HISTORICAL SEASONS
    # ========================================================

    latest_game = (
        unique_games["game_time"].max()
    )

    season_year = (
        latest_game.year
        if latest_game.month >= 9
        else latest_game.year - 1
    )

    seasons = tuple(
        range(
            season_year - 3,
            season_year + 1,
        )
    )

    # ========================================================
    # 3. LOAD CACHED HISTORICAL NBA MODEL DATA
    # ========================================================

    first_game_date = (
        unique_games["game_time"]
        .min()
        .date()
    )

    report(
        "Loading historical NBA data/features..."
    )

    (
        prepared_games,
        feature_games,
        season_summary,
        requests_used,
    ) = _load_nba_historical_features(
        seasons=seasons,
        cutoff_date=first_game_date,
        progress=report,
    )

    report(
        "Historical NBA data/features ready"
    )
    
    # ========================================================
    # TRAIN LIVE MODEL ONCE FOR ENTIRE UPCOMING SLATE
    # ========================================================

    report("Training NBA model...")

    trained_model = _train_cached_nba_model(
        feature_games=feature_games,
        prediction_date=first_game_date,
    )

    report("NBA model and chronological confidence assessment ready")


    # ========================================================
    # 6. GENERATE ONE PREDICTION PER GAME
    # ========================================================

    report(
        f"Generating predictions for "
        f"{len(unique_games)} games..."
    )

    unified_artifact, unified_status = _load_nba_unified_winner()
    box_artifact, box_status = _load_nba_box_context()
    prediction_rows = []
    prediction_errors = []

    for _, game in unique_games.iterrows():

        home_team = game["home_team"]
        away_team = game["away_team"]

        try:

            home_code = (
                normalize_nba_team_name(
                    home_team
                )
            )

            away_code = (
                normalize_nba_team_name(
                    away_team
                )
            )

            if not home_code or not away_code:
                raise ValueError(
                    "Unable to identify NBA team."
                )

            game_date = (
                game["game_time"].date()
            )

            matchup_features = (
                build_balldontlie_future_matchup_features(
                    prepared_games,
                    home_code,
                    away_code,
                    game_date,
                )
            )

            prediction = (
                predict_balldontlie_with_trained_model(
                    trained_model,
                    matchup_features,
                )
            )

            original_prediction = dict(prediction)
            learning_fields = {}
            try:
                from dual_agent.nba_winner_comparison import predict as learning_predict
                learning_fields = learning_predict(matchup_features, game['commence_time'])
            except Exception as exc:
                learning_fields = {'learning_model_issue':type(exc).__name__+': '+str(exc)}

            box_used = False
            box_issue = box_status
            if box_artifact is not None:
                try:
                    from dual_agent.nba_winner_box_context import predict as box_predict
                    expanded = box_predict(box_artifact, matchup_features, home_team,
                                           away_team, game['commence_time'])
                    if expanded is not None:
                        prediction = expanded
                        box_used = True
                    else:
                        box_issue = 'Recent verified box coverage unavailable; team-form model retained'
                except Exception as exc:
                    box_issue = 'Box context unavailable (' + type(exc).__name__ + '); team-form model retained'

            unified_forecast = None
            unified_used = False
            unified_issue = unified_status
            if unified_artifact is not None and learning_fields.get('learning_home_probability') is not None:
                try:
                    from dual_agent.nba_unified_winner import predict as unified_predict
                    unified_forecast = unified_predict(
                        unified_artifact, matchup_features,
                        original_prediction['home_win_probability'],
                        learning_fields['learning_home_probability'],
                        home_team, away_team, game['commence_time'])
                    if unified_forecast is not None:
                        unified_forecast['original_model_version'] = original_prediction.get('probability_version', 3)
                    activated = str(_nba_box_setting('NBA_UNIFIED_WINNER_ENABLED', 'false')).lower() == 'true'
                    if unified_forecast is not None and activated and unified_forecast['checks_passed']:
                        p = unified_forecast['combined_home_probability']
                        prediction = dict(original_prediction,
                            home_win_probability=p, away_win_probability=1-p,
                            predicted_side='HOME' if p >= .5 else 'AWAY', confidence=max(p,1-p),
                            raw_home_win_probability=p, calibration_games=0,
                            probability_version=unified_forecast['model_id'],
                            winner_model_id=unified_forecast['model_id'],
                            calibration_status='Three-source NBA ensemble · live accuracy tracked separately')
                        unified_used = True
                        box_used = False
                    unified_issue = 'Applied' if unified_used else 'Saved for comparison; current engine retained'
                except Exception as exc:
                    unified_issue = 'Unified winner unavailable (' + type(exc).__name__ + '); current engine retained'

            home_probability = float(
                prediction[
                    "home_win_probability"
                ]
            )

            away_probability = float(
                prediction[
                    "away_win_probability"
                ]
            )

            predicted_side = prediction[
                "predicted_side"
            ]

            predicted_team = (
                home_team
                if predicted_side == "HOME"
                else away_team
            )

            confidence = max(
                home_probability,
                away_probability,
            )


            prediction_rows.append({
                **learning_fields,
                'unified_forecast': unified_forecast,
                'unified_used': unified_used,
                'unified_status': unified_issue,
                'winner_model_id': prediction.get('winner_model_id', 'nba_team_form_v3'),
                'box_context_used': box_used,
                'box_context_status': 'Saved box context applied' if box_used else box_issue,
                "event_id":
                    game["event_id"],

                "commence_time":
                    game["commence_time"],

                "away_team":
                    away_team,

                "home_team":
                    home_team,

                "predicted_team":
                    predicted_team,

                "predicted_side":
                    predicted_side,

                "confidence":
                    float(confidence),

                "home_win_probability":
                    home_probability,

                "away_win_probability":
                    away_probability,

                "training_games":
                    prediction.get(
                        "training_games"
                    ),
                "input_status": prediction.get('calibration_status','Team-form estimate · confidence not calibrated'),
                'raw_home_win_probability': prediction.get('raw_home_win_probability'),
                'calibration_games': prediction.get('calibration_games',0),
                'probability_version': prediction.get('probability_version'),
            })

        except Exception as error:

            prediction_errors.append({
                "event_id":
                    game.get("event_id"),

                "away_team":
                    away_team,

                "home_team":
                    home_team,

                "error":
                    str(error),
            })

    predictions = pd.DataFrame(
        prediction_rows
    )

    if predictions.empty:
        raise ValueError(
            "NBA pipeline could not generate "
            "any upcoming predictions."
        )

    predictions = (
        predictions
        .sort_values(
            "confidence",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    report("NBA matchup predictions complete")

    # ========================================================
    # 7. BUILD SPORTSBOOK OPPORTUNITIES
    # ========================================================

    opportunity_rows = []

    prediction_lookup = {
        row["event_id"]: row
        for row in prediction_rows
    }

    for _, sportsbook in odds_df.iterrows():

        event_id = sportsbook[
            "event_id"
        ]

        prediction = prediction_lookup.get(
            event_id
        )

        if prediction is None:
            continue

        for (
            side,
            team,
            probability,
            odds_key,
        ) in [
            (
                "HOME",
                prediction["home_team"],
                prediction[
                    "home_win_probability"
                ],
                "home_odds",
            ),
            (
                "AWAY",
                prediction["away_team"],
                prediction[
                    "away_win_probability"
                ],
                "away_odds",
            ),
        ]:

            if odds_key not in sportsbook:
                continue

            american_odds = pd.to_numeric(
                sportsbook[odds_key],
                errors="coerce",
            )

            if pd.isna(american_odds):
                continue

            market_metrics = (
                _american_odds_metrics(
                    probability,
                    american_odds,
                )
            )

            if market_metrics is None:
                continue

            opportunity_rows.append({
                "event_id":
                    event_id,

                "commence_time":
                    prediction[
                        "commence_time"
                    ],

                "matchup":
                    (
                        f'{prediction["away_team"]} '
                        f'@ {prediction["home_team"]}'
                    ),

                "predicted_winner":
                    prediction[
                        "predicted_team"
                    ],

                "betting_side":
                    team,

                "side":
                    side,

                "model_probability":
                    float(probability),

                "sportsbook":
                    sportsbook.get(
                        "bookmaker"
                    ),

                "american_odds":
                    float(american_odds),

                "implied_probability":
                    market_metrics[
                        "implied_probability"
                    ],

                "model_edge":
                    market_metrics[
                        "model_edge"
                    ],

                "expected_roi":
                    market_metrics[
                        "expected_roi"
                    ],

                "training_games":
                    prediction[
                        "training_games"
                    ],
            })

    opportunities = pd.DataFrame(
        opportunity_rows
    )

    if not opportunities.empty:
        opportunities = (
            opportunities
            .sort_values(
                [
                    "expected_roi",
                    "model_edge",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

    # ========================================================
    # 8. RETURN COMPLETE PIPELINE
    # ========================================================

    comparison_issue = None
    try:
        from dual_agent.nba_winner_comparison import freeze_and_grade
        comparison_ledger = freeze_and_grade(predictions, prepared_games)
        if comparison_ledger.get('persistence_issue'):
            comparison_issue = comparison_ledger['persistence_issue']
    except Exception as exc:
        comparison_issue = 'Winner comparison could not be saved/graded: '+type(exc).__name__
    unified_save_issue = None
    if 'unified_forecast' in predictions and predictions['unified_forecast'].map(lambda x: isinstance(x, dict)).any():
        try:
            from dual_agent.nba_unified_winner import save_forecasts
            save_forecasts(_nba_unified_client(), predictions)
        except Exception as exc:
            unified_save_issue = 'Unified forecast persistence failed (' + type(exc).__name__ + ')'
    report("NBA prediction pipeline complete")

    return {
        'pipeline_version': NBA_PIPELINE_VERSION,
        'unified_forecast_save_issue': unified_save_issue,
        'winner_comparison_issue': comparison_issue,
        'validation_summary': trained_model.get('validation_summary', {}),
        'calibration_status': trained_model.get('calibration_status'),
        "live_odds":
            odds_df.reset_index(drop=True),

        "predictions":
            predictions,

        "opportunities":
            opportunities,

        "prediction_errors":
            prediction_errors,

        "historical_games":
            prepared_games,

        "feature_games":
            feature_games,

        "seasons":
            seasons,

        "season_summary":
            season_summary,

        "requests_used":
            requests_used,
    }
