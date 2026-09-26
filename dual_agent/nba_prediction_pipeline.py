import pandas as pd

from dual_agent.balldontlie_data import (
    get_multiple_historical_seasons,
)

from dual_agent.nba_research import (
    prepare_balldontlie_games_for_research,
    build_balldontlie_pregame_features,
    build_balldontlie_future_matchup_features,
    predict_balldontlie_matchup,
    get_live_nba_moneylines,
    normalize_nba_team_name,
)


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


def run_nba_prediction_pipeline():
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

    # ========================================================
    # 1. LOAD LIVE NBA MONEYLINES
    # ========================================================

    live_games = get_live_nba_moneylines()

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
        odds_df["game_time"] > current_time
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
    # 3. LOAD HISTORICAL NBA DATA
    # ========================================================

    (
        historical_games,
        season_summary,
        requests_used,
    ) = get_multiple_historical_seasons(
        list(seasons)
    )

    if (
        historical_games is None
        or len(historical_games) == 0
    ):
        raise ValueError(
            "No historical NBA games were returned."
        )

    historical_games = (
        historical_games.copy()
    )

    # ========================================================
    # 4. STRICT ANTI-LEAKAGE CUTOFF
    # ========================================================

    first_game_date = (
        unique_games["game_time"]
        .min()
        .date()
    )

    historical_dates = pd.to_datetime(
        historical_games["game_date"],
        utc=True,
        errors="coerce",
    )

    historical_games = historical_games[
        historical_dates.dt.date
        < first_game_date
    ].copy()

    if historical_games.empty:
        raise ValueError(
            "No historical NBA games are available "
            "before the upcoming matchups."
        )

    # ========================================================
    # 5. PREPARE MODEL TRAINING DATA
    # ========================================================

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

    if (
        feature_games is None
        or len(feature_games) == 0
    ):
        raise ValueError(
            "NBA historical feature generation "
            "returned no usable games."
        )

    # ========================================================
    # 6. GENERATE ONE PREDICTION PER GAME
    # ========================================================

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
                predict_balldontlie_matchup(
                    feature_games,
                    matchup_features,
                )
            )

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

    return {
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
