"""
Market Edge AI
NFL Player Props V2A

Purpose
-------
Experimental player-prop projection engine.

V2A improves on the original historical-hit-rate baseline by:

1. Building a recency-weighted player projection.
2. Estimating player-specific volatility.
3. Converting the sportsbook line into an Over/Under probability.
4. Removing sportsbook vig when both sides are available.
5. Calculating edge versus the no-vig market.
6. Calculating expected value from the actual sportsbook price.
7. Supporting leakage-safe walk-forward validation.

IMPORTANT:
This module does NOT replace nfl_player_props.py yet.
It reuses stable plumbing from V1 while the new model is validated.
"""

import math

import numpy as np
import pandas as pd


from dual_agent.nfl_player_props import (
    MARKETS,
    canonical_name,
    implied_probability,
    validate_history,
)


# ============================================================
# SUPPORTED V2A MARKETS
# ============================================================

V2_MARKETS = {
    "Passing yards",
    "Rushing yards",
    "Receiving yards",
    "Receptions",
    "Passing completions",
}


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_WINDOW = 12
DEFAULT_MIN_GAMES = 6

MIN_STANDARD_DEVIATION = {
    "Passing yards": 20.0,
    "Rushing yards": 8.0,
    "Receiving yards": 8.0,
    "Receptions": 1.0,
    "Passing completions": 2.0,
}


# ============================================================
# BASIC MATH HELPERS
# ============================================================

def _normal_cdf(value):
    """
    Standard normal cumulative distribution function.

    Uses math.erf so V2A does not require scipy.
    """

    return 0.5 * (
        1.0
        + math.erf(
            float(value)
            / math.sqrt(2.0)
        )
    )


def _clip_probability(value):
    """
    Prevent probabilities from reaching exactly 0 or 1.
    """

    return float(
        np.clip(
            float(value),
            0.01,
            0.99,
        )
    )


def american_profit_per_unit(american_odds):
    """
    Profit returned on a 1-unit stake if the bet wins.

    Examples
    --------
    -110 -> 0.9091 units profit
    +120 -> 1.20 units profit
    """

    odds = float(american_odds)

    if odds == 0:
        raise ValueError(
            "American odds cannot be zero."
        )

    if odds > 0:
        return odds / 100.0

    return 100.0 / abs(odds)


def expected_value_per_unit(
    probability,
    american_odds,
):
    """
    Expected profit per 1 unit risked.

    EV =
        P(win) * profit_if_win
        - P(loss) * 1 unit
    """

    probability = float(probability)

    profit = american_profit_per_unit(
        american_odds
    )

    return (
        probability * profit
        - (1.0 - probability)
    )


# ============================================================
# HISTORY
# ============================================================

def historical_sample_v2(
    history,
    player,
    market,
    cutoff,
    window=DEFAULT_WINDOW,
    min_games=DEFAULT_MIN_GAMES,
):
    """
    Return only games that occurred before the prediction
    cutoff.

    This preserves the leakage-safe behavior of V1.
    """

    if market not in V2_MARKETS:
        return None

    cutoff = pd.Timestamp(cutoff)

    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(
            "UTC"
        )

    else:
        cutoff = cutoff.tz_convert(
            "UTC"
        )

    player_key = canonical_name(
        player
    )

    sample = history[
        (
            history["player_key"]
            == player_key
        )
        &
        (
            history["market"]
            == market
        )
        &
        (
            history["game_time"]
            < cutoff
        )
    ].copy()

    sample = (
        sample
        .sort_values(
            "game_time"
        )
        .drop_duplicates(
            subset=[
                "game_time",
                "player_key",
                "market",
            ],
            keep="last",
        )
        .tail(window)
    )

    if len(sample) < min_games:
        return None

    return sample


# ============================================================
# RECENCY WEIGHTING
# ============================================================

def recency_weights(
    sample_size,
    decay=0.88,
):
    """
    Create exponential recency weights.

    Newer games receive more weight.

    Example:
    oldest game -> smaller weight
    newest game -> largest weight
    """

    sample_size = int(
        sample_size
    )

    if sample_size <= 0:
        return np.array(
            [],
            dtype=float,
        )

    powers = np.arange(
        sample_size - 1,
        -1,
        -1,
        dtype=float,
    )

    weights = np.power(
        float(decay),
        powers,
    )

    return (
        weights
        / weights.sum()
    )


def weighted_average(
    values,
    decay=0.88,
):
    """
    Recency-weighted average.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    weights = recency_weights(
        len(values),
        decay=decay,
    )

    return float(
        np.average(
            values,
            weights=weights,
        )
    )


def weighted_standard_deviation(
    values,
    decay=0.88,
):
    """
    Recency-weighted population standard deviation.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) <= 1:
        return 0.0

    weights = recency_weights(
        len(values),
        decay=decay,
    )

    mean = np.average(
        values,
        weights=weights,
    )

    variance = np.average(
        np.square(
            values - mean
        ),
        weights=weights,
    )

    return float(
        math.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


# ============================================================
# PLAYER PROJECTION
# ============================================================

def build_player_projection(
    sample,
    market,
    decay=0.88,
):
    """
    Build V2A projection statistics from historical games.

    Returns:
        projection
        volatility
        recent averages
        trend information
    """

    if sample is None:
        return None

    if sample.empty:
        return None

    values = (
        pd.to_numeric(
            sample["value"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(values) == 0:
        return None

    season_average = float(
        np.mean(values)
    )

    projection = weighted_average(
        values,
        decay=decay,
    )

    raw_std = weighted_standard_deviation(
        values,
        decay=decay,
    )

    std_floor = MIN_STANDARD_DEVIATION.get(
        market,
        1.0,
    )

    volatility = max(
        raw_std,
        float(std_floor),
    )

    recent_3 = float(
        np.mean(
            values[-3:]
        )
    )

    recent_5 = float(
        np.mean(
            values[-5:]
        )
    )

    trend = (
        recent_3
        - season_average
    )

    return {
        "projection": float(
            projection
        ),
        "volatility": float(
            volatility
        ),
        "season_average": float(
            season_average
        ),
        "recent_3_average": float(
            recent_3
        ),
        "recent_5_average": float(
            recent_5
        ),
        "trend": float(
            trend
        ),
        "historical_games": int(
            len(values)
        ),
        "minimum_value": float(
            np.min(values)
        ),
        "maximum_value": float(
            np.max(values)
        ),
    }


# ============================================================
# DISTRIBUTION PROBABILITY
# ============================================================

def probability_from_distribution(
    projection,
    volatility,
    line,
    side,
):
    """
    Estimate probability of clearing the sportsbook line.

    V2A uses a normal approximation around the player's
    recency-weighted projection.

    For OVER:
        P(X > line)

    For UNDER:
        P(X < line)

    Half-point lines naturally avoid pushes.

    Whole-number lines are treated as approximate
    continuous thresholds in V2A. Exact push modeling
    can be added later for count markets.
    """

    projection = float(
        projection
    )

    volatility = float(
        volatility
    )

    line = float(
        line
    )

    if volatility <= 0:
        return None

    z_score = (
        line - projection
    ) / volatility

    under_probability = (
        _normal_cdf(
            z_score
        )
    )

    over_probability = (
        1.0
        - under_probability
    )

    if side == "Over":

        return _clip_probability(
            over_probability
        )

    if side == "Under":

        return _clip_probability(
            under_probability
        )

    return None


# ============================================================
# MARKET PROBABILITY / VIG REMOVAL
# ============================================================

def remove_vig(
    over_odds,
    under_odds,
):
    """
    Convert both sportsbook prices into normalized
    no-vig probabilities.

    Returns:
        over_market_probability
        under_market_probability
        sportsbook_hold
    """

    over_raw = implied_probability(
        over_odds
    )

    under_raw = implied_probability(
        under_odds
    )

    total = (
        over_raw
        + under_raw
    )

    if total <= 0:
        return None

    return {
        "over_market_probability":
            float(
                over_raw
                / total
            ),

        "under_market_probability":
            float(
                under_raw
                / total
            ),

        "sportsbook_hold":
            float(
                total - 1.0
            ),

        "over_raw_implied":
            float(
                over_raw
            ),

        "under_raw_implied":
            float(
                under_raw
            ),
    }


# ============================================================
# PAIR SPORTSBOOK OVER / UNDER LINES
# ============================================================

def pair_prop_sides(
    odds,
):
    """
    Pair Over and Under prices from the SAME sportsbook
    for the SAME player / market / line.

    This allows proper vig removal.
    """

    if odds is None:
        return pd.DataFrame()

    if odds.empty:
        return pd.DataFrame()

    required = {
        "event_id",
        "game_time",
        "home_team",
        "away_team",
        "bookmaker",
        "bookmaker_key",
        "market_key",
        "player",
        "side",
        "line",
        "american_odds",
        "last_update",
    }

    missing = (
        required
        - set(
            odds.columns
        )
    )

    if missing:

        raise ValueError(
            "Odds data missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    working = odds.copy()

    working = working[
        working["side"].isin(
            [
                "Over",
                "Under",
            ]
        )
    ].copy()

    if working.empty:
        return pd.DataFrame()

    index_columns = [
        "event_id",
        "game_time",
        "home_team",
        "away_team",
        "bookmaker",
        "bookmaker_key",
        "market_key",
        "player",
        "line",
    ]

    over = (
        working[
            working["side"]
            == "Over"
        ]
        .copy()
    )

    under = (
        working[
            working["side"]
            == "Under"
        ]
        .copy()
    )

    over = over.rename(
        columns={
            "american_odds":
                "over_odds",
            "last_update":
                "over_last_update",
        }
    )

    under = under.rename(
        columns={
            "american_odds":
                "under_odds",
            "last_update":
                "under_last_update",
        }
    )

    keep_over = (
        index_columns
        + [
            "over_odds",
            "over_last_update",
        ]
    )

    keep_under = (
        index_columns
        + [
            "under_odds",
            "under_last_update",
        ]
    )

    paired = over[
        keep_over
    ].merge(
        under[
            keep_under
        ],
        on=index_columns,
        how="inner",
        validate="one_to_one",
    )

    return paired


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

def classify_opportunity(
    model_probability,
    market_probability,
    expected_value,
    historical_games,
):
    """
    Conservative V2A research classification.

    These thresholds are NOT claimed to be profitable.
    They simply create consistent buckets for validation.
    """

    edge = (
        float(model_probability)
        - float(market_probability)
    )

    expected_value = float(
        expected_value
    )

    historical_games = int(
        historical_games
    )

    if historical_games < 6:
        return "PASS"

    if (
        model_probability >= 0.57
        and edge >= 0.05
        and expected_value >= 0.05
    ):
        return "BET"

    if (
        model_probability >= 0.54
        and edge >= 0.025
        and expected_value > 0
    ):
        return "LEAN"

    return "PASS"


# ============================================================
# ANALYZE LIVE SPORTSBOOK PROPS
# ============================================================

def analyze_v2(
    odds,
    history,
    window=DEFAULT_WINDOW,
    min_games=DEFAULT_MIN_GAMES,
    decay=0.88,
):
    """
    Analyze live NFL player props with V2A.

    Input:
        normalized sportsbook odds from
        nfl_player_props.normalize_props()

        validated player history from
        nfl_player_props.validate_history()

    Output:
        one row per sportsbook/player/market/line/side
        with projection, probability, no-vig market
        probability, edge and EV.
    """

    if odds is None:
        return pd.DataFrame()

    if odds.empty:
        return pd.DataFrame()

    history = validate_history(
        history
    )

    paired = pair_prop_sides(
        odds
    )

    if paired.empty:
        return pd.DataFrame()

    market_lookup = {
        market_key: market_name
        for (
            market_name,
            market_key,
        )
        in MARKETS.items()
    }

    results = []

    for _, row in paired.iterrows():

        market = market_lookup.get(
            row["market_key"]
        )

        if market not in V2_MARKETS:
            continue

        sample = historical_sample_v2(
            history=history,
            player=row["player"],
            market=market,
            cutoff=row["game_time"],
            window=window,
            min_games=min_games,
        )

        if sample is None:
            continue

        projection_data = (
            build_player_projection(
                sample=sample,
                market=market,
                decay=decay,
            )
        )

        if projection_data is None:
            continue

        market_data = remove_vig(
            over_odds=row[
                "over_odds"
            ],
            under_odds=row[
                "under_odds"
            ],
        )

        if market_data is None:
            continue

        for side in [
            "Over",
            "Under",
        ]:

            if side == "Over":

                american_odds = (
                    row[
                        "over_odds"
                    ]
                )

                market_probability = (
                    market_data[
                        "over_market_probability"
                    ]
                )

                last_update = (
                    row[
                        "over_last_update"
                    ]
                )

            else:

                american_odds = (
                    row[
                        "under_odds"
                    ]
                )

                market_probability = (
                    market_data[
                        "under_market_probability"
                    ]
                )

                last_update = (
                    row[
                        "under_last_update"
                    ]
                )

            model_probability = (
                probability_from_distribution(
                    projection=projection_data[
                        "projection"
                    ],
                    volatility=projection_data[
                        "volatility"
                    ],
                    line=row["line"],
                    side=side,
                )
            )

            if model_probability is None:
                continue

            edge = (
                model_probability
                - market_probability
            )

            ev = expected_value_per_unit(
                probability=model_probability,
                american_odds=american_odds,
            )

            recommendation = (
                classify_opportunity(
                    model_probability=
                        model_probability,
                    market_probability=
                        market_probability,
                    expected_value=ev,
                    historical_games=
                        projection_data[
                            "historical_games"
                        ],
                )
            )

            results.append(
                {
                    "event_id":
                        row["event_id"],

                    "game_time":
                        row["game_time"],

                    "home_team":
                        row["home_team"],

                    "away_team":
                        row["away_team"],

                    "bookmaker":
                        row["bookmaker"],

                    "bookmaker_key":
                        row[
                            "bookmaker_key"
                        ],

                    "market":
                        market,

                    "market_key":
                        row["market_key"],

                    "player":
                        row["player"],

                    "side":
                        side,

                    "line":
                        float(
                            row["line"]
                        ),

                    "american_odds":
                        float(
                            american_odds
                        ),

                    "projection":
                        round(
                            projection_data[
                                "projection"
                            ],
                            2,
                        ),

                    "volatility":
                        round(
                            projection_data[
                                "volatility"
                            ],
                            2,
                        ),

                    "season_average":
                        round(
                            projection_data[
                                "season_average"
                            ],
                            2,
                        ),

                    "recent_3_average":
                        round(
                            projection_data[
                                "recent_3_average"
                            ],
                            2,
                        ),

                    "recent_5_average":
                        round(
                            projection_data[
                                "recent_5_average"
                            ],
                            2,
                        ),

                    "trend":
                        round(
                            projection_data[
                                "trend"
                            ],
                            2,
                        ),

                    "historical_games":
                        projection_data[
                            "historical_games"
                        ],

                    "model_probability":
                        round(
                            model_probability,
                            4,
                        ),

                    "market_probability":
                        round(
                            market_probability,
                            4,
                        ),

                    "edge_pp":
                        round(
                            edge * 100.0,
                            2,
                        ),

                    "expected_value":
                        round(
                            ev,
                            4,
                        ),

                    "expected_value_pct":
                        round(
                            ev * 100.0,
                            2,
                        ),

                    "sportsbook_hold_pct":
                        round(
                            market_data[
                                "sportsbook_hold"
                            ]
                            * 100.0,
                            2,
                        ),

                    "recommendation":
                        recommendation,

                    "last_update":
                        last_update,

                    "model_version":
                        "NFL_PROP_V2A",
                }
            )

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(
        results
    )

    recommendation_rank = {
        "BET": 0,
        "LEAN": 1,
        "PASS": 2,
    }

    result[
        "_recommendation_rank"
    ] = (
        result[
            "recommendation"
        ]
        .map(
            recommendation_rank
        )
        .fillna(3)
    )

    result = (
        result
        .sort_values(
            [
                "_recommendation_rank",
                "expected_value_pct",
                "edge_pp",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop(
            columns=[
                "_recommendation_rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# WALK-FORWARD PROJECTION TEST
# ============================================================

def walkforward_v2(
    history,
    window=DEFAULT_WINDOW,
    min_games=DEFAULT_MIN_GAMES,
    decay=0.88,
):
    """
    Leakage-safe walk-forward evaluation of the V2A
    STAT PROJECTION.

    For each historical game:

        1. Use only earlier games.
        2. Build the V2A projection.
        3. Compare projection to actual result.

    This does NOT claim historical sportsbook ROI because
    historical prop lines/odds are not present in the
    current player-history dataset.
    """

    history = validate_history(
        history
    )

    history = history[
        history["market"].isin(
            V2_MARKETS
        )
    ].copy()

    rows = []

    grouped = history.groupby(
        [
            "player_key",
            "market",
        ]
    )

    for (
        player_key,
        market,
    ), group in grouped:

        group = (
            group
            .sort_values(
                "game_time"
            )
            .drop_duplicates(
                subset=[
                    "game_time"
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        if len(group) <= min_games:
            continue

        for index in range(
            min_games,
            len(group),
        ):

            current = group.iloc[
                index
            ]

            previous = group.iloc[
                max(
                    0,
                    index - window,
                ):
                index
            ].copy()

            if len(previous) < min_games:
                continue

            projection_data = (
                build_player_projection(
                    sample=previous,
                    market=market,
                    decay=decay,
                )
            )

            if projection_data is None:
                continue

            actual = float(
                current["value"]
            )

            forecast = float(
                projection_data[
                    "projection"
                ]
            )

            error = (
                forecast
                - actual
            )

            rows.append(
                {
                    "player":
                        current[
                            "player"
                        ],

                    "player_key":
                        player_key,

                    "market":
                        market,

                    "game_time":
                        current[
                            "game_time"
                        ],

                    "forecast":
                        forecast,

                    "actual":
                        actual,

                    "error":
                        error,

                    "absolute_error":
                        abs(
                            error
                        ),

                    "squared_error":
                        error ** 2,

                    "volatility":
                        projection_data[
                            "volatility"
                        ],

                    "training_games":
                        projection_data[
                            "historical_games"
                        ],

                    "model_version":
                        "NFL_PROP_V2A",
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WALK-FORWARD SUMMARY
# ============================================================

def summarize_walkforward(
    predictions,
):
    """
    Summarize projection accuracy overall and by market.
    """

    if predictions is None:
        return {
            "prediction_count": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "by_market": pd.DataFrame(),
        }

    if predictions.empty:
        return {
            "prediction_count": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "by_market": pd.DataFrame(),
        }

    overall_mae = float(
        predictions[
            "absolute_error"
        ].mean()
    )

    overall_rmse = float(
        math.sqrt(
            predictions[
                "squared_error"
            ].mean()
        )
    )

    overall_bias = float(
        predictions[
            "error"
        ].mean()
    )

    by_market = (
        predictions
        .groupby(
            "market",
            as_index=False,
        )
        .agg(
            prediction_count=(
                "actual",
                "size",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            mse=(
                "squared_error",
                "mean",
            ),
            bias=(
                "error",
                "mean",
            ),
            average_actual=(
                "actual",
                "mean",
            ),
            average_forecast=(
                "forecast",
                "mean",
            ),
        )
    )

    by_market[
        "rmse"
    ] = np.sqrt(
        by_market[
            "mse"
        ]
    )

    by_market = by_market.drop(
        columns=[
            "mse"
        ]
    )

    numeric_columns = [
        "mae",
        "rmse",
        "bias",
        "average_actual",
        "average_forecast",
    ]

    for column in numeric_columns:

        by_market[column] = (
            by_market[column]
            .round(3)
        )

    return {
        "prediction_count":
            int(
                len(
                    predictions
                )
            ),

        "mae":
            overall_mae,

        "rmse":
            overall_rmse,

        "bias":
            overall_bias,

        "by_market":
            by_market,
    }


# ============================================================
# V1 VS V2 PROJECTION COMPARISON
# ============================================================

def compare_v1_v2_walkforward(
    history,
    window=DEFAULT_WINDOW,
    min_games=DEFAULT_MIN_GAMES,
    decay=0.88,
):
    """
    Compare V1 mean projection against V2A's
    recency-weighted projection on the SAME historical
    prediction opportunities.

    Lower MAE and RMSE are better.

    Bias closer to zero is better.
    """

    history = validate_history(
        history
    )

    v2_predictions = walkforward_v2(
        history=history,
        window=window,
        min_games=min_games,
        decay=decay,
    )

    if v2_predictions.empty:

        return {
            "v1": None,
            "v2": None,
            "comparison":
                pd.DataFrame(),
            "paired_predictions":
                pd.DataFrame(),
        }

    rows = []

    grouped = history[
        history["market"].isin(
            V2_MARKETS
        )
    ].groupby(
        [
            "player_key",
            "market",
        ]
    )

    for (
        player_key,
        market,
    ), group in grouped:

        group = (
            group
            .sort_values(
                "game_time"
            )
            .drop_duplicates(
                subset=[
                    "game_time"
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        for index in range(
            min_games,
            len(group),
        ):

            previous = (
                group.iloc[
                    max(
                        0,
                        index - window,
                    ):
                    index
                ]["value"]
                .to_numpy(
                    dtype=float
                )
            )

            if len(previous) < min_games:
                continue

            current = group.iloc[
                index
            ]

            actual = float(
                current["value"]
            )

            v1_forecast = float(
                np.mean(
                    previous
                )
            )

            rows.append(
                {
                    "player_key":
                        player_key,

                    "market":
                        market,

                    "game_time":
                        current[
                            "game_time"
                        ],

                    "actual":
                        actual,

                    "v1_forecast":
                        v1_forecast,
                }
            )

    v1_predictions = pd.DataFrame(
        rows
    )

    if v1_predictions.empty:

        return {
            "v1": None,
            "v2": None,
            "comparison":
                pd.DataFrame(),
            "paired_predictions":
                pd.DataFrame(),
        }

    v2_for_merge = (
        v2_predictions[
            [
                "player_key",
                "market",
                "game_time",
                "forecast",
            ]
        ]
        .rename(
            columns={
                "forecast":
                    "v2_forecast"
            }
        )
    )

    paired = v1_predictions.merge(
        v2_for_merge,
        on=[
            "player_key",
            "market",
            "game_time",
        ],
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:

        return {
            "v1": None,
            "v2": None,
            "comparison":
                pd.DataFrame(),
            "paired_predictions":
                pd.DataFrame(),
        }

    paired[
        "v1_error"
    ] = (
        paired[
            "v1_forecast"
        ]
        - paired[
            "actual"
        ]
    )

    paired[
        "v2_error"
    ] = (
        paired[
            "v2_forecast"
        ]
        - paired[
            "actual"
        ]
    )

    paired[
        "v1_absolute_error"
    ] = np.abs(
        paired[
            "v1_error"
        ]
    )

    paired[
        "v2_absolute_error"
    ] = np.abs(
        paired[
            "v2_error"
        ]
    )

    paired[
        "v1_squared_error"
    ] = np.square(
        paired[
            "v1_error"
        ]
    )

    paired[
        "v2_squared_error"
    ] = np.square(
        paired[
            "v2_error"
        ]
    )

    v1_mae = float(
        paired[
            "v1_absolute_error"
        ].mean()
    )

    v2_mae = float(
        paired[
            "v2_absolute_error"
        ].mean()
    )

    v1_rmse = float(
        math.sqrt(
            paired[
                "v1_squared_error"
            ].mean()
        )
    )

    v2_rmse = float(
        math.sqrt(
            paired[
                "v2_squared_error"
            ].mean()
        )
    )

    v1_bias = float(
        paired[
            "v1_error"
        ].mean()
    )

    v2_bias = float(
        paired[
            "v2_error"
        ].mean()
    )

    comparison = pd.DataFrame(
        [
            {
                "model":
                    "V1 Mean Baseline",

                "prediction_count":
                    len(
                        paired
                    ),

                "mae":
                    round(
                        v1_mae,
                        4,
                    ),

                "rmse":
                    round(
                        v1_rmse,
                        4,
                    ),

                "bias":
                    round(
                        v1_bias,
                        4,
                    ),
            },
            {
                "model":
                    "V2A Recency Weighted",

                "prediction_count":
                    len(
                        paired
                    ),

                "mae":
                    round(
                        v2_mae,
                        4,
                    ),

                "rmse":
                    round(
                        v2_rmse,
                        4,
                    ),

                "bias":
                    round(
                        v2_bias,
                        4,
                    ),
            },
        ]
    )

    return {
        "v1": {
            "prediction_count":
                int(
                    len(
                        paired
                    )
                ),

            "mae":
                v1_mae,

            "rmse":
                v1_rmse,

            "bias":
                v1_bias,
        },

        "v2": {
            "prediction_count":
                int(
                    len(
                        paired
                    )
                ),

            "mae":
                v2_mae,

            "rmse":
                v2_rmse,

            "bias":
                v2_bias,
        },

        "mae_change":
            v2_mae
            - v1_mae,

        "rmse_change":
            v2_rmse
            - v1_rmse,

        "comparison":
            comparison,

        "paired_predictions":
            paired,
    }
