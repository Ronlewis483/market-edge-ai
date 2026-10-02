"""
MARKET EDGE AI V5
NFL PLAYER PROP PREDICTION PIPELINE

Purpose:
Generate and rank today's strongest NFL player-prop forecasts
from live sportsbook lines and completed historical player data.

This module does not execute wagers.
"""

import re

import numpy as np
import pandas as pd

from dual_agent.nfl_player_props import (
    MARKETS,
    canonical_name,
)


def _player_keys(name):
    """Return possible historical identifiers for a player."""

    full_key = canonical_name(name)

    parts = re.findall(
        r"[A-Za-z]+",
        str(name),
    )

    keys = {full_key}

    if len(parts) >= 2:
        keys.add(
            canonical_name(
                parts[0][0] + parts[-1]
            )
        )

    return keys


def _historical_player_games(
    history,
    player,
    market,
    kickoff,
    window=12,
):
    """Return leakage-safe historical games before kickoff."""

    if history is None or history.empty:
        return pd.DataFrame()

    player_keys = _player_keys(player)

    games = history.loc[
        history["player_key"].isin(player_keys)
        & (history["market"] == market)
    ].copy()

    if games.empty:
        return games

    games["game_time"] = pd.to_datetime(
        games["game_time"],
        utc=True,
        errors="coerce",
    )

    games["value"] = pd.to_numeric(
        games["value"],
        errors="coerce",
    )

    games = games.dropna(
        subset=["game_time", "value"]
    )

    games = games.loc[
        games["game_time"] < kickoff
    ]

    games = (
        games
        .sort_values("game_time")
        .drop_duplicates(
            subset=["game_time"],
            keep="last",
        )
        .tail(window)
    )

    return games


def _estimate_prop(
    values,
    line,
    market,
):
    """
    Produce a ranked prop forecast.

    prediction_score is a ranking score built from:
    - historical support
    - projection distance from sportsbook line
    - sample reliability
    - historical consistency

    It is NOT a calibrated future probability.
    """

    sample_size = len(values)

    if sample_size < 8:
        return None

    weights = np.linspace(
        1.0,
        2.0,
        sample_size,
    )

    projection = float(
        np.average(
            values,
            weights=weights,
        )
    )

    mean_value = float(
        np.mean(values)
    )

    std_value = float(
        np.std(values)
    )

    if market == "Anytime touchdown":

        over_hits = int(
            (values >= 1).sum()
        )

        under_hits = int(
            (values < 1).sum()
        )

        eligible = sample_size

    else:

        over_hits = int(
            (values > line).sum()
        )

        under_hits = int(
            (values < line).sum()
        )

        pushes = int(
            (values == line).sum()
        )

        eligible = (
            sample_size - pushes
        )

    if eligible <= 0:
        return None

    over_support = (
        (over_hits + 1)
        / (eligible + 2)
    )

    under_support = (
        (under_hits + 1)
        / (eligible + 2)
    )

    if market == "Anytime touchdown":

        if over_support >= under_support:
            model_pick = "YES"
            support = over_support
        else:
            model_pick = "NO"
            support = under_support

        line_edge = abs(
            projection - 0.5
        )

        normalized_edge = min(
            line_edge / 0.5,
            1.0,
        )

    else:

        if projection > line:
            model_pick = "OVER"
            support = over_support

        elif projection < line:
            model_pick = "UNDER"
            support = under_support

        else:
            model_pick = "PASS"
            support = max(
                over_support,
                under_support,
            )

        line_edge = abs(
            projection - line
        )

        denominator = max(
            abs(line),
            1.0,
        )

        normalized_edge = min(
            line_edge / denominator,
            1.0,
        )

    sample_reliability = min(
        sample_size / 12.0,
        1.0,
    )

    if abs(mean_value) > 0:

        coefficient_variation = (
            std_value
            / abs(mean_value)
        )

        consistency = (
            1.0
            / (
                1.0
                + coefficient_variation
            )
        )

    else:
        consistency = 0.0

    consistency = float(
        np.clip(
            consistency,
            0.0,
            1.0,
        )
    )

    prediction_score = (
        0.55 * support
        + 0.20 * normalized_edge
        + 0.15 * sample_reliability
        + 0.10 * consistency
    )

    prediction_score = float(
        np.clip(
            prediction_score,
            0.0,
            1.0,
        )
    )

    return {
        "model_pick":
            model_pick,
        "projection":
            projection,
        "historical_support":
            float(support),
        "over_support":
            float(over_support),
        "under_support":
            float(under_support),
        "sample_size":
            int(sample_size),
        "line_edge":
            float(line_edge),
        "normalized_edge":
            float(normalized_edge),
        "consistency":
            consistency,
        "prediction_score":
            prediction_score,
    }


def _select_consensus_props(
    prop_lines,
):
    """
    Reduce sportsbook duplicates to one representative
    line per game / player / market.

    Median sportsbook line is used as the consensus line.
    """

    required_columns = [
        "event_id",
        "game_time",
        "home_team",
        "away_team",
        "player",
        "market_key",
        "line",
    ]

    available = [
        column
        for column in required_columns
        if column in prop_lines.columns
    ]

    if len(available) != len(
        required_columns
    ):
        return pd.DataFrame()

    props = prop_lines[
        required_columns
    ].copy()

    props["line"] = pd.to_numeric(
        props["line"],
        errors="coerce",
    )

    props = props.dropna(
        subset=[
            "event_id",
            "game_time",
            "player",
            "market_key",
            "line",
        ]
    )

    if props.empty:
        return props

    group_columns = [
        "event_id",
        "game_time",
        "home_team",
        "away_team",
        "player",
        "market_key",
    ]

    consensus = (
        props
        .groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            line=(
                "line",
                "median",
            ),
            sportsbook_line_count=(
                "line",
                "count",
            ),
        )
    )

    return consensus


def run_nfl_player_prop_prediction_pipeline(
    prop_lines,
    player_history,
    window=12,
    max_results=20,
    horizon_days=1,
):
    """
    Generate and rank future player-prop forecasts within the requested horizon.
    """

    if prop_lines is None:
        return pd.DataFrame()

    if not isinstance(
        prop_lines,
        pd.DataFrame,
    ):
        prop_lines = pd.DataFrame(
            prop_lines
        )

    if prop_lines.empty:
        return pd.DataFrame()

    if player_history is None:
        return pd.DataFrame()

    if not isinstance(
        player_history,
        pd.DataFrame,
    ):
        player_history = pd.DataFrame(
            player_history
        )

    if player_history.empty:
        return pd.DataFrame()

    market_lookup = {
        value: name
        for name, value in MARKETS.items()
    }

    current_time = pd.Timestamp.now(
        tz="UTC"
    )

    if not isinstance(horizon_days,int) or not 1<=horizon_days<=7:
        raise ValueError('horizon_days must be an integer from 1 to 7.')
    forecast_end=current_time+pd.Timedelta(days=horizon_days)

    consensus_props = (
        _select_consensus_props(
            prop_lines
        )
    )

    if consensus_props.empty:
        return pd.DataFrame()

    rows = []

    for _, prop in (
        consensus_props.iterrows()
    ):

        kickoff = pd.to_datetime(
            prop["game_time"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(kickoff):
            continue

        # Never forecast a game
        # that has already started.
        if kickoff <= current_time:
            continue

        if horizon_days==1:
            # Preserve legacy today-only behavior for callers that omit the argument.
            if kickoff.tz_convert('America/Chicago').date()!=current_time.tz_convert('America/Chicago').date():
                continue
        elif kickoff>forecast_end:
            continue

        market = market_lookup.get(
            prop["market_key"]
        )

        if not market:
            continue

        line = pd.to_numeric(
            prop["line"],
            errors="coerce",
        )

        if pd.isna(line):
            continue

        games = (
            _historical_player_games(
                history=player_history,
                player=prop["player"],
                market=market,
                kickoff=kickoff,
                window=window,
            )
        )

        if games.empty:
            continue

        values = (
            games["value"]
            .to_numpy(
                dtype=float
            )
        )

        estimate = _estimate_prop(
            values=values,
            line=float(line),
            market=market,
        )

        if estimate is None:
            continue

        # Avoid displaying extremely weak
        # or essentially coin-flip forecasts.
        if (
            estimate[
                "historical_support"
            ]
            < 0.55
        ):
            continue

        if (
            estimate[
                "model_pick"
            ]
            == "PASS"
        ):
            continue

        score = estimate[
            "prediction_score"
        ]

        if score >= 0.70:
            confidence_group = (
                "HIGH CONFIDENCE"
            )

        elif score >= 0.60:
            confidence_group = (
                "MODERATE"
            )

        else:
            confidence_group = (
                "CLOSE / PASS"
            )

        rows.append(
            {
                "event_id":
                    prop["event_id"],
                "game_time":
                    kickoff,
                "away_team":
                    prop["away_team"],
                "home_team":
                    prop["home_team"],
                "player":
                    prop["player"],
                "market":
                    market,
                "line":
                    float(line),
                "model_pick":
                    estimate[
                        "model_pick"
                    ],
                "projected_value":
                    estimate[
                        "projection"
                    ],
                "historical_support":
                    estimate[
                        "historical_support"
                    ],
                "over_support":
                    estimate[
                        "over_support"
                    ],
                "under_support":
                    estimate[
                        "under_support"
                    ],
                "sample_size":
                    estimate[
                        "sample_size"
                    ],
                "line_edge":
                    estimate[
                        "line_edge"
                    ],
                "consistency":
                    estimate[
                        "consistency"
                    ],
                "prediction_score":
                    score,
                "sportsbook_line_count":
                    int(
                        prop[
                            "sportsbook_line_count"
                        ]
                    ),
                "confidence_group":
                    confidence_group,
            }
        )

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        return results

        # ----------------------------------------------------------
    # RANK ALL QUALIFIED PROPS
    # ----------------------------------------------------------

    results = (
        results
        .sort_values(
            [
                "prediction_score",
                "historical_support",
                "sample_size",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "event_id",
                "player",
                "market",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # IDENTIFY EACH PLAYER'S STRONGEST PROP
    #
    # A player can only occupy ONE Top-10 position.
    # Their strongest individual prop determines their rank.
    # ----------------------------------------------------------

    best_prop_per_player = (
        results
        .sort_values(
            [
                "prediction_score",
                "historical_support",
                "sample_size",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "event_id",
                "player",
            ],
            keep="first",
        )
        .head(10)
        .copy()
    )

    if best_prop_per_player.empty:
        return pd.DataFrame()

    # Assign visible player ranking.
    best_prop_per_player[
        "player_rank"
    ] = range(
        1,
        len(best_prop_per_player) + 1,
    )

    # ----------------------------------------------------------
    # RETAIN OTHER QUALIFIED PROPS FOR EACH TOP PLAYER
    #
    # These become the dropdown props in the UI.
    # Maximum 4 props total per player.
    # ----------------------------------------------------------

    selected_players = (
        best_prop_per_player[
            [
                "event_id",
                "player",
                "player_rank",
            ]
        ]
        .copy()
    )

    top_player_props = results.merge(
        selected_players,
        on=[
            "event_id",
            "player",
        ],
        how="inner",
    )

    top_player_props = (
        top_player_props
        .sort_values(
            [
                "player_rank",
                "prediction_score",
                "historical_support",
                "sample_size",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .groupby(
            [
                "event_id",
                "player",
            ],
            group_keys=False,
        )
        .head(4)
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # MARK EACH PLAYER'S HEADLINE PROP
    # ----------------------------------------------------------

    top_player_props[
        "is_best_prop"
    ] = (
        top_player_props
        .groupby(
            [
                "event_id",
                "player",
            ]
        )
        .cumcount()
        == 0
    )

    # ----------------------------------------------------------
    # FINAL ORDER
    #
    # Player #1 first, then all of that player's props.
    # Player #2 next, etc.
    # ----------------------------------------------------------

    top_player_props = (
        top_player_props
        .sort_values(
            [
                "player_rank",
                "is_best_prop",
                "prediction_score",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return top_player_props
