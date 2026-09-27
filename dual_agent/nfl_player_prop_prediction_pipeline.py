"""
MARKET EDGE AI V5
NFL PLAYER PROP PREDICTION PIPELINE

Purpose:
Build automated player-prop forecasts from live sportsbook
lines and completed historical NFL player statistics.

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
    Produce a preliminary prop forecast.

    The displayed strength is historical support,
    not a calibrated future probability.
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

        eligible = sample_size - pushes

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

    elif projection > line:

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

    return {
        "model_pick": model_pick,
        "projection": projection,
        "historical_support": float(
            support
        ),
        "over_support": float(
            over_support
        ),
        "under_support": float(
            under_support
        ),
        "sample_size": int(
            sample_size
        ),
    }


def run_nfl_player_prop_prediction_pipeline(
    prop_lines,
    player_history,
    window=12,
):
    """
    Generate forecasts for a collection of live NFL props.
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

    current_date_ct = (
        current_time
        .tz_convert("America/Chicago")
        .date()
    )

    rows = []

    # One forecast per player / market / line / game.
    unique_props = (
        prop_lines[
            [
                "event_id",
                "game_time",
                "home_team",
                "away_team",
                "player",
                "market_key",
                "line",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    for _, prop in unique_props.iterrows():

        kickoff = pd.to_datetime(
            prop["game_time"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(kickoff):
            continue

        # Do not forecast games already underway.
        if kickoff <= current_time:
            continue

        # TODAY ONLY.
        kickoff_date_ct = (
            kickoff
            .tz_convert("America/Chicago")
            .date()
        )

        if kickoff_date_ct != current_date_ct:
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

        games = _historical_player_games(
            history=player_history,
            player=prop["player"],
            market=market,
            kickoff=kickoff,
            window=window,
        )

        if games.empty:
            continue

        values = games[
            "value"
        ].to_numpy(dtype=float)

        estimate = _estimate_prop(
            values=values,
            line=float(line),
            market=market,
        )

        if estimate is None:
            continue

        support = estimate[
            "historical_support"
        ]

        if support >= 0.70:
            confidence_group = (
                "HIGH CONFIDENCE"
            )

        elif support >= 0.58:
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
                    estimate["model_pick"],
                "projected_value":
                    estimate["projection"],
                "historical_support":
                    support,
                "over_support":
                    estimate["over_support"],
                "under_support":
                    estimate["under_support"],
                "sample_size":
                    estimate["sample_size"],
                "confidence_group":
                    confidence_group,
            }
        )

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        return results

    return (
        results
        .sort_values(
            [
                "historical_support",
                "game_time",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )
