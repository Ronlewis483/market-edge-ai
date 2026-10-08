"""
MARKET EDGE AI V5
NFL PLAYER PROP PREDICTION PIPELINE

Purpose:
Generate and rank today's strongest NFL player-prop forecasts
from live sportsbook lines and completed historical player data.

This module does not execute wagers.
"""

import re

NFL_PROP_VALUE_VERSION = 1

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
    side=None,
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

    if side is not None:
        valid_sides = {'YES','NO'} if market == 'Anytime touchdown' else {'OVER','UNDER'}
        if side not in valid_sides:
            return None
        model_pick = side
        support = over_support if side in {'YES','OVER'} else under_support

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


def _select_consensus_props(prop_lines):
    """Select a real, fresh offered line; never synthesize median handicaps."""
    keys = ['event_id', 'game_time', 'home_team', 'away_team', 'player', 'market_key']
    required = keys + ['line', 'side', 'bookmaker_key', 'american_odds', 'last_update']
    if any(c not in prop_lines for c in required):
        return pd.DataFrame()
    props = prop_lines[required].copy()
    props['line'] = pd.to_numeric(props['line'], errors='coerce')
    props['american_odds'] = pd.to_numeric(props['american_odds'], errors='coerce')
    props['side'] = props['side'].astype(str).str.upper()
    updated = pd.to_datetime(props['last_update'], utc=True, errors='coerce')
    age = pd.Timestamp.now(tz='UTC') - updated
    props = props.loc[age.between(pd.Timedelta(0), pd.Timedelta(minutes=30))
        & props['side'].isin(['OVER','UNDER','YES','NO'])
        & np.isfinite(props['line']) & np.isfinite(props['american_odds'])
        & (props['american_odds'].abs() >= 100)].dropna(subset=keys+['bookmaker_key'])
    props = props.loc[props.bookmaker_key.astype(str).str.strip().ne('')]
    rows = []
    for identity, candidates in props.groupby(keys, dropna=False):
        lines = []
        for line, offers in candidates.groupby('line'):
            sides = {}
            for side, quotes in offers.groupby('side'):
                quotes = quotes.drop_duplicates('bookmaker_key', keep='last')
                if len(quotes) >= 1:
                    sides[side] = quotes[['bookmaker_key','american_odds']].to_dict('records')
            if sides:
                lines.append((max(len(q) for q in sides.values()), float(line), sides))
        if not lines:
            continue
        # Score each real offered line; ranking later retains one per market.
        for _, line, sides in sorted(lines, key=lambda v:(-v[0], v[1])):
            rows.append(dict(zip(keys, identity), line=line,
                sportsbook_line_count=max(len(q) for q in sides.values()), offered_sides=sides))
    return pd.DataFrame(rows)


def run_nfl_player_prop_prediction_pipeline(
    prop_lines,
    player_history,
    window=12,
    max_results=20,
    horizon_days=1,
    retain_all_qualified=False,
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
        empty = pd.DataFrame()
        empty.attrs['reason'] = 'Offered lines were returned, but none had a usable side, price and quote updated within 30 minutes.'
        return empty

    rows = []
    exclusions = {'insufficient_history': 0, 'unoffered_direction': 0, 'below_65_percent': 0, 'nonpositive_value': 0}

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
            exclusions['insufficient_history'] += 1
            continue

        values = (
            games["value"]
            .to_numpy(
                dtype=float
            )
        )

        valid_sides = {'YES','NO'} if market == 'Anytime touchdown' else {'OVER','UNDER'}
        for offered_side in prop['offered_sides']:
            if offered_side not in valid_sides:
                continue
            estimate = _estimate_prop(
                values=values,
                line=float(line),
                market=market,
                side=offered_side,
            )

            if estimate is None:
                exclusions['insufficient_history'] += 1
                continue

            # A model direction is actionable only if books actually sell that bet.
            offered = prop['offered_sides'].get(estimate['model_pick'], [])
            if not offered:
                exclusions['unoffered_direction'] += 1
                continue

            # Avoid displaying extremely weak
            # or essentially coin-flip forecasts.
            if (
                estimate[
                    "historical_support"
                ]
                < 0.65
            ):
                exclusions['below_65_percent'] += 1
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
            def decimal_price(offer):
                price = float(offer['american_odds'])
                return 1 + price/100 if price > 0 else 1 + 100/abs(price)
            best_offer = max(offered, key=decimal_price)
            best_decimal = decimal_price(best_offer)
            push_chance = 0.0 if market == 'Anytime touchdown' else float(np.mean(values == float(line)))
            expected_return = (1-push_chance) * (estimate['historical_support'] * best_decimal - 1)
            # A listed bet must offer positive estimated value at its best actual price.
            if not np.isfinite(expected_return) or expected_return <= 0:
                exclusions['nonpositive_value'] += 1
                continue



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
                    "sportsbook_line_count": len(offered),
                    "sportsbook_offers": offered,
                    "best_sportsbook": best_offer['bookmaker_key'],
                    "best_american_odds": best_offer['american_odds'],
                    "best_decimal_odds": best_decimal,
                    "estimated_cover_chance": estimate['historical_support'],
                    "estimated_return_per_unit": expected_return,
                    "estimated_push_chance": push_chance,
                    "break_even_cover_chance": 1/best_decimal,
                    "availability_checked_at": current_time.isoformat(),
                    "confidence_group":
                        confidence_group,
                }
            )

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        results.attrs['exclusions'] = exclusions
        return results

        # ----------------------------------------------------------
    # RANK ALL QUALIFIED PROPS
    # ----------------------------------------------------------

    results = (
        results
        .sort_values(
            [
                "historical_support",
                "estimated_return_per_unit",
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

    if retain_all_qualified:
        # Preserve every qualifying market for per-game UI filters before display caps.
        all_props = results.head(max_results).copy().reset_index(drop=True)
        all_props['prop_rank'] = range(1,len(all_props)+1)
        all_props['is_best_prop'] = all_props.groupby(['event_id','player']).cumcount()==0
        identities = list(zip(all_props.event_id,all_props.player))
        ranks = {identity:i+1 for i,identity in enumerate(dict.fromkeys(identities))}
        all_props['player_rank'] = [ranks[identity] for identity in identities]
        all_props['selection_reason'] = 'Minimum 65% estimated cover chance and positive estimated value at an actual listed price; ranked by cover chance, then value.'
        return all_props

    best_prop_per_player = (
        results
        .sort_values(
            [
                "historical_support",
                "estimated_return_per_unit",
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
                "historical_support",
                "estimated_return_per_unit",
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
    # FINAL ORDER: rank cover chance across all bet types, with value breaking ties.
    # Player grouping must not put a weaker yards bet above a stronger NO TD bet.
    top_player_props = top_player_props.sort_values(
        ['historical_support', 'estimated_return_per_unit', 'sample_size'],
        ascending=[False, False, False], kind='stable',
    ).reset_index(drop=True)
    top_player_props['prop_rank'] = range(1, len(top_player_props) + 1)
    top_player_props['selection_reason'] = (
        'Ranked by estimated cover chance, then listed-odds value;  '
        'minimum 65%, positive estimated value, exact recommended side listed at a sportsbook.'
    )

    return top_player_props
