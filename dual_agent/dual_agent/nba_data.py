import requests
import pandas as pd
import streamlit as st


# ============================================================
# MARKET EDGE AI - NBA DATA LAYER
#
# Historical data: SportsDataIO
# Current odds:    The Odds API
# ============================================================


SPORTSDATA_BASE_URL = "https://api.sportsdata.io/v3/nba"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

NBA_SPORT_KEY = "basketball_nba"


# ------------------------------------------------------------
# API KEY HELPERS
# ------------------------------------------------------------

def get_sportsdata_key():
    """Return the SportsDataIO API key from Streamlit Secrets."""

    try:
        return st.secrets["SPORTSDATA_API_KEY"]
    except Exception:
        return None


def get_odds_api_key():
    """Return The Odds API key from Streamlit Secrets."""

    try:
        return st.secrets["ODDS_API_KEY"]
    except Exception:
        return None


def provider_status():
    """
    Check which NBA data providers are configured.

    Does NOT expose API keys.
    """

    return {
        "sportsdata_historical": bool(get_sportsdata_key()),
        "odds_api_current": bool(get_odds_api_key()),
    }


# ------------------------------------------------------------
# SPORTSDATAIO REQUEST
# ------------------------------------------------------------

def _sportsdata_get(path, params=None):
    key = get_sportsdata_key()

    if not key:
        raise RuntimeError(
            "SPORTSDATA_API_KEY is missing from Streamlit Secrets."
        )

    url = f"{SPORTSDATA_BASE_URL}/{path.lstrip('/')}"

    headers = {
        "Ocp-Apim-Subscription-Key": key
    }

    response = requests.get(
        url,
        headers=headers,
        params=params or {},
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            f"SportsDataIO error {response.status_code}: "
            f"{response.text[:300]}"
        )

    return response.json()


# ------------------------------------------------------------
# HISTORICAL NBA DATA
# ------------------------------------------------------------

def get_historical_games(season):
    """
    Get historical NBA games for a season.

    Example:
        get_historical_games(2025)
    """

    data = _sportsdata_get(
        f"scores/json/Games/{season}"
    )

    return pd.DataFrame(data)


def get_team_season_stats(season):
    """
    Get NBA team season statistics when available
    on the configured SportsDataIO plan.
    """

    data = _sportsdata_get(
        f"stats/json/TeamSeasonStats/{season}"
    )

    return pd.DataFrame(data)


def get_player_season_stats(season):
    """
    Get NBA player season statistics when available
    on the configured SportsDataIO plan.
    """

    data = _sportsdata_get(
        f"stats/json/PlayerSeasonStats/{season}"
    )

    return pd.DataFrame(data)


# ------------------------------------------------------------
# THE ODDS API REQUEST
# ------------------------------------------------------------

def _odds_get(path, params=None):
    key = get_odds_api_key()

    if not key:
        raise RuntimeError(
            "ODDS_API_KEY is missing from Streamlit Secrets."
        )

    url = f"{ODDS_BASE_URL}/{path.lstrip('/')}"

    query = dict(params or {})
    query["apiKey"] = key

    response = requests.get(
        url,
        params=query,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            f"The Odds API error {response.status_code}: "
            f"{response.text[:300]}"
        )

    return response


# ------------------------------------------------------------
# CURRENT NBA EVENTS
# ------------------------------------------------------------

def get_current_nba_events():
    """
    Get current/upcoming NBA events.

    This endpoint is useful for obtaining event IDs,
    which can later be used for player-prop requests.
    """

    response = _odds_get(
        f"sports/{NBA_SPORT_KEY}/events"
    )

    return response.json()


# ------------------------------------------------------------
# CURRENT NBA ODDS
# ------------------------------------------------------------

def get_current_nba_odds(
    markets="h2h,spreads,totals",
    regions="us",
):
    """
    Get current NBA featured markets.

    markets:
        h2h     = moneyline
        spreads = point spread
        totals  = game total
    """

    response = _odds_get(
        f"sports/{NBA_SPORT_KEY}/odds",
        params={
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )

    return response.json()


# ------------------------------------------------------------
# PLAYER PROPS
# ------------------------------------------------------------

def get_nba_player_props(
    event_id,
    markets="player_points,player_rebounds,player_assists",
    regions="us",
):
    """
    Get current player props for ONE NBA game.

    Player props use the event-specific odds endpoint.
    """

    response = _odds_get(
        f"sports/{NBA_SPORT_KEY}/events/{event_id}/odds",
        params={
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )

    return response.json()


# ------------------------------------------------------------
# AMERICAN ODDS -> IMPLIED PROBABILITY
# ------------------------------------------------------------

def american_to_probability(odds):
    """
    Convert American odds to raw implied probability.

    Examples:
        -150 -> 60%
        +150 -> 40%
    """

    odds = float(odds)

    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)

    return 100.0 / (odds + 100.0)


# ------------------------------------------------------------
# REMOVE SPORTSBOOK VIG
# ------------------------------------------------------------

def no_vig_two_way(prob_a, prob_b):
    """
    Normalize two sportsbook implied probabilities
    to estimate no-vig market probabilities.
    """

    total = prob_a + prob_b

    if total <= 0:
        return None, None

    return (
        prob_a / total,
        prob_b / total,
    )


# ------------------------------------------------------------
# FLATTEN CURRENT MONEYLINES
# ------------------------------------------------------------

def current_moneyline_dataframe():
    """
    Convert current NBA moneylines into a simple DataFrame.

    One row = one bookmaker's price for one team.
    """

    games = get_current_nba_odds(
        markets="h2h"
    )

    rows = []

    for game in games:

        event_id = game.get("id")
        commence_time = game.get("commence_time")
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        for bookmaker in game.get("bookmakers", []):

            book_name = bookmaker.get("title")
            book_key = bookmaker.get("key")

            for market in bookmaker.get("markets", []):

                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):

                    price = outcome.get("price")

                    if price is None:
                        continue

                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": book_name,
                        "bookmaker_key": book_key,
                        "team": outcome.get("name"),
                        "american_odds": price,
                        "implied_probability":
                            american_to_probability(price),
                    })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# CONNECTION TEST
# ------------------------------------------------------------

def test_nba_connections():
    """
    Safely test both configured providers.

    Returns status information without exposing API keys.
    """

    results = {
        "SportsDataIO": {
            "configured": bool(get_sportsdata_key()),
            "working": False,
            "message": "",
        },
        "The Odds API": {
            "configured": bool(get_odds_api_key()),
            "working": False,
            "message": "",
        },
    }

    # Historical provider
    if results["SportsDataIO"]["configured"]:
        try:
            # Lightweight request. If a particular account
            # restricts this feed, the error is returned cleanly.
            _sportsdata_get("scores/json/AreAnyGamesInProgress")

            results["SportsDataIO"]["working"] = True
            results["SportsDataIO"]["message"] = (
                "Historical NBA provider connected."
            )

        except Exception as e:
            results["SportsDataIO"]["message"] = str(e)

    else:
        results["SportsDataIO"]["message"] = (
            "SPORTSDATA_API_KEY not configured."
        )

    # Current odds provider
    if results["The Odds API"]["configured"]:
        try:
            response = _odds_get("sports")

            results["The Odds API"]["working"] = True

            remaining = response.headers.get(
                "x-requests-remaining",
                "unknown",
            )

            results["The Odds API"]["message"] = (
                f"Current odds provider connected. "
                f"Requests remaining: {remaining}"
            )

        except Exception as e:
            results["The Odds API"]["message"] = str(e)

    else:
        results["The Odds API"]["message"] = (
            "ODDS_API_KEY not configured."
        )

    return results
