import requests
import streamlit as st


THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"


def _get_odds_api_key():
    try:
        return st.secrets["ODDS_API_KEY"]
    except Exception:
        raise ValueError(
            "ODDS_API_KEY was not found in Streamlit Secrets."
        )


def get_live_nfl_moneylines():
    """
    Get current NFL moneyline odds.

    Returns one row per sportsbook/game with:
    - home_team
    - away_team
    - commence_time
    - sportsbook
    - home_moneyline
    - away_moneyline
    """

    api_key = _get_odds_api_key()

    url = f"{THE_ODDS_API_BASE_URL}/sports/americanfootball_nfl/odds"

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"NFL odds API error {response.status_code}: "
            f"{response.text}"
        )

    games = response.json()

    rows = []

    for game in games:

        home_team = game.get("home_team")
        away_team = game.get("away_team")

        for bookmaker in game.get("bookmakers", []):

            sportsbook = bookmaker.get("title")

            for market in bookmaker.get("markets", []):

                if market.get("key") != "h2h":
                    continue

                home_moneyline = None
                away_moneyline = None

                for outcome in market.get("outcomes", []):

                    if outcome.get("name") == home_team:
                        home_moneyline = outcome.get("price")

                    elif outcome.get("name") == away_team:
                        away_moneyline = outcome.get("price")

                if (
                    home_moneyline is not None
                    and away_moneyline is not None
                ):
                    rows.append(
                        {
                            "game_id": game.get("id"),
                            "commence_time": game.get(
                                "commence_time"
                            ),
                            "home_team": home_team,
                            "away_team": away_team,
                            "sportsbook": sportsbook,
                            "home_moneyline": home_moneyline,
                            "away_moneyline": away_moneyline,
                        }
                    )

    return rows
