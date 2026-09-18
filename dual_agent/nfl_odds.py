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

    def get_best_nfl_moneylines(odds_rows):
    """
    Reduce sportsbook-level NFL odds into one row per game.

    For each team, select the most favorable available American
    moneyline and remember which sportsbook offered it.
    """

    if not odds_rows:
        return []

    games = {}

    for row in odds_rows:
        game_id = row["game_id"]

        if game_id not in games:
            games[game_id] = {
                "game_id": game_id,
                "commence_time": row["commence_time"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "best_home_moneyline": None,
                "best_home_sportsbook": None,
                "best_away_moneyline": None,
                "best_away_sportsbook": None,
            }

        game = games[game_id]

        home_line = row["home_moneyline"]
        away_line = row["away_moneyline"]

        # With American odds, the numerically larger line is
        # always better for the bettor:
        # +130 is better than +120
        # -140 is better than -150
        if (
            game["best_home_moneyline"] is None
            or home_line > game["best_home_moneyline"]
        ):
            game["best_home_moneyline"] = home_line
            game["best_home_sportsbook"] = row["sportsbook"]

        if (
            game["best_away_moneyline"] is None
            or away_line > game["best_away_moneyline"]
        ):
            game["best_away_moneyline"] = away_line
            game["best_away_sportsbook"] = row["sportsbook"]

    return list(games.values())

    return rows
