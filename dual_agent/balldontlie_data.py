import requests
import pandas as pd
import streamlit as st


BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"


def _get_api_key():
    """
    Read the BALLDONTLIE API key from Streamlit Secrets.
    """
    try:
        return st.secrets["BALLDONTLIE_API_KEY"]
    except Exception:
        raise ValueError(
            "BALLDONTLIE_API_KEY was not found in Streamlit Secrets."
        )


def _balldontlie_get(endpoint, params=None):
    """
    Make an authenticated request to BALLDONTLIE.
    """
    api_key = _get_api_key()

    headers = {
        "Authorization": api_key
    }

    response = requests.get(
        f"{BALLDONTLIE_BASE_URL}/{endpoint}",
        headers=headers,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"BALLDONTLIE error {response.status_code}: "
            f"{response.text}"
        )

    return response.json()


def get_historical_games(season=2023, per_page=100):
    """
    Retrieve one page of historical NBA games.

    This is intentionally small for the first connection test.
    We will add full pagination after confirming the API works
    from Streamlit Cloud.
    """

    payload = _balldontlie_get(
        "games",
        params={
            "seasons[]": season,
            "per_page": per_page,
        },
    )

    games = payload.get("data", [])

    if not games:
        raise ValueError(
            f"No games were returned for NBA season {season}."
        )

    rows = []

    for game in games:
        home_team = game.get("home_team") or {}
        visitor_team = game.get("visitor_team") or {}

        rows.append(
            {
                "game_id": game.get("id"),
                "game_date": game.get("date"),
                "season": game.get("season"),
                "status": game.get("status"),
                "home_team_id": home_team.get("id"),
                "home_team": home_team.get("abbreviation"),
                "away_team_id": visitor_team.get("id"),
                "away_team": visitor_team.get("abbreviation"),
                "home_points": game.get("home_team_score"),
                "away_points": game.get("visitor_team_score"),
            }
        )

    df = pd.DataFrame(rows)

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    df["home_points"] = pd.to_numeric(
        df["home_points"],
        errors="coerce",
    )

    df["away_points"] = pd.to_numeric(
        df["away_points"],
        errors="coerce",
    )

    completed = df.dropna(
        subset=["home_points", "away_points"]
    ).copy()

    completed["home_win"] = (
        completed["home_points"] >
        completed["away_points"]
    ).astype(int)

    return completed


def test_balldontlie_connection():
    """
    Test an older season to prove historical access works.
    """

    games = get_historical_games(
        season=2023,
        per_page=25,
    )

    return {
        "success": True,
        "season": 2023,
        "games_returned": len(games),
        "first_game": games["game_date"].min(),
        "last_game": games["game_date"].max(),
        "sample": games.head(5),
    }
