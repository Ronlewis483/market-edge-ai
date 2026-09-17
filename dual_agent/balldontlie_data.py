import time

import pandas as pd
import requests
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


def _games_to_dataframe(games):
    """
    Convert BALLDONTLIE game records into our standard format.
    """
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

    if df.empty:
        return df

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

    df = df.dropna(
        subset=[
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
        ]
    ).copy()

    df["home_win"] = (
        df["home_points"] > df["away_points"]
    ).astype(int)

    df = df.drop_duplicates(
        subset=["game_id"]
    )

    df = df.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    return df


def get_historical_games(
    season=2023,
    per_page=100,
    fetch_all=True,
):
    """
    Retrieve historical NBA games for one season.

    When fetch_all=True, pagination continues until the
    entire available season has been downloaded.

    The pause between requests protects the free-tier
    BALLDONTLIE rate limit.
    """

    all_games = []
    cursor = None
    request_count = 0

    while True:
        params = {
            "seasons[]": season,
            "per_page": per_page,
        }

        if cursor is not None:
            params["cursor"] = cursor

        payload = _balldontlie_get(
            "games",
            params=params,
        )

        request_count += 1

        batch = payload.get("data", [])

        if not batch:
            break

        all_games.extend(batch)

        if not fetch_all:
            break

        meta = payload.get("meta") or {}

        next_cursor = meta.get("next_cursor")

        if next_cursor is None:
            break

        cursor = next_cursor

        # Free tier allows only a small number of requests
        # per minute, so wait before requesting the next page.
        time.sleep(13)

    if not all_games:
        raise ValueError(
            f"No games were returned for NBA season {season}."
        )

    games_df = _games_to_dataframe(all_games)

    if games_df.empty:
        raise ValueError(
            f"No completed games were found for NBA season {season}."
        )

    return games_df, request_count


def test_balldontlie_connection():
    """
    Quick connection test.

    Only retrieves a small page so testing does not require
    downloading an entire season.
    """

    games, request_count = get_historical_games(
        season=2023,
        per_page=25,
        fetch_all=False,
    )

    return {
        "success": True,
        "season": 2023,
        "games_returned": len(games),
        "requests_used": request_count,
        "first_game": games["game_date"].min(),
        "last_game": games["game_date"].max(),
        "sample": games.head(5),
    }


def test_full_historical_season(season=2023):
    """
    Download one complete NBA season to verify pagination.
    """

    games, request_count = get_historical_games(
        season=season,
        per_page=100,
        fetch_all=True,
    )

    return {
        "success": True,
        "season": season,
        "games_returned": len(games),
        "requests_used": request_count,
        "first_game": games["game_date"].min(),
        "last_game": games["game_date"].max(),
        "home_win_rate": float(games["home_win"].mean()),
        "sample": games.head(5),
        "games": games,
    }
