import requests
import pandas as pd
import streamlit as st


SPORTRADAR_BASE_URL = (
    "https://api.sportradar.com/americanfootball/trial/v2/en"
)


def _get_sportradar_api_key():
    """
    Read the Sportradar API key from Streamlit Secrets.
    """
    try:
        return st.secrets["SPORTRADAR_API_KEY"]
    except Exception:
        raise ValueError(
            "SPORTRADAR_API_KEY was not found in Streamlit Secrets."
        )


def _sportradar_get(endpoint, params=None):
    """
    Make an authenticated request to Sportradar.
    """

    api_key = _get_sportradar_api_key()

    url = f"{SPORTRADAR_BASE_URL}/{endpoint}"

    headers = {
        "x-api-key": api_key,
        "Accept": "application/json",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Sportradar error {response.status_code}: "
            f"{response.text}"
        )

    return response.json()


def test_sportradar_connection():
    """
    Test the Global American Football API
    and return available competitions.
    """

    data = _sportradar_get("competitions.json")

    competitions = data.get("competitions", [])

    rows = []

    for competition in competitions:
        rows.append(
            {
                "id": competition.get("id"),
                "name": competition.get("name"),
                "gender": competition.get("gender"),
            }
        )

    df = pd.DataFrame(rows)

    return {
        "success": True,
        "competition_count": len(df),
        "competitions": df,
    }

NFL_COMPETITION_ID = "sr:competition:31"
NCAA_COMPETITION_ID = "sr:competition:850"


def get_nfl_seasons():
    """
    Return the NFL seasons available through Sportradar.
    """

    endpoint = f"competitions/{NFL_COMPETITION_ID}/seasons.json"

    data = _sportradar_get(endpoint)

    seasons = data.get("seasons", [])

    rows = []

    for season in seasons:
        rows.append(
            {
                "id": season.get("id"),
                "name": season.get("name"),
                "start_date": season.get("start_date"),
                "end_date": season.get("end_date"),
                "year": season.get("year"),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty and "start_date" in df.columns:
        df = df.sort_values(
            "start_date",
            ascending=False,
        ).reset_index(drop=True)

    return {
        "success": True,
        "season_count": len(df),
        "seasons": df,
    }

def get_nfl_season_games(season_id):
    """
    Retrieve NFL games for a specific Sportradar season.
    """

    endpoint = f"seasons/{season_id}/summaries.json"

    data = _sportradar_get(endpoint)

    summaries = data.get("summaries", [])

    rows = []

    for summary in summaries:
        sport_event = summary.get("sport_event", {})
        status = summary.get("sport_event_status", {})

        competitors = sport_event.get("competitors", [])

        home_team = None
        home_team_id = None
        away_team = None
        away_team_id = None

        for competitor in competitors:
            qualifier = competitor.get("qualifier")

            if qualifier == "home":
                home_team = competitor.get("name")
                home_team_id = competitor.get("id")

            elif qualifier == "away":
                away_team = competitor.get("name")
                away_team_id = competitor.get("id")

        rows.append(
            {
                "game_id": sport_event.get("id"),
                "start_time": sport_event.get("start_time"),
                "home_team": home_team,
                "home_team_id": home_team_id,
                "away_team": away_team,
                "away_team_id": away_team_id,
                "status": status.get("status"),
                "home_score": status.get("home_score"),
                "away_score": status.get("away_score"),
                "winner_id": status.get("winner_id"),
            }
        )

    games_df = pd.DataFrame(rows)

    if not games_df.empty and "start_time" in games_df.columns:
        games_df["start_time"] = pd.to_datetime(
            games_df["start_time"],
            errors="coerce",
        )

        games_df = games_df.sort_values(
            "start_time"
        ).reset_index(drop=True)

    return {
        "success": True,
        "season_id": season_id,
        "game_count": len(games_df),
        "games": games_df,
    }
