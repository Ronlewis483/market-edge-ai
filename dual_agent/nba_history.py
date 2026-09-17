import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


def get_nba_team_game_logs(season="2024-25"):
    """
    Download NBA regular-season team game logs from NBA.com.

    Each NBA game appears twice in the raw data:
    once from the home team's perspective and once from the away team's.
    """

    game_log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        timeout=60,
    )

    df = game_log.get_data_frames()[0]

    if df is None or df.empty:
        raise ValueError(
            f"No NBA historical data returned for season {season}."
        )

    return df


def get_nba_historical_games(season="2024-25"):
    """
    Convert NBA.com team game logs into one row per NBA game.
    """

    df = get_nba_team_game_logs(season)

    required_columns = [
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "WL",
        "PTS",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"NBA data is missing required columns: {missing}"
        )

    df = df[required_columns].copy()

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"],
        errors="coerce",
    )

    df["PTS"] = pd.to_numeric(
        df["PTS"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["GAME_ID", "GAME_DATE", "MATCHUP", "PTS"]
    )

    home = df[
        df["MATCHUP"].str.contains("vs.", regex=False, na=False)
    ].copy()

    away = df[
        df["MATCHUP"].str.contains("@", regex=False, na=False)
    ].copy()

    home = home.rename(
        columns={
            "TEAM_ID": "home_team_id",
            "TEAM_ABBREVIATION": "home_team",
            "PTS": "home_points",
            "WL": "home_result",
        }
    )

    away = away.rename(
        columns={
            "TEAM_ID": "away_team_id",
            "TEAM_ABBREVIATION": "away_team",
            "PTS": "away_points",
            "WL": "away_result",
        }
    )

    games = home[
        [
            "GAME_ID",
            "GAME_DATE",
            "home_team_id",
            "home_team",
            "home_points",
            "home_result",
        ]
    ].merge(
        away[
            [
                "GAME_ID",
                "away_team_id",
                "away_team",
                "away_points",
                "away_result",
            ]
        ],
        on="GAME_ID",
        how="inner",
    )

    games = games.rename(
        columns={
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
        }
    )

    games["home_win"] = (
        games["home_points"] > games["away_points"]
    ).astype(int)

    games["season"] = season

    games = games.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    return games


def test_nba_history(season="2024-25"):
    """
    Small diagnostic used to verify NBA.com historical access.
    """

    games = get_nba_historical_games(season)

    return {
        "season": season,
        "games": len(games),
        "first_game": (
            games["game_date"].min()
            if len(games)
            else None
        ),
        "last_game": (
            games["game_date"].max()
            if len(games)
            else None
        ),
        "columns": list(games.columns),
        "sample": games.head(5),
    }

