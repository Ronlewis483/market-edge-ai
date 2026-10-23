import time
import json
import os
import threading
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_HISTORY_LOCK = threading.Lock()


def _setting(name, default):
    value = os.environ.get(name)
    if value is None:
        try:
            value = st.secrets.get(name, default)
        except Exception:
            value = default
    return value


def _pace_request():
    """One request budget for history and NBA prop-stat calls in this process."""
    global _LAST_REQUEST_AT
    try:
        rate = float(_setting('BALLDONTLIE_REQUESTS_PER_MINUTE', 5))
    except (TypeError, ValueError):
        rate = 5.
    if not 0 < rate <= 600:
        rate = 5.
    interval = 60. / rate + .05
    with _REQUEST_LOCK:
        delay = interval - (time.monotonic() - _LAST_REQUEST_AT)
        if delay > 0:
            time.sleep(delay)
        _LAST_REQUEST_AT = time.monotonic()


def _write_history(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, allow_nan=False))
    temporary.replace(path)


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


def _balldontlie_get(endpoint, params=None, api_key=None):
    """
    Make an authenticated request to BALLDONTLIE.
    Automatically retry if the API rate-limits us.
    """
    api_key = api_key or _get_api_key()

    headers = {
        "Authorization": api_key
    }

    max_retries = 5

    for attempt in range(max_retries):
        _pace_request()
        response = requests.get(
            endpoint if endpoint.startswith('https://api.balldontlie.io/') else f"{BALLDONTLIE_BASE_URL}/{endpoint}",
            headers=headers,
            params=params,
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")

            if retry_after:
                try:
                    wait_seconds = float(retry_after)
                except ValueError:
                    wait_seconds = 10
            else:
                wait_seconds = 10 * (attempt + 1)

            time.sleep(wait_seconds)
            continue

        raise RuntimeError(
            f"BALLDONTLIE error {response.status_code}: "
            f"{response.text}"
        )

    raise RuntimeError(
        "BALLDONTLIE rate limit continued after "
        f"{max_retries} retry attempts."
    )


def _games_to_dataframe(games):
    """
    Convert BALLDONTLIE game records into our standard format.
    """
    rows = []

    for game in games:
        # Scores are present even on scheduled/in-progress games. Only a
        # confirmed final may create a training outcome.
        if not str(game.get('status', '')).strip().lower().startswith('final'):
            continue
        if game.get('postponed') is True:
            continue
        home_team = game.get("home_team") or {}
        visitor_team = game.get("visitor_team") or {}

        rows.append(
            {
                "game_id": game.get("id"),
                "game_date": game.get("date"),
                "season": game.get("season"),
                "status": game.get("status"),
                "postseason": game.get("postseason"),
                "postponed": game.get("postponed", False),
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
        errors="coerce", utc=True,
    ).dt.tz_localize(None)

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

    df = df.loc[(df.home_points > 0) & (df.away_points > 0)
        & df.home_points.lt(float('inf')) & df.away_points.lt(float('inf'))
        & df.home_points.ne(df.away_points)
        & (df.game_date <= pd.Timestamp.now(tz='UTC').tz_localize(None))].copy()

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
    progress=None,
):
    """
    Retrieve historical NBA games for one season.

    When fetch_all=True, pagination continues until the
    entire available season has been downloaded.

    The pause between requests protects the free-tier
    BALLDONTLIE rate limit.
    """

    season = int(season)
    if not 1 <= int(per_page) <= 100:
        raise ValueError('NBA history page size must be between 1 and 100.')
    # Partial research probes never read or overwrite the complete cache.
    if not fetch_all:
        payload = _balldontlie_get('games', {'seasons[]': season, 'per_page': per_page,
            'end_date': pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')})
        return _games_to_dataframe(payload.get('data', [])), 1
    root = Path(_setting('NBA_HISTORY_CACHE_DIR', '.market_edge_cache/nba_history'))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'season_{season}_v1.json'
    # Avoid duplicate bootstraps and checkpoint races across app sessions.
    with _HISTORY_LOCK:
        state = {}
        if path.exists():
            try:
                state = json.loads(path.read_text())
                if state.get('version') != 1 or state.get('season') != season:
                    state = {}
            except (ValueError, OSError):
                state = {}
        now = pd.Timestamp.now(tz='UTC')
        checked = pd.to_datetime(state.get('checked_at'), utc=True, errors='coerce')
        current_season = now.year if now.month >= 9 else now.year - 1
        closed = season < current_season and pd.notna(checked) and checked >= pd.Timestamp(year=season+1, month=9, day=1, tz='UTC')
        fresh = pd.notna(checked) and pd.Timedelta(0) <= now-checked < pd.Timedelta(minutes=15)
        if state.get('complete') and (closed or fresh):
            if progress: progress(f'Using saved NBA {season} history')
            return _games_to_dataframe(state.get('games', [])), 0
        if not state.get('pending'):
            start = None
            if state.get('complete') and pd.notna(checked):
                # Overlap catches recently finalized games and score corrections.
                start = (checked-pd.Timedelta(days=3)).strftime('%Y-%m-%d')
                pending_dates = [g.get('date') for g in state.get('games', [])
                    if not str(g.get('status', '')).lower().startswith(('final', 'postponed', 'cancel'))]
                dates = pd.to_datetime(pending_dates, utc=True, errors='coerce').dropna()
                if len(dates):
                    start = min(start, (dates.min()-pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
            state.update(version=1, season=season, pending=True, next_cursor=None,
                query_start=start, query_end=now.strftime('%Y-%m-%d'), pages=0)
            state.setdefault('games', [])
        request_count = 0
        seen = set()
        while True:
            cursor = state.get('next_cursor')
            if cursor is not None and str(cursor) in seen:
                raise ValueError('NBA history pagination repeated a cursor; checkpoint retained.')
            if cursor is not None: seen.add(str(cursor))
            params = {'seasons[]': season, 'per_page': per_page, 'end_date': state['query_end']}
            if state.get('query_start'): params['start_date'] = state['query_start']
            if cursor is not None: params['cursor'] = cursor
            if progress:
                mode = 'updating recent results' if state.get('query_start') else 'building saved history'
                progress(f"NBA {season}: {mode}, page {state['pages']+1}")
            payload = _balldontlie_get('games', params)
            request_count += 1
            batch = payload.get('data', [])
            if not isinstance(batch, list):
                raise ValueError('Unexpected NBA history response; checkpoint retained.')
            merged = {str(g['id']): g for g in state['games'] if g.get('id') is not None}
            for game in batch:
                if game.get('id') is not None: merged[str(game['id'])] = game
            state['games'] = list(merged.values())
            next_cursor = (payload.get('meta') or {}).get('next_cursor')
            if not batch and next_cursor is not None:
                raise ValueError('Empty NBA history page has a next cursor; refusing incomplete history.')
            state['pages'] += 1
            state['next_cursor'] = next_cursor
            finished = next_cursor is None
            if finished:
                state.update(pending=False, complete=True, checked_at=now.isoformat())
            _write_history(path, state)
            if finished: break
        return _games_to_dataframe(state['games']), request_count


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

def get_multiple_historical_seasons(
    seasons,
    per_page=100,
    progress=None,
):
    """
    Download and combine multiple NBA seasons.

    Example:
        seasons = [2022, 2023, 2024, 2025]

    Returns:
        combined_games: DataFrame containing all seasons
        season_summary: DataFrame with one row per season
        total_requests: number of BALLDONTLIE API requests used
    """

    season_frames = []
    summary_rows = []
    total_requests = 0

    for season in seasons:
        games, request_count = get_historical_games(
            season=int(season),
            per_page=per_page,
            fetch_all=True,
            progress=progress,
        )

        total_requests += request_count

        if games.empty:
            continue

        games = games.copy()

        games["requested_season"] = int(season)

        season_frames.append(games)

        summary_rows.append(
            {
                "season": int(season),
                "games": int(len(games)),
                "first_game": games["game_date"].min(),
                "last_game": games["game_date"].max(),
                "home_win_rate": float(
                    games["home_win"].mean()
                ),
                "api_requests": int(request_count),
            }
        )

    if not season_frames:
        raise ValueError(
            "No historical NBA seasons were downloaded."
        )

    combined_games = pd.concat(
        season_frames,
        ignore_index=True,
    )

    combined_games = combined_games.drop_duplicates(
        subset=["game_id"]
    )

    combined_games = combined_games.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    season_summary = pd.DataFrame(summary_rows)

    return (
        combined_games,
        season_summary,
        total_requests,
    )


def test_multiple_historical_seasons(
    seasons=None,
):
    """
    Test multi-season historical NBA access.
    """

    if seasons is None:
        seasons = [2022, 2023]

    (
        games,
        season_summary,
        total_requests,
    ) = get_multiple_historical_seasons(seasons)

    return {
        "success": True,
        "seasons": seasons,
        "games_returned": int(len(games)),
        "total_requests": int(total_requests),
        "first_game": games["game_date"].min(),
        "last_game": games["game_date"].max(),
        "home_win_rate": float(
            games["home_win"].mean()
        ),
        "season_summary": season_summary,
        "sample": games.head(10),
        "games": games,
    }
def audit_historical_games(games):
    """
    Audit a BALLDONTLIE historical NBA dataset before
    allowing it into model training.
    """

    if games is None or games.empty:
        raise ValueError("No games were supplied for audit.")

    df = games.copy()

    # Make sure dates and scores have usable types.
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

    total_rows = len(df)

    unique_game_ids = (
        df["game_id"].nunique()
        if "game_id" in df.columns
        else total_rows
    )

    duplicate_game_ids = (
        total_rows - unique_game_ids
    )

    missing_dates = int(
        df["game_date"].isna().sum()
    )

    missing_scores = int(
        (
            df["home_points"].isna()
            | df["away_points"].isna()
        ).sum()
    )

    tied_games = int(
        (
            df["home_points"]
            == df["away_points"]
        ).sum()
    )

    # Look for unusual team abbreviations.
    normal_nba_teams = {
        "ATL", "BOS", "BKN", "CHA", "CHI",
        "CLE", "DAL", "DEN", "DET", "GSW",
        "HOU", "IND", "LAC", "LAL", "MEM",
        "MIA", "MIL", "MIN", "NOP", "NYK",
        "OKC", "ORL", "PHI", "PHX", "POR",
        "SAC", "SAS", "TOR", "UTA", "WAS",
    }

    teams_found = set()

    if "home_team" in df.columns:
        teams_found.update(
            df["home_team"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
        )

    if "away_team" in df.columns:
        teams_found.update(
            df["away_team"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
        )

    unusual_teams = sorted(
        teams_found - normal_nba_teams
    )

    # Count games by month. This helps expose preseason,
    # All-Star, postseason, or other unusual records.
    games_by_month = (
        df.dropna(subset=["game_date"])
        .assign(
            month=lambda x:
            x["game_date"].dt.to_period("M").astype(str)
        )
        .groupby("month")
        .size()
        .reset_index(name="games")
    )

    # Show status values returned by the API.
    if "status" in df.columns:
        status_summary = (
            df["status"]
            .fillna("MISSING")
            .astype(str)
            .value_counts()
            .rename_axis("status")
            .reset_index(name="games")
        )
    else:
        status_summary = pd.DataFrame()

    # Find games with zero/missing-looking scores.
    suspicious_scores = df[
        (df["home_points"].fillna(0) <= 0)
        | (df["away_points"].fillna(0) <= 0)
    ].copy()

    return {
        "total_rows": int(total_rows),
        "unique_game_ids": int(unique_game_ids),
        "duplicate_game_ids": int(duplicate_game_ids),
        "missing_dates": missing_dates,
        "missing_scores": missing_scores,
        "tied_games": tied_games,
        "teams_found": len(teams_found),
        "unusual_teams": unusual_teams,
        "games_by_month": games_by_month,
        "status_summary": status_summary,
        "suspicious_scores": suspicious_scores,
    }
