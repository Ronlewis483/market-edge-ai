"""Ticket-only prop collection and small helpers; never generate game winners."""
import copy
import json
from pathlib import Path
import time

import pandas as pd


def records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict('records')
    return list(value) if isinstance(value, (list, tuple)) else []


def prop_records(payload):
    """An empty game_picks array must not hide populated legacy picks."""
    return records((payload or {}).get('game_picks')) or records((payload or {}).get('picks'))


def fresh_candidates(candidates, sports, now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    clean = []
    for row in candidates:
        if row.get('sport') not in sports:
            continue
        start = pd.to_datetime(row.get('start'), utc=True, errors='coerce')
        quote = pd.to_datetime(row.get('quote_time'), utc=True, errors='coerce')
        live = row.get('game_phase') == 'live'
        age = pd.Timedelta(seconds=90) if live else pd.Timedelta(minutes=30)
        if pd.isna(start) or pd.isna(quote) or not pd.Timedelta(0) <= now-quote <= age:
            continue
        if live:
            snapshot = pd.to_datetime(row.get('live_snapshot_at'), utc=True, errors='coerce')
            if start > now or quote < start or pd.isna(snapshot) or not pd.Timedelta(0) <= now-snapshot <= age:
                continue
        elif not now < start <= now+pd.Timedelta(days=7) or quote >= start:
            continue
        clean.append(row)
    return clean


def offer_identity(row):
    return tuple(str(row.get(k, '')) for k in ('sport', 'book', 'game', 'start', 'player', 'market', 'side', 'line'))


def current_ticket_legs(legs, candidates):
    """Unrelated new offers must not invalidate a ticket; changed legs must."""
    available = {offer_identity(r): r for r in candidates}
    return [available[offer_identity(r)] for r in legs if offer_identity(r) in available]


def save_ticket_source(sport, result):
    root = Path('sports_prop_workspace')
    root.mkdir(exist_ok=True)
    payload = dict(result)
    for key in ('nfl_player_props', 'all_predictions', 'predictions'):
        if isinstance(payload.get(key), pd.DataFrame):
            payload[key] = json.loads(payload[key].to_json(orient='records', date_format='iso'))
    path = root/(sport.lower()+'_ticket_latest.json')
    temp = path.with_suffix('.tmp')
    # pandas normalizes missing values; never replace provider quote timestamps.
    temp.write_text(pd.Series(payload).to_json(date_format='iso', double_precision=15))
    temp.replace(path)


def collect_nba_ticket_props(odds_key, nba_key, publish=None, context=None):
    """Discover NBA events directly, then reuse the existing statistical estimator."""
    if not odds_key or not nba_key:
        raise ValueError('NBA ticket scan needs Odds API and BallDontLie API keys')
    if context is not None:
        import threading
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(threading.current_thread(), context)
    from .sports_player_prop_predictions import _get, generate_player_prop_picks
    now = pd.Timestamp.now(tz='UTC')
    result = {'player_props': {}, 'all_predictions': [], 'ticket_scan_complete': False}
    last_saved = 0.

    def send(props=None, stage=None, complete=False):
        nonlocal last_saved
        if props is not None:
            previous = result['player_props']
            merged = dict(previous, **props)
            # Status-only updates do not erase already completed estimates.
            if not prop_records(props) and prop_records(previous):
                merged.update(game_picks=prop_records(previous), picks=prop_records(previous)[:10])
            result['player_props'] = merged
        result.update(ticket_scan_complete=complete, ticket_checked_at=pd.Timestamp.now(tz='UTC').isoformat())
        if stage:
            result['ticket_collection_stage'] = stage
        if publish:
            publish(copy.deepcopy(result))
        if complete or prop_records(result['player_props']) and time.monotonic()-last_saved >= 5:
            try:
                save_ticket_source('NBA', result)
                last_saved = time.monotonic()
            except OSError:
                pass  # Completed estimates still reach the UI queue.
        return copy.deepcopy(result)

    send(stage='Finding upcoming NBA sportsbook events; winner predictions are not required')
    events = _get('https://api.the-odds-api.com/v4/sports/basketball_nba/events', odds_key, {'apiKey': odds_key})
    games = []
    seen = set()
    for event in events:
        start = pd.to_datetime(event.get('commence_time'), utc=True, errors='coerce')
        event_id = event.get('id')
        if (not event_id or event_id in seen or pd.isna(start) or not now < start <= now+pd.Timedelta(days=7)
                or not event.get('home_team') or not event.get('away_team')):
            continue
        seen.add(event_id)
        games.append(dict(event_id=event_id, home_team=event['home_team'], away_team=event['away_team'],
                          commence_time=start.isoformat(), start_time=start.isoformat()))
    games.sort(key=lambda g: g['start_time'])
    result['all_predictions'] = games  # Event descriptors only; no winner estimates.
    if not games:
        return send({'game_picks': [], 'picks': [], 'message': 'No upcoming NBA events were returned by the sportsbook feed.'},
                    stage='NBA scan finished: no upcoming sportsbook events', complete=True)

    def progress(props):
        stage = (props.get('loading_progress') or {}).get('stage', 'Calculating NBA player props')
        send(props, stage)

    try:
        props = generate_player_prop_picks('NBA', games, odds_key, nba_key=nba_key,
                                          progress=progress, nba_history_enabled=True, foreground=True)
    except Exception as exc:
        send(stage='NBA scan stopped ('+type(exc).__name__+'); completed estimates are preserved', complete=True)
        raise
    return send(props, 'NBA prop scan finished', complete=True)
