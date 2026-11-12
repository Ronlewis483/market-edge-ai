"""Ticket-only prop collection and small helpers; never generate game winners."""
import copy
import json
import re
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import pandas as pd


_context_lock = threading.Lock()
_context_pool = None
_context_job = None
_context_key = None
_context_started = 0.


def start_nba_context_refresh(api_key, script_context=None):
    """At most one background ticket context refresh per process; never wait.

    The worker only updates the context cache. Collector queues are owned by
    the collector, so a late worker cannot publish into a finished scan.
    Completed attempts are reused for 15 minutes, including failed attempts.
    """
    global _context_pool, _context_job, _context_key, _context_started
    fingerprint = sha256(api_key.encode()).hexdigest()
    with _context_lock:
        if _context_job is not None:
            if not _context_job.done():
                return _context_job if fingerprint == _context_key else None
            if fingerprint == _context_key and time.monotonic()-_context_started < 900:
                return _context_job
        if _context_pool is None:
            _context_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nba-ticket-context')

        def refresh():
            if script_context is not None:
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                add_script_run_ctx(threading.current_thread(), script_context)
            from .sports_player_prop_predictions import load_nba_matchup_context
            return load_nba_matchup_context(api_key)

        _context_key = fingerprint
        _context_started = time.monotonic()
        _context_job = _context_pool.submit(refresh)
        return _context_job


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


def safe_ticket_error(exc):
    # Provider exceptions can include URLs with credentials. Report only type
    # and HTTP status; detailed stat/identity errors arrive in the scan payload.
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status is None:
        match = re.search(r'HTTP[ :]+([1-5][0-9]{2})', str(exc), re.IGNORECASE)
        status = int(match.group(1)) if match else None
    return type(exc).__name__ + ('; HTTP '+str(status) if status else '')


def collect_nba_ticket_props(odds_key, nba_key, publish=None, context=None):
    """Discover NBA events directly, then reuse the existing statistical estimator."""
    if not odds_key or not nba_key:
        raise ValueError('NBA ticket scan needs Odds API and BallDontLie API keys')
    if context is not None:
        import threading
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(threading.current_thread(), context)
    from .sports_player_prop_predictions import (
        _get, fetch_prop_quotes, nba_cached_histories, nba_score_cached_quotes,
        _nba_history_batches, nba_identity_key, load_nba_matchup_context,
    )
    now = pd.Timestamp.now(tz='UTC')
    result = {'player_props': {}, 'all_predictions': [], 'ticket_scan_complete': False}
    last_saved = 0.

    def send(props=None, stage=None, complete=False):
        nonlocal last_saved
        if props is not None:
            previous = result['player_props']
            merged = dict(previous, **props)
            # Status-only updates do not erase already completed estimates.
            if not complete and not prop_records(props) and prop_records(previous):
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

    from .odds_request_cache import refresh_game_props
    # One discovery pass identifies only players actually offered by the books.
    # Do not pay for repeated quote collection while history is being prepared.
    send(stage='Finding listed NBA players for history preparation')
    for game in games:
        refresh_game_props(game['home_team'],game['away_team'])
    quotes, issues = fetch_prop_quotes('NBA', games, odds_key)
    if not quotes:
        return send({'game_picks':[], 'picks':[], 'errors':issues,
                     'message':'No usable sportsbook player-prop lines were returned; player-history downloads were skipped.'},
                    stage='NBA scan finished: no listed player lines returned', complete=True)
    # Prepare widely listed players first, so a completed batch is useful at
    # more than one platform. Aliases share one history request, never votes.
    named = {}; coverage = {}; first_start = {}
    for q in quotes:
        key = nba_identity_key(q['player'])
        named.setdefault(key,q['player'])
        coverage.setdefault(key,set()).add(q['book'])
        start = pd.Timestamp(q['start_time'])
        first_start[key] = min(first_start.get(key,start),start)
    names = [named[key] for key in sorted(named,key=lambda k:(-len(coverage[k]),first_start[k],named[k]))]
    histories = nba_cached_histories(names)
    try:
        matchup = load_nba_matchup_context(nba_key, cache_only=True)
    except Exception as exc:
        matchup = {'issue':'NBA matchup context unavailable: '+safe_ticket_error(exc)}
    context_job = None

    def update_context():
        nonlocal matchup, context_job
        if context_job is not None and context_job.done():
            try:
                matchup = context_job.result()
            except Exception as exc:
                # Retain cached known exclusions if the refresh fails.
                matchup = dict(matchup, injury_complete=False,
                    issue='NBA matchup refresh unavailable: '+safe_ticket_error(exc))
            context_job = None
        if matchup.get('injury_complete'):
            checked = pd.to_datetime(matchup.get('checked_at'), utc=True, errors='coerce')
            age = pd.Timestamp.now(tz='UTC')-checked
            if pd.isna(checked) or not pd.Timedelta(0) <= age < pd.Timedelta(minutes=15):
                matchup = dict(matchup, injury_complete=False, injury_stale=True,
                    issue='NBA injury snapshot needs refreshing; estimates are conditional on playing.')
                context_job = start_nba_context_refresh(nba_key, context)

    def prepared_count():
        return len({nba_identity_key(name) for name,rows in histories.items() if len(rows)>=10})

    def score(lines, stage, complete=False):
        update_context()
        context_issues = [matchup['issue']] if matchup.get('issue') else []
        props = nba_score_cached_quotes(games,lines,histories,matchup,issues+context_issues)
        props['availability_refresh_pending'] = context_job is not None
        props['loading_progress'].update(prepared_players=prepared_count(), total_players=len(names),
            players_with_history=sum(bool(rows) for rows in histories.values()))
        props['ticket_quote_passes'] = result.get('ticket_quote_passes',1)
        return send(props,stage,complete)

    result['ticket_quote_passes'] = 1
    score(quotes,'Scored saved histories first; preparing missing or outdated player histories')
    # Start optional provider work only AFTER publishing cached estimates.
    # No result()/executor shutdown wait may block scoring or completion.
    if not matchup.get('injury_complete'):
        context_job = start_nba_context_refresh(nba_key, context)
    last_count = prepared_count(); last_scored = time.monotonic()
    errors = set(issues)
    history_network_work = False

    def history_status(message):
        nonlocal history_network_work
        history_network_work = True
        send(stage='Preparing player history: '+str(message))

    try:
        # Five cold players share each stats request. Provider pacing remains in
        # the existing loader; incomplete pagination is never used as history.
        for player, rows, error in _nba_history_batches(names,nba_key,status=history_status,
                                                       foreground=True,cold_batch_size=5):
            if rows or player not in histories:
                histories[player] = rows
            if error and error not in errors:
                issues.append(error); errors.add(error)
            count = prepared_count()
            if count > last_count or time.monotonic()-last_scored >= 10:
                score(quotes,f'{count} player histories prepared; completed estimates are available')
                last_count = count; last_scored = time.monotonic()
    except Exception as exc:
        issues.append('Player-history preparation stopped: '+safe_ticket_error(exc))

    # History may have taken longer than a quote's lifetime. Collect current
    # lines once after preparation, and recompute for the NEW exact thresholds.
    # Only games with usable history need this second (and final) quote pass.
    ready_names = {nba_identity_key(name) for name,rows in histories.items() if len(rows)>=10}
    ready_events = {q['event_id'] for q in quotes if nba_identity_key(q['player']) in ready_names}
    ready_games = [game for game in games if game['event_id'] in ready_events]
    if not ready_games:
        return score([], 'NBA scan finished: no players have sufficient usable prepared history', complete=True)
    quote_now = pd.Timestamp.now(tz='UTC')
    quotes_current = all(pd.Timedelta(0) <= quote_now-pd.Timestamp(q['updated_at']) <= pd.Timedelta(minutes=30) for q in quotes)
    if not history_network_work and quotes_current:
        # A fully warm scan does not need to pay for identical quotes twice.
        return score(quotes,'NBA ticket scan finished using prepared history and current lines',complete=True)
    update_context()
    send(stage=f'{prepared_count()} player histories prepared; collecting current sportsbook lines')
    for game in ready_games:
        refresh_game_props(game['home_team'],game['away_team'])
    result['ticket_quote_passes'] = 2
    try:
        current_quotes, quote_issues = fetch_prop_quotes('NBA',ready_games,odds_key)
        issues.extend(quote_issues)
    except Exception as exc:
        issues.append('Current sportsbook quotes unavailable: '+safe_ticket_error(exc))
        current_quotes = []
    # No player-history calls occur after this point. Updated lines are scored
    # from the prepared rows rather than attaching new quotes to old chances.
    stage = ('NBA ticket scan finished: current quotes scored from prepared histories' if current_quotes
             else 'History prepared, but no current sportsbook lines were returned; see input issues')
    return score(current_quotes,stage,complete=True)
