"""Live MLB inference from saved matchup coefficients and fresh pregame data.

No fitting, historical replay, accuracy comparisons, or synthetic outcomes.
"""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import skellam

ACTIVE_MODEL_PATH = 'mlb/live_matchup/active.json'
MLB_PIPELINE_VERSION = 10
_HISTORY_CACHE = {}
_CACHE_LOCK = RLock()
CORE = ['home_field', 'offense_runs_per_game', 'opponent_runs_allowed',
        'offense_recent_run_difference', 'lineup_ops', 'lineup_obp', 'lineup_slg',
        'opposing_starter_era', 'opposing_starter_whip', 'opposing_starter_k9',
        'opposing_starter_bb9', 'lineup_ops_x_opposing_starter_whip']


def _cached_history(key, seconds, loader):
    """Cache reusable history only; fresh decision feeds and odds are never reused."""
    with _CACHE_LOCK:
        cached = _HISTORY_CACHE.get(key)
        if cached and time.monotonic()-cached[0]<seconds:
            return cached[1]
    value = loader()
    with _CACHE_LOCK:
        _HISTORY_CACHE[key] = (time.monotonic(), value)
    return value


def collect_prediction_inputs(dates, now, research, update):
    """One range schedule, bounded parallel feeds, then eligible-game updates.

    Core player rates already live in the feed. Do not fetch each roster player's
    profile and duplicate stat pages before checking whether a lineup exists.
    """
    response = requests.get('https://statsapi.mlb.com/api/v1/schedule', params={
        'sportId': 1, 'startDate': str(min(dates)), 'endDate': str(max(dates)), 'hydrate': 'probablePitcher'}, timeout=20)
    response.raise_for_status()
    scheduled = {int(g['gamePk']):g for day in response.json().get('dates', []) for g in day.get('games', [])
                 if g.get('status', {}).get('abstractGameState')=='Preview'
                 and now<pd.to_datetime(g['gameDate'], utc=True)<=now+pd.Timedelta(days=7)}
    package = {'games': [], 'errors': [], 'skipped': [], 'game_date': str(min(dates)),
               'snapshot_time': now.isoformat(), 'capture_mode': 'prediction_button', 'source': 'MLB Stats API',
               'scheduled_games': [{'game_id': g['gamePk'], 'start_time': g['gameDate'],
                    'home_team': g['teams']['home']['team']['name'], 'away_team': g['teams']['away']['team']['name']} for g in scheduled.values()]}
    response_cache = {}; request_locks = {}; cache_lock = RLock()
    class CollectionSession(requests.Session):
        def get(self, url, **kwargs):
            key = (url, json.dumps(kwargs.get('params'), sort_keys=True, default=str))
            with cache_lock:
                lock = request_locks.setdefault(key, RLock())
            with lock:
                if key not in response_cache:
                    response = super().get(url, **kwargs)
                    if response.status_code == 200:
                        response_cache[key] = response
                    return response
                return response_cache[key]
    def collect(g):
        snapshot = research.fetch_mlb_pregame_game_snapshot(int(g['gamePk']), timeout=15)
        box = snapshot.get('raw_feed', {}).get('liveData', {}).get('boxscore', {}).get('teams', {})
        for side in ['home', 'away']:
            probable = g.get('teams', {}).get(side, {}).get('probablePitcher', {})
            if probable and not snapshot.get('probable_pitchers', {}).get(side):
                snapshot.setdefault('probable_pitchers', {})[side] = probable
        if snapshot.get('status', {}).get('abstractGameState')!='Preview' or pd.to_datetime(snapshot['start_time'], utc=True)<=pd.Timestamp.now(tz='UTC'):
            return None, {'game_id': g['gamePk'], 'reason': 'Game started or changed status.'}
        snapshot['schedule'] = {'start_time': g['gameDate'], 'season_id': g.get('season'), 'venue_id': g.get('venue', {}).get('id')}
        with CollectionSession() as session:
            snapshot = research._attach_mlb_current_information(snapshot, session, {}, timeout=10)
        finished = pd.Timestamp.now(tz='UTC')
        snapshot['capture_finished_at'] = finished.isoformat()
        snapshot['eligible_pregame_capture'] = finished<pd.to_datetime(snapshot['start_time'], utc=True)
        return snapshot, None
    update(f'Checking {len(scheduled)} upcoming games; collecting eligible lineups in parallel...')
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(collect,g):g for g in scheduled.values()}
        for count, future in enumerate(as_completed(pending), 1):
            try:
                snapshot, skipped = future.result()
                if skipped: package['skipped'].append(skipped)
                if snapshot:
                    package['games'].append(snapshot)
                    package['errors'].extend(snapshot.get('current_information', {}).get('errors', []))
            except Exception as exc:
                package['errors'].append({'game_id': pending[future]['gamePk'], 'error': str(exc)})
            update(f'Checked {count}/{len(scheduled)} upcoming games; {len(package["games"])} pregame feeds collected.')
    package['games'].sort(key=lambda g:(g['start_time'],g['game_id']))
    return package


def validate_model(model):
    """Accept existing full matchup archives or frozen-base probability bundles."""
    bundle = model.get('model_bundle', {})
    columns = bundle.get('columns', [])
    from dual_agent.mlb_matchup_model import BULLPEN_FEATURES
    from dual_agent.mlb_handedness import HAND_FEATURES
    from dual_agent.mlb_matchup_context import FORM_FEATURES, ENVIRONMENT_FEATURES
    allowed = CORE + BULLPEN_FEATURES + HAND_FEATURES + FORM_FEATURES + ENVIRONMENT_FEATURES
    if not columns or len(set(columns)) != len(columns) or set(columns)-set(allowed):
        raise ValueError('Upload the saved full matchup model or probability-upgrade JSON, with supported coefficients.')
    if not set(CORE).issubset(columns):
        raise ValueError('Saved model lacks the core pitcher and lineup features.')
    for key in ['median', 'mean', 'scale', 'weights']:
        values = np.asarray(bundle.get(key), dtype=float)
        if values.shape != (len(columns),) or not np.isfinite(values).all():
            raise ValueError('Invalid saved run coefficients: ' + key)
    if (np.asarray(bundle['scale']) <= 0).any() or not np.isfinite(float(bundle['intercept'])):
        raise ValueError('Invalid run scaling or intercept.')
    if 'selected_layer' in model:
        label = model['selected_layer']
        if label not in ['Frozen matchup runs', 'Calibrated matchup', 'Consistent matchup + Elo']:
            raise ValueError('Unsupported saved probability layer.')
        layer = model.get('layer_bundles', {}).get(label)
        if label != 'Frozen matchup runs' and not layer:
            raise ValueError('Saved probability layer coefficients missing.')
    else:
        if model.get('version', 0) < 6:
            raise ValueError('Use the full version 6 or newer matchup model archive.')
        layer = model.get('win_layer_bundle') if model.get('win_layer_selected', True) else None
        if model.get('win_layer_selected', True) and not layer:
            raise ValueError('Saved win-layer coefficients missing.')
    for layer in [layer, (model.get('market') or {}).get('bundle')]:
        if layer:
            if layer['columns'] not in [['matchup_log_odds'], ['matchup_log_odds', 'pregame_elo_difference', 'projected_run_difference'], ['matchup_log_odds', 'market_log_odds']]:
                raise ValueError('Unsupported saved win-layer inputs.')
            width = len(layer['columns'])
            for key in ['mean', 'scale', 'weights']:
                values = np.asarray(layer[key], dtype=float)
                if values.shape != (width,) or not np.isfinite(values).all():
                    raise ValueError('Invalid saved probability coefficients.')
            if min(layer['scale']) <= 0 or not np.isfinite(float(layer['intercept'])):
                raise ValueError('Invalid saved probability scaling.')
    threshold = model.get('threshold') if 'selected_layer' in model else model.get('confidence_selection', {}).get('threshold')
    if threshold is not None and not .5 <= float(threshold) <= 1:
        raise ValueError('Invalid saved recommendation threshold.')
    return model


def _bucket():
    from dual_agent import supabase_db as storage
    ready = storage.ensure_market_edge_storage_bucket()
    if not ready.get('success'):
        raise RuntimeError(ready.get('error', 'Prediction storage unavailable'))
    return storage.get_supabase_client().storage.from_(storage.MLB_STORAGE_BUCKET)


def activate_model(model):
    validate_model(model)
    payload = json.dumps(model, allow_nan=False).encode()
    _bucket().upload(ACTIVE_MODEL_PATH, payload, {'content-type': 'application/json', 'upsert': 'true'})
    with _CACHE_LOCK:
        for key in list(_HISTORY_CACHE):
            if isinstance(key, tuple) and key[0] == 'completed_predictions':
                del _HISTORY_CACHE[key]
    return {'model_id': hashlib.sha256(payload).hexdigest(), 'created_at': model.get('created_at')}


def load_active_model():
    bucket = _bucket()
    try:
        payload = bucket.download(ACTIVE_MODEL_PATH)
    except Exception:
        # Existing deployments already archive full matchup bundles. Reuse the
        # newest compatible one once, then persist an explicit active model.
        files = bucket.list('mlb/matchup_model', {'limit': 1000, 'sortBy': {'column': 'name', 'order': 'desc'}})
        for entry in files:
            name = entry.get('name', '')
            if not name.endswith('.json'):
                continue
            candidate = json.loads(bucket.download('mlb/matchup_model/'+name))
            try:
                validate_model(candidate)
            except (ValueError, KeyError, TypeError):
                continue
            activate_model(candidate)
            return candidate
        raise RuntimeError('No compatible saved MLB model found. Open Saved MLB model setup and activate your full matchup JSON once.')
    return validate_model(json.loads(payload))


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def index_team_results(schedules):
    """Parse each completed scoring result once, preserving the training buffer."""
    indexed = {}
    for prior in schedules:
        if (prior.get('gameType') != 'R' or prior.get('resumeDate')
                or prior.get('status', {}).get('abstractGameState') != 'Final'):
            continue
        stamp = pd.to_datetime(prior['gameDate'], utc=True)
        for side, other in [('home', 'away'), ('away', 'home')]:
            runs = prior['teams'][side].get('score'); against = prior['teams'][other].get('score')
            if runs is None or against is None or runs == against:
                continue
            key = (int(prior['season']), int(prior['teams'][side]['team']['id']))
            indexed.setdefault(key, []).append({'date': prior['gameDate'], 'stamp': stamp,
                'id': prior['gamePk'], 'for': runs, 'against': against,
                'diff': runs-against, 'win': int(runs>against)})
    for rows in indexed.values():
        rows.sort(key=lambda r: (r['date'], r['id']))
    return indexed


def build_live_row(snapshot, schedules, team_results=None):
    """Use the same side order, player rates and 48-hour team buffer as training."""
    start = pd.to_datetime(snapshot['start_time'], utc=True)
    capture = pd.to_datetime(snapshot['capture_finished_at'], utc=True)
    now = pd.Timestamp.now(tz='UTC')
    feed = snapshot['raw_feed']
    if (not snapshot.get('eligible_pregame_capture') or capture > now or now-capture > pd.Timedelta(minutes=15)
            or capture >= start or start <= now or snapshot['status'].get('abstractGameState') != 'Preview'):
        raise ValueError('Game requires a fresh pregame capture.')
    for play in feed.get('liveData', {}).get('plays', {}).get('allPlays', []):
        if play.get('about', {}).get('isComplete') or any(event.get('isPitch') for event in play.get('playEvents', [])):
            raise ValueError('Gameplay already occurred; pregame predictions stopped.')
    game = {'game_id': int(snapshot['game_id']), 'season': start.year, 'start_time': start.isoformat(),
            'timecode': capture.strftime('%Y%m%d_%H%M%S'), 'input_notes': [], 'home_team': snapshot['home_team']['name'],
            'away_team': snapshot['away_team']['name'], 'team_sides': {}, 'player_sides': {}, 'recorded_identities': {}}
    box = feed.get('liveData', {}).get('boxscore', {}).get('teams', {})
    for side, other in [('home', 'away'), ('away', 'home')]:
        team = box.get(side, {})
        players = team.get('players', {})
        order = team.get('battingOrder', [])
        pitchers = team.get('pitchers', [])
        if len(order) != 9 or len(set(order)) != 9:
            order = []
            game['input_notes'].append(side+' lineup unannounced')
        probable = snapshot.get('probable_pitchers', {}).get(side, {}).get('id')
        pid = int(pitchers[0]) if pitchers else int(probable) if probable else None
        if probable and pitchers and int(probable) != pid:
            raise ValueError('Pitcher identity changed or feed identities disagree.')
        if pid is None: game['input_notes'].append(side+' starter unannounced')
        stat = players.get('ID'+str(pid), {}).get('seasonStats', {}).get('pitching', {})
        rates = [_number(stat.get(key)) for key in ['era', 'whip']]
        if None in rates: game['input_notes'].append(side+' starter rates unavailable')
        innings = str(stat.get('inningsPitched', ''))
        try:
            whole, _, partial = innings.partition('.')
            outs = int(whole)*3+int(partial or 0)
            if int(partial or 0) not in [0, 1, 2] or outs <= 0:
                raise ValueError()
        except ValueError:
            outs = 0
        def per9(key):
            value = _number(stat.get(key))
            return value*27/outs if value is not None and outs else None
        def lineup(key):
            values = [_number(players.get('ID'+str(p), {}).get('seasonStats', {}).get('batting', {}).get(key)) for p in order]
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if len(values) >= 8 else None
        ops, obp, slg = [lineup(key) for key in ['ops', 'obp', 'slg']]
        if any(v is None for v in [ops, obp, slg, per9('strikeOuts'), per9('baseOnBalls')]):
            game['input_notes'].append(side+' player-rate coverage incomplete')
        game['player_sides'][side] = [np.nan if v is None else v for v in rates + [ops, per9('strikeOuts'), per9('baseOnBalls'), obp, slg]]
        team_id = int(snapshot[side+'_team']['id'])
        game['recorded_identities'][side] = {'team_id': team_id, 'pitcher_id': pid, 'lineup_ids': order, 'bullpen_ids': team.get('bullpen', [])}
        if team_results is None:
            team_results = index_team_results(schedules)
        cutoff = capture-pd.Timedelta(hours=48)
        history = [row for row in team_results.get((start.year, team_id), []) if row['stamp'] < cutoff]
        if not history:
            raise ValueError('Prior regular-season team scoring history missing.')
        def mean(key, count=None):
            return float(np.mean([r[key] for r in (history[-count:] if count else history)]))
        game['team_sides'][side] = [len(history), mean('win'), mean('for'), mean('against'), mean('diff'), mean('win', 5), mean('win', 10), mean('diff', 5)]
    return game


def _layer(bundle, features):
    X = np.column_stack([features[name] for name in bundle['columns']])
    z = ((X-np.asarray(bundle['mean']))/np.asarray(bundle['scale']))@np.asarray(bundle['weights'])+bundle['intercept']
    return 1/(1+np.exp(-np.clip(z, -40, 40)))


def score_live_games(model, games, market=None):
    """Evaluate saved equations once. Never learn from upcoming-game outcomes."""
    validate_model(model)
    from dual_agent.mlb_matchup_model import matchup_rows, BULLPEN_FEATURES
    from dual_agent.mlb_handedness import HAND_FEATURES
    from dual_agent.mlb_matchup_context import FORM_FEATURES, ENVIRONMENT_FEATURES
    columns = model['model_bundle']['columns']
    flags = [bool(set(columns)&set(group)) for group in [BULLPEN_FEATURES, HAND_FEATURES, FORM_FEATURES+ENVIRONMENT_FEATURES]]
    names = CORE + (BULLPEN_FEATURES if flags[0] else []) + (HAND_FEATURES if flags[1] else []) + (FORM_FEATURES+ENVIRONMENT_FEATURES if flags[2] else [])
    # matchup_rows requires targets for its historical caller; they are discarded here.
    X, _ = matchup_rows([dict(g, home_runs=0, away_runs=0) for g in games], *flags)
    X = X[:, [names.index(name) for name in columns]]
    missing = ~np.isfinite(X)
    b = model['model_bundle']
    X = np.where(missing, np.asarray(b['median']), X)
    z = ((X-np.asarray(b['mean']))/np.asarray(b['scale']))@np.asarray(b['weights'])+b['intercept']
    rates = np.exp(np.clip(z, -20, 20)).reshape(-1, 2)
    raw = np.clip(skellam.sf(0, rates[:, 0], rates[:, 1])+.5*skellam.pmf(0, rates[:, 0], rates[:, 1]), 1e-6, 1-1e-6)
    features = {'matchup_log_odds': np.log(raw/(1-raw)), 'pregame_elo_difference': np.asarray([g['elo_difference'] for g in games]), 'projected_run_difference': rates[:, 0]-rates[:, 1]}
    label = model.get('selected_layer', 'Matchup + Elo' if model.get('win_layer_selected', True) else 'Matchup runs')
    bundle = model.get('layer_bundles', {}).get(label) if 'selected_layer' in model else model.get('win_layer_bundle') if model.get('win_layer_selected', True) else None
    p = _layer(bundle, features) if bundle else raw.copy()
    threshold = model.get('threshold') if 'selected_layer' in model else model.get('confidence_selection', {}).get('threshold')
    labels = [label]*len(games); thresholds = [threshold]*len(games)
    mp = np.asarray(market if market is not None else [np.nan]*len(games), dtype=float)
    report = model.get('market', {}) or {}
    selected = report.get('selected_on_late_2025')
    if selected in ['Market consensus', 'Matchup + market', 'Matchup on odds-covered games']:
        covered = np.isfinite(mp)
        if selected == 'Matchup + market' and not report.get('bundle'):
            raise ValueError('Saved market blend coefficients missing.')
        if covered.any():
            market_features = {key: value[covered] for key, value in features.items()}
            market_features['market_log_odds'] = np.log(np.clip(mp[covered], 1e-6, 1-1e-6)/(1-np.clip(mp[covered], 1e-6, 1-1e-6)))
            p[covered] = (mp[covered] if selected == 'Market consensus' else
                          raw[covered] if selected == 'Matchup on odds-covered games' else
                          _layer(report['bundle'], market_features))
            for i in np.flatnonzero(covered):
                labels[i] = selected; thresholds[i] = report.get('threshold')
    return [{'game_id': g['game_id'], 'Home': g['home_team'], 'Away': g['away_team'], 'start_time': g['start_time'],
             'Home win probability': float(p[i]), 'Predicted winner': g['home_team'] if p[i] >= .5 else g['away_team'],
             'Winner probability': float(max(p[i], 1-p[i])), 'Projected home runs': float(rates[i, 0]), 'Projected away runs': float(rates[i, 1]),
             'Equation': labels[i], 'Recommendation threshold': thresholds[i],
             'Should we make the pick': bool(not g.get('input_notes') and thresholds[i] is not None and max(p[i], 1-p[i]) >= thresholds[i]),
             'Input status': 'Early forecast — player inputs incomplete' if g.get('input_notes') else 'Recorded pregame player inputs available',
             'Input notes': g.get('input_notes', []),
             'Market home probability': float(mp[i]) if np.isfinite(mp[i]) else None,
             'Captured UTC': pd.to_datetime(g['timecode'], format='%Y%m%d_%H%M%S', utc=True).isoformat(),
             'Missing model inputs': [name for j, name in enumerate(columns) if missing[2*i:2*i+2, j].any()]}
            for i, g in enumerate(games)]


def _generate_mlb_predictions(game_date=None, api_key=None, progress=None, on_team_predictions=None):
    from dual_agent import mlb_research as research, supabase_db as storage
    from dual_agent.mlb_matchup_elo import attach_elo
    from dual_agent.mlb_matchup_context import attach_matchup_context
    from dual_agent.mlb_matchup_model import attach_bullpen_history, BULLPEN_FEATURES
    from dual_agent.mlb_bullpen_freshness import collect_bullpen_completion_times
    from dual_agent.mlb_handedness import attach_handedness, HAND_FEATURES
    started = time.monotonic(); timings = {}; stage_started = started; stage_name = 'Load saved model'
    def update(message):
        if progress: progress(f"{message} ({time.monotonic()-started:.1f}s elapsed)")
    def mark(next_stage):
        nonlocal stage_started, stage_name
        instant = time.monotonic()
        timings[stage_name] = round(instant-stage_started, 3)
        update(f"{stage_name} completed in {timings[stage_name]:.1f}s. {next_stage}...")
        stage_started = instant; stage_name = next_stage
    model = load_active_model()
    mark('Collect fresh feeds and prepare history')
    update('Collecting current lineups, pitchers, player updates and stadium forecasts...')
    now = pd.Timestamp.now(tz='UTC')
    automatic = game_date is None
    game_date = game_date or now.tz_convert('America/Chicago').date()
    dates = [game_date]
    if automatic:
        end_date = (now+pd.Timedelta(days=7)).tz_convert('America/Chicago').date()
        dates = [date.date() for date in pd.date_range(game_date, end_date, freq='D')]
    def load_schedules():
        schedules = []
        first_year = min([int(g['season']) for g in model.get('retained_feature_rows', [])] or [2024])
        with requests.Session() as session:
            for year in range(first_year, pd.Timestamp(game_date).year+1):
                def load_year(year=year):
                    response = session.get('https://statsapi.mlb.com/api/v1/schedule', params={'sportId': 1, 'startDate': f'{year}-03-01', 'endDate': f'{year}-12-31', 'gameType': 'R'}, timeout=20)
                    response.raise_for_status()
                    return [g for day in response.json().get('dates', []) for g in day.get('games', [])]
                schedules.extend(_cached_history(('schedule', year), 180 if year==now.year else 86400, load_year))
        return schedules
    # History preparation is independent of fresh feeds: overlap their I/O.
    with ThreadPoolExecutor(max_workers=2) as preparation:
        history_job = preparation.submit(_cached_history, 'player_warehouse', 180, storage.load_mlb_player_history)
        schedule_job = preparation.submit(load_schedules)
        package = collect_prediction_inputs(dates, now, research, update)
        update('Finishing historical input preparation...')
        schedules = schedule_job.result()
        logs = history_job.result()
    mark('Save fresh game information')
    errors = list(package.get('errors', [])); skipped = list(package.get('skipped', []))
    saved_capture = storage.save_mlb_pregame_intelligence(package, snapshot_date=str(game_date))
    if not saved_capture.get('success'):
        raise RuntimeError(saved_capture.get('message', 'Could not save fresh pregame information.'))
    if not package['games']:
        scheduled = package.get('scheduled_games', [])
        return {'predictions': [], 'scheduled_games': scheduled, 'skipped': skipped, 'errors': errors,
                'message': f'{len(scheduled)} upcoming games found; their pregame feeds could not be captured.' if scheduled else 'No scheduled MLB games in the next seven days.'}
    mark('Build matchup features')
    team_results = index_team_results(schedules)
    games = []; snapshots = {}
    for snapshot in package['games']:
        try:
            games.append(build_live_row(snapshot, schedules, team_results)); snapshots[int(snapshot['game_id'])] = snapshot
        except (ValueError, KeyError, TypeError) as exc:
            skipped.append({'game_id': snapshot.get('game_id'), 'reason': str(exc)})
    if not games:
        return {'predictions': [], 'scheduled_games': package.get('scheduled_games', []), 'skipped': skipped, 'errors': errors,
                'message': f'{len(package.get("scheduled_games", []))} upcoming games found; see game information coverage for forecast exclusions.'}
    columns = model['model_bundle']['columns']
    if set(columns)&set(BULLPEN_FEATURES):
        update('Connecting verified bullpen workload...')
        ready = [g for g in games if all(g['recorded_identities'][s]['pitcher_id'] is not None for s in ['home','away'])]
        completion, details = (collect_bullpen_completion_times(ready, schedules, 'mlb_accuracy_results')
                               if ready and not logs.empty else ({}, {'conservative_game_ids': []}))
        for game in games: game['bullpen_sides'] = {s: [None]*len(BULLPEN_FEATURES) for s in ['home','away']}
        if not logs.empty and ready:
            needed_teams = {identity['team_id'] for g in ready for identity in g['recorded_identities'].values()}
            bullpen_logs = logs.loc[pd.to_numeric(logs.team_id, errors='coerce').isin(needed_teams)]
            attach_bullpen_history(ready, bullpen_logs, completion, details['conservative_game_ids'])
        # A missing warehouse game must not turn unrecorded workload into zero.
        for game in games:
            capture = pd.to_datetime(game['timecode'], format='%Y%m%d_%H%M%S', utc=True)
            for side in ['home', 'away']:
                tid = game['recorded_identities'][side]['team_id']
                expected = {int(g['gamePk']) for g in schedules if int(g['gamePk']) in completion and pd.to_datetime(completion[int(g['gamePk'])], utc=True)<capture and tid in [int(g['teams'][s]['team']['id']) for s in ['home', 'away']]}
                observed = set(pd.to_numeric(logs.loc[pd.to_numeric(logs.team_id, errors='coerce')==tid, 'game_id'], errors='coerce').dropna().astype(int)) if not logs.empty else set()
                if expected-observed:
                    game['bullpen_sides'][side][1:4] = [None]*3
                    errors.append({'game_id': game['game_id'], 'error': 'Recent bullpen warehouse incomplete for '+side+'; workload left unknown.'})
    if set(columns)&set(HAND_FEATURES):
        records = []
        for path in (Path('mlb_accuracy_results')/'handedness_play_cache').glob('*_appearances_v1.json'):
            records.extend(json.loads(path.read_text()))
        # Partial cached histories bias splits. Require all eligible prior regular-season games.
        available = {int(r['game_id']) for r in records}
        for game in games:
            cutoff = pd.to_datetime(game['timecode'], format='%Y%m%d_%H%M%S', utc=True)-pd.Timedelta(hours=48)
            expected = {int(g['gamePk']) for g in schedules if int(g['season'])==game['season'] and g.get('gameType')=='R' and g['status'].get('abstractGameState')=='Final' and not g.get('resumeDate') and pd.to_datetime(g['gameDate'], utc=True)<cutoff}
            if expected and expected.issubset(available):
                attach_handedness([game], records)
            else:
                game['hand_sides'] = {s: [None]*len(HAND_FEATURES) for s in ['home', 'away']}
                errors.append({'game_id': game['game_id'], 'error': 'Complete prior handedness cache unavailable; split inputs use saved model medians.'})
    from dual_agent.mlb_matchup_context import FORM_FEATURES, ENVIRONMENT_FEATURES
    if set(columns)&set(FORM_FEATURES+ENVIRONMENT_FEATURES):
        context = {'rows': [{'game_id': s['game_id'], 'start_time': s['start_time'], 'capture_finished_at': s['capture_finished_at'], 'context_version': 1, 'context_features': research.build_mlb_complete_context_features(s)} for s in snapshots.values()]}
        # Sentinel used only for a missing-player lookup, never presented as an identity.
        context_games = [dict(g, recorded_identities={s:dict(g['recorded_identities'][s], pitcher_id=g['recorded_identities'][s]['pitcher_id'] if g['recorded_identities'][s]['pitcher_id'] is not None else -1) for s in ['home','away']}) for g in games]
        if not logs.empty:
            needed_players = {pid for g in games for identity in g['recorded_identities'].values()
                              for pid in identity['lineup_ids']+[identity['pitcher_id']] if pid is not None}
            form_logs = logs.loc[pd.to_numeric(logs.player_id, errors='coerce').isin(needed_players)]
            attach_matchup_context(context_games, form_logs, schedules, context)
        else:
            values_by_game = {row['game_id']: row['context_features'] for row in context['rows']}
            for game in context_games:
                game['form_sides'] = {s: [None]*len(FORM_FEATURES) for s in ['home','away']}
                game['environment'] = [None]+[values_by_game[game['game_id']].get(name) for name in ENVIRONMENT_FEATURES[1:]]
        for game, enriched in zip(games, context_games):
            game['form_sides'] = enriched['form_sides']; game['environment'] = enriched['environment']
    attach_elo(games, schedules)
    market = None
    if api_key:
        update('Reading current sportsbook consensus...')
        try:
            market = fetch_live_consensus(games, api_key)
        except Exception:
            errors.append({'error': 'Current odds unavailable; using the saved matchup equation.'})
    mark('Score team winners')
    predictions = score_live_games(model, games, market)
    now = pd.Timestamp.now(tz='UTC')
    predictions = [r for r in predictions if pd.Timestamp(r['start_time'])>now and now-pd.Timestamp(r['Captured UTC'])<=pd.Timedelta(minutes=15)]
    predictions.sort(key=lambda r: (-r['Winner probability'], r['start_time'], r['game_id']))
    all_eligible_predictions = list(predictions)
    predictions = predictions[:10]
    mark('Collect player props')
    if on_team_predictions:
        on_team_predictions(predictions)
    update('Team forecasts ready. Collecting player props in parallel...')
    try:
        import importlib
        prop_module = importlib.import_module('dual_agent.sports_player_prop_predictions')
        if getattr(prop_module, 'MLB_PROP_MATCH_VERSION', None) != 2:
            prop_module = importlib.reload(prop_module)
        if getattr(prop_module, 'MLB_PROP_MATCH_VERSION', None) != 2:
            raise RuntimeError('Install the matching sports_player_prop_predictions.py update.')
        generate_player_prop_picks = prop_module.generate_player_prop_picks
        prop_games = [{'game_id': r['game_id'], 'start_time': r['start_time'], 'home_team': r['Home'], 'away_team': r['Away']} for r in all_eligible_predictions]
        props = generate_player_prop_picks('MLB', prop_games, api_key, player_logs=logs, snapshots=snapshots)
    except Exception as exc:
        props = {'picks': [], 'errors': [{'stage': 'MLB prop generation', 'error_type': type(exc).__name__}],
                 'message': 'MLB player-prop generation failed ('+type(exc).__name__+'). See player-prop data coverage.'}
    now = pd.Timestamp.now(tz='UTC')
    predictions = [r for r in predictions if pd.Timestamp(r['start_time'])>now and now-pd.Timestamp(r['Captured UTC'])<=pd.Timedelta(minutes=15)]
    props['picks'] = [r for r in props.get('picks', []) if pd.Timestamp(r['start_time'])>now and now-pd.Timestamp(r['Captured UTC'])<=pd.Timedelta(minutes=15)]
    mark('Save predictions')
    result = {'stage_timings_seconds': dict(timings), 'version': 1, 'created_at': now.isoformat(), 'game_date': str(game_date),
              'model_id': hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'predictions': predictions, 'all_predictions': all_eligible_predictions, 'player_props': props,
              'scheduled_games': package.get('scheduled_games', []), 'skipped': skipped, 'errors': errors,
              'message': f'{len(all_eligible_predictions)} games scored; showing the {len(predictions)} strongest forecasts.', 'historical_tests_run': False}
    path = 'mlb/live_matchup/predictions/'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json'
    _bucket().upload(path, json.dumps(storage._json_safe(result), allow_nan=False).encode(), {'content-type': 'application/json', 'upsert': 'false'})
    result['archive_file'] = path
    return result


def fetch_live_consensus(games, api_key):
    """One live US moneyline request. Only fresh two-sided quotes from >=2 books."""
    from dual_agent.mlb_historical_odds import team as _normalize_team
    response = requests.get('https://api.the-odds-api.com/v4/sports/baseball_mlb/odds', params={'apiKey': api_key, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'decimal'}, timeout=25)
    if response.status_code != 200:
        raise RuntimeError('Live odds request failed (HTTP '+str(response.status_code)+').')
    events = response.json(); probabilities = []
    for game in games:
        capture = pd.to_datetime(game['timecode'], format='%Y%m%d_%H%M%S', utc=True)
        start = pd.Timestamp(game['start_time'])
        matched = [e for e in events if _normalize_team(e.get('home_team'))==_normalize_team(game['home_team']) and _normalize_team(e.get('away_team'))==_normalize_team(game['away_team']) and abs(pd.Timestamp(e['commence_time'])-start)<=pd.Timedelta(minutes=15)]
        books = {}
        if len(matched)==1:
            for book in matched[0].get('bookmakers', []):
                for market in book.get('markets', []):
                    if market.get('key')!='h2h' or len(market.get('outcomes', []))!=2: continue
                    stamp = pd.to_datetime(market.get('last_update') or book.get('last_update'), utc=True, errors='coerce')
                    # Capture times define the model decision; later quotes cannot enter it.
                    if pd.isna(stamp) or stamp>capture or stamp>=start or capture-stamp>pd.Timedelta(minutes=30): continue
                    prices = {_normalize_team(o.get('name')): _number(o.get('price')) for o in market['outcomes']}
                    h = prices.get(_normalize_team(game['home_team'])); a = prices.get(_normalize_team(game['away_team']))
                    if h is not None and a is not None and min(h, a)>1 and book.get('key'):
                        books[book['key']] = (1/h)/(1/h+1/a)
        probabilities.append(float(np.median(list(books.values()))) if len(books)>=2 else np.nan)
    return probabilities


def run_mlb_prediction_pipeline(game_date=None, api_key=None, progress=None, on_team_predictions=None):
    """Reuse completed pregame results for three minutes, retaining source times."""
    import copy
    now = pd.Timestamp.now(tz='UTC')
    date_key = str(game_date or now.tz_convert('America/Chicago').date())
    key = ('completed_predictions', date_key, hashlib.sha256((api_key or '').encode()).hexdigest())
    with _CACHE_LOCK:
        cached = _HISTORY_CACHE.get(key)
        reused = bool(cached and time.monotonic()-cached[0]<180)
    if reused and progress:
        progress('Using predictions collected within the last three minutes...')
    result = copy.deepcopy(_cached_history(key, 180, lambda: _generate_mlb_predictions(game_date, api_key, progress, on_team_predictions)))
    now = pd.Timestamp.now(tz='UTC')
    def eligible(row):
        return pd.Timestamp(row['start_time'])>now and now-pd.Timestamp(row['Captured UTC'])<=pd.Timedelta(minutes=15)
    if 'all_predictions' in result:
        result['all_predictions'] = [r for r in result['all_predictions'] if eligible(r)]
        result['predictions'] = sorted(result['all_predictions'], key=lambda r:-r['Winner probability'])[:10]
    else:
        result['predictions'] = [r for r in result.get('predictions', []) if eligible(r)]
    if result.get('player_props'):
        result['player_props']['picks'] = [r for r in result['player_props'].get('picks', []) if eligible(r)]
    result['reused_recent_predictions'] = reused
    return result
