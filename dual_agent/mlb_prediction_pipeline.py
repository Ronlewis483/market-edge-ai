"""Live MLB inference from saved matchup coefficients and fresh pregame data.

No fitting, historical replay, accuracy comparisons, or synthetic outcomes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import skellam

ACTIVE_MODEL_PATH = 'mlb/live_matchup/active.json'
CORE = ['home_field', 'offense_runs_per_game', 'opponent_runs_allowed',
        'offense_recent_run_difference', 'lineup_ops', 'lineup_obp', 'lineup_slg',
        'opposing_starter_era', 'opposing_starter_whip', 'opposing_starter_k9',
        'opposing_starter_bb9', 'lineup_ops_x_opposing_starter_whip']


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


def build_live_row(snapshot, schedules):
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
            'timecode': capture.strftime('%Y%m%d_%H%M%S'), 'home_team': snapshot['home_team']['name'],
            'away_team': snapshot['away_team']['name'], 'team_sides': {}, 'player_sides': {}, 'recorded_identities': {}}
    box = feed.get('liveData', {}).get('boxscore', {}).get('teams', {})
    for side, other in [('home', 'away'), ('away', 'home')]:
        team = box.get(side, {})
        players = team.get('players', {})
        order = team.get('battingOrder', [])
        pitchers = team.get('pitchers', [])
        if len(order) != 9 or len(set(order)) != 9 or not pitchers:
            raise ValueError('Recorded starting pitcher and nine-player lineup required.')
        pid = int(pitchers[0])
        probable = snapshot.get('probable_pitchers', {}).get(side, {}).get('id')
        if probable and int(probable) != pid:
            raise ValueError('Pitcher identity changed or feed identities disagree.')
        stat = players.get('ID'+str(pid), {}).get('seasonStats', {}).get('pitching', {})
        rates = [_number(stat.get(key)) for key in ['era', 'whip']]
        if None in rates:
            raise ValueError('Starter season ERA or WHIP missing.')
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
            raise ValueError('Insufficient prior lineup or starter rate coverage.')
        game['player_sides'][side] = rates + [ops, per9('strikeOuts'), per9('baseOnBalls'), obp, slg]
        team_id = int(snapshot[side+'_team']['id'])
        game['recorded_identities'][side] = {'team_id': team_id, 'pitcher_id': pid, 'lineup_ids': order, 'bullpen_ids': team.get('bullpen', [])}
        history = []
        for prior in schedules:
            if (int(prior['season']) != start.year or prior.get('gameType') != 'R' or prior.get('resumeDate')
                    or prior.get('status', {}).get('abstractGameState') != 'Final'
                    or pd.to_datetime(prior['gameDate'], utc=True) >= capture-pd.Timedelta(hours=48)):
                continue
            for prior_side, opponent in [('home', 'away'), ('away', 'home')]:
                if int(prior['teams'][prior_side]['team']['id']) != team_id:
                    continue
                runs = prior['teams'][prior_side].get('score'); against = prior['teams'][opponent].get('score')
                if runs is not None and against is not None and runs != against:
                    history.append({'date': prior['gameDate'], 'id': prior['gamePk'], 'for': runs, 'against': against, 'diff': runs-against, 'win': int(runs>against)})
        history.sort(key=lambda r: (r['date'], r['id']))
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
             'Should we make the pick': bool(thresholds[i] is not None and max(p[i], 1-p[i]) >= thresholds[i]),
             'Market home probability': float(mp[i]) if np.isfinite(mp[i]) else None,
             'Captured UTC': pd.to_datetime(g['timecode'], format='%Y%m%d_%H%M%S', utc=True).isoformat(),
             'Missing model inputs': [name for j, name in enumerate(columns) if missing[2*i:2*i+2, j].any()]}
            for i, g in enumerate(games)]


def run_mlb_prediction_pipeline(game_date, api_key=None, progress=None):
    from dual_agent import mlb_research as research, supabase_db as storage
    from dual_agent.mlb_matchup_elo import attach_elo
    from dual_agent.mlb_matchup_context import attach_matchup_context
    from dual_agent.mlb_matchup_model import attach_bullpen_history, BULLPEN_FEATURES
    from dual_agent.mlb_bullpen_freshness import collect_bullpen_completion_times
    from dual_agent.mlb_handedness import attach_handedness, HAND_FEATURES
    def update(message):
        if progress: progress(message)
    model = load_active_model()
    update('Collecting current lineups, pitchers, player updates and stadium forecasts...')
    package = research.collect_mlb_upcoming_game_information(game_date)
    errors = list(package.get('errors', [])); skipped = list(package.get('skipped', []))
    saved_capture = storage.save_mlb_pregame_intelligence(package, snapshot_date=str(game_date))
    if not saved_capture.get('success'):
        raise RuntimeError(saved_capture.get('message', 'Could not save fresh pregame information.'))
    if not package['games']:
        return {'predictions': [], 'skipped': skipped, 'errors': errors, 'message': 'No upcoming MLB games for this date.'}
    update('Loading prior team results and saved player history...')
    schedules = []
    first_year = min([int(g['season']) for g in model.get('retained_feature_rows', [])] or [2024])
    with requests.Session() as session:
        for year in range(first_year, pd.Timestamp(game_date).year+1):
            response = session.get('https://statsapi.mlb.com/api/v1/schedule', params={'sportId': 1, 'startDate': f'{year}-03-01', 'endDate': f'{year}-12-31', 'gameType': 'R'}, timeout=30)
            response.raise_for_status()
            schedules.extend(g for day in response.json().get('dates', []) for g in day.get('games', []))
    games = []; snapshots = {}
    for snapshot in package['games']:
        try:
            games.append(build_live_row(snapshot, schedules)); snapshots[int(snapshot['game_id'])] = snapshot
        except (ValueError, KeyError, TypeError) as exc:
            skipped.append({'game_id': snapshot.get('game_id'), 'reason': str(exc)})
    if not games:
        return {'predictions': [], 'skipped': skipped, 'errors': errors, 'message': 'No games have a fresh recorded lineup, starter and required player rates.'}
    columns = model['model_bundle']['columns']
    logs = storage.load_mlb_player_history()
    if set(columns)&set(BULLPEN_FEATURES):
        update('Connecting verified bullpen workload...')
        completion, details = collect_bullpen_completion_times(games, schedules, 'mlb_accuracy_results')
        attach_bullpen_history(games, logs, completion, details['conservative_game_ids'])
        # A missing warehouse game must not turn unrecorded workload into zero.
        for game in games:
            capture = pd.to_datetime(game['timecode'], format='%Y%m%d_%H%M%S', utc=True)
            for side in ['home', 'away']:
                tid = game['recorded_identities'][side]['team_id']
                expected = {int(g['gamePk']) for g in schedules if int(g['gamePk']) in completion and pd.to_datetime(completion[int(g['gamePk'])], utc=True)<capture and tid in [int(g['teams'][s]['team']['id']) for s in ['home', 'away']]}
                observed = set(pd.to_numeric(logs.loc[pd.to_numeric(logs.team_id, errors='coerce')==tid, 'game_id'], errors='coerce').dropna().astype(int))
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
        attach_matchup_context(games, logs, schedules, context)
    attach_elo(games, schedules)
    market = None
    if api_key:
        update('Reading current sportsbook consensus...')
        try:
            market = fetch_live_consensus(games, api_key)
        except Exception:
            errors.append({'error': 'Current odds unavailable; using the saved matchup equation.'})
    update('Applying saved matchup coefficients and saving predictions...')
    predictions = score_live_games(model, games, market)
    now = pd.Timestamp.now(tz='UTC')
    predictions = [r for r in predictions if pd.Timestamp(r['start_time'])>now and now-pd.Timestamp(r['Captured UTC'])<=pd.Timedelta(minutes=15)]
    result = {'version': 1, 'created_at': now.isoformat(), 'game_date': str(game_date),
              'model_id': hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'predictions': predictions, 'skipped': skipped, 'errors': errors,
              'message': f'{len(predictions)} games scored with the saved matchup model.', 'historical_tests_run': False}
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
