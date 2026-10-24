"""Saved NFL/MLB display results and frozen model-prop tracking. No wagers."""
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_LOCK = threading.RLock()
_ROOT = Path('sports_prop_workspace')
_LEAGUES = {'NFL', 'MLB'}


def _path(league, kind):
    if league not in _LEAGUES: raise ValueError('Unsupported workspace sport')
    _ROOT.mkdir(exist_ok=True)
    return _ROOT/(league.lower()+'_'+kind+'.json')


def _write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(pd.Series(payload).to_json(date_format='iso', double_precision=15))
    temporary.replace(path)


def _remote(league, kind, data=None):
    from dual_agent.balldontlie_data import _setting
    url, key = _setting('SUPABASE_URL', ''), _setting('SUPABASE_KEY', '')
    if not url or not key: return None
    suffix = '/market-edge-data/'+league.lower()+'/props/'+kind+'_v1.json'
    headers = {'apikey': key, 'Authorization': 'Bearer '+key}
    if data is None:
        response = requests.get(str(url).rstrip('/')+'/storage/v1/object/authenticated'+suffix,
                                headers=headers, timeout=3)
        return response.json() if response.status_code == 200 else None
    response = requests.post(str(url).rstrip('/')+'/storage/v1/object'+suffix,
        headers=dict(headers, **{'Content-Type':'application/json','x-upsert':'true'}),
        data=data, timeout=3)
    return response.status_code in (200,201)


def _load(league, kind, restore=False):
    path = _path(league, kind)
    try:
        payload = json.loads(path.read_text())
        if payload.get('version') == 1: return payload
    except (OSError, ValueError, AttributeError): pass
    if restore:
        try:
            payload = _remote(league, kind)
            if isinstance(payload, dict) and payload.get('version') == 1:
                _write(path, payload); return payload
        except Exception: pass
    return {'version':1, 'records':{}} if kind == 'tracking' else None


def rows(value):
    if isinstance(value, pd.DataFrame): return value.to_dict('records')
    return list(value or [])


def prop_rows(league, result):
    if league == 'NFL': return rows((result or {}).get('nfl_player_props'))
    props = (result or {}).get('player_props', {})
    return rows(props.get('game_picks', props.get('picks', [])))


def normalized(league, row):
    start = pd.to_datetime(row.get('game_time', row.get('start_time')), utc=True, errors='coerce')
    home, away = row.get('home_team'), row.get('away_team')
    if not home or not away:
        pair = str(row.get('Game','')).split(' @ ',1)
        if len(pair) == 2: away, home = pair
    stamp = row.get('Captured UTC', row.get('availability_checked_at'))
    return {'home_team':home,'away_team':away,'start_time':start,
        'player':row.get('player',row.get('Player')), 'market':row.get('market',row.get('Market')),
        'line':row.get('line',row.get('Line')), 'side':str(row.get('model_pick',row.get('Pick',''))).upper(),
        'probability':row.get('historical_support',row.get('Estimated chance')),
        'book':row.get('best_sportsbook',row.get('Best sportsbook')),
        'captured_at':pd.to_datetime(stamp,utc=True,errors='coerce'),
        'game_id':row.get('game_id'),'player_id':row.get('player_id')}


def _identity(record):
    return '|'.join(str(record.get(k,'')) for k in ('home_team','away_team','start_time','player','market','line'))


def merge_result(league, previous, incoming):
    """Keep existing cards while replacing matching line/side calculations."""
    previous, incoming = previous or {}, incoming or {}
    merged = dict(previous); merged.update(incoming)
    if league=='NFL' and 'predictions' in incoming:
        merged['all_predictions'] = incoming['predictions']
    combined = {}
    for result in (previous, incoming):
        for row in prop_rows(league, result):
            n = normalized(league, row)
            combined[(_identity(n), n['side'])] = row
    values = sorted(combined.values(), key=lambda r:-float(normalized(league,r)['probability'] or 0))
    if league == 'NFL':
        merged['nfl_player_props'] = pd.DataFrame(values)
    else:
        props = dict(previous.get('player_props', {})); props.update(incoming.get('player_props', {}))
        props.update(game_picks=values,picks=values[:10]); merged['player_props'] = props
    return merged


def checkpoint(league, result, remote=False):
    """Local first; remote backup is optional and uses existing private storage."""
    with _LOCK:
        keys = ['predictions','all_predictions','opportunities','nfl_player_props','player_props',
                'spreads','prop_diagnostics','prop_issues','message','pipeline_version','created_at',
                'validation_summary','model_source']
        payload = {k:result[k] for k in keys if k in result}
        for k in ('predictions','all_predictions','opportunities','nfl_player_props'):
            if isinstance(payload.get(k),pd.DataFrame):
                payload[k] = json.loads(payload[k].to_json(orient='records',date_format='iso',double_precision=15))
        payload.update(version=1,saved_at=pd.Timestamp.now(tz='UTC').isoformat())
        path = _path(league,'latest'); _write(path,payload)
        if remote:
            try:
                _remote(league,'latest',path.read_bytes())
                ledger = _path(league,'tracking')
                if ledger.exists(): _remote(league,'tracking',ledger.read_bytes())
            except Exception: pass


def restore_result(league):
    payload = _load(league,'latest',restore=True)
    if not payload: return None
    frame = pd.DataFrame(payload.get('all_predictions',payload.get('predictions',[])))
    if frame.empty: return None
    column = 'commence_time' if 'commence_time' in frame else 'start_time'
    if column not in frame: return None
    stamp = pd.to_datetime(frame[column],utc=True,errors='coerce'); now = pd.Timestamp.now(tz='UTC')
    frame = frame.loc[(stamp >= now-pd.Timedelta(days=1)) & (stamp <= now+pd.Timedelta(days=7))]
    if frame.empty: return None
    payload['all_predictions'] = frame if league == 'NFL' else frame.to_dict('records')
    payload['predictions'] = frame if league == 'NFL' else frame.head(10).to_dict('records')
    payload['opportunities'] = pd.DataFrame(payload.get('opportunities') or [])
    if league == 'NFL': payload['nfl_player_props'] = pd.DataFrame(payload.get('nfl_player_props',[]))
    payload['workspace_saved_at'] = payload.get('saved_at'); payload['workspace_pending'] = False
    return payload


def tracking_records(league):
    return list(_load(league,'tracking').get('records',{}).values())


def track(league, result):
    """Freeze the first stronger offered side for each exact line before kickoff."""
    from dual_agent.preferred_sportsbooks import allowed
    now = pd.Timestamp.now(tz='UTC')
    with _LOCK:
        path = _path(league,'tracking')
        ledger = _load(league,'tracking',restore=not path.exists())
        candidates = [normalized(league,r) for r in prop_rows(league,result)]
        candidates.sort(key=lambda n:-float(n['probability'] or 0))
        for n in candidates:
            start, captured = n['start_time'], n['captured_at']
            try: p, line = float(n['probability']), float(n['line'])
            except (TypeError,ValueError): continue
            if not np.isfinite([p,line]).all() or not 0<=p<=1 or not n['player']: continue
            if n['side'] not in {'OVER','UNDER','YES','NO'} or not allowed(n['book']): continue
            if pd.isna(start) or pd.isna(captured) or not captured<=now<start or now-captured>pd.Timedelta(minutes=30): continue
            n.update(start_time=start.isoformat(),captured_at=captured.isoformat(),line=line,probability=p)
            key = hashlib.sha256(_identity(n).encode()).hexdigest()
            if key not in ledger['records']:
                ledger['records'][key] = dict(n,status='Pending',actual=None,recorded_at=now.isoformat())
        _write(path,ledger)


def _decision(record, actual):
    if record['market'] == 'Anytime touchdown':
        win = actual>=1 if record['side'] in {'OVER','YES'} else actual<1
        return 'Covered' if win else 'Missed'
    if actual == record['line']: return 'Push'
    win = actual>record['line'] if record['side'] in {'OVER','YES'} else actual<record['line']
    return 'Covered' if win else 'Missed'


def settle_nfl(history):
    """Match completed nflverse stats by player, market, both teams and game date."""
    from dual_agent.nfl_live_context import team_code as code
    from dual_agent.nfl_player_props import canonical_name
    if history is None or history.empty: return
    required = {'player','market','game_time','team','opponent','is_home','value'}
    if not required.issubset(history): return
    with _LOCK:
        ledger = _load('NFL','tracking'); frame = history.copy()
        frame['_player'] = frame.player.map(canonical_name)
        available = pd.to_datetime(frame.game_time,utc=True,errors='coerce')
        frame['_day'] = (available-pd.Timedelta(days=1)).dt.date
        frame = frame.loc[available <= pd.Timestamp.now(tz='UTC')].copy()
        frame['_home'] = frame.apply(lambda r: code(str(r.team if r.is_home else r.opponent)),axis=1)
        frame['_away'] = frame.apply(lambda r: code(str(r.opponent if r.is_home else r.team)),axis=1)
        for r in ledger['records'].values():
            start = pd.Timestamp(r['start_time'])
            if r['status']!='Pending' or start>=pd.Timestamp.now(tz='UTC'): continue
            h,a = code(str(r['home_team'])),code(str(r['away_team']))
            if not h or not a: continue
            match = frame[(frame._player==canonical_name(r['player'])) & (frame.market==r['market']) &
                (frame._day==start.tz_convert('America/Chicago').date()) & (frame._home==h) & (frame._away==a)]
            values = pd.to_numeric(match.value,errors='coerce').dropna().unique()
            if len(values)!=1 or not np.isfinite(values[0]): continue
            actual = float(values[0])
            participation = ('participation_verified' in match and match['participation_verified'].eq(True).any())
            if actual == 0 and not participation: continue
            r.update(status=_decision(r,actual),actual=actual,settled_at=pd.Timestamp.now(tz='UTC').isoformat())
        _write(_path('NFL','tracking'),ledger)


def _mlb_final(game_id):
    path = _path('MLB','final_'+str(int(game_id)))
    try:
        if time.time()-path.stat().st_mtime<1800: return json.loads(path.read_text())
    except (OSError,ValueError): pass
    response = requests.get(f'https://statsapi.mlb.com/api/v1.1/game/{int(game_id)}/feed/live',timeout=8)
    response.raise_for_status(); data=response.json()
    if data.get('gamePk')!=int(game_id) or data.get('gameData',{}).get('status',{}).get('abstractGameState')!='Final': return None
    _write(path,data); return data


def _mlb_value(data, record):
    pid = int(record['player_id']); entry = None
    for team in data.get('liveData',{}).get('boxscore',{}).get('teams',{}).values():
        if 'ID'+str(pid) in team.get('players',{}): entry=team['players']['ID'+str(pid)]; break
    if not entry: return None
    stats = entry.get('stats',{}); pitching = record['market'] in {'Pitcher strikeouts','Hits allowed','Pitcher walks','Earned runs allowed','Outs recorded'}
    values = stats.get('pitching' if pitching else 'batting',{})
    def number(key):
        v = pd.to_numeric(values.get(key),errors='coerce')
        return float(v) if pd.notna(v) and np.isfinite(v) else None
    if pitching:
        participated = any((number(k) or 0)>0 for k in ('numberOfPitches','pitchesThrown','outs','baseOnBalls','hits','strikeOuts'))
    else:
        participated = any((number(k) or 0)>0 for k in ('plateAppearances','atBats','baseOnBalls','hitByPitch','sacFlies','sacBunts'))
    if not participated: return None
    direct = {'Hits':'hits','Home runs':'homeRuns','RBIs':'rbi','Runs scored':'runs','Doubles':'doubles','Triples':'triples','Batter walks':'baseOnBalls',
        'Batter strikeouts':'strikeOuts','Stolen bases':'stolenBases','Pitcher strikeouts':'strikeOuts','Hits allowed':'hits','Pitcher walks':'baseOnBalls','Earned runs allowed':'earnedRuns'}
    market = record['market']
    if market in direct: return number(direct[market])
    if market=='Outs recorded':
        if number('outs') is not None: return number('outs')
        innings = str(values.get('inningsPitched','')).split('.')
        if len(innings)==2 and innings[0].isdigit() and innings[1] in {'0','1','2'}: return int(innings[0])*3+int(innings[1])
        return None
    keys = {'Singles':['hits','doubles','triples','homeRuns'],'Total bases':['hits','doubles','triples','homeRuns'],
            'Hits + runs + RBIs':['hits','runs','rbi']}.get(market)
    if not keys: return None
    v=[number(k) for k in keys]
    if any(x is None for x in v): return None
    return sum(v) if market=='Hits + runs + RBIs' else v[0]-sum(v[1:]) if market=='Singles' else v[0]+v[1]+2*v[2]+3*v[3]


def settle_mlb():
    """One shared final-game feed for all tracked player/category outcomes."""
    with _LOCK:
        ledger = _load('MLB','tracking'); feeds = {}
        for r in ledger['records'].values():
            if r['status']!='Pending' or pd.Timestamp(r['start_time'])>=pd.Timestamp.now(tz='UTC'): continue
            if r.get('game_id') is None or r.get('player_id') is None: continue
            try:
                gid=int(r['game_id'])
                if gid not in feeds: feeds[gid]=_mlb_final(gid)
                if feeds[gid] is None: continue
                actual=_mlb_value(feeds[gid],r)
                if actual is not None: r.update(status=_decision(r,actual),actual=actual,settled_at=pd.Timestamp.now(tz='UTC').isoformat())
            except (ValueError,TypeError,requests.RequestException): continue
        _write(_path('MLB','tracking'),ledger)
