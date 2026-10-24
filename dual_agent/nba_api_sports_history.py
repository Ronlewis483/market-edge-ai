"""API-Sports NBA completed player logs with shared caching and a free-plan budget."""
import hashlib
import json
from pathlib import Path
import threading
import time
import unicodedata
from datetime import datetime, timezone

import pandas as pd
import requests

BASE = 'https://v2.nba.api-sports.io'
_LOCK = threading.RLock()
_LAST_REQUEST = 0.0


def player_key(name):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(name)).lower() if c.isalnum())


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def _read(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return None


def _request(path, key, params, root):
    global _LAST_REQUEST
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    budget_path = root/('request_budget_'+day+'.json')
    budget = _read(budget_path) or {'used':0}
    if budget['used'] >= 90:
        raise RuntimeError('Daily local NBA statistics budget reached (90 requests). Saved complete history is retained; retry after the UTC quota reset.')
    delay = 6.2-(time.monotonic()-_LAST_REQUEST)
    if delay>0: time.sleep(delay)
    budget['used'] += 1
    _save(budget_path,budget)
    _LAST_REQUEST = time.monotonic()
    response = requests.get(BASE+path, headers={'x-apisports-key':key.strip()}, params=params, timeout=(10,45))
    if response.status_code in (401,403):
        raise RuntimeError(f'API-Sports NBA HTTP {response.status_code}: verify NBA endpoint access for API_SPORTS_KEY. This is separate from The Odds API.')
    if response.status_code==429:
        raise RuntimeError('API-Sports NBA rate or quota limit reached. Saved history is retained; no automatic paid retries were made.')
    if response.status_code!=200:
        raise RuntimeError(f'API-Sports NBA returned HTTP {response.status_code}.')
    payload = response.json()
    if payload.get('errors'):
        # Do not echo arbitrary response bodies, which may include credentials.
        raise RuntimeError('API-Sports NBA rejected this request. Check the NBA plan, season coverage and remaining quota in its Live Tester.')
    if not isinstance(payload.get('response'),list):
        raise RuntimeError('API-Sports NBA returned an unexpected response format.')
    paging=payload.get('paging') or {}
    if int(paging.get('total',1))>1:
        raise RuntimeError('NBA statistics response is paginated; incomplete history was not accepted.')
    return payload['response']


def normalize_player_stats(stats, games, cutoff):
    """Accept played, final regular/postseason logs; preserve missing components."""
    game_index = {str(game.get('id')):game for game in games}
    result = {}; identities = {}
    fields = {'pts':'points','reb':'totReb','ast':'assists','fg3m':'tpm',
              'blk':'blocks','stl':'steals','turnover':'turnovers'}
    for stat in stats:
        game = game_index.get(str((stat.get('game') or {}).get('id')))
        if not game or str((game.get('status') or {}).get('short'))!='3': continue
        if str(game.get('stage'))=='1': continue  # preseason
        stamp = pd.to_datetime((game.get('date') or {}).get('start'),utc=True,errors='coerce')
        if pd.isna(stamp) or stamp>=cutoff or stamp<cutoff-pd.Timedelta(days=365): continue
        person = stat.get('player') or {}; pid=person.get('id')
        name = str(person.get('firstname') or '')+' '+str(person.get('lastname') or '')
        if pid is None or not name.strip(): continue
        try:
            parts=str(stat.get('min') or '0').split(':')
            minutes=float(parts[0])+(float(parts[1])/60 if len(parts)>1 else 0)
        except (ValueError,TypeError): continue
        if minutes<=0: continue
        row={'game':{'id':game['id'],'date':stamp.isoformat(),'status':'Final'},
             'player':{'id':pid},'source_date':stamp.isoformat(),'min':minutes,
             'history_provider':'api_sports'}
        for target, source in fields.items():
            value=pd.to_numeric(stat.get(source), errors='coerce')
            row[target]=float(value) if pd.notna(value) and float('-inf')<float(value)<float('inf') else None
        k=player_key(name);identities.setdefault(k,set()).add(str(pid))
        result.setdefault(k,{})[(str(pid),str(game['id']))]=row
    # Never merge two distinct athletes who normalize to the same full name.
    return {k:sorted(rows.values(),key=lambda r:(r['source_date'],str(r['game']['id'])))
            for k,rows in result.items() if len(identities[k])==1}


def load_histories(names, api_key, progress=None):
    """Load shared season data, then update only newly final current-season games."""
    if not api_key: raise ValueError('Set API_SPORTS_KEY in Streamlit Secrets.')
    now=pd.Timestamp.now(tz='UTC')
    season=now.year if now.month>=9 else now.year-1
    fingerprint=hashlib.sha256(api_key.strip().encode()).hexdigest()[:12]
    root=Path('.market_edge_cache')/'api_sports_nba'/fingerprint
    root.mkdir(parents=True,exist_ok=True)
    all_stats=[]; all_games=[]; errors=[]
    def report(message):
        if progress: progress(message)
    with _LOCK:
        # Prior season first provides support at the start of a new season.
        for year in (season-1,season):
            path=root/f'season_{year}_v1.json'; saved=_read(path)
            try:
                checked=pd.to_datetime((saved or {}).get('checked_at'),utc=True,errors='coerce')
                fresh = saved and (year<season or (pd.notna(checked) and now-checked<pd.Timedelta(minutes=10)))
                if fresh:
                    report(f'Using saved API-Sports NBA {year} player logs')
                    games=saved['games'];stats=saved['stats']
                else:
                    report(f'Checking completed NBA {year} games')
                    games=_request('/games',api_key,{'league':'standard','season':year},root)
                    if saved:
                        stats=list(saved['stats'])
                        loaded=set(saved.get('loaded_games',[]))
                        recent=[g for g in games if str((g.get('status') or {}).get('short'))=='3' and str(g.get('id')) not in loaded and str(g.get('stage'))!='1']
                        for i,game in enumerate(recent,1):
                            report(f'Updating final NBA game statistics {i} of {len(recent)}')
                            new=_request('/players/statistics',api_key,{'game':game['id']},root)
                            if not new: continue
                            stats.extend(new);loaded.add(str(game['id']))
                    else:
                        report(f'Downloading shared NBA {year} player history')
                        stats=_request('/players/statistics',api_key,{'season':year},root)
                        loaded={str((r.get('game') or {}).get('id')) for r in stats}
                    if games and stats:
                        _save(path,{'checked_at':now.isoformat(),'games':games,'stats':stats,'loaded_games':sorted(loaded)})
                all_games.extend(games);all_stats.extend(stats)
            except Exception as exc:
                errors.append(str(exc))
                # Only a previously complete response is safe to reuse.
                if saved:
                    all_games.extend(saved['games']);all_stats.extend(saved['stats'])
                if any(word in str(exc) for word in ('quota','budget')): break
    normalized=normalize_player_stats(all_stats,all_games,now)
    matched={name:normalized.get(player_key(name),[]) for name in names}
    for name,rows in matched.items():
        if not rows: errors.append('No uniquely matched completed API-Sports NBA game logs for '+name+'.')
    return matched,errors
