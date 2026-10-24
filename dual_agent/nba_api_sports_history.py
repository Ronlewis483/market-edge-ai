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


def _provider_error(payload, key):
    """Keep the actual provider reason, but redact keys, tokens and URLs."""
    import re
    details = payload.get('errors') if isinstance(payload,dict) else None
    if not details: return ''
    text = json.dumps(details,ensure_ascii=False) if isinstance(details,(dict,list)) else str(details or '')
    text = text.replace(key.strip(),'[REDACTED]') if key.strip() else text
    text = re.sub(r'https?://[^\s\"<>]+','[URL]',text)
    text = re.sub(r'(?i)(api[_-]?key|token|authorization)\s*[:=]\s*[^,;\s]+',r'\1=[REDACTED]',text)
    return text[:700]


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
    try: payload = response.json()
    except ValueError: payload = {}
    reason = _provider_error(payload,key)
    context = path+' '+json.dumps(params,sort_keys=True)
    if response.status_code!=200 or reason:
        message = f'API-Sports NBA HTTP {response.status_code} · {context}'
        raise RuntimeError(message+' · '+(reason or 'The provider did not return a usable JSON response.'))
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


def load_histories(names, api_key, progress=None, games=None):
    """Cache explicitly scoped team-season logs; update only newly final games."""
    if not api_key: raise ValueError('Set API_SPORTS_KEY in Streamlit Secrets.')
    if not games: raise ValueError('NBA matchup teams are required for API-Sports history requests.')
    now=pd.Timestamp.now(tz='UTC'); season=now.year if now.month>=9 else now.year-1
    fingerprint=hashlib.sha256(api_key.strip().encode()).hexdigest()[:12]
    root=Path('.market_edge_cache')/'api_sports_nba'/fingerprint
    root.mkdir(parents=True,exist_ok=True)
    all_stats=[]; all_games=[]; errors=[]
    def report(message):
        if progress: progress(message)
    def fresh(saved,closed=False):
        if not saved: return False
        checked=pd.to_datetime(saved.get('checked_at'),utc=True,errors='coerce')
        return closed or (pd.notna(checked) and now-checked<pd.Timedelta(minutes=10))
    def team_key(name): return player_key(name).replace('losangeles','la')
    ordered_games=sorted(games,key=lambda game:str(game.get('commence_time',game.get('start_time',''))))
    labels=list(dict.fromkeys(game[side+'_team'] for game in ordered_games for side in ('away','home')))
    with _LOCK:
        team_path=root/'teams_v2.json'; teams_saved=_read(team_path)
        try:
            if fresh(teams_saved): teams=teams_saved['teams']
            else:
                report('Resolving API-Sports NBA team identities')
                teams=_request('/teams',api_key,{'league':'standard'},root)
                if teams: _save(team_path,{'checked_at':now.isoformat(),'teams':teams})
        except Exception as exc:
            return {name:[] for name in names},[str(exc)]
        team_index={}
        for team in teams:
            if team.get('id') is not None and team.get('name'):
                team_index.setdefault(team_key(team['name']),set()).add(int(team['id']))
        wanted=[]
        for label in labels:
            ids=team_index.get(team_key(label),set())
            if len(ids)==1: wanted.append((label,next(iter(ids))))
            else: errors.append('No unique API-Sports NBA team identity for '+label+'.')
        fatal=False
        for year in (season-1,season):
            legacy=_read(root/f'season_{year}_v1.json')
            if legacy and fresh(legacy,year<season):
                all_games.extend(legacy['games']);all_stats.extend(legacy['stats'])
                report(f'Using saved API-Sports NBA {year} history')
                continue
            game_path=root/f'games_{year}_v2.json'; saved_games=_read(game_path)
            try:
                if fresh(saved_games,year<season): season_games=saved_games['games']
                else:
                    report(f'Checking completed NBA {year} games')
                    season_games=_request('/games',api_key,{'league':'standard','season':year},root)
                    if season_games: _save(game_path,{'checked_at':now.isoformat(),'games':season_games})
                all_games.extend(season_games)
            except Exception as exc:
                errors.append(str(exc))
                if saved_games: all_games.extend(saved_games['games'])
                if any(token in str(exc) for token in ('HTTP 401','HTTP 403','HTTP 429','budget','quota')): break
                continue
            for i,(label,tid) in enumerate(wanted,1):
                path=root/f'team_{tid}_{year}_v2.json'; saved=_read(path)
                try:
                    if fresh(saved,year<season):
                        stats=saved['stats']
                    else:
                        report(f'Loading NBA {year} player logs · {label} · team {i} of {len(wanted)}')
                        if saved:
                            stats=list(saved['stats']);loaded=set(saved.get('loaded_games',[]))
                            recent=[g for g in season_games if str((g.get('status') or {}).get('short'))=='3'
                                and str(g.get('stage'))!='1' and str(g.get('id')) not in loaded
                                and any(str((g.get('teams') or {}).get(side,{}).get('id'))==str(tid) for side in ('home','visitors'))]
                            for game in recent:
                                rows=_request('/players/statistics',api_key,{'game':game['id']},root)
                                if rows: stats.extend(rows);loaded.add(str(game['id']))
                        else:
                            # Explicit team + season scope; never issue a season-only stats request.
                            stats=_request('/players/statistics',api_key,{'team':tid,'season':year},root)
                            loaded={str((r.get('game') or {}).get('id')) for r in stats}
                        if stats: _save(path,{'checked_at':now.isoformat(),'stats':stats,'loaded_games':sorted(loaded)})
                    all_stats.extend(stats)
                except Exception as exc:
                    errors.append(str(exc))
                    if saved: all_stats.extend(saved['stats'])
                    if any(token in str(exc) for token in ('HTTP 401','HTTP 403','HTTP 429','budget','quota')):
                        fatal=True;break
            if fatal: break
    normalized=normalize_player_stats(all_stats,all_games,now)
    matched={name:normalized.get(player_key(name),[]) for name in names}
    # A failed source request is the primary reason. Do not bury it under per-player errors.
    if not errors:
        for name,rows in matched.items():
            if not rows: errors.append('No uniquely matched completed API-Sports NBA game logs for '+name+'.')
    return matched,list(dict.fromkeys(errors))
