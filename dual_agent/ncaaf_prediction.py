"""Independent FBS winner baseline from CFBD completed games; no NFL weights.

Whole UTC days are processed together so same-day outcomes cannot enter earlier
inputs. Historical label timing is conservatively approximated at kickoff+24h;
this is retrospective evaluation, not an untouched prospective benchmark.
"""
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import math
import threading
import time
import unicodedata
import re

_LOCK = threading.RLock()
_MODELS = OrderedDict()
FEATURES = ['elo_gap','win_rate_gap','margin_gap','scoring_gap','allowed_gap','rest_gap','home_advantage']


def stamp(value):
    try:
        dt = datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (TypeError,ValueError):return None


def name_key(value):
    text = unicodedata.normalize('NFKD',str(value or '')).encode('ascii','ignore').decode().lower()
    return re.sub('[^a-z0-9]','',text)


def _cfbd(path, params, key, ttl):
    if not key:raise ValueError('Add CFBD_API_KEY to app secrets to generate independent NCAA predictions.')
    import requests
    identity = hashlib.sha256(json.dumps([path,params],sort_keys=True).encode()).hexdigest()
    target = Path('ncaaf_history_cache')/(identity+'.json')
    with _LOCK:
        try:
            saved = json.loads(target.read_text())
            if 0<=time.time()-float(saved['saved_at'])<ttl and isinstance(saved['data'],list):return saved['data']
        except (OSError,ValueError,TypeError,KeyError):pass
        response = requests.get('https://api.collegefootballdata.com'+path,
            params=params,headers={'Authorization':'Bearer '+key.strip()},timeout=(5,25))
        if response.status_code!=200:
            raise RuntimeError('CollegeFootballData returned HTTP '+str(response.status_code)+'. Check CFBD_API_KEY, endpoint access and usage allowance.')
        data = response.json()
        if not isinstance(data,list):raise ValueError('Unexpected CollegeFootballData response schema.')
        try:
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps({'saved_at':time.time(),'data':data},allow_nan=False))
            temporary.replace(target)
        except OSError:pass
        return data


def load_history(key, now=None):
    now = now or datetime.now(timezone.utc)
    season = now.year if now.month>=7 else now.year-1
    # Completed seasons are reusable; the current season is checked every 15 min.
    rows, issues = [], []
    for year in range(season-3,season+1):
        try:
            rows.extend(_cfbd('/games',{'year':year,'seasonType':'both','classification':'fbs'},key,900 if year==season else 86400))
        except Exception as exc:
            if year==season:raise
            issues.append('Historical season '+str(year)+': '+str(exc))
    try:teams = _cfbd('/teams/fbs',{'year':season},key,86400)
    except Exception:teams=[];issues.append('Team alias directory unavailable; market matching uses exact school names only.')
    return rows, teams, issues


def _game(row):
    if not isinstance(row,dict) or row.get('id') is None:return None
    start = stamp(row.get('startDate'))
    if start is None or row.get('startTimeTBD') is True:return None
    if row.get('homeId') is None or row.get('awayId') is None:return None
    if row['homeId']==row['awayId']:return None
    if row.get('homeClassification')!='fbs' or row.get('awayClassification')!='fbs':return None
    if not row.get('homeTeam') or not row.get('awayTeam'):return None
    return dict(id=str(row['id']),start=start,home=str(row['homeId']),away=str(row['awayId']),
        home_name=row['homeTeam'],away_name=row['awayTeam'],neutral=row.get('neutralSite') is True,
        completed=row.get('completed') is True,home_score=row.get('homePoints'),away_score=row.get('awayPoints'))


def build_dataset(records, now=None):
    now = now or datetime.now(timezone.utc)
    games = {}
    for row in records:
        game = _game(row)
        if game is not None:games[game['id']] = game
    elo = defaultdict(lambda:1500.)
    history = defaultdict(list)
