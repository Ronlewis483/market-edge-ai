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
    last_season = None
    def features(game, cutoff):
        def team(tid):
            prior = history[tid][-12:]
            # Explicit prior support; new/unseen teams do not get invented form.
            if len(prior)<3:return None
            n = len(prior)
            return ((sum(r[1] for r in prior)+2)/(n+4),sum(r[2]-r[3] for r in prior)/n,
                sum(r[2] for r in prior)/n,sum(r[3] for r in prior)/n,
                min(28.,max(0.,(game['start']-prior[-1][0]).total_seconds()/86400)))
        h,a = team(game['home']),team(game['away'])
        if h is None or a is None:return None
        return [(elo[game['home']]-elo[game['away']])/400.,h[0]-a[0],(h[1]-a[1])/20.,
                (h[2]-a[2])/20.,(h[3]-a[3])/20.,(h[4]-a[4])/14.,0. if game['neutral'] else 1.]
    # A final observed now may support a future input. Historical inputs use only
    # finals whose conservative 24-hour availability precedes that input day.
    by_day = defaultdict(list)
    completed = []
    upcoming = []
    for game in games.values():
        if game['completed'] and game['start']<now:
            try:
                hs,aws = float(game['home_score']),float(game['away_score'])
                if not math.isfinite(hs+aws) or min(hs,aws)<0 or hs==aws:continue
                game.update(home_score=hs,away_score=aws)
                by_day[game['start'].date()].append(game)
            except (ValueError,TypeError):continue
        elif not game['completed'] and now<game['start']<=now+timedelta(days=7):upcoming.append(game)
    pending = []
    def apply(game):
        hs,aws = game['home_score'],game['away_score'];win=float(hs>aws)
        h,a = game['home'],game['away']
        expectation = 1/(1+10**(-((elo[h]-elo[a])+(0 if game['neutral'] else 55))/400))
        delta=20*(win-expectation);elo[h]+=delta;elo[a]-=delta
        history[h].append((game['start'],win,hs,aws));history[a].append((game['start'],1-win,aws,hs))
    for day in sorted(by_day):
        boundary = datetime.combine(day,datetime.min.time(),tzinfo=timezone.utc)
        ready = [g for g in pending if g['start']+timedelta(hours=24)<boundary]
        pending = [g for g in pending if g not in ready]
        for game in sorted(ready,key=lambda g:(g['start'],g['id'])):apply(game)
        for game in sorted(by_day[day],key=lambda g:(g['start'],g['id'])):
            x = features(game,boundary)
            if x is not None:
                completed.append(dict(game,x=x,y=int(game['home_score']>game['away_score']),label_time=game['start']+timedelta(hours=24)))
            pending.append(game)
    # All completed finals in this response were observed before live inference.
    for game in sorted(pending,key=lambda g:(g['start'],g['id'])):apply(game)
    future = []
    for game in sorted(upcoming,key=lambda g:(g['start'],g['id'])):
        x=features(game,now)
        if x is not None:future.append(dict(game,x=x))
    return completed, future, len(upcoming)-len(future)


def attach_markets(predictions, market_games, teams):
    aliases=defaultdict(set)
    for g in predictions:
        aliases[name_key(g['home_team'])].add(str(g['home_id']));aliases[name_key(g['away_team'])].add(str(g['away_id']))
    for team in teams:
        if team.get('id') is None:continue
        identity=str(team['id'])
        names=[team.get('school'),team.get('abbreviation'),team.get('altName1'),team.get('altName2'),team.get('altName3')]
        if team.get('school') and team.get('mascot'):names.append(str(team['school'])+' '+str(team['mascot']))
        names.extend(team.get('alternateNames',[]) or [])
        for n in names:
            if n:aliases[name_key(n)].add(identity)
    def resolve(name):
        key=name_key(name)
        ids=aliases.get(key,set())
        # If no explicit alias exists, accept only one unambiguous school-prefix.
        if not ids:
            matches=[(len(alias),values) for alias,values in aliases.items() if len(alias)>=5 and key.startswith(alias)]
            if matches:
                longest=max(length for length,_ in matches)
                ids=set().union(*(values for length,values in matches if length==longest))
        return next(iter(ids)) if len(ids)==1 else None
    output=[]
    for g in predictions:
        matches=[m for m in market_games if resolve(m['home_team'])==g['home_id'] and resolve(m['away_team'])==g['away_id']
                 and abs((stamp(m['commence_time'])-stamp(g['commence_time'])).total_seconds())<=900]
        fields=dict(moneylines=[],spreads=[],totals=[],market_favorite=None,favorite_probability=None,home_probability=None,away_probability=None)
        if len(matches)==1:
            fields.update({k:matches[0][k] for k in fields})
        output.append(dict(g,**fields))
    return output


def update_forecast_ledger(games, records, now):
    """Grade frozen pregame forecasts using already fetched exact CFBD finals."""
    target=Path('ncaaf_history_cache/forecast_results_v1.json')
    try:
        saved=json.loads(target.read_text())
        ledger=saved['records'] if saved.get('schema')==1 else {}
        if not isinstance(ledger,dict):ledger={}
    except (OSError,ValueError,TypeError,KeyError):ledger={}
    finals={str(r.get('id')):r for r in records if r.get('completed') is True}
    for game in games:
        if stamp(game['commence_time'])<=now:continue
        identity=game['game_id']+'|'+game['model_id']
        ledger.setdefault(identity,dict(game,captured_at=now.isoformat(),status='Pending'))
    for row in ledger.values():
        if row.get('status')!='Pending':continue
        final=finals.get(row['game_id'])
        if not final:continue
        at,start,observed_start=stamp(row['captured_at']),stamp(row['commence_time']),stamp(final.get('startDate'))
        if not at or not start or not observed_start or not at<start or observed_start>=now:continue
        if abs((observed_start-start).total_seconds())>900:continue
        if str(final.get('homeId'))!=row['home_id'] or str(final.get('awayId'))!=row['away_id']:continue
        try:
            hs,aws=float(final['homePoints']),float(final['awayPoints'])
            if not math.isfinite(hs+aws) or min(hs,aws)<0 or hs==aws:continue
        except (TypeError,ValueError,KeyError):continue
        y=int(hs>aws)
        row.update(status='Graded',home_won=y,label_observed_at=now.isoformat(),
            correct=int((row['model_home_probability']>=.5)==bool(y)))
    counts=defaultdict(lambda:{'graded_games':0,'correct':0,'pending_games':0})
    for r in ledger.values():
        c=counts[r['model_id']]
        if r.get('status')=='Graded':c['graded_games']+=1;c['correct']+=r['correct']
        else:c['pending_games']+=1
    try:
        target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps({'schema':1,'records':ledger},allow_nan=False));temp.replace(target)
    except OSError:pass
    return dict(counts)


def generate_predictions(cfbd_key, odds_key=None, now=None):
    now=now or datetime.now(timezone.utc)
    records,teams,issues=load_history(cfbd_key,now)
    from dual_agent.ncaaf_connected import fit_connected
    result=fit_connected(records,now)
    market_games=[]
    if odds_key:
        try:
            from dual_agent.ncaaf_station import load_markets
            market_games=load_markets(odds_key)['games']
        except Exception as exc:issues.append('Market comparison unavailable: '+str(exc))
    result['games']=attach_markets(result['games'],market_games,teams)
    result.update(checked_at=now.isoformat(),issues=issues)
    result['prospective_results'] = update_forecast_ledger(result['games'], records, now)
    # Freeze generated predictions without retraining, settling or placing bets.
    try:
        root=Path('ncaaf_history_cache/forecasts');root.mkdir(parents=True,exist_ok=True)
        target=root/(now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        target.write_text(json.dumps(result,allow_nan=False))
    except OSError:result['issues'].append('Forecast is displayed; local snapshot saving unavailable.')
    return result
