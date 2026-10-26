"""Scheduled NBA pregame dataset collector. No Streamlit imports or wager execution.

ALL-STAR endpoints only: games, stats, active players, player injuries.
Supabase Storage holds immutable snapshots, final-result sidecars and a gzip
checkpoint. Run with: python dual_agent/nba_pregame_snapshots.py
"""
import gzip
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
import time
import uuid
from urllib.parse import quote
import requests

UTC = timezone.utc
BDL = 'https://api.balldontlie.io/v1'
ODDS = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
PREFIX = 'nba/pregame_v1/'
FIELDS = ['pts','fga','fgm','fg3a','fg3m','fta','ftm','oreb','dreb','reb','ast','stl','blk','turnover']


def safe_issue(exc):
    # HTTP-library exceptions can embed a URL containing the Odds API key.
    return str(exc) if type(exc) is RuntimeError else type(exc).__name__


def storage_error(response):
    """Classify provider errors without logging URLs, keys or arbitrary bodies."""
    try: payload=response.json()
    except (ValueError, TypeError): payload={}
    if not isinstance(payload,dict): payload={}
    code=str(payload.get('code','')).lower()
    message=str(payload.get('message','')).strip().lower()
    kind=str(payload.get('error','')).strip().lower()
    status=str(payload.get('statusCode',''))
    if code in {'accessdenied','invalidjwt','invalidsignature','jwtExpired'.lower()} or status in {'401','403'} or response.status_code in (401,403):
        return 'access_denied'
    if code=='nosuchbucket' or message=='bucket not found': return 'bucket_missing'
    if code=='nosuchkey' or (status=='404' and message=='object not found' and kind in {'not_found','object not found'}):
        return 'object_missing'
    if code=='invalidkey': return 'invalid_object_path'
    return 'unclassified'


def stamp(value):
    try:
        dt = datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    except (ValueError, TypeError): return None


def number(value):
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError): return None


def minutes(value):
    try:
        if ':' in str(value):
            m,s=map(float,str(value).split(':'))
            if not 0<=s<60: return None
            n=m+s/60
        else: n=number(value)
        return n if n is not None and 0<n<=70 else None
    except (TypeError,ValueError): return None


def final(game):
    return (game.get('status_state')=='final' or str(game.get('status','')).lower().startswith('final')) and not game.get('postponed')


def scheduled(game):
    state=game.get('status_state')
    if state: return state=='scheduled'
    return not final(game) and not game.get('postponed') and game.get('period',0)==0


def played_date(game):
    value=stamp(game.get('date'))
    return value.date() if value else None


def start_time(game):
    # A date-only game date is NEVER a verified tipoff time.
    return stamp(game.get('datetime'))


def team_name(value):
    text=''.join(c.lower() for c in str(value) if c.isalnum())
    return {'laclippers':'losangelesclippers'}.get(text,text)


class Client:
    def __init__(self, env=None):
        self.env=dict(os.environ if env is None else env)
        for key in ('BALLDONTLIE_API_KEY','SUPABASE_URL','SUPABASE_KEY'):
            if not self.env.get(key): raise RuntimeError('Missing required setting: '+key)
        self.http=requests.Session(); self.last=0.; self.calls=0; self.odds_calls=0
        self.max_calls=int(self.env.get('NBA_SNAPSHOT_BDL_BUDGET','100'))
        self.bucket=self.env.get('NBA_SNAPSHOT_BUCKET','market-edge-data')
        self.base=self.env['SUPABASE_URL'].rstrip('/')+'/storage/v1/object/'

    def headers(self):
        key=self.env['SUPABASE_KEY']
        return {'apikey':key,'Authorization':'Bearer '+key}

    def read(self,path,compressed=False):
        response=self.http.get(self.base+'authenticated/'+quote(self.bucket,safe='')+'/'+quote(path,safe='/'),headers=self.headers(),timeout=(5,20))
        if response.status_code!=200:
            category=storage_error(response)
            # Some Supabase versions wrap NoSuchKey/statusCode=404 in HTTP 400.
            # Only an explicit missing OBJECT is a valid first-run checkpoint.
            if category=='object_missing' and response.status_code in (400,404): return None
            raise RuntimeError('Supabase read failed (HTTP '+str(response.status_code)+', '+category+'); existing history was not reset')
        raw=gzip.decompress(response.content) if compressed else response.content
        return json.loads(raw)

    def write(self,path,payload,replace=False,compressed=False):
        raw=json.dumps(payload,allow_nan=False,separators=(',',':')).encode()
        if compressed: raw=gzip.compress(raw)
        headers=dict(self.headers(),**{'Content-Type':'application/gzip' if compressed else 'application/json','x-upsert':'true' if replace else 'false'})
        response=self.http.post(self.base+quote(self.bucket,safe='')+'/'+quote(path,safe='/'),headers=headers,data=raw,timeout=(5,25))
        if response.status_code not in (200,201):
            raise RuntimeError('Supabase write failed (HTTP '+str(response.status_code)+', '+storage_error(response)+')')

    def pages(self,endpoint,params=None):
        result=[];cursor=None;seen=set()
        while True:
            if self.calls>=self.max_calls: raise RuntimeError('BALLDONTLIE run budget reached; completed games remain checkpointed for the next run')
            params=dict(params or {},per_page=100)
            if cursor is not None: params['cursor']=cursor
            delay=max(0.,1.4-(time.monotonic()-self.last))
            if delay: time.sleep(delay)
            self.last=time.monotonic();self.calls+=1
            response=self.http.get(BDL+'/'+endpoint,params=params,headers={'Authorization':self.env['BALLDONTLIE_API_KEY']},timeout=(5,20))
            if response.status_code!=200:
                raise RuntimeError('BALLDONTLIE '+endpoint+' failed (HTTP '+str(response.status_code)+'); no partial page set used')
            payload=response.json();batch=payload.get('data')
            if not isinstance(batch,list): raise ValueError('Invalid BALLDONTLIE '+endpoint+' response')
            result.extend(batch);cursor=(payload.get('meta') or {}).get('next_cursor')
            if cursor is None: return result
            if str(cursor) in seen: raise ValueError('Repeated BALLDONTLIE cursor')
            seen.add(str(cursor))

    def odds(self):
        key=self.env.get('ODDS_API_KEY')
        if not key: return [],{'status':'missing_key'}
        self.odds_calls+=1
        response=self.http.get(ODDS,params={'apiKey':key,'markets':'h2h','oddsFormat':'decimal',
            'dateFormat':'iso','bookmakers':self.env.get('NBA_SNAPSHOT_BOOKMAKERS','draftkings,fanduel,underdog,prizepicks,polymarket')},timeout=(5,20))
        if response.status_code!=200: raise RuntimeError('Odds API failed (HTTP '+str(response.status_code)+')')
        data=response.json()
        if not isinstance(data,list): raise ValueError('Invalid Odds API response')
        return data,{'status':'fetched','checked_at':datetime.now(UTC).isoformat(),
            'requests_remaining':response.headers.get('x-requests-remaining'),
            'last_request_cost':response.headers.get('x-requests-last')}


def history_before(games,cutoff):
    return sorted([g for g in games if final(g) and played_date(g) and played_date(g)<cutoff.date()
        and number(g.get('home_team_score')) is not None and number(g.get('visitor_team_score')) is not None
        and min(float(g['home_team_score']),float(g['visitor_team_score']))>0
        and g['home_team_score']!=g['visitor_team_score']],key=lambda g:(g['date'],str(g['id'])))


def elo_ratings(games):
    ratings={}; counts={}
    for game in games:
        home=game['home_team']['id']; away=game['visitor_team']['id']
        rh,ra=ratings.get(home,1500.),ratings.get(away,1500.)
        expectation=1/(1+10**((ra-rh-60)/400))
        change=20*(int(game['home_team_score']>game['visitor_team_score'])-expectation)
        ratings[home]=rh+change;ratings[away]=ra-change
        counts[home]=counts.get(home,0)+1;counts[away]=counts.get(away,0)+1
    return ratings,counts


def form(games,tid,target_date):
    rows=[]
    for g in games:
        home=g['home_team']['id']==tid
        if not home and g['visitor_team']['id']!=tid: continue
        pf=g['home_team_score'] if home else g['visitor_team_score']
        pa=g['visitor_team_score'] if home else g['home_team_score']
        rows.append({'game_id':g['id'],'date':g['date'],'home':home,'win':int(pf>pa),'margin':pf-pa,
            'points_for':pf,'points_against':pa,'venue_team_id':g['home_team']['id']})
    recent=rows[-8:]; support=rows[-30:]
    if not recent: return {'status':'unknown','games':0}
    last=stamp(recent[-1]['date']).date(); gap=(target_date-last).days
    return {'status':'available','recent_window':8,'recent_games':len(recent),'support_games':len(support),
        'win_rate_8':mean(r['win'] for r in recent),'margin_8':mean(r['margin'] for r in recent),
        'points_for_8':mean(r['points_for'] for r in recent),'points_against_8':mean(r['points_against'] for r in recent),
        'win_rate_30':mean(r['win'] for r in support),'margin_sd_30':pstdev(r['margin'] for r in support),
        'days_since_last_game':gap,'back_to_back':gap==1,'games_last_4_days':sum(0<(target_date-stamp(r['date']).date()).days<=4 for r in rows),
        'home_games_8':sum(r['home'] for r in recent),'last_venue_team_id':recent[-1]['venue_team_id'],
        'travel_distance_km':None,'travel_status':'unknown; neutral sites and travel itineraries not verified',
        'source_game_ids':[r['game_id'] for r in support]}


def team_box(rows,tid):
    selected=[r for r in rows if (r.get('team') or {}).get('id')==tid and minutes(r.get('min')) is not None]
    if not selected: return None
    total_minutes=sum(minutes(r['min']) for r in selected)
    if not 230<=total_minutes<=380: return None
    if any(number(r.get(f)) is None for r in selected for f in FIELDS): return None
    totals={f:sum(float(r[f]) for r in selected) for f in FIELDS}
    poss=totals['fga']+.44*totals['fta']-totals['oreb']+totals['turnover']
    totals.update(team_minutes=total_minutes,estimated_possessions=poss,
        estimated_pace_48=poss/(total_minutes/5)*48,
        effective_fg=(totals['fgm']+.5*totals['fg3m'])/totals['fga'] if totals['fga'] else None)
    return totals


def matchup_stats(games,stats,tid):
    pairs=[]
    for g in games:
        teams=[g['home_team']['id'],g['visitor_team']['id']]
        if tid not in teams: continue
        rows=stats.get(str(g['id']))
        if rows is None: continue
        own=team_box(rows,tid);opp=team_box(rows,teams[1] if teams[0]==tid else teams[0])
        if own is not None and opp is not None:
            expected=g['home_team_score'] if teams[0]==tid else g['visitor_team_score']
            other=g['visitor_team_score'] if teams[0]==tid else g['home_team_score']
            if own['pts']==expected and opp['pts']==other: pairs.append((g,own,opp))
    pairs=pairs[-8:]
    if not pairs: return {'status':'unknown','reason':'No complete prior player-stat box totals cached'}
    return {'status':'derived_estimates','games':len(pairs),'source_game_ids':[g['id'] for g,_,_ in pairs],
        'estimated_pace_48':mean((a['estimated_pace_48']+b['estimated_pace_48'])/2 for _,a,b in pairs),
        'estimated_offensive_rating':mean(100*a['pts']/max((a['estimated_possessions']+b['estimated_possessions'])/2,1) for _,a,b in pairs),
        'estimated_defensive_rating':mean(100*b['pts']/max((a['estimated_possessions']+b['estimated_possessions'])/2,1) for _,a,b in pairs),
        'effective_fg':mean(a['effective_fg'] for _,a,_ in pairs if a['effective_fg'] is not None),
        'rebounds':mean(a['reb'] for _,a,_ in pairs),'turnovers':mean(a['turnover'] for _,a,_ in pairs),
        'note':'Derived from complete prior basic box scores; not official GOAT advanced statistics'}


def rotation(games,stats,tid,active,injuries):
    players={}; team_games=[g for g in games if tid in [g['home_team']['id'],g['visitor_team']['id']]][-8:]
    for g in team_games:
        for row in stats.get(str(g['id']),[]):
            player=row.get('player') or {};pid=player.get('id');m=minutes(row.get('min'))
            if pid is None or m is None or (row.get('team') or {}).get('id')!=tid: continue
            players.setdefault(pid,[]).append((m,row))
    report={}
    for item in injuries or []:
        pid=(item.get('player') or {}).get('id')
        if pid is not None: report.setdefault(pid,[]).append(item)
    result=[]
    for person in active or []:
        if (person.get('team') or {}).get('id')!=tid: continue
        pid=person.get('id');sample=players.get(pid,[]);reports=report.get(pid,[])
        blocked=any(str(r.get('status','')).lower() in {'out','doubtful','inactive','suspended'} for r in reports)
        vals=[m for m,_ in sample]
        result.append({'player_id':pid,'name':(person.get('first_name','')+' '+person.get('last_name','')).strip(),
            'prior_appearances':len(vals),'recent_minutes_if_playing':sum(m*(i+1) for i,m in enumerate(vals))/sum(range(1,len(vals)+1)) if vals else None,
            'minutes_sd':pstdev(vals) if vals else None,'availability_reports':reports,
            'unavailable_reported':blocked,'participation_probability':None,
            'minutes_restriction_reported':any(re.search(r'minutes? restriction|minutes? limit|limited to .*minutes',str(r.get('description','')),re.I) for r in reports),
            'status':'reported unavailable' if blocked else 'participation unconfirmed',
            'replacement_quality':None,'replacement_status':'not trained; no redistribution of absent-player minutes'})
    result.sort(key=lambda p:-(p['recent_minutes_if_playing'] or 0))
    return {'status':'history_based_projections' if active is not None else 'unknown_roster',
        'window':8,'players':result,'confirmed_starters':None,
        'starter_status':'unknown; ALL-STAR does not include confirmed lineups'}


def market_snapshot(game,events,captured):
    start=start_time(game);home=team_name(game['home_team'].get('full_name'));away=team_name(game['visitor_team'].get('full_name'))
    matches=[e for e in events if team_name(e.get('home_team'))==home and team_name(e.get('away_team'))==away
        and stamp(e.get('commence_time')) and abs((stamp(e['commence_time'])-start).total_seconds())<=900]
    if len(matches)!=1: return {'status':'unknown','reason':'No unique odds event matched exact teams and tipoff'}
    offers=[]
    for book in matches[0].get('bookmakers',[]):
        for market in book.get('markets',[]):
            updated=stamp(market.get('last_update') or book.get('last_update'))
            if market.get('key')!='h2h' or updated is None or not 0<=(captured-updated).total_seconds()<=1800 or updated>=start: continue
            prices={team_name(o.get('name')):number(o.get('price')) for o in market.get('outcomes',[])}
            hp,ap=prices.get(home),prices.get(away)
            if hp is None or ap is None or min(hp,ap)<=1: continue
            ih,ia=1/hp,1/ap
            offers.append({'bookmaker':book.get('key'),'home_decimal':hp,'away_decimal':ap,
                'home_no_vig_probability':ih/(ih+ia),'away_no_vig_probability':ia/(ih+ia),
                'updated_at':updated.isoformat()})
    return {'status':'available' if offers else 'unknown','event_id':matches[0].get('id'),'offers':offers,
        'note':'Each margin adjustment uses both teams from the same book; no cross-book price pairing'}


def make_snapshot(game,games,stats,active,injuries,events,source_times,captured):
    start=start_time(game)
    if not scheduled(game) or start is None or captured>=start: raise ValueError('Only verified future scheduled games may be frozen')
    prior=history_before(games,captured);ratings,counts=elo_ratings(prior)
    context={}
    for side in ('home','visitor'):
        team=game[side+'_team'];tid=team['id']
        context[side]={'team':team,'form_and_schedule':form(prior,tid,played_date(game)),
            'opponent_adjusted_strength':{'elo':ratings.get(tid),'prior_games':counts.get(tid,0),
                'method':'Elo K=20, home advantage=60, seeded=1500; descriptive feature, not calibrated win probability'},
            'matchup_performance':matchup_stats(prior,stats,tid),
            'availability_and_rotation':rotation(prior,stats,tid,active,injuries)}
    market=market_snapshot(game,events,captured)
    return {'schema':1,'sport':'NBA','snapshot_id':uuid.uuid4().hex,'game_id':game['id'],
        'scheduled_start':start.isoformat(),'scheduled_date':str(played_date(game)),
        'captured_at':captured.isoformat(),'source_times':source_times,'teams':context,'sportsbook_context':market,
        'confidence_reliability':{'model_probability':None,'probability_reliability':'not established',
            'prediction_join_status':'pending; join frozen model forecasts by exact game identity before evaluation',
            'injury_feed_complete':injuries is not None,'active_roster_complete':active is not None,
            'market_available':market['status']=='available',
            'training_ready_full_package':False,
            'missing':['confirmed pregame starters','trained replacement quality','historical pregame availability backfill','frozen model probability join']},
        'notes':['History excludes same-day results and all future results.',
            'Projected minutes are conditional on playing, not confirmed participation.',
            'No current model weights or recommendations are changed by collecting this snapshot.']}


def due(game,previous,now):
    start=start_time(game)
    if start is None or not scheduled(game) or not now<start<=now+timedelta(days=7): return False
    hours=(start-now).total_seconds()/3600
    interval=timedelta(minutes=30 if hours<=4 else 120 if hours<=24 else 1440)
    last=stamp((previous or {}).get('captured_at'))
    return last is None or now-last>=interval or (previous or {}).get('scheduled_start')!=start.isoformat()


def run(client=None,now=None):
    client=client or Client();now=now or datetime.now(UTC)
    state=client.read(PREFIX+'checkpoint.json.gz',compressed=True)
    if state is None: state={'schema':1,'games':{},'stats':{},'captures':{},'settled':{}}
    if state.get('schema')!=1: raise ValueError('Checkpoint schema mismatch; existing history not reset')
    issues=[];saved=0;settled=0;times={}
    def checkpoint(): client.write(PREFIX+'checkpoint.json.gz',state,replace=True,compressed=True)
    checked=stamp(state.get('history_checked_at'))
    since=now-timedelta(days=400 if checked is None else 7)
    games=client.pages('games',{'start_date':since.date().isoformat(),'end_date':(now+timedelta(days=7)).date().isoformat()})
    for g in games: state['games'][str(g['id'])]=g
    state['history_checked_at']=datetime.now(UTC).isoformat();times['games']=state['history_checked_at']
    all_games=list(state['games'].values());checkpoint()
    targets=[g for g in all_games if due(g,state['captures'].get(str(g['id'])),now)]
    # Prior completed basic stats bootstrap gradually and resume by game ID.
    needed=sorted([g for g in all_games if final(g) and played_date(g) and now.date()-timedelta(days=90)<=played_date(g)<now.date()],key=lambda g:g['date'],reverse=True)
    for g in needed:
        gid=str(g['id'])
        if gid in state['stats']: continue
        if client.calls>=client.max_calls-10: break
        try:
            rows=client.pages('stats',{'game_ids[]':[g['id']]})
            if not rows: raise ValueError('Final game '+gid+' returned no player stats; will retry next run')
            if any(str((r.get('game') or {}).get('id'))!=gid for r in rows):
                raise ValueError('Player statistics do not match requested game '+gid)
            state['stats'][gid]=rows;checkpoint()
        except Exception as exc:
            issues.append(safe_issue(exc));break
    # Settlement is a sidecar; never write actual minutes/results into pregame inputs.
    for g in all_games:
        gid=str(g['id'])
        if not final(g) or gid not in state['captures']: continue
        rows=state['stats'].get(gid)
        score=[g.get('home_team_score'),g.get('visitor_team_score')]
        if any(number(x) is None for x in score) or min(score)<=0 or score[0]==score[1]: continue
        content={'game_id':g['id'],'scheduled_date':str(played_date(g)),'home_score':score[0],'away_score':score[1],
            'home_win':score[0]>score[1],'player_stats':rows,'player_stats_complete':rows is not None}
        digest=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
        if state['settled'].get(gid)==digest: continue
        client.write(PREFIX+'outcomes/'+gid+'/'+digest+'.json',dict(content,observed_at=datetime.now(UTC).isoformat()))
        state['settled'][gid]=digest;settled+=1
    if targets:
        try:
            injuries=client.pages('player_injuries');times['injuries']=datetime.now(UTC).isoformat()
        except Exception as exc: injuries=None;issues.append(safe_issue(exc))
        cached_at=stamp(state.get('active_checked_at'))
        if cached_at and now-cached_at<timedelta(hours=6):
            active=state.get('active');times['active_roster']=state['active_checked_at']
        else:
            try:
                active=client.pages('players/active');times['active_roster']=datetime.now(UTC).isoformat()
                state.update(active=active,active_checked_at=times['active_roster'])
            except Exception as exc: active=None;issues.append(safe_issue(exc))
        try: events,odds_status=client.odds();times['odds']=odds_status
        except Exception as exc: events=[];issues.append(safe_issue(exc))
        for game in targets:
            captured=datetime.now(UTC)
            if start_time(game)<=captured: continue
            payload=make_snapshot(game,all_games,state['stats'],active,injuries,events,times,captured)
            gid=str(game['id']);path=PREFIX+'snapshots/'+gid+'/'+payload['snapshot_id']+'.json'
            client.write(path,payload)
            previous=state['captures'].get(gid,{})
            state['captures'][gid]={'captured_at':captured.isoformat(),'scheduled_start':payload['scheduled_start'],
                'paths':previous.get('paths',[])+[path]};saved+=1
    checkpoint()
    status={'schema':1,'finished_at':datetime.now(UTC).isoformat(),'snapshots_saved':saved,'outcomes_saved':settled,
        'games_with_snapshots':len(state['captures']),'completed_games_cached':len(state['stats']),
        'bdl_requests':client.calls,'odds_requests':client.odds_calls,'issues':issues,
        'scope':'All seven strengthening areas; unavailable fields remain explicitly unknown. Not yet a trained full-package winner model.'}
    client.write(PREFIX+'status.json',status,replace=True)
    print(json.dumps(status),flush=True)
    return status


if __name__=='__main__':
    try: run()
    except Exception as exc:
        # Never print requests exceptions or URLs which could contain API keys.
        print('NBA snapshot collector stopped: '+safe_issue(exc),flush=True)
        raise SystemExit(1)
