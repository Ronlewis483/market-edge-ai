"""Selected-book NFL ticket collection, including explicit live estimates.

Approved market/phase pairs use saved neural weights and historical residual
calibration. Other pairs retain the existing estimator or the explicitly
uncalibrated live clock approximation. No wagers are executed. Feed failures
and unsupported live stats stay unknown.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
import math
import threading
import time
import unicodedata
import re
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, nbinom, poisson
from dual_agent.odds_request_cache import get as odds_get
from dual_agent.preferred_sportsbooks import allowed
from dual_agent.nfl_player_props import MARKETS, normalize_props, canonical_name
from dual_agent.nfl_player_prop_prediction_pipeline import _estimate_prop
from dual_agent.nfl_live_context import team_code
from dual_agent.nfl_prop_neural import estimate as neural_estimate

BASE = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl'
ESPN = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/'
CORE = ['Passing yards','Rushing yards','Receiving yards','Receptions',
        'Passing attempts','Passing completions','Rushing attempts',
        'Passing touchdowns','Rushing touchdowns','Receiving touchdowns',
        'Interceptions thrown','Rushing + receiving yards','Passing + rushing yards',
        'Passing + rushing + receiving yards','Rushing + receiving touchdowns',
        'Passing + rushing + receiving touchdowns','Anytime touchdown']
FIELDS = {'Passing yards':('passingYards',),'Rushing yards':('rushingYards',),
          'Receiving yards':('receivingYards',),'Receptions':('receptions',),
          'Passing attempts':('passingAttempts',),'Passing completions':('completions',),
          'Rushing attempts':('rushingAttempts',),'Passing touchdowns':('passingTouchdowns',),
          'Rushing touchdowns':('rushingTouchdowns',),'Receiving touchdowns':('receivingTouchdowns',),
          'Interceptions thrown':('interceptions',),
          'Rushing + receiving yards':('rushingYards','receivingYards'),
          'Passing + rushing yards':('passingYards','rushingYards'),
          'Passing + rushing + receiving yards':('passingYards','rushingYards','receivingYards'),
          'Rushing + receiving touchdowns':('rushingTouchdowns','receivingTouchdowns'),
          'Passing + rushing + receiving touchdowns':('passingTouchdowns','rushingTouchdowns','receivingTouchdowns')}
_http_cache = {}
_http_lock = threading.Lock()
_history_lock = threading.Lock()


def name_key(value):
    value=re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)$','',str(value).strip(),flags=re.I)
    return canonical_name(unicodedata.normalize('NFKD',value))


def _json(url, params=None, ttl=30):
    identity=(url,repr(sorted((params or {}).items())))
    with _http_lock:
        previous=_http_cache.get(identity)
        if previous and time.monotonic()-previous[0]<ttl: return previous[1],previous[2]
    response=requests.get(url,params=params,timeout=(4,10))
    if response.status_code!=200: raise ValueError('Live statistics feed HTTP '+str(response.status_code))
    payload=response.json();checked=pd.Timestamp.now(tz='UTC').isoformat()
    with _http_lock: _http_cache[identity]=(time.monotonic(),payload,checked)
    return payload,checked


def _odds(path,key,book,markets=None,live=False):
    params={'apiKey':key,'dateFormat':'iso','bookmakers':book,'oddsFormat':'american'}
    if markets: params['markets']=','.join(markets)
    response=odds_get(BASE+path,params=params,cache_ttl=45 if live else 300,timeout=(4,12))
    if response.status_code!=200:
        raise ValueError('Sportsbook feed HTTP '+str(response.status_code))
    return response.json(),response.headers.get('X-Market-Edge-Fetched-At')


def load_ticket_history():
    """Reuse the existing estimator history; keep a six-hour local checkpoint."""
    with _history_lock:
        path=Path('nfl_ticket_history/history_v1.parquet')
        if path.exists() and time.time()-path.stat().st_mtime<21600:
            try:
                saved = pd.read_parquet(path)
                if saved.attrs.get('player_alias_schema') == 1:
                    return saved
            except Exception: pass
        from dual_agent.nfl_player_props_ui import cached_player_history
        now=pd.Timestamp.now(tz='UTC');year=now.year if now.month>=8 else now.year-1
        history=cached_player_history((year-1,year))
        try:
            path.parent.mkdir(exist_ok=True);temp=path.with_suffix('.tmp.parquet')
            history.to_parquet(temp,index=False);temp.replace(path)
        except Exception: pass
        return history


def history_index(history,now):
    """Index once, rather than scanning the full history for every offered line."""
    if history is None or history.empty: return {}
    h=history.copy();h['game_time']=pd.to_datetime(h['game_time'],utc=True,errors='coerce')
    h['value']=pd.to_numeric(h['value'],errors='coerce')
    h=h.loc[(h.game_time<now)&np.isfinite(h.value)].sort_values('game_time')
    h['ticket_key']=h['player'].map(name_key)
    result = {key:group.drop_duplicates('game_time',keep='last').tail(12)
              for key,group in h.groupby(['ticket_key','market'],sort=False)}
    if 'provider_player_id' in h:
        from dual_agent.player_aliases import resolver_from_history
        result['__resolver__'] = resolver_from_history(h)
    return result


def board_games(payload):
    result={}
    for event in payload.get('events',[]):
        competitions=event.get('competitions',[])
        if not competitions: continue
        c=competitions[0];teams={r.get('homeAway'):team_code(r.get('team',{}).get('displayName')) for r in c.get('competitors',[])}
        if not teams.get('home') or not teams.get('away'): continue
        date=pd.to_datetime(event.get('date'),utc=True,errors='coerce')
        result[(teams['home'],teams['away'])]={'id':str(event['id']),'date':date,
            'state':event.get('status',c.get('status',{})).get('type',{}).get('state')}
    return result


def parse_summary(payload,checked,event):
    competitions=payload.get('header',{}).get('competitions',[])
    if len(competitions)!=1: raise ValueError('Live game identity unavailable')
    c=competitions[0]
    if str(c.get('id'))!=str(event['id']): raise ValueError('Live event id mismatch')
    status=c.get('status',{});kind=status.get('type',{})
    if kind.get('state')!='in' or kind.get('completed'): raise ValueError('Game is not active')
    period=int(status.get('period',0))
    if period not in (1,2,3,4): raise ValueError('Overtime requires a separate remaining-time model')
    clock=str(status.get('displayClock','')).split(':')
    if len(clock)!=2: raise ValueError('Live game clock unavailable')
    minutes,seconds=int(clock[0]),int(clock[1])
    if not 0<=minutes*60+seconds<=900 or not 0<=seconds<60: raise ValueError('Invalid game clock')
    elapsed=(period-1)*900+900-(minutes*60+seconds)
    if not 60<=elapsed<3600: raise ValueError('Need a running regulation game clock')
    # Name and team form the player identity; no initial-only guessing.
    players={};blocked=set()
    for group in payload.get('injuries',[]):
        for r in group.get('injuries',[]):
            if str(r.get('status','')).lower() in ('out','doubtful','inactive','suspended'):
                blocked.add(str(r.get('athlete',{}).get('id')))
    for team in payload.get('boxscore',{}).get('players',[]):
        code=team_code(team.get('team',{}).get('displayName'))
        for category in team.get('statistics',[]):
            keys=category.get('keys',[])
            for item in category.get('athletes',[]):
                athlete=item.get('athlete',{});pid=str(athlete.get('id',''))
                identity=(code,name_key(athlete.get('displayName')))
                row=players.setdefault(identity,{'id':pid,'stats':{},'seen':False,'blocked':False})
                if row['id']!=pid: row['blocked']=True
                row['blocked']=row['blocked'] or pid in blocked or bool(item.get('didNotPlay'))
                for key,value in zip(keys,item.get('stats',[])):
                    try:
                        if key=='completions/passingAttempts':
                            made,attempted=map(float,str(value).split('/'))
                            row['stats'].update(completions=made,passingAttempts=attempted)
                        else: row['stats'][key]=float(value)
                    except (ValueError,TypeError): pass
                row['seen']=any(row['stats'].get(k,0)>0 for k in ('passingAttempts','rushingAttempts','receptions','receivingTargets'))
    return {'players':players,'elapsed':elapsed,'remaining':3600-elapsed,'period':period,
            'clock':status['displayClock'],'checked_at':checked,'event_id':str(event['id'])}


def live_estimate(values,current,line,market,side,elapsed):
    """Uncalibrated remaining-production forecast with strict pushes.

Blend one game of prior production with observed game exposure. A count
distribution is used for counts; a discrete normal approximation allows
negative yardage. This does not model possession, score effects or overtime.
"""
    values=np.asarray(values,dtype=float)
    if len(values)<8 or not np.isfinite(values).all() or not np.isfinite([current,line,elapsed]).all(): return None
    if not 60<=elapsed<3600: return None
    elapsed_fraction=elapsed/3600;remaining=1-elapsed_fraction
    prior=float(np.average(values,weights=np.linspace(1,2,len(values))))
    # Clock exposure shrinks noisy in-game pace toward the player's history.
    full_rate=(prior+current)/(1+elapsed_fraction)
    mean=full_rate*remaining
    low=math.ceil(line-current)-1;high=math.floor(line-current)
    if 'yards' in market.lower():
        scale=max(float(np.std(values,ddof=1)),abs(prior)*.25,5.0)
        sd=math.sqrt(scale*scale*remaining+(.35*abs(mean))**2)
        under=float(norm.cdf(low+.5,loc=mean,scale=sd))
        over=float(norm.sf(high+.5,loc=mean,scale=sd))
    else:
        # Already attained nondecreasing count lines are settled observations,
        # not newly actionable "100%" recommendations.
        if current>line: return None
        mean=max(0.,mean)
        variance=max(mean,float(np.var(values,ddof=1))*remaining+(.35*mean)**2)
        if mean<=0: return None
        dist=(nbinom(mean*mean/(variance-mean),mean/variance) if variance>mean+1e-9 else poisson(mean))
        under=float(dist.cdf(low));over=float(dist.sf(high))
    mass=over+under
    if not mass>1e-8: return None
    p=over/mass if side in ('OVER','YES') else under/mass
    return {'historical_support':min(.99,max(.01,p)),
            'over_support':min(.99,max(.01,over/mass)),
            'under_support':min(.99,max(.01,under/mass)),
            'projection':current+mean,'sample_size':len(values),'estimated_push_chance':max(0.,1-mass),
            'estimate_method':'Live clock + observed production + prior history; uncalibrated'}


def _score(frame,event,index,snapshot,checked,now):
    rows=[];reasons=Counter();lookup={v:k for k,v in MARKETS.items()}
    active=snapshot is not None
    if active and not pd.Timedelta(0)<=now-pd.Timestamp(snapshot['checked_at'])<=pd.Timedelta(seconds=90):
        return [],{'live statistics snapshot expired':len(frame)}
    # Memoize estimates across duplicate books/quotes and alternatives.
    memo={}
    for quote in frame.to_dict('records'):
        market=lookup.get(quote['market_key']);player=quote['player'];side=str(quote['side']).upper()
        if not market: reasons['unsupported market']+=1;continue
        if side not in ('OVER','UNDER','YES','NO'): continue
        kickoff=pd.Timestamp(quote['game_time']);line=float(quote['line'])
        history_name = player
        if '__resolver__' in index:
            matched = index['__resolver__'].resolve(player)
            if matched is None:
                reasons[market+': player identity missing or ambiguous']+=1;continue
            history_name = matched['name']
        sample=index.get((name_key(history_name),market))
        if sample is None: reasons[market+': no exact player history']+=1;continue
        vals=sample.loc[sample.game_time<min(kickoff,now),'value'].to_numpy(float)
        if len(vals)<8: reasons[market+': fewer than 8 prior games']+=1;continue
        estimate_key=(name_key(player),market,line,side)
        live_fields={}
        if active:
            if market not in FIELDS: reasons[market+': no supported live statistic']+=1;continue
            found=[v for (team,name),v in snapshot['players'].items()
                   if name==name_key(player) and team in (team_code(event['home_team']),team_code(event['away_team']))]
            if len(found)!=1 or not found[0]['seen'] or found[0]['blocked']:
                reasons[market+': participation unverified or unavailable']+=1;continue
            stats=found[0]['stats'];fields=FIELDS[market]
            # Missing a category in a box score is not proof of a zero.
            if any(k not in stats for k in fields): reasons[market+': live statistic missing']+=1;continue
            current=sum(stats[k] for k in fields)
            if estimate_key not in memo:
                try: trained=neural_estimate(vals,current,line,market,side,snapshot['elapsed'])
                except (ValueError,OSError,KeyError): trained=None
                memo[estimate_key]=trained or live_estimate(vals,current,line,market,side,snapshot['elapsed'])
            live_fields={'game_phase':'live','live_current':current,'live_period':snapshot['period'],
                'live_clock':snapshot['clock'],'live_remaining_seconds':snapshot['remaining'],
                'live_snapshot_at':snapshot['checked_at'],'live_event_id':snapshot['event_id']}
        elif estimate_key not in memo:
            try: trained=neural_estimate(vals,0.,line,market,side,0)
            except (ValueError,OSError,KeyError): trained=None
            memo[estimate_key]=trained or _estimate_prop(vals,line,market,side=side)
        estimate=memo.get(estimate_key)
        if estimate is None: reasons[market+': insufficient usable history or resolved live line']+=1;continue
        offer={'bookmaker_key':quote['bookmaker_key'],'american_odds':quote.get('american_odds'),
            'last_update':quote.get('last_update'),'observed_at':checked,'side':side,'line':line,'market':market}
        updated=pd.to_datetime(quote.get('last_update'),utc=True,errors='coerce')
        age=pd.Timedelta(seconds=90) if active else pd.Timedelta(minutes=30)
        if pd.isna(updated) or not pd.Timedelta(0)<=now-updated<=age:
            reasons[market+': expired sportsbook quote']+=1;continue
        if active and updated<kickoff: reasons[market+': quote predates kickoff']+=1;continue
        row=dict(quote);row.update(estimate);row.update(market=market,model_pick=side,line=line)
        row.update(sportsbook_offers=[offer],game_phase='pregame',
            historical_average=float(np.mean(vals)),quote_updated_at=quote.get('last_update'),
            historical_hit_rate=float(np.mean(vals>line) if side in ('OVER','YES') else np.mean(vals<line)),
            ticket_book=quote['bookmaker_key'])
        row.update(live_fields)
        if 'estimate_method' not in row: row['estimate_method']='Existing NFL historical estimator (neural model not promoted for this market/phase)'
        rows.append(row)
    return rows,dict(reasons)


def collect_ticket_props(key,book,include_specialty=False,publish=None):
    """No injuries/weather/spreads/winner retraining; one selected bookmaker.

    Completed games are excluded using explicit status. Active games are
    scored with current box-score data rather than discarded on kickoff time.
    """
    if not key or (book != 'all' and not allowed(book)): raise ValueError('Choose configured sportsbooks and API key')
    from dual_agent.preferred_sportsbooks import PLATFORMS, BOOKMAKERS
    requested_books = set(PLATFORMS) if book == 'all' else {book}
    request_book = BOOKMAKERS if book == 'all' else book
    started=time.monotonic();now=pd.Timestamp.now(tz='UTC');all_rows=[];issues=[];diagnostics=[]
    def send(stage):
        payload={'nfl_player_props':pd.DataFrame(all_rows),'ticket_book':book,
            'ticket_collection_stage':stage,'ticket_diagnostics':list(diagnostics),
            'ticket_issues':list(issues),'ticket_checked_at':pd.Timestamp.now(tz='UTC').isoformat()}
        if publish: publish(payload)
        return payload
    with ThreadPoolExecutor(max_workers=3) as setup:
        history_job=setup.submit(load_ticket_history)
        events_job=setup.submit(_odds,'/events',key,request_book)
        board_job=setup.submit(_json,ESPN+'scoreboard',
            {'dates':(now-pd.Timedelta(days=1)).strftime('%Y%m%d')+'-'+now.strftime('%Y%m%d'),'limit':1000},30)
        send('Loading saved player history and game status')
        events,_=events_job.result()
        try: board=board_games(board_job.result()[0])
        except Exception: board={};issues.append('Active-game status unavailable; future games can still be scored.')
        index=history_index(history_job.result(),now)
    selected=[]
    for event in events:
        kickoff=pd.to_datetime(event.get('commence_time'),utc=True,errors='coerce')
        if pd.isna(kickoff) or kickoff>now+pd.Timedelta(days=7): continue
        game=board.get((team_code(event.get('home_team')),team_code(event.get('away_team'))))
        exact=game and pd.notna(game['date']) and abs(game['date']-kickoff)<=pd.Timedelta(minutes=15)
        if exact and game['state']=='post': continue
        live=kickoff<=now
        if live and (not exact or game['state']!='in'):
            diagnostics.append({'game':event.get('id'),'reason':'Started game needs verified active status'});continue
        selected.append((event,game if live else None))
    markets=list(MARKETS.values()) if include_specialty else [MARKETS[k] for k in CORE if k in MARKETS]
    first=[MARKETS[k] for k in ('Passing yards','Rushing yards','Receiving yards','Receptions','Passing touchdowns','Passing attempts','Passing completions','Rushing attempts')]
    batches=[first]+[markets[i:i+8] for i in range(0,len(markets),8)]
    # A category is requested once; offensive totals are the first batch.
    seen=set();batches=[[m for m in batch if m not in seen and not seen.add(m)] for batch in batches]
    batches=[b for b in batches if b]
    def fetch(event,game,batch):
        snapshot=None
        if game:
            summary,checked=_json(ESPN+'summary',{'event':game['id']},30)
            snapshot=parse_summary(summary,checked,game)
        payload,checked=_odds('/events/'+event['id']+'/odds',key,request_book,batch,live=bool(game))
        if not checked: raise ValueError('Original quote retrieval time unavailable; update odds_request_cache.py')
        # A response must match the requested event and its exact participants.
        if payload.get('id')!=event['id'] or payload.get('home_team')!=event['home_team'] or payload.get('away_team')!=event['away_team']:
            raise ValueError('Sportsbook event identity mismatch')
        frame=normalize_props(payload)
        if not frame.empty: frame=frame.loc[frame.bookmaker_key.isin(requested_books)]
        return _score(frame,event,index,snapshot,checked,pd.Timestamp.now(tz='UTC')) if not frame.empty else ([],{'no sportsbook lines in this batch':1})
    # Active games first, then nearest upcoming game. Each batch publishes
    # immediately; no wait for all games or for a full historical model refit.
    selected.sort(key=lambda item:(item[1] is None,item[0]['commence_time']))
    send('Collecting configured-book offensive props, including active games')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in batches:
            jobs={pool.submit(fetch,event,game,batch):event for event,game in selected}
            for job in as_completed(jobs):
                event=jobs[job]
                try:
                    rows,excluded=job.result();all_rows.extend(rows)
                    diagnostics.append({'game':event['away_team']+' @ '+event['home_team'],
                        'categories':list(batch),'estimates':len(rows),'excluded':excluded})
                except Exception as exc:
                    issues.append(event['away_team']+' @ '+event['home_team']+': '+str(exc) if isinstance(exc,ValueError)
                                  else event['away_team']+' @ '+event['home_team']+': '+type(exc).__name__)
                send('Ready estimates: '+str(len(all_rows))+'; collecting remaining configured-book markets')
    result=send('Finished in '+str(round(time.monotonic()-started,1))+' seconds')
    result['ticket_scan_complete']=True
    return result
