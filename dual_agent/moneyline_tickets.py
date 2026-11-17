"""Moneyline ticket previews: saved model forecasts, exact fresh book offers.

Building never collects player history or places/records a wager. Combined
probabilities are independence scenarios, not validated joint forecasts.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations
import json
import math
from pathlib import Path
import re
import pandas as pd

SPORTS={'NFL':'americanfootball_nfl','NBA':'basketball_nba','MLB':'baseball_mlb'}
BOOKS={'DraftKings':'draftkings','FanDuel':'fanduel','Polymarket':'polymarket'}
FILES={'NFL':'sports_prop_workspace/nfl_latest.json',
       'NBA':'nba_player_history_cache/latest_prop_results_v1.json',
       'MLB':'sports_prop_workspace/mlb_latest.json'}
STATES={'NFL':'nfl_prediction_pipeline_result','NBA':'nba_prediction_pipeline_result',
        'MLB':'mlb_live_pipeline_result'}


def records(value):
    if isinstance(value,pd.DataFrame):return value.to_dict('records')
    return list(value) if isinstance(value,(list,tuple)) else []


def number(value):
    if isinstance(value,bool):return None
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError):return None


def stamp(value):
    return pd.to_datetime(value,utc=True,errors='coerce')


def team(value):
    key=re.sub('[^a-z0-9]','',str(value).lower())
    return {'lalakers':'losangeleslakers','laclippers':'losangelesclippers',
        'oaklandathletics':'athletics','sacramentoathletics':'athletics'}.get(key,key)


@lru_cache(maxsize=8)
def _saved_file(path,mtime,size):
    payload=json.loads(Path(path).read_text())
    rows=records(payload.get('all_predictions')) or records(payload.get('predictions'))
    # Cache only the forecast rows, never a second copy of player histories.
    return rows


def saved_forecasts(session,sports):
    sources={}
    for sport in sports:
        payload=session.get(STATES[sport]) or {}
        rows=records(payload.get('all_predictions')) or records(payload.get('predictions'))
        if not rows:
            try:
                path=Path(FILES[sport]);stat=path.stat()
                rows=_saved_file(str(path),stat.st_mtime_ns,stat.st_size)
            except (OSError,ValueError,AttributeError):rows=[]
        sources[sport]=rows
    return sources


def fetch_moneyline_quotes(sports,api_key):
    """Explicit refresh only: one bounded h2h request per chosen sport."""
    if not api_key:raise ValueError('Odds API key is missing')
    from dual_agent.odds_request_cache import get
    def fetch(sport):
        try:
            response=get('https://api.the-odds-api.com/v4/sports/'+SPORTS[sport]+'/odds',
                params={'apiKey':api_key,'bookmakers':','.join(BOOKS.values()),
                        'markets':'h2h','oddsFormat':'decimal','dateFormat':'iso'},
                cache_ttl=0,timeout=(3,8))
            if response.status_code!=200:return sport,[],sport+' odds unavailable (HTTP '+str(response.status_code)+')'
            payload=response.json()
            if not isinstance(payload,list):return sport,[],sport+' odds response was not an event list'
            return sport,payload,None
        except Exception as exc:
            return sport,[],sport+' odds request failed ('+type(exc).__name__+')'
    events={};issues=[]
    with ThreadPoolExecutor(max_workers=min(3,len(sports) or 1)) as pool:
        for sport,payload,issue in pool.map(fetch,sports):
            events[sport]=payload
            if issue:issues.append(issue)
    return events,issues


def moneyline_candidates(forecasts,events,book,now=None):
    now=stamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    candidates={}
    for sport,games in forecasts.items():
        for game in records(games):
            home=game.get('home_team',game.get('Home'));away=game.get('away_team',game.get('Away'))
            start=stamp(game.get('commence_time',game.get('start_time')))
            if not home or not away or team(home)==team(away) or pd.isna(start) or not now<start<=now+pd.Timedelta(days=7):continue
            source=str(game.get('probability_source',game.get('Probability source',''))).lower()
            if any(label in source for label in ('sportsbook','market_implied','market implied')):continue
            p=number(game.get('home_win_probability',game.get('Home win probability')))
            if p is None or not 0<p<1:continue
            other=number(game.get('away_win_probability'))
            if other is not None and abs(p+other-1)>1e-6:continue
            matched=[]
            for event in events.get(sport,[]):
                at=stamp(event.get('commence_time'))
                if (pd.notna(at) and abs(at-start)<=pd.Timedelta(minutes=15)
                        and team(home)==team(event.get('home_team')) and team(away)==team(event.get('away_team'))):
                    matched.append(event)
            if len(matched)!=1:continue
            event=matched[0]
            game_id=(sport,team(home),team(away),start.isoformat())
            for platform in event.get('bookmakers',[]):
                if platform.get('key')!=book:continue
                for market in platform.get('markets',[]):
                    if market.get('key')!='h2h':continue
                    quoted=stamp(market.get('last_update') or platform.get('last_update'))
                    if pd.isna(quoted) or not pd.Timedelta(0)<=now-quoted<=pd.Timedelta(minutes=30) or quoted>=start:continue
                    outcomes=market.get('outcomes',[])
                    # Three-way/draw markets are not equivalent to our binary model.
                    if len(outcomes)!=2 or {team(row.get('name')) for row in outcomes}!={team(home),team(away)}:continue
                    for outcome in outcomes:
                        price=number(outcome.get('price'))
                        if price is None or price<=1:continue
                        home_side=team(outcome['name'])==team(home)
                        chosen=home if home_side else away;chance=p if home_side else 1-p
                        key=(game_id,team(chosen))
                        candidate=dict(sport=sport,book=book,game=str(away)+' @ '+str(home),
                            game_id=game_id,teams=(team(home),team(away)),start=start.isoformat(),
                            team=chosen,chance=chance,decimal_odds=price,quote_time=quoted.isoformat())
                        if key not in candidates or candidate['quote_time']>candidates[key]['quote_time']:
                            candidates[key]=candidate
    return sorted(candidates.values(),key=lambda row:(-row['chance'],row['start'],row['team']))


def build_target_ticket(candidates,book,stake,target_total,max_legs=3,now=None):
    stake=number(stake);target_total=number(target_total)
    if stake is None or stake<=0 or target_total is None or target_total<=stake:
        raise ValueError('Total return must exceed a positive stake')
    if max_legs not in (1,2,3):raise ValueError('Choose one to three maximum selections')
    now=stamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    pool=[]
    for row in candidates:
        quote=stamp(row.get('quote_time'));start=stamp(row.get('start'))
        p=number(row.get('chance'));price=number(row.get('decimal_odds'))
        if (row.get('book')!=book or p is None or not 0<p<1 or price is None or price<=1
            or pd.isna(quote) or pd.isna(start) or not now<start<=now+pd.Timedelta(days=7)
            or not pd.Timedelta(0)<=now-quote<=pd.Timedelta(minutes=30) or quote>=start):continue
        pool.append(row)
    pool.sort(key=lambda row:(-row['chance'],row['start'],row['team']))
    limited=len(pool)>80;pool=pool[:80]
    best=None;checked=0
    for size in range(1,max_legs+1):
        for legs in combinations(pool,size):
            game_ids={tuple(r['game_id']) for r in legs}
            participants=[(r['sport'],t) for r in legs for t in r['teams']]
            if len(game_ids)!=size or len(set(participants))!=len(participants):continue
            probability=math.prod(r['chance'] for r in legs)
            if best is not None and probability<best['estimated_probability']:continue
            decimal=math.prod(r['decimal_odds'] for r in legs)
            total=stake*decimal;checked+=1
            if total+1e-8<target_total:continue
            ticket=dict(book=book,legs=list(legs),stake=stake,total_return=round(total,2),
                profit=round(total-stake,2),decimal_odds=decimal,estimated_probability=probability,
                break_even_probability=1/decimal,estimated_return_per_unit=probability*decimal-1,
                independence_assumed=size>1)
            priority=(-probability,size,total-target_total)
            if best is None or priority<best['_priority']:
                ticket['_priority']=priority;best=ticket
    if best is not None:best.pop('_priority',None)
    return dict(ticket=best,available_offers=len(pool),search_limited=limited,checked_combinations=checked)


def render_moneyline_builder(st,api_key):
    st.caption('Compare singles and short moneyline parlays against a total-return target. Uses saved winner forecasts; player-prop scans are separate.')
    left,right=st.columns(2)
    with left:label=st.selectbox('Sportsbook',list(BOOKS),key='ml_ticket_book')
    with right:sport=st.selectbox('Sports',['All sports']+list(SPORTS),key='ml_ticket_sport')
    left,right=st.columns(2)
    with left:stake=st.number_input('Stake ($)',min_value=1.,value=400.,step=10.,key='ml_ticket_stake')
    with right:target=st.number_input('Target total return ($), including stake',min_value=1.,value=600.,step=10.,key='ml_ticket_target')
    maximum=st.selectbox('Maximum selections',[1,2,3],index=2,key='ml_ticket_maximum')
    st.caption('A single is included in every search. Amounts are editable calculations, not stake recommendations.')
    selected=list(SPORTS) if sport=='All sports' else [sport]
    if st.button('Refresh moneyline odds',key='ml_ticket_refresh'):
        with st.spinner('Collecting moneyline odds…'):
            try:
                events,issues=fetch_moneyline_quotes(selected,api_key)
                saved=dict(st.session_state.get('ml_ticket_events') or {})
                saved.update(events);st.session_state['ml_ticket_events']=saved
                st.session_state['ml_ticket_issues']=issues
            except ValueError as exc:st.session_state['ml_ticket_issues']=[str(exc)]
        st.session_state.pop('ml_ticket_preview',None)
    forecasts=saved_forecasts(st.session_state,selected)
    quotes=st.session_state.get('ml_ticket_events') or {}
    candidates=moneyline_candidates(forecasts,quotes,BOOKS[label])
    st.caption(str(len(candidates))+' fresh team offers matched to saved model forecasts at '+label+'. Building uses collected data only.')
    if not any(forecasts.values()):
        st.info('Generate winner predictions on a sport page first, then return here. Player props are not required.')
    for issue in st.session_state.get('ml_ticket_issues',[]):st.caption(issue)
    if st.button('Find target-return ticket',key='ml_ticket_build',type='primary'):
        try:
            result=build_target_ticket(candidates,BOOKS[label],stake,target,maximum)
            st.session_state['ml_ticket_preview']={'settings':(label,sport,stake,target,maximum),'result':result}
        except ValueError as exc:
            st.error(str(exc));st.session_state.pop('ml_ticket_preview',None)
    preview=st.session_state.get('ml_ticket_preview')
    if not preview or preview['settings']!=(label,sport,stake,target,maximum):return
    result=preview['result'];ticket=result['ticket']
    if ticket is None:
        st.info('No target ticket was found in the searched fresh offers with this maximum selection count. Refresh moneyline odds or adjust the target.')
        return
    # Recheck exact saved offers and original quote ages on every display.
    identities={(r['book'],tuple(r['game_id']),r['team'],r['decimal_odds'],r['quote_time'],r['chance']) for r in candidates}
    if any((r['book'],tuple(r['game_id']),r['team'],r['decimal_odds'],r['quote_time'],r['chance']) not in identities for r in ticket['legs']):
        st.info('The ticket’s forecasts or odds changed or expired. Refresh odds and build again.');return
    st.subheader(label+' · '+('Single moneyline' if len(ticket['legs'])==1 else str(len(ticket['legs']))+' moneyline selections'))
    st.write(f"Stake ${ticket['stake']:,.2f} · calculated total return ${ticket['total_return']:,.2f} · profit if all win ${ticket['profit']:,.2f}")
    st.write(f"Model-estimated {'independent ticket scenario' if ticket['independence_assumed'] else 'win chance'}: {ticket['estimated_probability']:.1%}")
    st.caption(f"Break-even win probability at these prices: {ticket['break_even_probability']:.1%}")
    if ticket['estimated_return_per_unit']<=0:
        st.warning('This reaches the payout target, but the model does not estimate a positive return at these prices.')
    for leg in ticket['legs']:
        with st.container(border=True):
            price=leg['decimal_odds'];american=(price-1)*100 if price>=2 else -100/(price-1)
            st.markdown('**'+leg['team']+' to win**')
            st.caption(leg['sport']+' · '+leg['game']+' · '+str(leg['start']))
            st.write(f"Model chance {leg['chance']:.1%} · moneyline {american:+.0f} · {label}")
            st.caption('Original quote: '+leg['quote_time'])
    if result['search_limited']:st.caption('Preview compares the highest-confidence collected offers; it is not an exhaustive search of the full schedule.')
    st.caption('Preview only. Parlay probability assumes independent games; model probabilities are not proven betting accuracy. The sportsbook must confirm current odds and combination acceptance. A loss risks the full stake. This app does not place a wager.')
