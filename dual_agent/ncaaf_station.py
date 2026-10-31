"""NCAA football market station using the existing Odds API credentials.

Market probabilities are explicitly separate from trained model predictions.
No extra stats subscription, model training, automatic polls or wager placement.
"""
from datetime import datetime, timedelta, timezone
from statistics import median
import math

SPORT = 'americanfootball_ncaaf'


def parse_markets(events, now=None):
    now = now or datetime.now(timezone.utc)
    def stamp(value):
        try:
            dt = datetime.fromisoformat(str(value).replace('Z','+00:00'))
            return dt.astimezone(timezone.utc) if dt.tzinfo else None
        except (ValueError, TypeError):
            return None
    def price(value):
        try:
            number = float(value)
            return number if math.isfinite(number) and number > 1 else None
        except (ValueError, TypeError):
            return None
    def number(value):
        try:
            n = float(value)
            return n if math.isfinite(n) else None
        except (ValueError, TypeError):
            return None
    games = {}
    for event in events:
        start = stamp(event.get('commence_time'))
        home, away = event.get('home_team'), event.get('away_team')
        if not event.get('id') or not home or not away or home==away or start is None or not now < start <= now+timedelta(days=7):
            continue
        identity = str(event['id'])
        game = {'game_id':identity,'home_team':home,'away_team':away,'commence_time':start.isoformat(),
                'moneylines':[],'spreads':[],'totals':[],'probability_source':'sportsbook_consensus'}
        # One contribution per book, even if provider data repeats an entry.
        books = {b.get('key'):b for b in event.get('bookmakers',[]) if b.get('key')}
        probabilities = []
        for key, book in books.items():
            markets = {m.get('key'):m for m in book.get('markets',[]) if m.get('key')}
            for market_key, market in markets.items():
                updated = stamp(market.get('last_update') or book.get('last_update'))
                if updated is None or not timedelta(0) <= now-updated <= timedelta(minutes=30) or updated>=start:
                    continue
                outcomes = market.get('outcomes',[])
                by_name = {o.get('name'):o for o in outcomes}
                common = {'Sportsbook':book.get('title',key),'book':key,'Quote UTC':updated.isoformat()}
                if market_key in ('h2h','spreads'):
                    if len(outcomes)!=2 or set(by_name)!={home,away}: continue
                    hp, ap = price(by_name[home].get('price')), price(by_name[away].get('price'))
                    if hp is None or ap is None: continue
                    if market_key == 'h2h':
                        fair_home = (1/hp)/(1/hp+1/ap)
                        probabilities.append(fair_home)
                        game['moneylines'].append(dict(common, home_decimal=hp, away_decimal=ap, home_probability=fair_home))
                    else:
                        hl, al = number(by_name[home].get('point')), number(by_name[away].get('point'))
                        if hl is not None and al is not None and abs(hl+al)<1e-8:
                            game['spreads'].append(dict(common, home_line=hl, away_line=al, home_decimal=hp, away_decimal=ap))
                elif market_key == 'totals':
                    if len(outcomes)!=2 or set(by_name)!={'Over','Under'}:continue
                    over, under = by_name['Over'], by_name['Under']
                    line, other = number(over.get('point')), number(under.get('point'))
                    op, up = price(over.get('price')),price(under.get('price'))
                    if None not in (line,other,op,up) and line==other:
                        game['totals'].append(dict(common,total=line,over_decimal=op,under_decimal=up))
        if not any(game[k] for k in ('moneylines','spreads','totals')):continue
        p = median(probabilities) if probabilities else None
        game.update(home_probability=p,away_probability=1-p if p is not None else None,
                    market_favorite=home if p is not None and p>=.5 else away if p is not None else None,
                    favorite_probability=max(p,1-p) if p is not None else None)
        games[identity] = game
    return sorted(games.values(),key=lambda g:(g['commence_time'],g['game_id']))


def load_station(api_key):
    if not api_key:raise ValueError('The existing ODDS_API_KEY is not configured.')
    from dual_agent.odds_request_cache import get
    now = datetime.now(timezone.utc)
    # Stable five-minute request window permits reuse of the shared cache.
    window = now.replace(minute=now.minute//5*5,second=0,microsecond=0)
    params = {'apiKey':api_key,'markets':'h2h,spreads,totals','oddsFormat':'decimal','dateFormat':'iso',
              'commenceTimeFrom':window.strftime('%Y-%m-%dT%H:%M:%SZ'),
              'commenceTimeTo':(window+timedelta(days=7,minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')}
    response = get('https://api.the-odds-api.com/v4/sports/'+SPORT+'/odds',params=params,cache_ttl=300,timeout=(5,20))
    if response.status_code!=200:
        raise RuntimeError('NCAA odds request returned HTTP '+str(response.status_code)+'. Check existing Odds API access and remaining credits.')
    events = response.json()
    if not isinstance(events,list):raise ValueError('The NCAA odds response was not a game list.')
    return {'games':parse_markets(events,now),'checked_at':now.isoformat(),
            'remaining_credits':response.headers.get('x-requests-remaining'),
            'reported_request_cost':response.headers.get('x-requests-last')}


def render_station():
    import os
    import pandas as pd
    import streamlit as st
    from concurrent.futures import ThreadPoolExecutor
    @st.cache_resource(show_spinner=False)
    def worker_pool():
        return ThreadPoolExecutor(max_workers=2)
    st.title('🏈 NCAA Football')
    st.caption('FBS college football · upcoming seven days · sportsbook information from your existing Odds API.')
    st.info('This station currently shows market-based win probabilities and offered lines. An independent NCAA prediction model and player-prop history are not connected yet; NFL model weights are not used.')
    if st.button('Load / refresh NCAA games and lines',type='primary',key='ncaaf_load'):
        pending = st.session_state.get('ncaaf_station_future')
        if pending is None or pending.done():
            key = os.environ.get('ODDS_API_KEY') or os.environ.get('THE_ODDS_API_KEY')
            if not key:
                try:key = st.secrets.get('ODDS_API_KEY') or st.secrets.get('THE_ODDS_API_KEY')
                except Exception:key = None
            st.session_state['ncaaf_station_future'] = worker_pool().submit(load_station,key)
            st.session_state.pop('ncaaf_station_error',None)
    st.button('Show latest NCAA results',key='ncaaf_show')
    pending = st.session_state.get('ncaaf_station_future')
    if pending is not None and pending.done():
        try:
            st.session_state['ncaaf_station_result'] = pending.result()
            st.session_state.pop('ncaaf_station_error',None)
        except Exception as exc:
            st.session_state['ncaaf_station_error'] = str(exc)
        st.session_state.pop('ncaaf_station_future',None)
        pending = None
    if pending is not None:
        st.caption('NCAA lines are loading in the background. Use Show latest NCAA results when ready; this page does not refresh automatically.')
    if st.session_state.get('ncaaf_station_error'):
        st.error(st.session_state['ncaaf_station_error'])
    result = st.session_state.get('ncaaf_station_result')
    st.caption('Refreshing sportsbook lines may use API credits. Viewing saved results does not make requests.')
    if not result:return
    checked = pd.Timestamp(result['checked_at']).tz_convert('America/Chicago')
    st.caption('Last check: '+checked.strftime('%b %d, %I:%M %p CT'))
    # Reapply time and freshness rules while viewing saved results.
    now = pd.Timestamp.now(tz='UTC')
    games = []
    for original in result['games']:
        if not now<pd.Timestamp(original['commence_time'])<=now+pd.Timedelta(days=7):continue
        game = dict(original)
        for field in ('moneylines','spreads','totals'):
            game[field] = [q for q in original[field] if pd.Timedelta(0)<=now-pd.Timestamp(q['Quote UTC'])<=pd.Timedelta(minutes=30)]
        if not any(game[k] for k in ('moneylines','spreads','totals')):continue
        p = median(q['home_probability'] for q in game['moneylines']) if game['moneylines'] else None
        game.update(home_probability=p,away_probability=1-p if p is not None else None,
                    market_favorite=game['home_team'] if p is not None and p>=.5 else game['away_team'] if p is not None else None,
                    favorite_probability=max(p,1-p) if p is not None else None)
        games.append(game)
    if not games:
        st.info('No upcoming games with fresh supported sportsbook lines were returned. No picks were invented.')
        return
    def summary(g):
        return {'Game':g['away_team']+' @ '+g['home_team'],
                'Kickoff CT':pd.Timestamp(g['commence_time']).tz_convert('America/Chicago').strftime('%b %d, %I:%M %p'),
                'Market favorite':g['market_favorite'], 'Market-implied chance':g['favorite_probability'],
                'Moneyline books':len(g['moneylines'])}
    favorites = sorted([g for g in games if g['favorite_probability'] is not None],key=lambda g:-g['favorite_probability'])[:10]
    if favorites:
        st.subheader('Top 10 market favorites')
        st.dataframe(pd.DataFrame([summary(g) for g in favorites]),hide_index=True,use_container_width=True)
    st.subheader('All upcoming games')
    st.dataframe(pd.DataFrame([summary(g) for g in games]),hide_index=True,use_container_width=True)
    chosen = st.selectbox('Game details',games,format_func=lambda g:g['away_team']+' @ '+g['home_team']+' · '+pd.Timestamp(g['commence_time']).tz_convert('America/Chicago').strftime('%b %d, %I:%M %p CT'),key='ncaaf_game')
    st.caption('Win probabilities remove each book’s margin, then take the median across returned books. They are not independent model confidence or proof of a betting edge.')
    for field,label in [('moneylines','Moneylines'),('spreads','Spreads'),('totals','Game totals')]:
        st.markdown('**'+label+'**')
        fresh = [q for q in chosen[field] if pd.Timedelta(0)<=now-pd.Timestamp(q['Quote UTC'])<=pd.Timedelta(minutes=30)]
        if fresh:st.dataframe(pd.DataFrame(fresh).drop(columns=['book']),hide_index=True,use_container_width=True)
        else:st.caption('No current quotes in this category. Refresh to check for new listings.')
    st.caption('Player props and NCAA ticket recommendations are not generated without verified college player statistics and participation checks.')
