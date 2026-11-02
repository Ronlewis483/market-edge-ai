"""NCAA football market station using the existing Odds API credentials.

Market probabilities are explicitly separate from trained model predictions.
Connected forecasts are generated explicitly; viewing cards never starts API requests.
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


def load_markets(api_key):
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


def load_station(api_key, cfbd_key=None):
    from dual_agent.ncaaf_prediction import generate_predictions
    return generate_predictions(cfbd_key,api_key)


def _player_ui(function_name):
    """Keep the winner page usable when a deployed prop module is incomplete."""
    import importlib
    import streamlit as st
    try:
        module = importlib.import_module('dual_agent.ncaaf_player_props')
        function = getattr(module, function_name, None)
        if not callable(function):
            raise ImportError('Missing public NCAAF player UI function')
        return function
    except ImportError:
        st.warning('NCAAF player tools could not load. Replace dual_agent/ncaaf_player_props.py with the complete matching file, then reboot the app. NCAA winner predictions remain available.')
        return None


def render_station():
    import os
    import pandas as pd
    import streamlit as st
    from concurrent.futures import ThreadPoolExecutor
    @st.cache_resource(show_spinner=False)
    def worker_pool():
        return ThreadPoolExecutor(max_workers=2)
    st.title('NCAA Football | Connected Engine')
    st.caption('FBS college football \xb7 upcoming seven days \xb7 CollegeFootballData + your existing Odds API.')
    st.caption('One final forecast: college base engine \u2192 NCAA-trained neural signal \u2192 bounded context correction. No probability averaging or NBA/NFL weights. Winner inputs cover team history, rest and neutral venues. College-only player-prop regression is configured separately below; injury and starter status remain unverified.')
    if st.button('Generate NCAA connected predictions',type='primary',key='ncaaf_load'):
        pending = st.session_state.get('ncaaf_station_future')
        if pending is None or pending.done():
            key = os.environ.get('ODDS_API_KEY') or os.environ.get('THE_ODDS_API_KEY')
            if not key:
                try:key = st.secrets.get('ODDS_API_KEY') or st.secrets.get('THE_ODDS_API_KEY')
                except Exception:key = None
            cfbd_key = os.environ.get('CFBD_API_KEY')
            if not cfbd_key:
                try:cfbd_key = st.secrets.get('CFBD_API_KEY')
                except Exception:cfbd_key = None
            st.session_state['ncaaf_station_future'] = worker_pool().submit(load_station,key,cfbd_key)
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
        st.caption('Loading saved NCAA weights and calculating predictions in the background. Training runs in GitHub Actions. Use Show latest NCAA results when ready.')
    if st.session_state.get('ncaaf_station_error'):
        st.error(st.session_state['ncaaf_station_error'])
    result = st.session_state.get('ncaaf_station_result')
    st.caption('Generation may use CFBD and Odds API requests. Viewing saved results makes no requests. CFBD_API_KEY is required; odds comparisons are optional.')
    st.caption('Use Generate NCAA connected predictions for game winners. Data collection and training run separately in GitHub Actions.')
    training_controls = _player_ui('render_training_controls')
    if training_controls is not None:
        training_controls()
    if not result:return
    if result.get('architecture') != 'original_offset_neural_context_residual_v1':
        st.info('Generate NCAA connected predictions to replace the earlier market-only results.')
        return
    for issue in result.get('issues', []):st.warning(str(issue))
    if not result.get('checks_passed'):
        st.warning('Experimental NCAA candidate: it has not demonstrated improvement over the base model on both historical partitions. Forecasts are not validated for betting performance.')
    with st.expander('Connected NCAA model and historical comparison'):
        st.write('Model: '+result['model_id'])
        st.write('Usable historical games: '+str(result['training_games']))
        st.json(result.get('comparison', {}))
        st.caption(result.get('limitations',''))
        st.write('Frozen pregame results by model version')
        st.json(result.get('prospective_results', {}))
        st.caption('Pregame winner forecasts are backed up to Supabase and graded by the daily Actions job after CFBD marks games final. Weekly training evaluates challengers separately.')
    if result.get('history_checked_at'):
        history_time=pd.Timestamp(result['history_checked_at'])
        st.caption('Saved schedule/results snapshot: '+history_time.tz_convert('America/Chicago').strftime('%b %d, %I:%M %p CT'))
        if pd.Timestamp.now(tz='UTC')-history_time>pd.Timedelta(days=2):
            st.warning('The saved NCAA history is more than two days old. Check the daily Actions job before relying on this slate.')
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
        # Connected predictions remain visible when market quotes are unavailable.
        p = median(q['home_probability'] for q in game['moneylines']) if game['moneylines'] else None
        game.update(home_probability=p,away_probability=1-p if p is not None else None,
                    market_favorite=game['home_team'] if p is not None and p>=.5 else game['away_team'] if p is not None else None,
                    favorite_probability=max(p,1-p) if p is not None else None)
        games.append(game)
    if not games:
        st.info('No supported upcoming FBS predictions were produced. Games with unconfirmed kickoff times or insufficient prior history are excluded.')
        return
    chosen = render_prediction_cards(games, result)
    if chosen is None:
        return
    st.caption('Win probabilities remove each book\u2019s margin, then take the median across returned books. They are not independent model confidence or proof of a betting edge.')
    for field,label in [('moneylines','Moneylines'),('spreads','Spreads'),('totals','Game totals')]:
        st.markdown('**'+label+'**')
        fresh = [q for q in chosen[field] if pd.Timedelta(0)<=now-pd.Timestamp(q['Quote UTC'])<=pd.Timedelta(minutes=30)]
        if fresh:st.dataframe(pd.DataFrame(fresh).drop(columns=['book']),hide_index=True,use_container_width=True)
        else:st.caption('No current quotes in this category. Refresh to check for new listings.')
    st.caption('Winner predictions and player-prop regression are separate models. College props are provisional; no NCAA ticket placement or verified combined-ticket probability is provided.')


def card_groups(games, now=None):
    """Today first, then strongest ten remaining, then the rest; no duplicates."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ZoneInfo('America/Chicago')).date()
    unique = {str(g['game_id']): g for g in games}
    ranked = sorted(unique.values(), key=lambda g: (-float(g['model_confidence']), g['commence_time'], str(g['game_id'])))
    current = [g for g in ranked if datetime.fromisoformat(g['commence_time'].replace('Z','+00:00')).astimezone(ZoneInfo('America/Chicago')).date() == today]
    seen = {str(g['game_id']) for g in current}
    remaining = [g for g in ranked if str(g['game_id']) not in seen]
    return [("Today's picks", current), ('Strongest upcoming picks \xb7 up to 10', remaining[:10]), ('Remaining games', remaining[10:])]


def render_prediction_cards(games, result):
    """Match the other sport workspaces: selectable cards and game spotlight."""
    import pandas as pd
    import streamlit as st
    from html import escape
    st.markdown('<div class="me-workspace-header"><div class="me-eyebrow">PREDICTION WORKSPACE \xb7 NCAA FOOTBALL</div><h2>Find your next pick</h2><p>Select a game to compare its connected forecast and listed sportsbook lines.</p></div>', unsafe_allow_html=True)
    controls = st.columns([2,2,3])
    with controls[0]:
        window = st.selectbox('Game window', ['Next 7 days', 'Today'], key='ncaaf_card_window')
    with controls[1]:
        threshold = st.selectbox('Winner confidence', ['All predictions','70% or higher','80% or higher'], key='ncaaf_card_confidence')
    with controls[2]:
        search = st.text_input('Find a team', key='ncaaf_card_search', placeholder='Search either team\u2026').strip().casefold()
    minimum = .8 if threshold.startswith('80') else .7 if threshold.startswith('70') else 0
    filtered = [g for g in games if float(g['model_confidence']) >= minimum and
                (not search or search in g['home_team'].casefold() or search in g['away_team'].casefold())]
    groups = card_groups(filtered)
    if window == 'Today': groups = groups[:1]
    visible = [g for _, rows in groups for g in rows]
    if not visible:
        st.info('No games match these filters. Try another team, window or confidence level.')
        return None
    by_id = {str(g['game_id']):g for g in visible}
    selection = 'ncaaf_selected_card'
    if st.session_state.get(selection) not in by_id:
        st.session_state[selection] = str(visible[0]['game_id'])
    def select_game(identity):
        st.session_state[selection] = identity
    def badge(label, probability):
        # Same badge class used by the other prediction workspaces.
        tone = 'strong' if probability >= .70 else 'moderate' if probability >= .58 else 'close'
        st.markdown('<div class="me-confidence me-'+tone+'">'+escape(str(label))+' <span>'+f'{float(probability):.1%}'+'</span></div>', unsafe_allow_html=True)
    left, right = st.columns([1,1.25], gap='large')
    with left:
        st.subheader('NCAA game picks')
        st.caption(f'{len(visible)} games \xb7 today first, then the strongest upcoming forecasts')
        with st.container(height=650, border=True):
            for title, rows in groups:
                if not rows: continue
                st.markdown('**'+title+'**')
                for game in rows:
                    gid = str(game['game_id'])
                    with st.container(border=True):
                        kickoff = pd.Timestamp(game['commence_time']).tz_convert('America/Chicago')
                        st.caption(kickoff.strftime('%b %d, %Y \xb7 %I:%M %p CT')+' \xb7 Upcoming')
                        st.markdown('**'+game['away_team']+' @ '+game['home_team']+'**')
                        badge(game['predicted_team'], game['model_confidence'])
                        st.button('Selected game' if st.session_state[selection]==gid else 'View game & sportsbook lines',
                            key='ncaaf_pick_'+gid, type='primary' if st.session_state[selection]==gid else 'secondary',
                            use_container_width=True, on_click=select_game, args=(gid,))
    chosen = by_id[st.session_state[selection]]
    with right:
        st.subheader('Game spotlight')
        with st.container(border=True):
            st.markdown('### '+chosen['away_team']+' @ '+chosen['home_team'])
            st.caption(pd.Timestamp(chosen['commence_time']).tz_convert('America/Chicago').strftime('%b %d, %Y \xb7 %I:%M %p CT'))
            st.markdown('**\U0001f3c6 Predicted winner: '+chosen['predicted_team']+'**')
            badge('Estimated win chance', chosen['model_confidence'])
            st.progress(min(1., max(0., float(chosen['model_confidence']))))
            from dual_agent.shared_experience import render as render_experience
            render_experience('NCAAF',chosen)
            market = chosen.get('market_favorite')
            if market and chosen.get('favorite_probability') is not None:
                st.caption(f"Sportsbook consensus: {market} \xb7 {chosen['favorite_probability']:.1%} implied chance")
            else:
                st.caption('No fresh moneyline comparison. The connected forecast remains available.')
            with st.expander('View model details'):
                st.write('Home win chance:', f"{chosen['model_home_probability']:.1%}")
                st.write('Away win chance:', f"{chosen['model_away_probability']:.1%}")
                st.caption('Connected NCAA engine \xb7 '+str(result['model_id']))
                st.caption('Winner inputs: team history, rest and neutral venues. Individual player-strength and injury corrections are not inputs to this winner model.')
            st.caption('Listed spreads and totals below are sportsbook offers, not model spread or total predictions.')
            props_panel = _player_ui('render_props_panel')
            if props_panel is not None:
                props_panel(chosen)
    return chosen
