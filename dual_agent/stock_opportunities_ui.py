"""Automatic hourly and weekly stock opportunity cards."""
import os
import pandas as pd
import streamlit as st
from dual_agent import stock_opportunity_engine as engine


def setting(name, default=None):
    try:
        return st.secrets.get(name) or os.environ.get(name) or default
    except Exception:
        return os.environ.get(name) or default


@st.cache_data(ttl=300, show_spinner=False)
def history(symbols, key, secret, horizon, feed):
    return engine.fetch_bars(list(symbols), key, secret, horizon, feed)


@st.cache_data(ttl=60, show_spinner=False)
def current_data(symbols, key, secret, feed, url):
    return engine.fetch_quotes(list(symbols), key, secret, feed), engine.fetch_clock(key, secret, url)


def render_card(pick, horizon, top=False):
    with st.container(border=True):
        st.caption('TOP QUALIFYING CANDIDATE' if top else 'ALSO QUALIFIES')
        st.markdown('### '+pick['symbol'])
        st.write('**Strategy:** '+pick['strategy'])
        a,b=st.columns(2)
        a.metric('Entry reference · ask', f"${pick['entry_reference']:,.2f}")
        b.metric('Historical net return per trade', f"{pick['historical_mean_net_return']:+.2%}")
        st.caption(f"{pick['verification_trades']} separate-period trades · historical win rate {pick['historical_win_rate']:.0%}")
        st.write('**Planned hold:** '+pick['holding_period'])
        if horizon=='hour':
            st.write(f"**Stop reference:** ${pick['stop_reference']:,.2f}")
            exit_time=pd.Timestamp(pick['exit_time']).tz_convert('America/Chicago')
            st.write('**Time exit:** '+exit_time.strftime('%-I:%M %p CT'))
        stamp=pd.Timestamp(pick['quote_time']).tz_convert('America/Chicago')
        st.caption('Quote: '+stamp.strftime('%b %d · %-I:%M:%S %p CT')+f" · spread {pick['spread_pct']:.2%}")
        st.caption('Historical results are estimates from bar data, not a forecast of your realized profit.')


def render_result(result, horizon):
    evidence=result['strategy_evidence']
    result=dict(result)
    result['picks']=[p for p in result['picks'] if pd.Timestamp.now(tz='UTC')-pd.Timestamp(p['quote_time'])<=pd.Timedelta(seconds=90)]
    if not result['picks'] and result.get('message','').startswith('Candidates ranked'):
        result['message']='Entry quotes expired while the full scan completed. Waiting for the next update.'
    if result['picks']:
        render_card(result['picks'][0],horizon,True)
        if len(result['picks'])>1:
            with st.expander('Other qualifying candidates'):
                for pick in result['picks'][1:]:render_card(pick,horizon)
    else:
        st.info(result['message'])
    if evidence.get('strategy'):
        held=evidence['verification']
        with st.expander('Why this strategy was selected'):
            st.write('**'+evidence['strategy']+'** ranked first by average net return in the earlier selection period.')
            if held['mean_net_return'] is not None:
                st.write(f"Separate recent period: {held['trades']} completed trades, average {held['mean_net_return']:+.2%} after estimated costs.")
            st.caption(f"Compared {len(engine.STRATEGIES[horizon])} strategies in the scanned symbol list. Estimated round-trip cost: {evidence['cost_assumption_bps']} basis points. Positions may overlap across symbols; this is not a portfolio return.")
            st.caption('This comparison uses currently listed symbols and does not establish the most profitable strategy across the entire market.')


@st.cache_resource(show_spinner=False)
def market_worker(_key, _secret, feed, url, version, credential_identity):
    from concurrent.futures import ThreadPoolExecutor
    return {'pool':ThreadPoolExecutor(max_workers=1), 'cache':{}}


@st.fragment(run_every='10s')
def render_stock_opportunities():
    st.title('📈 Stock Opportunities')
    st.caption('Automatic research candidates · regular-session long positions · no orders are submitted.')
    key=setting('ALPACA_API_KEY') or setting('APCA_API_KEY_ID')
    secret=setting('ALPACA_SECRET_KEY') or setting('ALPACA_API_SECRET') or setting('APCA_API_SECRET_KEY')
    if not key or not secret:
        st.info('Connect Alpaca market data to populate the hourly and weekly picks automatically.')
        with st.expander('Connect market data'):
            st.write('Add your Alpaca API key and secret to the app’s Streamlit Secrets settings. Either secret-key name below is accepted.')
            st.code('ALPACA_API_KEY = "your key"\nALPACA_SECRET_KEY = "your secret"\nALPACA_DATA_FEED = "iex"\nALPACA_TRADING_URL = "https://paper-api.alpaca.markets"', language='toml')
            st.caption('Use the live API URL if your credentials belong to a live account. Keys are used for reading market data and the market clock only.')
        return
    feed=setting('ALPACA_DATA_FEED','iex')
    if feed not in ['iex','sip']:
        st.error('Choose iex or sip for ALPACA_DATA_FEED. Delayed quotes are excluded from these live entry candidates.')
        return
    url=setting('ALPACA_TRADING_URL','https://paper-api.alpaca.markets').rstrip('/')
    st.caption('Scanning all active exchange-listed U.S. equities available through Alpaca, including ETFs. No preset symbol list.')
    st.caption('Stocks priced below five dollars or with less than one million dollars of previous-session dollar volume on your feed are excluded before strategy evaluation.')
    import hashlib
    identity=hashlib.sha256((key+'|'+secret).encode()).hexdigest()
    state=market_worker(key,secret,feed,url,engine.ENGINE_VERSION,identity)
    future=state.get('future')
    if future is not None and future.done():
        try:
            state['result']=future.result();state.pop('error',None)
        except Exception as exc:
            state['error']=str(exc)
            import time
            state['retry_after']=time.monotonic()+120
        state['future']=None
    import time
    busy=state.get('future') is not None
    cooling=time.monotonic()<state.get('retry_after',0)
    requested=st.button('🔎 Scan stocks now', key='scan_full_stock_market', type='primary', disabled=busy or cooling)
    if cooling:st.info('Pausing requests to let the data connection recover. Automatic retry in '+str(max(1,int(state['retry_after']-time.monotonic())))+' seconds.')
    if not cooling and state.get('future') is None and (requested or time.monotonic()-state.get('submitted',-1000)>=60):
        state.pop('error',None)
        state['submitted']=time.monotonic()
        state['future']=state['pool'].submit(engine.scan_market,key,secret,feed,url,state['cache'])
    if state.get('future') is not None:
        progress=state['cache'].get('progress',{'text':'Starting market scan','done':0,'total':0})
        elapsed=int(time.monotonic()-state['submitted'])
        st.write('**'+progress['text']+f'** · {elapsed//60}m {elapsed%60}s elapsed')
        if progress['total']:
            st.progress(min(progress['done']/progress['total'],1.0),text=f"{progress['done']:,} of {progress['total']:,} eligible stocks in this stage")
        st.caption('A scan is already running. The button becomes available when it finishes; results update automatically.')
    if state.get('error'):st.error('Market scan could not complete: '+state['error'])
    available=[r for r in [state['cache'].get('partial_result'),state.get('result')] if r]
    saved=max(available,key=lambda r:r['as_of']) if available else None
    if not saved:
        hour,weekly=st.columns(2)
        with hour:
            st.subheader('⚡ Day-trade suggestions')
            st.info('Preparing the first scan. Qualifying day trades will appear here.')
        with weekly:
            st.subheader('🌱 Buy-and-hold suggestions')
            st.info('Preparing the first scan. Qualifying longer-term candidates will appear here.')
        return
    coverage=saved['coverage']
    checked=pd.Timestamp(saved['as_of'])
    st.caption(f"{coverage['listed']:,} listed symbols screened · {coverage.get('eligible',0):,} liquid candidates · {coverage.get('detailed_candidates',0):,} shortlisted for detailed strategy evaluation · updated "+checked.tz_convert('America/Chicago').strftime('%-I:%M:%S %p CT'))
    if feed=='iex':st.caption('IEX is one exchange. Stocks without sufficient coverage on that feed cannot qualify.')
    if saved['closed']:
        st.info('Market closed. Fresh entry candidates will be scanned during the regular session.');return
    stale=pd.Timestamp.now(tz='UTC')-checked>pd.Timedelta(seconds=90)
    hour,weekly=st.columns(2)
    for column,horizon,title in [(hour,'hour','⚡ Best qualifying setup for the next hour'),(weekly,'hold','🌱 Buy-and-hold candidate this week')]:
        with column:
            st.subheader(title)
            result=saved['results'].get(horizon)
            if result is None:
                st.info('This section is still being prepared. It will appear as soon as it is ready.');continue
            if stale:
                st.info('Refreshing the market scan. Previous entry quotes have expired.')
            else:
                render_result(result,horizon)
            failures=result.get('failed_batches',[])
            if failures:
                st.caption(f"History unavailable for {sum(len(x['symbols']) for x in failures)} symbols; this scan has partial coverage.")
                with st.expander('Stocks with unavailable history'):
                    for failure in failures:st.write(', '.join(failure['symbols'])+': '+failure['reason'])
    if state.get('future') is not None:st.caption('The next full-market scan is running in the background.')
    st.caption('Scan status updates every ten seconds; fresh quotes are checked about once a minute, and the full-market screen is reused for five minutes. The worker can finish a started scan after navigation, but this is not a scheduled service when the app is shut down. Detailed history is loaded for up to 40 candidates selected from the full-market screen. Saved history is reused; intraday updates fetch only recent bars. Local saved files may be lost when the hosting environment resets.')
