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


@st.fragment(run_every='60s')
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
    with st.expander('Scan settings'):
        text=st.text_input('Symbols to scan',','.join(engine.DEFAULT_SYMBOLS),key='automatic_stock_symbols')
        st.caption('Hourly setup: up to 60 minutes. Weekly review: buy-and-hold candidates evaluated over 63 trading sessions.')
    symbols=tuple(sorted({s.strip().upper() for s in text.split(',') if s.strip()}))
    if not symbols or len(symbols)>30 or any(not s.replace('.','').replace('-','').isalnum() for s in symbols):
        st.error('Enter between 1 and 30 valid stock symbols.');return
    try:
        with st.spinner('Reading current market conditions…'):
            quotes,clock=current_data(symbols,key,secret,feed,url)
        checked=pd.Timestamp.now(tz='UTC')
        clock_time=pd.to_datetime(clock.get('timestamp'),utc=True,errors='coerce')
        if pd.isna(clock_time) or abs(checked-clock_time)>pd.Timedelta(seconds=90):
            st.warning('The market clock is stale. Waiting for a fresh check.');return
        st.caption(('Market open' if clock.get('is_open') else 'Market closed')+' · '+checked.tz_convert('America/Chicago').strftime('%b %d · %-I:%M %p CT')+' · '+feed.upper()+' feed')
        if feed=='iex':st.caption('IEX covers one exchange; its prices and volumes differ from consolidated market data.')
        hour,weekly=st.columns(2)
        for column,horizon,title in [(hour,'hour','⚡ Best setup for the next hour'),(weekly,'hold','🌱 Buy-and-hold candidate this week')]:
            with column:
                st.subheader(title)
                if not clock.get('is_open'):
                    st.info('Market closed. Fresh entry candidates will appear during the regular session.');continue
                try:
                    with st.spinner('Preparing '+('intraday' if horizon=='hour' else 'longer-term')+' history and strategy evidence…'):
                        raw=history(symbols,key,secret,horizon,feed)
                        # History can take time on first load: re-read quotes before ranking.
                        quotes,clock=current_data(symbols,key,secret,feed,url)
                        result=engine.opportunities(raw,quotes,clock,horizon)
                    render_result(result,horizon)
                except Exception as exc:
                    st.error('Unable to prepare this view: '+str(exc))
        st.caption('Checks refresh every minute while this screen is open. History is cached for five minutes. First-time history preparation takes longer than a cached visit.')
    except Exception as exc:
        st.error('Market data connection failed: '+str(exc))
