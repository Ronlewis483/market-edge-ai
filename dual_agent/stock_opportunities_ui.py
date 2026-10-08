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
            if 'evidence_score' in pick:
                st.write(f"**Evidence ranking score:** {pick['evidence_score']:.0%} · {pick['evidence_label']}")
                st.caption('Ranks historical win evidence with a penalty for small samples. This is not a predicted probability of profit.')
            earlier=pick.get('strategy_earlier_net_return')
            if earlier is not None and earlier<=0:
                st.caption(f"This strategy averaged {earlier:+.2%} in the older period; its recent-period average was positive.")
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
    if horizon=='hour':
        from collections import Counter
        with st.expander('Why day trades are waiting',expanded=not bool(result['picks'])):
            st.write('**'+result['message']+'**')
            checked=evidence.get('strategies',[])
            if checked:
                passed=sum(bool(item['qualified']) for item in checked)
                st.write(f"{passed} of {len(checked)} strategies passed the historical checks.")
                if not passed:
                    reasons=Counter()
                    for item in checked:
                        a=item['selection'];b=item['verification']
                        if a['trades']<25:reasons['Insufficient earlier-period trades']+=1
                        elif horizon!='hour' and (a['mean_net_return'] is None or a['mean_net_return']<=0):reasons['Earlier-period average return was not positive after costs']+=1
                        elif b['trades']<8:reasons['Insufficient recent-period trades']+=1
                        else:reasons['Recent-period average return was not positive after costs']+=1
                    for reason,count in reasons.items():st.write(f"• {count} strategies: {reason}")
            exclusions=result.get('exclusions',[])
            counts=Counter(item['reason'] for item in exclusions)
            for reason,count in counts.most_common():st.write(f"• {count} stock/strategy checks: {reason}")
            if exclusions:
                st.caption('Each count represents the first blocking rule for one stock and strategy; a stock can appear under several strategies.')
                st.caption('Individual examples (up to 30):')
                for item in exclusions[:30]:st.write(item['symbol']+' · '+item['strategy']+' · '+item['reason'])
            st.caption('Day trades currently cover long positions with up to a one-hour hold. These messages distinguish historical rejection, missing entry signals and live-data problems.')
    if evidence.get('strategies'):
        with st.expander('Strategies checked'):
            st.write(f"{len(evidence['strategies'])} strategies evaluated independently; {len(evidence['qualified_strategies'])} passed the historical checks.")
            for item in evidence['strategies']:
                held=item['verification']
                value=held['mean_net_return']
                average=f"{value:+.2%}" if value is not None else 'not available'
                st.write('**'+item['strategy']+'** · '+('Passed' if item['qualified'] else 'Waiting for sufficient positive evidence')+f" · {held['trades']} recent-period trades · average net return {average}")
            if horizon=='hour':st.caption('Day trades require any positive recent average net return after costs, at least 25 older-period trades and 8 recent-period trades. Negative older-period returns are disclosed rather than blocking a strategy.')
            st.caption('These are historical estimates after fixed costs, not proof of future profitability. Comparing more strategies increases the risk of finding a result by chance; the recent-period checks are screening evidence, not an untouched final validation.')


def render_daytrading(results,forex_config):
    st.subheader('⚡ Top 10 Day Trades · Options + Forex')
    st.caption('Ranked by sample-adjusted historical win evidence. Scores are not calibrated probabilities and do not establish equivalent risk across markets.')
    now=pd.Timestamp.now(tz='UTC');picks=[]
    for market in ['options','forex']:
        result=results.get(market)
        if result is None:
            st.info(market.title()+' scan is being prepared.');continue
        for pick in result.get('picks',[]):
            stamp=pd.Timestamp(pick['quote_time'])
            if stamp<=now and now-stamp<=pd.Timedelta(seconds=90) and pd.Timestamp(pick['exit_time'])>now:picks.append(pick)
        with st.expander(market.title()+' scan status'):
            st.write(result['message'])
            st.caption(str(result.get('contracts_checked',result.get('pairs_checked',0)))+' contracts or pairs assessed.')
            for reason in result.get('exclusions',[])[:20]:st.write(reason)
    if not forex_config.get('token') or not forex_config.get('account'):
        with st.expander('Connect forex data · OANDA'):
            st.write('Add an OANDA v20 account ID and API token to Streamlit Secrets. Use practice for a demo account, or live for a live account. No orders are submitted.')
            st.code('OANDA_API_TOKEN = "your token"\nOANDA_ACCOUNT_ID = "your account ID"\nOANDA_ENVIRONMENT = "practice"',language='toml')
    picks=sorted(picks,key=lambda p:(-p['evidence_score'],-p['verification_trades'],p['spread_pct']))
    seen=set();ranked=[]
    for pick in picks:
        identity=(pick['market'],pick['symbol'])
        if identity in seen:continue
        seen.add(identity);ranked.append(pick)
        if len(ranked)==10:break
    if not ranked:st.info('No fresh qualifying options or forex candidates yet. Scan status above shows what is pending or excluded.')
    for index,pick in enumerate(ranked,1):
        with st.container(border=True):
            st.markdown(f"### #{index} · {pick['market']} · {pick['symbol']}")
            st.write('**Strategy:** '+pick['strategy'])
            if pick['market']=='Options':
                st.write(f"Buy {pick['type']} · strike ${pick['strike']:,.2f} · expiry {pick['expiration']}")
                st.write(f"Premium ${pick['entry_reference']:.2f} · standard 100-share contract cost approximately ${pick['contract_cost_reference']:.2f}")
                st.caption('This screen assumes a standard contract multiplier; confirm contract specifications before acting. Long-option loss can reach the full premium. Historical option costs use current spread plus 0.5% friction, not recorded historical quotes.')
            else:
                st.write(pick['direction']+' · entry reference '+f"{pick['entry_reference']:.5f}")
                st.caption('Returns are unleveraged price returns with historical bid/ask spreads and an additional 0.02% round-trip cost assumption. Position size, margin and financing are not modeled.')
            st.write(f"**Evidence ranking score:** {pick['evidence_score']:.0%} · {pick['verification_trades']} recent historical trades")
            st.write(f"Historical average net return: {pick['historical_mean_net_return']:+.2%} · observed win rate {pick['historical_win_rate']:.0%}")
            st.write(f"Stop reference: {pick['stop_reference']:.5f} · time exit: "+pd.Timestamp(pick['exit_time']).tz_convert('America/Chicago').strftime('%-I:%M %p CT'))
            st.caption('Quote '+pd.Timestamp(pick['quote_time']).tz_convert('America/Chicago').strftime('%-I:%M:%S %p CT')+f" · spread {pick['spread_pct']:.2%}")


@st.cache_resource(show_spinner=False)
def market_worker(_key, _secret, feed, url, version, credential_identity):
    from concurrent.futures import ThreadPoolExecutor
    return {'pool':ThreadPoolExecutor(max_workers=1), 'cache':{}}


@st.fragment(run_every='10s')
def render_stock_opportunities():
    st.title('📈 Trading Opportunities')
    st.caption('Automatic stock, options and forex research · no orders are submitted.')
    key=setting('ALPACA_API_KEY') or setting('APCA_API_KEY_ID')
    secret=setting('ALPACA_SECRET_KEY') or setting('ALPACA_API_SECRET') or setting('APCA_API_SECRET_KEY')
    if not key or not secret:
        st.info('Connect Alpaca market data to populate the hourly and weekly picks automatically.')
        with st.expander('Connect market data'):
            st.write('Add your Alpaca API key and secret to the app’s Streamlit Secrets settings. Either secret-key name below is accepted.')
            st.code('ALPACA_API_KEY = "your key"\nALPACA_SECRET_KEY = "your secret"\nALPACA_DATA_FEED = "iex"\nALPACA_TRADING_URL = "https://paper-api.alpaca.markets"', language='toml')
            st.caption('Use the live API URL if your credentials belong to a live account. Keys are used for reading market data and the market clock only.')
        return
    feed=st.selectbox('Stock market data feed',['sip','iex'],key='paid_stock_data_feed')
    if feed not in ['iex','sip']:
        st.error('Choose iex or sip for ALPACA_DATA_FEED. Delayed quotes are excluded from these live entry candidates.')
        return
    url=setting('ALPACA_TRADING_URL','https://paper-api.alpaca.markets').rstrip('/')
    from dual_agent.stock_intraday_outlook import render_intraday_outlook
    render_intraday_outlook(key, secret, feed, url)
    st.divider()
    st.caption('Scanning all active exchange-listed U.S. equities available through Alpaca, including ETFs. No preset symbol list.')
    st.caption('Stocks priced below five dollars or with less than one million dollars of previous-session dollar volume on your feed are excluded before strategy evaluation.')
    import hashlib
    forex_config={'token':setting('OANDA_API_TOKEN'),'account':setting('OANDA_ACCOUNT_ID'),'environment':setting('OANDA_ENVIRONMENT','practice')}
    identity=hashlib.sha256((key+'|'+secret+'|'+str(forex_config)).encode()).hexdigest()
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
        state['future']=state['pool'].submit(engine.scan_market,key,secret,feed,url,state['cache'],forex_config)
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
    render_daytrading(saved['results'],forex_config)
    hour,weekly=st.columns(2)
    for column,horizon,title in [(hour,'hour','📈 Stock-only intraday candidates'),(weekly,'hold','🌱 Buy-and-hold candidate this week')]:
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
    st.caption('Scan status updates every ten seconds; fresh quotes are checked about once a minute, and the full-market screen is reused for five minutes. The worker can finish a started scan after navigation, but this is not a scheduled service when the app is shut down. Detailed evaluation starts with 40 candidates and expands on subsequent scans to 80, then 120 when either section has no qualifying picks. Saved history is reused; intraday updates fetch only recent bars. Local saved files may be lost when the hosting environment resets.')
