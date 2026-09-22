"""Trading Center Phase 1: existing stock scan + research-only risk planner."""
import streamlit as st
from dual_agent.trade_risk import calculate_trade


def render_trading_center(latest_scan, stock_decision, universe, clean_symbols, model):
    st.title('📈 Trading Center')
    st.caption('Stock research and paper-trade planning. No orders are placed.')
    day, swing, long_term, short = st.tabs(['Day Trade', 'Swing Trade', 'Long-Term', 'Short'])
    with day:
        st.subheader('Day Trade · Risk Planner')
        st.info('This calculator is not an intraday prediction model. Enter proposed prices to explore risk before paper trading.')
        _risk_planner('day', 'Long')
    with swing:
        st.subheader('Swing Research · Existing Five-Day Scanner')
        st.caption('The existing model estimates five-day relative performance versus SPY; it does not predict stop/target hits.')
        symbols = st.text_input('Symbols', 'AAPL,MSFT,NVDA,AMZN', key='tc_symbols')
        if st.button('Run existing stock scan', key='tc_scan'):
            try:
                with st.spinner('Scanning market data…'):
                    df = latest_scan(universe, clean_symbols(symbols))
                    st.session_state['tc_scan_result'] = stock_decision(df)
            except Exception as exc:
                st.error(f'Scan failed: {exc}')
        result = st.session_state.get('tc_scan_result')
        if result:
            st.write('Research signal:', result.get('status', 'Unavailable'))
            st.write('Symbol:', result.get('symbol', '—'))
            if result.get('probability') is not None:
                st.metric('Five-day model probability', f"{result['probability']:.1%}")
            st.caption('Not a validated trade win probability or evidence of positive expected profit.')
        _risk_planner('swing', 'Long')
    with long_term:
        st.subheader('Long-Term Investment Research')
        st.info('Fundamental analysis and multi-year models are not yet connected. No long-term holding recommendation is generated.')
    with short:
        st.subheader('Short-Selling Risk Planner')
        st.warning('Research only. Borrow availability, fees, margin and squeeze risk are not verified.')
        _risk_planner('short', 'Short')
    with st.expander('Existing model information'):
        st.json(model)


def _risk_planner(prefix, default_direction):
    with st.container(border=True):
        st.markdown('#### Adjustable stop-loss & position size')
        a, b = st.columns(2)
        account = a.number_input('Account balance ($)', min_value=1.0, value=10000.0, step=100.0, key=f'{prefix}_account')
        risk = b.number_input('Risk per trade (%)', min_value=0.1, max_value=100.0, value=1.0, step=0.1, key=f'{prefix}_risk')
        direction = st.selectbox('Direction', ['Long', 'Short'], index=0 if default_direction == 'Long' else 1, key=f'{prefix}_direction')
        c, d, e = st.columns(3)
        entry = c.number_input('Entry ($)', min_value=0.01, value=100.0, key=f'{prefix}_entry')
        stop = d.number_input('Stop ($)', min_value=0.01, value=98.0 if direction == 'Long' else 102.0, key=f'{prefix}_stop')
        target = e.number_input('Target ($)', min_value=0.01, value=105.0 if direction == 'Long' else 95.0, key=f'{prefix}_target')
        try:
            r = calculate_trade(account, risk, entry, stop, target, direction)
        except ValueError as exc:
            st.warning(str(exc))
            return
        x, y, z = st.columns(3)
        x.metric('Whole shares', r['shares'])
        y.metric('Planned loss at stop', f"${r['planned_loss']:,.2f}")
        z.metric('Potential profit at target', f"${r['potential_profit']:,.2f}")
        st.caption(f"Risk budget ${r['risk_budget']:,.2f} · Stop distance {r['stop_distance_pct']:.2f}% · Reward/risk {r['reward_risk']:.2f}:1")
        st.caption('Illustrative only: stops can fill at worse prices; fees, slippage, gaps and short borrow costs are excluded. No trade is executed.')
