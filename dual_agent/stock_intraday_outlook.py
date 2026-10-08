"""Single-stock intraday trend and historical-analogue outlooks; read-only."""
import re
import numpy as np
import pandas as pd

HORIZONS = (15, 30, 45, 60)
FEATURES = ['momentum_15', 'momentum_30', 'momentum_60', 'session_return', 'vwap_gap']


def resolve_symbol(question, fallback):
    if re.search(r'\b(bitcoin|btc|ethereum|eth|crypto|dogecoin|doge|solana|sol)\b', question, re.I) or str(fallback).strip().upper() in {'BTC','BTC/USD','ETH','ETH/USD','DOGE','SOL'}:
        raise ValueError('This trend check currently covers stocks and ETFs. Bitcoin and other cryptocurrencies need a crypto data connection; no AAPL forecast was substituted.')
    explicit = re.search(r'\$([A-Za-z][A-Za-z0-9.\-]{0,9})\b', question)
    words = re.findall(r'\b[A-Z][A-Z0-9.\-]{0,9}\b', question)
    ignored = {'IS','IN','AN','A','I','THE','TODAY','UPTREND','DOWNTREND','STOCK','MIN','MINUTES','HR','HOUR','FOR','NEXT','PREDICT','VWAP','US','AM','PM'}
    candidates = [word for word in words if word not in ignored]
    if explicit:
        value = explicit.group(1)
    elif len(candidates) == 1:
        value = candidates[0]
    elif len(candidates) > 1:
        raise ValueError('Ask about one ticker at a time, or use $TICKER to identify it.')
    else:
        value = fallback
    symbol = str(value).strip().upper()
    if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', symbol):
        raise ValueError('Enter a valid stock ticker, such as AAPL or BRK.B.')
    return symbol


def prepare_bars(raw, symbol, now):
    data = raw.loc[raw.symbol == symbol].copy()
    data['timestamp'] = pd.to_datetime(data.timestamp, utc=True, errors='coerce')
    for column in ['open','high','low','close','volume']:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna(subset=['timestamp','open','high','low','close','volume'])
    data = data.loc[(data.close > 0) & (data.open > 0) & (data.volume >= 0)]
    data = data.drop_duplicates('timestamp').sort_values('timestamp')
    data['bar_end'] = data.timestamp + pd.Timedelta(minutes=5)
    data = data.loc[data.bar_end <= now]
    local = data.timestamp.dt.tz_convert('America/New_York')
    minutes = local.dt.hour*60 + local.dt.minute
    data = data.loc[(minutes >= 570) & (minutes < 960)].copy()
    data['session'] = data.timestamp.dt.tz_convert('America/New_York').dt.date
    # Never manufacture missing bars or use overnight returns as intraday momentum.
    data = data.reset_index(drop=True)
    for _, indexes in data.groupby('session').groups.items():
        g = data.loc[indexes]
        for period in [15,30,60]:
            shifted_time = g.timestamp.shift(period//5)
            continuous = (g.timestamp-shifted_time) == pd.Timedelta(minutes=period)
            data.loc[indexes,'momentum_'+str(period)] = (g.close/g.close.shift(period//5)-1).where(continuous)
        opening = float(g.open.iloc[0])
        data.loc[indexes,'session_return'] = g.close/opening-1
        # Use provider bar VWAP when present, otherwise the bar's typical price.
        typical = (g.high+g.low+g.close)/3
        price = pd.to_numeric(g['vwap'], errors='coerce').fillna(typical) if 'vwap' in g else typical
        cumulative_volume = g.volume.cumsum().replace(0,np.nan)
        vwap = (price*g.volume).cumsum()/cumulative_volume
        data.loc[indexes,'session_vwap'] = vwap
        data.loc[indexes,'vwap_gap'] = g.close/vwap-1
    return data


def fifteen_minute_bars(data):
    rows = []
    for _, group in data.groupby('session'):
        for stamp, bars in group.groupby(group.timestamp.dt.floor('15min')):
            expected = pd.date_range(stamp, periods=3, freq='5min')
            if len(bars) != 3 or not bars.timestamp.reset_index(drop=True).equals(pd.Series(expected)):
                continue
            rows.append({'Time':stamp+pd.Timedelta(minutes=15),'Open':float(bars.open.iloc[0]),
                'High':float(bars.high.max()),'Low':float(bars.low.min()),'Close':float(bars.close.iloc[-1]),'Volume':float(bars.volume.sum())})
    return pd.DataFrame(rows)


def analyze(raw, symbol, clock, now=None):
    now = pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    data = prepare_bars(raw, symbol, now)
    if data.empty:
        raise ValueError('No completed regular-session bars are available for this stock.')
    today = now.tz_convert('America/New_York').date()
    current = data.loc[data.session == today]
    latest = data.iloc[-1]
    result = {'symbol':symbol,'as_of':latest.bar_end.isoformat(),'reference_price':float(latest.close),
        'trend':'No current-session data','today_return':None,'vwap':None,'bars_15':fifteen_minute_bars(current), 'outlooks':[]}
    is_open = bool(clock.get('is_open'))
    close = pd.to_datetime(clock.get('next_close'), utc=True, errors='coerce')
    fresh = not current.empty and pd.Timedelta(0) <= now-latest.bar_end <= pd.Timedelta(minutes=7)
    if not current.empty:
        today_change = float(latest.session_return)
        momentum = latest.momentum_15
        gap = latest.vwap_gap
        if pd.notna(momentum) and pd.notna(gap):
            trend = 'Uptrend today' if today_change>0 and momentum>0 and gap>0 else 'Downtrend today' if today_change<0 and momentum<0 and gap<0 else 'Mixed / sideways today'
        else:
            trend = 'Not enough completed bars to determine today’s trend'
        result.update(trend=trend,today_return=today_change,vwap=float(latest.session_vwap) if pd.notna(latest.session_vwap) else None)
    # Predictions are anchored to the most recent completed five-minute close.
    for minutes in HORIZONS:
        target = latest.bar_end+pd.Timedelta(minutes=minutes)
        output = {'minutes':minutes,'target_time':target.isoformat(),'status':'Unavailable'}
        result['outlooks'].append(output)
        if not is_open:
            output['reason'] = 'Regular market session is closed. Historical trend remains visible.'; continue
        if not fresh:
            output['reason'] = 'Current-session bars are missing or stale. No live outlook generated.'; continue
        if pd.isna(close) or target > close or target <= now:
            output['reason'] = 'This horizon is outside the remaining regular trading session.'; continue
        if not np.isfinite(latest[FEATURES].to_numpy(dtype=float)).all():
            output['reason'] = 'Need a full hour of continuous bars to compare intraday conditions.'; continue
        candidates = []
        for _, group in data.groupby('session'):
            future = group.close.shift(-minutes//5)
            future_end = group.bar_end.shift(-minutes//5)
            valid = ((future_end-group.bar_end)==pd.Timedelta(minutes=minutes)) & (future_end<=latest.bar_end)
            features = group[FEATURES].copy()
            features['future_return'] = future/group.close-1
            features['time'] = group.bar_end
            candidates.append(features.loc[valid].dropna())
        history = pd.concat(candidates, ignore_index=True)
        if len(history)<30:
            output['reason'] = 'Insufficient completed historical outcomes for this horizon.'; continue
        matrix = history[FEATURES].to_numpy(dtype=float)
        scale = np.std(matrix,axis=0);scale = np.where(scale>1e-8,scale,1.)
        distance = np.mean(((matrix-latest[FEATURES].to_numpy(dtype=float))/scale)**2,axis=1)
        history['distance'] = distance
        # Avoid counting heavily overlapping outcomes as separate analogue evidence.
        selected = []; used = []
        for _, row in history.sort_values('distance').iterrows():
            stamp = row['time']
            if all(abs(stamp-other)>=pd.Timedelta(minutes=minutes) for other in used):
                selected.append(float(row.future_return));used.append(stamp)
            if len(selected)>=60:break
        if len(selected)<30:
            output['reason'] = 'Fewer than 30 non-overlapping historical comparisons.'; continue
        values = np.asarray(selected)
        median = float(np.median(values));low,high = np.quantile(values,[.1,.9])
        output.update(status='Ready',direction='Upward' if median>0 else 'Downward' if median<0 else 'Flat',
            expected_return=median,projected_price=float(latest.close*(1+median)),
            range_low=float(latest.close*(1+low)),range_high=float(latest.close*(1+high)),
            historical_up_frequency=float(np.mean(values>0)), comparisons=len(values))
    return result


def render_intraday_outlook(key, secret, feed, url):
    import streamlit as st
    from dual_agent import stock_opportunity_engine as engine
    st.subheader('🔎 Ask about a stock · intraday trend')
    st.caption('Stocks and ETFs · enter a ticker or use $TICKER in your question. Crypto is not connected yet.')
    with st.form('stock_intraday_question'):
        ticker = st.text_input('Stock ticker',value='AAPL',key='stock_intraday_ticker')
        question = st.text_input('Your question',placeholder='Is AAPL in an uptrend today? What is the next 30-minute outlook?')
        submit = st.form_submit_button('Check trend & next hour',type='primary',use_container_width=True)
    if submit:
        # Remove the old answer before requesting another ticker; failures cannot leave a misleading old card.
        st.session_state.pop('stock_intraday_answer',None)
        try:
            symbol = resolve_symbol(question,ticker)
            with st.spinner('Checking '+symbol+' completed bars…'):
                raw = engine.fetch_bars([symbol],key,secret,'hour',feed)
                clock = engine.fetch_clock(key,secret,url)
                answer = analyze(raw,symbol,clock)
                answer['feed'] = feed;answer['question'] = question
                st.session_state['stock_intraday_answer'] = answer
        except Exception as exc:
            st.error('Stock trend check could not finish: '+str(exc))
    answer = st.session_state.get('stock_intraday_answer')
    if not answer:return
    if answer.get('feed')!=feed:
        st.info('Check again to use the newly selected data feed.');return
    stamp = pd.Timestamp(answer['as_of']).tz_convert('America/Chicago')
    st.markdown('### '+answer['symbol']+' · '+answer['trend'])
    st.caption('Completed-bar reference: '+stamp.strftime('%b %d, %Y · %-I:%M %p CT')+' · '+answer['feed'].upper())
    st.metric('Reference close',f"${answer['reference_price']:,.2f}")
    if answer['today_return'] is not None:st.caption(f"Change from regular-session open: {answer['today_return']:+.2%}")
    if answer['vwap'] is not None:st.caption(f"Session VWAP: ${answer['vwap']:,.2f}")
    age = pd.Timestamp.now(tz='UTC')-pd.Timestamp(answer['as_of'])
    expired = age>pd.Timedelta(minutes=7)
    if expired:st.info('This answer is a saved snapshot. Check again for a current outlook.')
    for column, forecast in zip(st.columns(4),answer['outlooks']):
        with column:
            st.markdown('**Next '+str(forecast['minutes'])+' min**')
            if forecast['status']!='Ready':
                st.caption(forecast['reason']);continue
            if expired:
                st.caption('Snapshot expired · refresh before using this outlook.');continue
            st.metric(forecast['direction'],f"${forecast['projected_price']:,.2f}",f"{forecast['expected_return']:+.2%}")
            target = pd.Timestamp(forecast['target_time']).tz_convert('America/Chicago')
            st.caption('Target: '+target.strftime('%-I:%M %p CT'))
            st.caption(f"Historical up frequency: {forecast['historical_up_frequency']:.0%} · {forecast['comparisons']} comparisons")
            st.caption(f"Historical 10th–90th price range: ${forecast['range_low']:.2f}–${forecast['range_high']:.2f}")
    bars = answer['bars_15']
    if not bars.empty:
        display = bars.copy();display['Time'] = pd.to_datetime(display.Time,utc=True).dt.tz_convert('America/Chicago')
        st.markdown('**Today in 15-minute increments**')
        st.line_chart(display.set_index('Time')[['Close']])
        with st.expander('View 15-minute bars'):
            st.dataframe(display,hide_index=True,use_container_width=True)
    st.caption('Historical-analogue outlooks use completed five-minute bars and are anchored to the reference close. Up frequency is historical evidence, not a calibrated probability or measured forecast accuracy. Price ranges describe historical comparisons and exclude trading costs.')
