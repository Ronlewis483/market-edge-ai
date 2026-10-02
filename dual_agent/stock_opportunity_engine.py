"""Automatic stock research: next-hour trades and weekly-reviewed 63-session holds.

Read-only Alpaca data. No orders. Strategy selection and verification use separate
chronological periods, include fixed round-trip costs, and never use live outcomes.
"""
import numpy as np
import pandas as pd
import requests

ENGINE_VERSION = 3
DEFAULT_SYMBOLS = ['AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AVGO','AMD','JPM','LLY','XOM']
STRATEGIES = {'hour': ['Trend + VWAP', 'Range breakout', 'VWAP reclaim'],
              'hold': ['Long-term momentum', '63-day breakout', 'Trend pullback']}


def _get(base, path, key, secret, params=None):
    response = requests.get(base+path, headers={'APCA-API-KEY-ID':key,
        'APCA-API-SECRET-KEY':secret}, params=params, timeout=20)
    if response.status_code != 200:
        raise RuntimeError('Alpaca request failed (HTTP '+str(response.status_code)+'). Check credentials and data access.')
    return response.json()


def fetch_clock(key, secret, trading_url='https://paper-api.alpaca.markets'):
    if trading_url not in ['https://paper-api.alpaca.markets','https://api.alpaca.markets']:
        raise ValueError('Use the official Alpaca paper or live API URL.')
    return _get(trading_url, '/v2/clock', key, secret)


def fetch_quotes(symbols, key, secret, feed='iex'):
    return _get('https://data.alpaca.markets', '/v2/stocks/quotes/latest', key, secret,
        {'symbols':','.join(symbols), 'feed':feed}).get('quotes', {})


def fetch_bars(symbols, key, secret, horizon, feed='iex', now=None):
    now = pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    days = 50 if horizon=='hour' else 365*6
    params = {'symbols':','.join(symbols), 'timeframe':'5Min' if horizon=='hour' else '1Day',
        'start':(now-pd.Timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'end':now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'adjustment':'all', 'feed':feed,
        'sort':'asc', 'limit':10000}
    rows = []
    for page in range(40):
        payload = _get('https://data.alpaca.markets', '/v2/stocks/bars', key, secret, params)
        for symbol, bars in payload.get('bars', {}).items():
            rows.extend(dict(bar, symbol=symbol) for bar in bars)
        token = payload.get('next_page_token')
        if not token: break
        params['page_token'] = token
    else:
        raise RuntimeError('Market history is incomplete; reduce the symbol list.')
    if not rows: raise RuntimeError('Alpaca returned no history for this feed.')
    frame = pd.DataFrame(rows).rename(columns={'t':'timestamp','o':'open','h':'high','l':'low','c':'close','v':'volume','vw':'vwap'})
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='coerce')
    frame = frame.dropna(subset=['timestamp']).drop_duplicates(['symbol','timestamp']).sort_values(['symbol','timestamp'])
    local = frame.timestamp.dt.tz_convert('America/New_York')
    if horizon == 'hour':
        minute = local.dt.hour*60+local.dt.minute
        frame = frame[(minute>=570)&(minute<960)&(frame.timestamp+pd.Timedelta(minutes=5)<=now)]
    else:
        # Today's daily bar is incomplete until the exchange session finishes.
        frame = frame[local.dt.date < now.tz_convert('America/New_York').date()]
    return frame.reset_index(drop=True)


def indicators(frame, horizon):
    g = frame.copy().sort_values('timestamp').reset_index(drop=True)
    c = g.close
    prior = c.shift()
    g['atr'] = pd.concat([g.high-g.low, (g.high-prior).abs(), (g.low-prior).abs()],axis=1).max(axis=1).rolling(14).mean()
    if horizon == 'hour':
        day = g.timestamp.dt.tz_convert('America/New_York').dt.date
        # Reset intraday indicators each session, with no overnight bars in VWAP.
        g['session'] = day
        g['fast'] = g.groupby('session').close.transform(lambda v:v.ewm(span=9,adjust=False).mean())
        g['slow'] = g.groupby('session').close.transform(lambda v:v.ewm(span=20,adjust=False).mean())
        weighted = (g.vwap if 'vwap' in g else (g.high+g.low+g.close)/3)*g.volume
        g['session_vwap'] = weighted.groupby(day).cumsum()/g.volume.groupby(day).cumsum().replace(0,np.nan)
        g['volume_mean'] = g.groupby('session').volume.transform(lambda v:v.shift().rolling(20,min_periods=12).mean())
        prev_close = g.groupby('session').close.shift()
        prev_vwap = g.groupby('session').session_vwap.shift()
        high = g.groupby('session').high.transform(lambda v:v.shift().rolling(20,min_periods=20).max())
        rising = g['slow'] > g.groupby('session').slow.shift(3)
        g[STRATEGIES['hour'][0]] = (c>g.fast)&(g.fast>g.slow)&(c>g.session_vwap)&rising&(g.volume>.8*g.volume_mean)
        g[STRATEGIES['hour'][1]] = (c>high)&(c>g.session_vwap)&(g.volume>.8*g.volume_mean)
        g[STRATEGIES['hour'][2]] = (prev_close<=prev_vwap)&(c>g.session_vwap)&rising
    else:
        g['ma50'] = c.rolling(50).mean();g['ma200'] = c.rolling(200).mean()
        change = c.diff(); gain=change.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-change.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
        rsi = 100-100/(1+gain/loss.replace(0,np.nan)); rsi = rsi.where(loss!=0,100)
        trend = (c>g.ma200)&(g.ma50>g.ma200)
        g[STRATEGIES['hold'][0]] = trend&(c.pct_change(63)>0)
        g[STRATEGIES['hold'][1]] = trend&(c>g.high.shift().rolling(63).max())
        g[STRATEGIES['hold'][2]] = trend&(rsi<50)&(rsi>25)
    return g


def historical_trades(g, strategy, horizon, split, section):
    """Next-bar entries and non-overlapping positions per symbol; costs included.

    Hour trades close at a 1.5 ATR stop or after 12 five-minute bars. Hold trades
    exit after 63 sessions. Entry prices cannot use the signal bar's closing price.
    """
    length = 12 if horizon=='hour' else 63
    fee = .001 if horizon=='hour' else .002
    records=[]; i=0
    signals=g[strategy].fillna(False).to_numpy(); stamps=g.timestamp.tolist()
    o=g.open.to_numpy();h=g.high.to_numpy();lo=g.low.to_numpy();c=g.close.to_numpy();atr=g.atr.to_numpy()
    while i+length<len(g):
        if not signals[i] or not np.isfinite(atr[i]) or atr[i]<=0:
            i+=1;continue
        entry=i+1; end=i+length
        if horizon=='hour' and (g.session.iloc[entry]!=g.session.iloc[end] or stamps[end]-stamps[i]>pd.Timedelta(hours=1)):
            i+=1;continue
        if section=='selection' and stamps[end]>=split:
            i+=1;continue
        if section=='verification' and stamps[i]<split:
            i+=1;continue
        if not np.isfinite(o[entry]) or o[entry]<=0:
            i+=1;continue
        exit_price=c[end]; actual_end=end
        if horizon=='hour':
            stop=o[entry]-1.5*atr[i]
            if stop<=0: i+=1;continue
            for j in range(entry,end+1):
                if lo[j]<=stop:
                    exit_price=min(o[j],stop);actual_end=j;break
        net=exit_price/o[entry]-1-fee
        if np.isfinite(net):records.append({'return':float(net),'signal':stamps[i],'exit':stamps[actual_end]})
        i=actual_end+1
    return records


def summarize(trades):
    values=np.asarray([r['return'] for r in trades],dtype=float)
    return {'trades':len(values),'mean_net_return':float(values.mean()) if len(values) else None,
            'win_rate':float((values>0).mean()) if len(values) else None}


def select_strategy(frames, horizon):
    dates=sorted({t for g in frames.values() for t in g.timestamp})
    if len(dates)<300:return {'qualified':False,'reason':'More market history is needed for strategy selection.'}
    split=dates[int(len(dates)*.7)]
    selection={}; verification={}
    for strategy in STRATEGIES[horizon]:
        chosen=[]; held=[]
        for g in frames.values():
            chosen.extend(historical_trades(g,strategy,horizon,split,'selection'))
            held.extend(historical_trades(g,strategy,horizon,split,'verification'))
        selection[strategy]=summarize(chosen);verification[strategy]=summarize(held)
    candidates=[s for s in STRATEGIES[horizon] if selection[s]['trades']>=25]
    if not candidates:return {'qualified':False,'reason':'No strategy has 25 completed selection-period trades.'}
    winner=max(candidates,key=lambda s:selection[s]['mean_net_return'])
    holdout=verification[winner]
    passed=holdout['trades']>=8 and holdout['mean_net_return'] is not None and holdout['mean_net_return']>0 and selection[winner]['mean_net_return']>0
    return {'qualified':passed,'strategy':winner,'selection':selection[winner],'verification':holdout,
            'split_utc':split.isoformat(), 'reason':None if passed else ('The selected strategy needs at least 8 recent-period trades.' if holdout['trades']<8 else 'The selected strategy did not show a positive average return after estimated costs in both historical periods.'),
            'cost_assumption_bps':10 if horizon=='hour' else 20}


def opportunities(raw, quotes, clock, horizon, now=None):
    now=pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    frames={symbol:indicators(g,horizon) for symbol,g in raw.groupby('symbol')}
    evidence=select_strategy(frames,horizon)
    result={'picks':[],'strategy_evidence':evidence,'as_of':now.isoformat(),'horizon':horizon,'warnings':[]}
    if not evidence.get('qualified'):
        result['message']=evidence['reason'];return result
    if not clock.get('is_open'):
        result['message']='Market closed. These strategies will be checked again during the regular session.';return result
    if horizon=='hour' and pd.Timestamp(clock['next_close'])-now<pd.Timedelta(hours=1):
        result['message']='Less than one hour remains in the regular session. No new hour-long trade.';return result
    strategy=evidence['strategy']
    for symbol,g in frames.items():
        if g.empty or not bool(g.iloc[-1][strategy]):continue
        row=g.iloc[-1];quote=quotes.get(symbol,{})
        stamp=pd.to_datetime(quote.get('t'),utc=True,errors='coerce')
        try:ask=float(quote['ap']);bid=float(quote['bp'])
        except (KeyError,ValueError,TypeError):continue
        if pd.isna(stamp) or stamp>now or now-stamp>pd.Timedelta(seconds=90) or not 0<bid<=ask:continue
        spread=(ask-bid)/((ask+bid)/2)
        if spread>.005 or ask<5:continue
        if horizon=='hour' and now-pd.Timestamp(row.timestamp)>pd.Timedelta(minutes=12):continue
        if horizon=='hold' and now-pd.Timestamp(row.timestamp)>pd.Timedelta(days=5):continue
        if horizon=='hour' and float(row.volume)*float(row.close)<50000:continue
        atr=float(row.atr)
        if not np.isfinite(atr) or atr<=0:continue
        # Don't chase a price far from the latest completed signal bar.
        if abs(ask/float(row.close)-1)>(.01 if horizon=='hour' else .05):continue
        split=pd.Timestamp(evidence['split_utc'])
        sample=summarize(historical_trades(g,strategy,horizon,split,'verification'))
        if sample['trades']<2 or sample['mean_net_return'] is None or sample['mean_net_return']<=0:continue
        picks={'symbol':symbol,'strategy':strategy,'entry_reference':ask,'spread_pct':spread,
               'quote_time':stamp.isoformat(),'signal_time':pd.Timestamp(row.timestamp).isoformat(),
               'historical_mean_net_return':sample['mean_net_return'],'historical_win_rate':sample['win_rate'],
               'verification_trades':sample['trades'],'holding_period':'Up to 1 hour' if horizon=='hour' else 'About 63 trading sessions',
               'direction':'Long','rank_score':sample['mean_net_return']}
        if horizon=='hour':
            stop=ask-1.5*atr
            if stop<=0:continue
            picks.update(stop_reference=stop,exit_time=min(now+pd.Timedelta(hours=1),pd.Timestamp(clock['next_close'])).isoformat())
        result['picks'].append(picks)
    result['picks'].sort(key=lambda p:(-p['rank_score'],-p['verification_trades'],p['symbol']))
    result['picks']=result['picks'][:5]
    result['message']='Candidates ranked by observed mean net return in the separate verification period.' if result['picks'] else 'No current setup passes the signal, history, quote and liquidity checks.'
    return result


def fetch_universe(key, secret, trading_url):
    if trading_url not in ['https://paper-api.alpaca.markets','https://api.alpaca.markets']:
        raise ValueError('Use the official Alpaca paper or live API URL.')
    assets = _get(trading_url, '/v2/assets', key, secret,
                  {'status':'active','asset_class':'us_equity'})
    return sorted({a['symbol'] for a in assets if a.get('tradable') and
                   a.get('exchange') not in ['OTC', 'CRYPTO']})


def screen_universe(symbols, key, secret, feed):
    """Screen every provider-listed symbol; never truncate to a preset shortlist."""
    snapshots = {}
    for i in range(0,len(symbols),100):
        snapshots.update(_get('https://data.alpaca.markets','/v2/stocks/snapshots',key,secret,
                              {'symbols':','.join(symbols[i:i+100]),'feed':feed}))
    eligible=[]
    for symbol in symbols:
        snap=snapshots.get(symbol) or {}
        bar=snap.get('prevDailyBar') or {}
        try:
            price=float(bar['c']); dollars=price*float(bar['v'])
        except (KeyError, TypeError, ValueError):continue
        if price>=5 and dollars>=1000000:eligible.append(symbol)
    return eligible, {'listed':len(symbols),'snapshots':len(snapshots),'eligible':len(eligible)}


def scan_market(key, secret, feed, trading_url, cache):
    """Background-safe scan; cache is private to a single sequential worker."""
    import time
    started=time.monotonic()
    cached=cache.get('universe')
    if not cached or time.monotonic()-cached[0]>86400:
        cache['universe']=(time.monotonic(),fetch_universe(key,secret,trading_url))
    symbols=cache['universe'][1]
    clock=fetch_clock(key,secret,trading_url)
    if not clock.get('is_open'):
        return {'closed':True,'coverage':{'listed':len(symbols)},'as_of':pd.Timestamp.now(tz='UTC').isoformat()}
    eligible,coverage=screen_universe(symbols,key,secret,feed)
    results={}; histories={}
    for horizon in ['hour','hold']:
        frames=[]; failed=[]
        for i in range(0,len(eligible),10):
            batch=tuple(eligible[i:i+10]); token=(horizon,batch)
            cached=cache.get(token); ttl=300 if horizon=='hour' else 3600
            try:
                if not cached or time.monotonic()-cached[0]>ttl:
                    cached=(time.monotonic(),fetch_bars(batch,key,secret,horizon,feed))
                    cache[token]=cached
                frames.append(cached[1])
            except Exception as exc:
                failed.append({'symbols':list(batch),'reason':str(exc)})
        if not frames:
            results[horizon]={'picks':[],'strategy_evidence':{},'message':'No usable history was returned for the eligible stocks.'}
        else:
            raw=pd.concat(frames,ignore_index=True)
            histories[horizon]=raw
            results[horizon]={'picks':[],'strategy_evidence':{},'history_symbols':raw.symbol.nunique()}
        results[horizon]['failed_batches']=failed
    for horizon,raw in histories.items():
        quotes={}
        present=sorted(raw.symbol.unique())
        for i in range(0,len(present),100):
            quotes.update(fetch_quotes(present[i:i+100],key,secret,feed))
        clock=fetch_clock(key,secret,trading_url)
        metadata=results[horizon]
        results[horizon]=opportunities(raw,quotes,clock,horizon)
        results[horizon].update(history_symbols=len(present),failed_batches=metadata['failed_batches'])
    return {'closed':False,'results':results,'coverage':coverage,
            'as_of':pd.Timestamp.now(tz='UTC').isoformat(),'seconds':time.monotonic()-started}
