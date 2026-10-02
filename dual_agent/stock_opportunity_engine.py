"""Automatic stock research: next-hour trades and weekly-reviewed 63-session holds.

Read-only Alpaca data. No orders. Strategy selection and verification use separate
chronological periods, include fixed round-trip costs, and never use live outcomes.
"""
import numpy as np
import pandas as pd
import requests

ENGINE_VERSION = 9
DEFAULT_SYMBOLS = ['AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AVGO','AMD','JPM','LLY','XOM']
STRATEGIES = {'hour': ['Trend + VWAP', 'Range breakout', 'VWAP reclaim', 'EMA pullback', 'Opening range breakout', 'Momentum continuation', 'RSI recovery', 'Bollinger recovery'],
              'hold': ['Long-term momentum', '63-day breakout', 'Trend pullback', '50-day reclaim', '20-day breakout']}


# Shared request pacing across workers in this process.
import threading
import time
_REQUEST_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0


class RateLimitError(RuntimeError):
    pass


def _get(base, path, key, secret, params=None):
    global _NEXT_REQUEST
    for attempt in range(3):
        with _REQUEST_LOCK:
            delay=max(0.0,_NEXT_REQUEST-time.monotonic())
            if delay>0:time.sleep(delay)
            _NEXT_REQUEST=time.monotonic()+.7
        response=requests.get(base+path,headers={'APCA-API-KEY-ID':key,
            'APCA-API-SECRET-KEY':secret},params=params,timeout=20)
        if response.status_code==429:
            try:wait=max(5.0,float(response.headers.get('Retry-After',15)))
            except (TypeError,ValueError):wait=15.0
            with _REQUEST_LOCK:
                _NEXT_REQUEST=max(_NEXT_REQUEST,time.monotonic()+min(wait,30))
            if attempt==2 or wait>30:
                raise RateLimitError('Alpaca temporarily limited requests (HTTP 429). The scanner will pause and retry automatically; this does not indicate invalid credentials.')
            continue
        if response.status_code!=200:
            raise RuntimeError('Alpaca request failed (HTTP '+str(response.status_code)+'). Check credentials and data access.')
        return response.json()


def fetch_clock(key, secret, trading_url='https://paper-api.alpaca.markets'):
    if trading_url not in ['https://paper-api.alpaca.markets','https://api.alpaca.markets']:
        raise ValueError('Use the official Alpaca paper or live API URL.')
    return _get(trading_url, '/v2/clock', key, secret)


def fetch_quotes(symbols, key, secret, feed='iex'):
    return _get('https://data.alpaca.markets', '/v2/stocks/quotes/latest', key, secret,
        {'symbols':','.join(symbols), 'feed':feed}).get('quotes', {})


def fetch_bars(symbols, key, secret, horizon, feed='iex', now=None, start=None):
    now = pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    days = 50 if horizon=='hour' else 365*6
    params = {'symbols':','.join(symbols), 'timeframe':'5Min' if horizon=='hour' else '1Day',
        'start':(pd.Timestamp(start) if start is not None else now-pd.Timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ'),
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
        previous_fast=g.groupby('session').fast.shift()
        g['EMA pullback']=(prev_close<=previous_fast)&(c>g.fast)&(g.fast>g.slow)&(c>g.session_vwap)
        minute=g.timestamp.dt.tz_convert('America/New_York').dt.hour*60+g.timestamp.dt.tz_convert('America/New_York').dt.minute
        opening=g.high.where(minute<600).groupby(day).cummax().groupby(day).ffill()
        g['Opening range breakout']=(minute>=600)&(c>opening)&(prev_close<=opening)&(c>g.session_vwap)
        prior_three=g.groupby('session').close.shift(3)
        g['Momentum continuation']=(c>prior_three)&(g.fast>g.slow)&rising&(c>g.session_vwap)&(g.volume>.8*g.volume_mean)
        delta=g.groupby('session').close.diff()
        gains=delta.clip(lower=0).groupby(day).transform(lambda v:v.ewm(alpha=1/14,adjust=False).mean())
        losses=(-delta.clip(upper=0)).groupby(day).transform(lambda v:v.ewm(alpha=1/14,adjust=False).mean())
        rsi=100-100/(1+gains/losses.replace(0,np.nan))
        previous_rsi=rsi.groupby(day).shift()
        g['RSI recovery']=(previous_rsi<35)&(rsi>=35)&rising
        mean=g.groupby('session').close.transform(lambda v:v.rolling(20).mean())
        std=g.groupby('session').close.transform(lambda v:v.rolling(20).std())
        lower=mean-2*std
        g['Bollinger recovery']=(prev_close<lower.groupby(day).shift())&(c>=lower)&rising
    else:
        g['ma50'] = c.rolling(50).mean();g['ma200'] = c.rolling(200).mean()
        change = c.diff(); gain=change.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-change.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
        rsi = 100-100/(1+gain/loss.replace(0,np.nan)); rsi = rsi.where(loss!=0,100)
        trend = (c>g.ma200)&(g.ma50>g.ma200)
        g[STRATEGIES['hold'][0]] = trend&(c.pct_change(63)>0)
        g[STRATEGIES['hold'][1]] = trend&(c>g.high.shift().rolling(63).max())
        g[STRATEGIES['hold'][2]] = trend&(rsi<50)&(rsi>25)
        g['50-day reclaim']=(c>g.ma200)&(prior<=g.ma50.shift())&(c>g.ma50)
        g['20-day breakout']=trend&(c>g.high.shift().rolling(20).max())
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
    checked=[]
    for strategy in STRATEGIES[horizon]:
        chosen=selection[strategy];held=verification[strategy]
        passed=(chosen['trades']>=25 and held['trades']>=8 and
                (horizon=='hour' or (chosen['mean_net_return'] is not None and chosen['mean_net_return']>0)) and
                held['mean_net_return'] is not None and held['mean_net_return']>0)
        checked.append({'qualified':passed,'strategy':strategy,'selection':chosen,'verification':held,
                        'split_utc':split.isoformat(),'cost_assumption_bps':10 if horizon=='hour' else 20})
    qualifying=[item for item in checked if item['qualified']]
    return {'qualified':bool(qualifying),'strategies':checked,'qualified_strategies':qualifying,
            'reason':None if qualifying else 'None of the independently evaluated strategies passed the history and positive net-return checks.'}


def _single_opportunities(raw, quotes, clock, horizon, now=None, prepared_frames=None, prepared_evidence=None):
    now=pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    frames=prepared_frames if prepared_frames is not None else {symbol:indicators(g,horizon) for symbol,g in raw.groupby('symbol')}
    evidence=prepared_evidence if prepared_evidence is not None else select_strategy(frames,horizon)
    result={'picks':[],'strategy_evidence':evidence,'as_of':now.isoformat(),'horizon':horizon,'warnings':[]}
    if not evidence.get('qualified'):
        result['message']=evidence['reason'];return result
    if not clock.get('is_open'):
        result['message']='Market closed. These strategies will be checked again during the regular session.';return result
    if horizon=='hour' and pd.Timestamp(clock['next_close'])-now<pd.Timedelta(hours=1):
        result['message']='Less than one hour remains in the regular session. No new hour-long trade.';return result
    strategy=evidence['strategy']
    result['exclusions']=[]
    def reject(symbol,reason):
        result['exclusions'].append({'symbol':symbol,'strategy':strategy,'reason':reason})
    for symbol,g in frames.items():
        if g.empty or not bool(g.iloc[-1][strategy]):reject(symbol,'No current entry signal');continue
        row=g.iloc[-1];quote=quotes.get(symbol,{})
        stamp=pd.to_datetime(quote.get('t'),utc=True,errors='coerce')
        try:ask=float(quote['ap']);bid=float(quote['bp'])
        except (KeyError,ValueError,TypeError):reject(symbol,'Live bid or ask unavailable');continue
        if pd.isna(stamp) or stamp>now or now-stamp>pd.Timedelta(seconds=90) or not 0<bid<=ask:reject(symbol,'Quote missing, older than 90 seconds, or invalid');continue
        spread=(ask-bid)/((ask+bid)/2)
        if spread>.005 or ask<5:reject(symbol,'Spread above 0.5% or price below five dollars');continue
        if horizon=='hour' and now-pd.Timestamp(row.timestamp)>pd.Timedelta(minutes=12):reject(symbol,'Completed intraday bars are older than 12 minutes');continue
        if horizon=='hold' and now-pd.Timestamp(row.timestamp)>pd.Timedelta(days=5):reject(symbol,'Daily history is stale');continue
        if horizon=='hour' and float(row.volume)*float(row.close)<50000:reject(symbol,'Latest five-minute dollar volume below 50,000 on this feed');continue
        atr=float(row.atr)
        if not np.isfinite(atr) or atr<=0:reject(symbol,'Not enough usable volatility history');continue
        # Don't chase a price far from the latest completed signal bar.
        if abs(ask/float(row.close)-1)>(.01 if horizon=='hour' else .05):reject(symbol,'Current price moved too far from the signal');continue
        split=pd.Timestamp(evidence['split_utc'])
        sample=summarize(historical_trades(g,strategy,horizon,split,'verification'))
        if sample['trades']<2:reject(symbol,'Fewer than two recent historical trades for this stock and strategy');continue
        if sample['mean_net_return'] is None or sample['mean_net_return']<=0:reject(symbol,'This stock and strategy have no positive recent average net return');continue
        picks={'symbol':symbol,'strategy':strategy,'entry_reference':ask,'spread_pct':spread,
               'quote_time':stamp.isoformat(),'signal_time':pd.Timestamp(row.timestamp).isoformat(),
               'historical_mean_net_return':sample['mean_net_return'],'historical_win_rate':sample['win_rate'],
               'verification_trades':sample['trades'],'holding_period':'Up to 1 hour' if horizon=='hour' else 'About 63 trading sessions',
               'direction':'Long','rank_score':sample['mean_net_return']}
        if horizon=='hour':
            # Wilson lower bound: discounts high observed win rates from tiny samples.
            n=sample['trades']; rate=sample['win_rate']; z=1.96
            support=(rate+z*z/(2*n)-z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n)))/(1+z*z/n)
            picks.update(rank_score=float(support),evidence_score=float(support),
                         evidence_label='Limited history' if n<10 else 'Historical evidence',
                         strategy_earlier_net_return=evidence['selection']['mean_net_return'])
            stop=ask-1.5*atr
            if stop<=0:reject(symbol,'Stop reference is invalid');continue
            picks.update(stop_reference=stop,exit_time=min(now+pd.Timedelta(hours=1),pd.Timestamp(clock['next_close'])).isoformat())
        result['picks'].append(picks)
    result['picks'].sort(key=lambda p:(-p['rank_score'],-p['verification_trades'],p['symbol']))
    result['picks']=result['picks'][:5]
    result['message']='Candidates ranked by observed mean net return in the separate verification period.' if result['picks'] else 'No current setup passes the signal, history, quote and liquidity checks.'
    return result


def opportunities(raw,quotes,clock,horizon,now=None,prepared_frames=None,prepared_evidence=None):
    frames=prepared_frames if prepared_frames is not None else {symbol:indicators(g,horizon) for symbol,g in raw.groupby('symbol')}
    evidence=prepared_evidence if prepared_evidence is not None else select_strategy(frames,horizon)
    result={'picks':[],'strategy_evidence':evidence,'horizon':horizon,'warnings':[]}
    if not evidence.get('qualified'):
        result['message']=evidence['reason'];return result
    groups=[_single_opportunities(raw,quotes,clock,horizon,now,frames,item) for item in evidence['qualified_strategies']]
    result['exclusions']=[item for group in groups for item in group.get('exclusions',[])]
    picks=sorted([pick for group in groups for pick in group['picks']],key=lambda p:(-p['rank_score'],-p['verification_trades'],p['symbol'],p['strategy']))
    seen=set()
    for pick in picks:
        if pick['symbol'] in seen:continue
        seen.add(pick['symbol']);result['picks'].append(pick)
        if len(result['picks'])==5:break
    result['message']='Candidates ranked across independently qualifying strategies.' if result['picks'] else groups[0]['message']
    result['as_of']=pd.Timestamp(now or pd.Timestamp.now(tz='UTC')).isoformat()
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
    for i in range(0,len(symbols),500):
        snapshots.update(_get('https://data.alpaca.markets','/v2/stocks/snapshots',key,secret,
                              {'symbols':','.join(symbols[i:i+500]),'feed':feed}))
    eligible=[]
    for symbol in symbols:
        snap=snapshots.get(symbol) or {}
        bar=snap.get('prevDailyBar') or {}
        try:
            price=float(bar['c']); dollars=price*float(bar['v'])
        except (KeyError, TypeError, ValueError):continue
        if price>=5 and dollars>=1000000:eligible.append(symbol)
    return eligible, {'listed':len(symbols),'snapshots':len(snapshots),'eligible':len(eligible),'snapshot_data':snapshots}


def shortlist(eligible, snapshots, limit=40):
    """Current-data screening, not a claim of historical strategy performance."""
    ranks=[]
    for symbol in eligible:
        snap=snapshots.get(symbol) or {}
        prev=snap.get('prevDailyBar') or {}; today=snap.get('dailyBar') or {}
        trade=snap.get('latestTrade') or {}
        try:
            prior=float(prev['c']); price=float(trade.get('p') or today.get('c') or prior)
            if price<=0 or prior<=0:continue
            dollars=prior*float(prev['v']); move=price/prior-1
        except (KeyError,ValueError,TypeError):continue
        ranks.append((symbol,dollars,move))
    # Mix liquid names with movers; include pullbacks and positive momentum.
    liquid=sorted(ranks,key=lambda r:(-r[1],r[0]))
    movers=sorted(ranks,key=lambda r:(-abs(r[2]),-r[1],r[0]))
    chosen=[]
    for i in range(max(len(liquid),len(movers))):
        for ranking in [liquid,movers]:
            if i<len(ranking) and ranking[i][0] not in chosen:chosen.append(ranking[i][0])
            if len(chosen)>=limit:return chosen
    return chosen


def reusable_history(symbols,key,secret,horizon,feed,cache):
    """Stable per-symbol cache; update recent bars instead of reloading all history."""
    import os, hashlib, time
    from pathlib import Path
    root=Path(os.environ.get('MARKET_EDGE_STOCK_CACHE','.cache/market_edge_stock_history'))
    root.mkdir(parents=True,exist_ok=True)
    now=pd.Timestamp.now(tz='UTC'); frames=[]; failed=[]; pending=[]
    for symbol in symbols:
        token=('bars_v5',feed,horizon,symbol)
        path=root/(hashlib.sha256((feed+'|'+horizon+'|'+symbol).encode()).hexdigest()+'.json.gz')
        old=cache.get(token)
        if old is None and path.exists():
            try:
                data=pd.read_json(path,orient='table',compression='gzip')
                old=(path.stat().st_mtime,data)
                cache[token]=old
            except Exception:old=None
        # Refresh complete adjusted history daily; within the day fetch only overlap.
        fresh_day=old is not None and pd.Timestamp(old[0],unit='s',tz='UTC').tz_convert('America/New_York').date()==now.tz_convert('America/New_York').date()
        if fresh_day and horizon=='hold':frames.append(old[1]);continue
        if fresh_day and time.time()-old[0]<300:frames.append(old[1]);continue
        pending.append((symbol,token,path,old if fresh_day else None))
    for i in range(0,len(pending),10):
        batch=pending[i:i+10]
        update=all(item[3] is not None for item in batch)
        start=now-pd.Timedelta(days=2 if horizon=='hour' else 7) if update else None
        cache['progress']={'text':('Updating day-trade bars' if update else 'Loading day-trade history') if horizon=='hour' else 'Loading buy-and-hold history','done':len(frames),'total':len(symbols)}
        try:
            fetched=fetch_bars([b[0] for b in batch],key,secret,horizon,feed,now,start)
            for symbol,token,path,old in batch:
                data=fetched[fetched.symbol==symbol].copy()
                if data.empty and old is None:
                    failed.append({'symbols':[symbol],'reason':'No usable completed bars returned.'});continue
                if old is not None:
                    data=pd.concat([old[1],data],ignore_index=True).drop_duplicates(['symbol','timestamp'],keep='last')
                minimum=now-pd.Timedelta(days=50 if horizon=='hour' else 365*6)
                data=data[data.timestamp>=minimum].sort_values('timestamp').reset_index(drop=True)
                cache[token]=(time.time(),data);frames.append(data)
                try:
                    temporary=path.with_suffix('.tmp.gz')
                    data.to_json(temporary,orient='table',compression='gzip')
                    temporary.replace(path)
                except OSError:
                    cache['cache_warning']='Local history could not be saved; this process still reuses its memory cache.'
        except RateLimitError:raise
        except Exception as exc:failed.append({'symbols':[b[0] for b in batch],'reason':str(exc)})
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()),failed


def scan_market(key, secret, feed, trading_url, cache):
    import time
    started=time.monotonic()
    cache['progress']={'text':'Loading the active stock list','done':0,'total':0}
    cached=cache.get('universe')
    if not cached or time.monotonic()-cached[0]>86400:
        cache['universe']=(time.monotonic(),fetch_universe(key,secret,trading_url))
    symbols=cache['universe'][1]
    clock=fetch_clock(key,secret,trading_url)
    if not clock.get('is_open'):
        return {'closed':True,'coverage':{'listed':len(symbols)},'as_of':pd.Timestamp.now(tz='UTC').isoformat()}
    cache['progress']={'text':f'Screening current data for {len(symbols):,} stocks','done':0,'total':0}
    screen=cache.get('market_screen')
    if not screen or time.monotonic()-screen[0]>=300:
        eligible,coverage=screen_universe(symbols,key,secret,feed)
        cache['market_screen']=(time.monotonic(),eligible,coverage)
    else:
        eligible,coverage=screen[1],screen[2]
    coverage=dict(coverage)
    snapshots=coverage.pop('snapshot_data')
    limit=cache.get('candidate_limit',40)
    candidates=shortlist(eligible,snapshots,limit)
    coverage['detailed_candidates']=len(candidates)
    results={}
    for horizon in ['hour','hold']:
        raw,failed=reusable_history(candidates,key,secret,horizon,feed,cache)
        if raw.empty:
            result={'picks':[],'strategy_evidence':{},'message':'No usable history was returned for the screened candidates.'}
        else:
            cache['progress']={'text':('Evaluating day-trade strategies' if horizon=='hour' else 'Evaluating buy-and-hold strategies'),'done':len(candidates),'total':len(candidates)}
            # Evaluate history first; read fresh quotes only after strategy selection.
            frames={symbol:indicators(g,horizon) for symbol,g in raw.groupby('symbol')}
            evidence=select_strategy(frames,horizon)
            quotes=fetch_quotes(sorted(frames),key,secret,feed) if evidence.get('qualified') else {}
            clock=fetch_clock(key,secret,trading_url)
            result=opportunities(raw,quotes,clock,horizon,prepared_frames=frames,prepared_evidence=evidence)
            result['history_symbols']=len(frames)
        result['failed_batches']=failed
        results[horizon]=result
        # Publish each section when ready, without waiting for the other horizon.
        cache['partial_result']={'closed':False,'results':dict(results),'coverage':dict(coverage),
                                 'as_of':pd.Timestamp.now(tz='UTC').isoformat(),'seconds':time.monotonic()-started}
    cache['candidate_limit']=min(limit+40,120) if any(not r['picks'] for r in results.values()) else 40
    return cache['partial_result']
