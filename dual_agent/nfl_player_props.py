"""Independent NFL player-props research module. No wager execution."""
import math
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from dual_agent.odds_request_cache import get as cached_odds_get
from dual_agent.preferred_sportsbooks import allowed

SPORT = 'americanfootball_nfl'
BASE = 'https://api.the-odds-api.com/v4'
MARKETS = {
 'Passing yards':'player_pass_yds', 'Rushing yards':'player_rush_yds',
 'Receiving yards':'player_reception_yds', 'Receptions':'player_receptions',
 'Rushing + receiving yards':'player_rush_reception_yds',
 'Passing touchdowns':'player_pass_tds', 'Anytime touchdown':'player_anytime_td',
 'Passing completions':'player_pass_completions',
 'Interceptions thrown':'player_pass_interceptions',
}
MARKETS.update({
 'Passing attempts':'player_pass_attempts', 'Rushing attempts':'player_rush_attempts',
 'Rushing touchdowns':'player_rush_tds', 'Receiving touchdowns':'player_reception_tds',
 'Passing + rushing yards':'player_pass_rush_yds',
 'Passing + rushing + receiving yards':'player_pass_rush_reception_yds',
 'Passing + rushing + receiving touchdowns':'player_pass_rush_reception_tds',
 'Rushing + receiving touchdowns':'player_rush_reception_tds',
 'Field goals':'player_field_goals', 'Extra points':'player_pats',
 'Kicking points':'player_kicking_points',
 'Longest completion':'player_pass_longest_completion',
 'Longest rush':'player_rush_longest', 'Longest reception':'player_reception_longest',
 'Sacks':'player_sacks', 'Solo tackles':'player_solo_tackles',
 'Assisted tackles':'player_assists', 'Tackles + assists':'player_tackles_assists',
 'Defensive interceptions':'player_defensive_interceptions',
})

COUNT = {'Receptions','Passing touchdowns','Passing completions','Interceptions thrown'}
REQUIRED = {'player','market','game_time','value'}

def fetch_json(path, key, **params):
    if not key: raise ValueError('Set ODDS_API_KEY in Streamlit Secrets.')
    response = cached_odds_get(BASE + path, params={'apiKey':key, **params}, timeout=25)
    response.raise_for_status()
    return response.json(), response.headers.get('x-requests-remaining', '?')

def events(key):
    return fetch_json(f'/sports/{SPORT}/events', key, dateFormat='iso')[0]

def event_props(key, event_id, markets, regions='us'):
    return fetch_json(f'/sports/{SPORT}/events/{event_id}/odds', key,
                      regions=regions, markets=','.join(markets), oddsFormat='american', dateFormat='iso')

def normalize_props(event):
    rows=[]
    for book in event.get('bookmakers', []):
        if not allowed(book.get('key')):
            continue
        for market in book.get('markets', []):
            for outcome in market.get('outcomes', []):
                player = outcome.get('description')
                if not player: continue
                side = outcome.get('name','')
                if side not in ('Over','Under','Yes','No'): continue
                point = outcome.get('point')
                if side in ('Over','Under') and point is None: continue
                # Numeric Yes/No totals need their actual threshold. Never
                # silently turn a missing passing/rushing/etc. line into 0.5.
                market_key = str(market.get('key','')).removesuffix('_alternate')
                if point is None and market_key != 'player_anytime_td': continue
                rows.append({'event_id':event.get('id'), 'game_time':event.get('commence_time'),
                  'home_team':event.get('home_team'), 'away_team':event.get('away_team'),
                  'bookmaker':book.get('title'), 'bookmaker_key':book.get('key'),
                  'market_key':market_key, 'player':player, 'side':side,
                  'line':float(point) if point is not None else 0.5,
                  'american_odds':outcome.get('price'), 'last_update':market.get('last_update')})
    return pd.DataFrame(rows)

def implied_probability(american):
    a=float(american)
    if a == 0: raise ValueError('Invalid American odds')
    return (-a)/((-a)+100) if a < 0 else 100/(a+100)

def canonical_name(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())

def validate_history(df):
    missing=REQUIRED-set(df.columns)
    if missing: raise ValueError('History CSV missing columns: '+', '.join(sorted(missing)))
    h=df.copy()
    h['game_time']=pd.to_datetime(h['game_time'], utc=True, errors='coerce')
    h['value']=pd.to_numeric(h['value'], errors='coerce')
    h['market']=h['market'].astype(str).str.strip()
    h['player_key']=h['player'].map(canonical_name)
    h=h.dropna(subset=['game_time','value'])
    h=h[h['market'].isin(MARKETS)]
    h=h[h['value']>=0]
    return h.sort_values('game_time')

def historical_sample(history, player, market, cutoff, window=12, min_games=6):
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None: cutoff=cutoff.tz_localize('UTC')
    else: cutoff=cutoff.tz_convert('UTC')
    h=history[(history['player_key']==canonical_name(player)) &
              (history['market']==market) & (history['game_time']<cutoff)]
    h=h.drop_duplicates(subset=['game_time','player_key','market'],keep='last').tail(window)
    if len(h)<min_games: return None
    return h['value'].to_numpy(dtype=float)

def probability(values, line, market, side):
    """Laplace-smoothed prior frequency, conditional on a non-push result.

    For numeric totals, Yes means above the listed line and No means below
    it, equivalent to Over/Under. Equality is a push. Anytime touchdown is
    the binary event of scoring at least once. Exact offered sides are still
    required by the caller; calculating a side does not invent a bet.
    This remains a descriptive baseline, not a calibrated game forecast.
    """
    if market not in MARKETS or side not in ('Over', 'Under', 'Yes', 'No'):
        return None
    try:
        sample = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if (sample.ndim != 1 or sample.size == 0
            or not np.isfinite(sample).all() or (sample < 0).any()):
        return None
    if market == 'Anytime touchdown':
        if side in ('Over', 'Under'):
            try:
                threshold = float(line)
            except (TypeError, ValueError):
                return None
            # An anytime market means at least one TD, not another total.
            if not math.isfinite(threshold) or threshold != 0.5:
                return None
        wins = int((sample >= 1).sum()) if side in ('Yes', 'Over') else int((sample < 1).sum())
        eligible = sample.size
    else:
        try:
            threshold = float(line)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(threshold) or threshold < 0:
            return None
        wins = int((sample > threshold).sum()) if side in ('Over', 'Yes') else int((sample < threshold).sum())
        eligible = sample.size - int((sample == threshold).sum())
    if eligible <= 0:
        return None
    return float((wins + 1) / (eligible + 2))

def analyze(odds, history, window=12, min_games=6):
    if odds.empty: return pd.DataFrame()
    lookup={v:k for k,v in MARKETS.items()}
    results=[]
    for _,r in odds.iterrows():
        market=lookup.get(r['market_key'])
        if not market: continue
        vals=historical_sample(history,r['player'],market,r['game_time'],window,min_games)
        if vals is None: continue
        p=probability(vals,r['line'],market,r['side'])
        if p is None: continue
        try: implied=implied_probability(r['american_odds'])
        except (TypeError,ValueError): continue
        row=r.to_dict()
        row.update(market=market, historical_games=len(vals), historical_average=round(float(vals.mean()),2),
                   historical_hit_rate=round(p,4), implied_probability=round(implied,4),
                   descriptive_gap_pp=round(100*(p-implied),2),
                   status='RESEARCH ONLY — uncalibrated historical baseline')
        results.append(row)
    return pd.DataFrame(results)

def walkforward(history, window=12, min_games=6):
    """Time-ordered leave-future-out baseline evaluation; no sportsbook ROI claims."""
    rows=[]
    for (player,market), group in history.groupby(['player_key','market']):
        group=group.sort_values('game_time').drop_duplicates('game_time',keep='last')
        for i in range(min_games,len(group)):
            previous=group.iloc[max(0,i-window):i]['value'].to_numpy(dtype=float)
            actual=float(group.iloc[i]['value'])
            forecast=float(np.mean(previous))
            rows.append({'player':group.iloc[i]['player'],'market':market,
                         'game_time':group.iloc[i]['game_time'], 'forecast':forecast,
                         'actual':actual,'absolute_error':abs(forecast-actual),
                         'training_games':len(previous)})
    return pd.DataFrame(rows)
