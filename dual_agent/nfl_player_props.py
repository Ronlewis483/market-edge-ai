"""Independent NFL player-props research module. No wager execution."""
import math
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

from dual_agent.nfl_player_history import load_player_history
from dual_agent.nfl_player_props import canonical_name

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
COUNT = {'Receptions','Passing touchdowns','Passing completions','Interceptions thrown'}
REQUIRED = {'player','market','game_time','value'}

def fetch_json(path, key, **params):
    if not key: raise ValueError('Set ODDS_API_KEY in Streamlit Secrets.')
    response = requests.get(BASE + path, params={'apiKey':key, **params}, timeout=25)
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
        for market in book.get('markets', []):
            for outcome in market.get('outcomes', []):
                player = outcome.get('description')
                if not player: continue
                side = outcome.get('name','')
                if side not in ('Over','Under','Yes','No'): continue
                point = outcome.get('point')
                if side in ('Over','Under') and point is None: continue
                rows.append({'event_id':event.get('id'), 'game_time':event.get('commence_time'),
                  'home_team':event.get('home_team'), 'away_team':event.get('away_team'),
                  'bookmaker':book.get('title'), 'bookmaker_key':book.get('key'),
                  'market_key':market.get('key'), 'player':player, 'side':side,
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
    """Smoothed empirical frequency. Descriptive baseline, NOT calibrated forecast."""
    if side not in ('Over','Under','Yes','No'): return None
    if market=='Anytime touchdown':
        wins=(values>=1).sum() if side in ('Yes','Over') else (values<1).sum()
    else:
        wins=(values>line).sum() if side=='Over' else (values<line).sum()
    pushes=(values==line).sum() if market!='Anytime touchdown' else 0
    eligible=len(values)-pushes
    if eligible<=0: return None
    return (wins+1)/(eligible+2)  # Laplace smoothing; excludes pushes

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
