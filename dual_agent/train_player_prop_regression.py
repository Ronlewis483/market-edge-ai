"""Offline ridge training from saved completed appearances; no API requests.

python dual_agent/train_player_prop_regression.py --league NBA --history nba_player_history_cache/player_histories_v3.json
NFL/MLB: --history saved_history.csv (or .parquet). Canonical long columns:
player_id, game_id, date, market, value. NFL game_time/provider_player_id and
MLB wide player-log start_time/stat columns are also accepted.
Requires numpy, pandas and scikit-learn. Outputs complete JSON weights + report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
try:
    from .player_prop_regression import FEATURES, WINDOW, features, probabilities
except ImportError:
    from player_prop_regression import FEATURES, WINDOW, features, probabilities

NBA = {'pts':'player_points','reb':'player_rebounds','ast':'player_assists',
       'fg3m':'player_threes','blk':'player_blocks','stl':'player_steals','turnover':'player_turnovers',
       'pts,reb,ast':'player_points_rebounds_assists','pts,reb':'player_points_rebounds',
       'pts,ast':'player_points_assists','reb,ast':'player_rebounds_assists','blk,stl':'player_blocks_steals'}
MLB = {'hits':'batter_hits','batting_home_runs':'batter_home_runs','total_bases':'batter_total_bases',
       'rbi':'batter_rbis','batting_runs':'batter_runs_scored','hits_runs_rbis':'batter_hits_runs_rbis',
       'singles':'batter_singles','doubles':'batter_doubles','triples':'batter_triples',
       'batting_walks':'batter_walks','batting_strikeouts':'batter_strikeouts',
       'stolen_bases':'batter_stolen_bases','pitcher_strikeouts':'pitcher_strikeouts',
       'pitcher_hits':'pitcher_hits_allowed','pitcher_walks':'pitcher_walks',
       'earned_runs':'pitcher_earned_runs','pitching_outs':'pitcher_outs'}


def read_history(path, league):
    path = Path(path)
    if league == 'NBA' and path.suffix == '.json':
        archive = json.loads(path.read_text()); records = []
        for entry in archive.get('players', {}).values():
            for r in entry.get('rows', []):
                game = r.get('game') or {}; player = r.get('player') or {}
                if not str(game.get('status', '')).startswith('Final'): continue
                for fields, market in NBA.items():
                    vals = [pd.to_numeric(r.get(f), errors='coerce') for f in fields.split(',')]
                    if all(pd.notna(v) and np.isfinite(v) and v >= 0 for v in vals):
                        records.append({'player_id':player.get('id') or entry.get('player_id'),
                            'game_id':game.get('id'), 'date':r.get('source_date') or game.get('date'),
                            'market':market, 'value':sum(vals)})
        return pd.DataFrame(records)
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    df = df.rename(columns={k:v for k,v in {'provider_player_id':'player_id',
        'game_time':'date','start_time':'date'}.items() if k in df and v not in df})
    if league == 'MLB' and 'market' not in df:
        parts = []
        for stat, market in MLB.items():
            if stat not in df: continue
            eligibility = 'games_started' if market.startswith('pitcher_') else 'plate_appearances'
            if eligibility not in df: continue
            part = df[pd.to_numeric(df[eligibility], errors='coerce') > 0].copy()
            part['market'] = market; part['value'] = pd.to_numeric(part[stat], errors='coerce')
            parts.append(part[['player_id','game_id','date','market','value']])
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    # NFL's saved frame identifies appearances by player and game_time.
    if 'game_id' not in df and 'date' in df: df['game_id'] = df['date'].astype(str)
    return df


def dataset(df):
    needed = ['player_id','game_id','date','market','value']
    if not set(needed) <= set(df): raise ValueError('History needs '+', '.join(needed))
    df = df[needed].copy().dropna()
    df['date'] = pd.to_datetime(df.date, utc=True, errors='coerce').dt.floor('D')
    df['value'] = pd.to_numeric(df.value, errors='coerce')
    df = df.dropna(); df = df[np.isfinite(df.value)]
    df = df[df.date < pd.Timestamp.now(tz='UTC').floor('D')-pd.Timedelta(days=2)]
    # Conflicting duplicates are rejected rather than picking a convenient label.
    keys = ['player_id','game_id','market']
    conflicts = df.groupby(keys).value.transform('nunique') > 1
    df = df[~conflicts].drop_duplicates(keys)
    rows = []
    for (_, market), group in df.groupby(['player_id','market']):
        prior = []
        for date, day in group.sort_values(['date','game_id']).groupby('date', sort=True):
            available = [value for day_stamp,value in prior if day_stamp < date-pd.Timedelta(days=2)]
            if len(available) >= WINDOW:
                x, base, scale = features(available)
                for r in day.itertuples():
                    rows.append((market, date, x, base, scale, float(r.value)))
            prior.extend((date, value) for value in day.value.tolist())
    return rows


def metric(rows, samples):
    actual = np.array([r[5] for r in rows]); base = np.array([r[3] for r in rows])
    scale = np.array([r[4] for r in rows]); preds = np.mean(np.rint(samples), axis=1)
    # Constructed half-point thresholds are not archived sportsbook lines.
    thresholds = np.maximum(.5, np.floor(base[:,None]+scale[:,None]*np.array([-.5,0,.5]))+.5)
