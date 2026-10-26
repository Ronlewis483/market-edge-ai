"""Offline NBA prop training. Never imported by Streamlit.

Run: python dual_agent/train_nba_prop_neural.py --data-dir nba_training_data --output-dir dual_agent
Downloads public SportsDataverse ESPN box-score archives (2020-2026), not paid APIs.
Requires numpy, pandas, scipy, scikit-learn, threadpoolctl. No GPU required.
The optional JSON report is evidence only; the app reads only the NPZ weights.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from nba_prop_neural import STATS, MARKETS, FEATURES, MATRIX, feature_batch, network

SOURCE = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/'
COLUMNS = {'pts': 'points', 'reb': 'rebounds', 'ast': 'assists',
           'fg3m': 'three_point_field_goals_made', 'blk': 'blocks',
           'stl': 'steals', 'turnover': 'turnovers'}


def collect(root):
    root.mkdir(parents=True, exist_ok=True)
    def download(year):
        path = root/f'player_box_{year}.csv'
        if not path.exists():
            with urllib.request.urlopen(SOURCE+path.name, timeout=60) as response:
                body = response.read()
            temp = path.with_suffix('.tmp'); temp.write_bytes(body); temp.replace(path)
        return path
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(download, range(2020, 2027)))
    keep = ['game_id', 'athlete_id', 'season', 'season_type', 'game_date', 'minutes',
            'team_id', 'opponent_team_id', 'team_score', 'opponent_team_score', *COLUMNS.values()]
    frames = [pd.read_csv(p, usecols=keep, dtype={'game_id': str, 'athlete_id': str}) for p in paths]
    frame = pd.concat(frames, ignore_index=True).rename(columns={v:k for k,v in COLUMNS.items()})
    frame['date'] = pd.to_datetime(frame['game_date'], utc=True)
    # Ended seasons only; exclude exhibitions, all-star games and unknown phases.
    frame = frame[frame['season_type'].isin([2, 3]) & (frame['date'] < pd.Timestamp('2026-07-01', tz='UTC'))]
    frame = frame.drop_duplicates(['game_id', 'athlete_id'])
    return frame, [{'name':p.name, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]


def opponent_factors(frame):
    """Same conservative scoring allowance as the app, using earlier dates only."""
    games = frame[['game_id', 'team_id', 'opponent_team_id', 'date', 'team_score', 'opponent_team_score']].drop_duplicates(['game_id', 'team_id'])
    games = games.sort_values(['date', 'game_id', 'team_id'])
    dates = games['date'].astype('int64').to_numpy()/86400e9
    team = games['team_id'].to_numpy(); opponent = games['opponent_team_id'].to_numpy()
    scored = pd.to_numeric(games['team_score'], errors='coerce').to_numpy()
    conceded = pd.to_numeric(games['opponent_team_score'], errors='coerce').to_numpy()
    valid = np.isfinite(scored) & np.isfinite(conceded) & (scored > 0) & (conceded > 0)
    result = {}
    for i, (gid, tid) in enumerate(zip(games.game_id, team)):
        prior = valid & (dates < dates[i]) & (dates >= dates[i]-365)
        # Each game has two team rows; scored therefore has both team totals once.
        idx = np.flatnonzero(prior & (team == opponent[i]))[-20:]
        if len(idx) < 5 or prior.sum() < 60:
            factor = 1.
        else:
            w = 2. ** (-(dates[i]-dates[idx])/90.)
            eff = w.sum()**2/(w @ w)
            league_w = 2. ** (-(dates[i]-dates[prior])/90.)
            league = np.average(scored[prior], weights=league_w)
            allowance = np.average(conceded[idx], weights=w)
            factor = np.clip(1+(allowance/league-1)*eff/(eff+20), .92, 1.08)
        result[(gid, tid)] = float(factor)
    return result


def make_dataset(frame, factors):
    usable = frame.dropna(subset=STATS+['minutes', 'athlete_id'])
    usable = usable[(usable.minutes > 0) & (usable.minutes <= 70) & (usable[STATS] >= 0).all(axis=1)]
    output = {key:[] for key in ['x','y','base','scale','market','season','date','game','role','rate','shape']}
    for _, group in usable.groupby('athlete_id', sort=True):
        group = group.sort_values(['date', 'game_id'])
        history = deque()
        for row in group.to_dict('records'):
            day = row['date']
            while history and (day-history[0]['date']).days > 365:
                history.popleft()
            prior = [r for r in history if r['date'] < day][-30:]
            if len(prior) >= 10:
                vals = np.array([[r[s] for s in STATS] for r in prior])
                mins = np.array([r['minutes'] for r in prior])
                factor = factors[(row['game_id'], row['team_id'])]
                x, base, scale, aux = feature_batch(vals, mins, factor, (day-prior[-1]['date']).days)
                actual = np.array([row[s] for s in STATS]) @ MATRIX
                for key, value in [('x',x), ('y',actual), ('base',base), ('scale',scale),
                    ('market',np.arange(12)), ('season',np.full(12,row['season'])),
                    ('date',np.full(12,day.strftime('%Y-%m-%d'))), ('game',np.full(12,row['game_id'])),
                    ('role',np.tile(aux['role'],(12,1))), ('rate',aux['rate']), ('shape',aux['shape'])]:
                    output[key].append(value)
            history.append(row)
    data = {key:np.concatenate(value) for key,value in output.items()}
    for key in ['x','y','base','scale','role','rate','shape']:
        data[key] = data[key].astype(np.float32)
    return data


def split_masks(data):
    season, date = data['season'], data['date']
    return {'train':season <= 2023, 'validation':(season == 2024) & (date < '2024-01-01'),
            'calibration':(season == 2024) & (date >= '2024-01-01'),
            'promotion':season == 2025, 'test':season == 2026}


def baseline_probability(data, idx, lines):
    roles = data['role'][idx]
    weights = 2. ** (-np.arange(roles.shape[1]-1, -1, -1)/3.); weights /= weights.sum()
    means = np.maximum(data['rate'][idx,None]*roles, 1e-12)
    shape = data['shape'][idx,None]
    parameter = shape/(shape+means)
    over = (1-nbinom.cdf(np.floor(lines)[:,None], shape, parameter)) @ weights
    under = nbinom.cdf(np.ceil(lines)[:,None]-1, shape, parameter) @ weights
    return over/np.maximum(over+under, 1e-12)


def residual_probability(location, scale, residuals, lines):
    # Discrete counts after clipping; all evaluation lines are half points.
    ordered = np.sort(residuals)
    bounds = (lines-location)/scale
    below = np.searchsorted(ordered, bounds, side='left')
    return (len(ordered)-below+1)/(len(ordered)+2)


def evaluate(data, mask, predictions, residuals):
    metrics = {}
    for market in range(12):
        idx = np.flatnonzero(mask & (data['market'] == market))
        base, scale, actual = data['base'][idx], data['scale'][idx], data['y'][idx]
        neural = np.maximum(0., base+predictions[idx]*scale)
        neural_mae = float(np.mean(np.abs(neural-actual)))
        base_mae = float(np.mean(np.abs(base-actual)))
        briers, old_briers = [], []
        for offset in [-.5, 0., .5]:
            lines = np.maximum(.5, np.floor(base+offset*scale)+.5)
            p = residual_probability(neural, scale, residuals[market], lines)
            old = baseline_probability(data, idx, lines)
            outcome = actual > lines
            briers.append(np.mean((p-outcome)**2)); old_briers.append(np.mean((old-outcome)**2))
        metrics[MARKETS[market]] = {'player_games':len(idx), 'neural_mae':neural_mae,
            'baseline_mae':base_mae, 'neural_brier':float(np.mean(briers)),
            'baseline_brier':float(np.mean(old_briers))}
    return metrics


def run(args):
    data_root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cached = data_root/'training_features_rolling8_v1.npz'
    if cached.exists():
        with np.load(cached, allow_pickle=False) as archive:
            data = {k:archive[k] for k in archive.files if k != 'source_metadata'}
            sources = json.loads(str(archive['source_metadata'].item()))
    else:
        print('Loading public completed-game archives', flush=True)
        frame, sources = collect(data_root)
        print('Constructing strictly prior-game features', len(frame), flush=True)
        data = make_dataset(frame, opponent_factors(frame))
        np.savez_compressed(cached, **data, source_metadata=json.dumps(sources))
    masks = split_masks(data)
    partitions = {}
    previous = set()
    for name, mask in masks.items():
        games = set(data['game'][mask])
        assert not (previous & games), 'Game leakage across chronological partitions'
        previous |= games
        partitions[name] = {'samples':int(mask.sum()), 'games':len(games),
            'start':str(min(data['date'][mask])), 'end':str(max(data['date'][mask]))}
    print('Partitions', partitions, flush=True)
    rng = np.random.default_rng(42)
    train_idx = np.flatnonzero(masks['train'])
    if len(train_idx) > args.max_train:
        train_idx = np.sort(rng.choice(train_idx, args.max_train, replace=False))
    scaler = StandardScaler().fit(data['x'][train_idx])
    x = scaler.transform(data['x'][train_idx])
    y = (data['y'][train_idx]-data['base'][train_idx])/data['scale'][train_idx]
    vi = np.flatnonzero(masks['validation'])
    vx = scaler.transform(data['x'][vi])
    vy = (data['y'][vi]-data['base'][vi])/data['scale'][vi]
    model = MLPRegressor(hidden_layer_sizes=(64,32,16), activation='relu', solver='adam',
        alpha=.02, batch_size=1024, learning_rate_init=.0006, random_state=42)
    best, stale, best_loss, best_epoch = None, 0, float('inf'), 0
    for epoch in range(args.epochs):
        model.partial_fit(x, y)
        loss = float(np.mean((model.predict(vx)-vy)**2))
        print('Epoch', epoch+1, 'validation MSE', round(loss,6), flush=True)
        if loss < best_loss-1e-5:
            best_loss, stale, best_epoch = loss, 0, epoch+1
            best = [(w.copy(), b.copy()) for w,b in zip(model.coefs_,model.intercepts_)]
        else:
            stale += 1
            if stale >= 6: break
    weights = {'x_mean':scaler.mean_, 'x_scale':scaler.scale_}
    for i, (w,b) in enumerate(best):
        weights['w'+str(i)], weights['b'+str(i)] = w, b
    predicted = np.concatenate([network(data['x'][i:i+20000], weights) for i in range(0,len(data['x']),20000)])
    residuals = []
    for market in range(12):
        idx = np.flatnonzero(masks['calibration'] & (data['market'] == market))
        center = np.maximum(0.,data['base'][idx]+predicted[idx]*data['scale'][idx])
        residual = (data['y'][idx]-center)/data['scale'][idx]
        if len(residual) > 2000:
            residual = residual[np.sort(rng.choice(len(residual),2000,replace=False))]
        residuals.append(residual.astype(float))
        weights['residual_'+str(market)] = residuals[-1]
    promotion = evaluate(data, masks['promotion'], predicted, residuals)
    promoted = {m:(r['player_games'] >= 1000 and r['neural_mae'] <= r['baseline_mae']
        and r['neural_brier'] <= r['baseline_brier']) for m,r in promotion.items()}
    # Test is read after selection; no threshold or architecture tuning on test.
    test = evaluate(data, masks['test'], predicted, residuals)
    meta = {'schema':1, 'model_id':'nba_neural_v1', 'features':FEATURES, 'markets':MARKETS,
        'architecture':[len(FEATURES),64,32,16,1], 'promoted':promoted,
        'training_end':partitions['train']['end'], 'validation_end':partitions['validation']['end'],
        'calibration_end':partitions['calibration']['end'], 'promotion_end':partitions['promotion']['end'],
        'training_samples_used':len(train_idx), 'best_epoch':best_epoch,
        'trained_at':datetime.now(timezone.utc).isoformat()}
    weights['metadata'] = np.array(json.dumps(meta))
    np.savez_compressed(out/'nba_prop_neural_weights.npz', **weights)
    report = {'model':meta, 'partitions':partitions, 'sources':sources,
        'promotion':promotion, 'untouched_test':test,
        'limitations':['Constructed half-point thresholds, not historical sportsbook offers.',
            'Player-game samples across markets are correlated; not independent bets.',
            'Conditional on playing; injuries, substitutions and future role changes are not learned inputs.',
            'ESPN training and BALLDONTLIE app statistics may differ after stat corrections.',
            'No live NBA, automatic retraining, profit or 70% hit-rate claim.',
            'Comparator uses the same eight-game minutes window; the older app fallback retains its ten-game minutes window.',
            'Promotion decisions precede the untouched test; later category results can regress.']}
    (out/'nba_prop_neural_report.json').write_text(json.dumps(report, indent=2))
    print('PROMOTED', [m for m in MARKETS if promoted[m]], flush=True)
    for m, result in test.items(): print(m, result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--max-train', type=int, default=300000)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
