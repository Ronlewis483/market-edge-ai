"""NBA pregame prop neural inference. Saved NumPy weights; no API or training calls.

Forecasts are conditional on participation. Promotion uses chronological games
and constructed thresholds, not historical sportsbook lines or ticket returns.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np

STATS = ['pts', 'reb', 'ast', 'fg3m', 'blk', 'stl', 'turnover']
MARKETS = ['Points', 'Rebounds', 'Assists', 'Three-pointers made', 'Blocks',
           'Steals', 'Turnovers', 'Points + rebounds + assists',
           'Points + rebounds', 'Points + assists', 'Rebounds + assists',
           'Blocks + steals']
COMPONENTS = [('pts',), ('reb',), ('ast',), ('fg3m',), ('blk',), ('stl',),
              ('turnover',), ('pts', 'reb', 'ast'), ('pts', 'reb'),
              ('pts', 'ast'), ('reb', 'ast'), ('blk', 'stl')]
MATRIX = np.array([[float(s in c) for c in COMPONENTS] for s in STATS])
FLOORS = np.array([4., 2., 2., 1., .8, .8, 1., 6., 5., 5., 3., 1.])
FEATURES = ['mean_scaled', 'sd_scaled', 'recent3_scaled', 'recent8_scaled',
            'trend_scaled', 'min_scaled', 'max_scaled', 'zero_fraction',
            'history_count', 'baseline_scaled', 'rate_scaled', 'mean_minutes',
            'recent_minutes', 'minutes_sd', 'minutes_trend', 'low_minutes_fraction',
            'opponent_points_factor', 'rest_days', 'dispersion', 'recent_zero_fraction']
FEATURES += ['market_' + m for m in MARKETS]


def feature_batch(values, minutes, factor=1., rest_days=3.):
    """Identical prior-game transform for offline training and app inference."""
    values = np.asarray(values, dtype=float)[-30:]
    minutes = np.asarray(minutes, dtype=float)[-30:]
    if (values.ndim != 2 or values.shape != (len(minutes), len(STATS))
            or len(minutes) < 10 or not np.isfinite(values).all()
            or not np.isfinite(minutes).all() or (values < 0).any()
            or (minutes <= 0).any() or (minutes > 70).any()
            or not np.isfinite([factor, rest_days]).all() or not .92 <= factor <= 1.08):
        raise ValueError('Neural inference needs 10 complete, positive-minute prior games')
    actual = values @ MATRIX
    n = len(actual)
    weight = 2. ** (-np.arange(n-1, -1, -1)/10.)
    effective = weight.sum()**2 / (weight @ weight)
    weight *= effective / weight.sum()
    total = weight @ actual
    exposure = float(weight @ minutes)
    rate = (total + .5)/(exposure + 1.)
    observed = minutes[:, None]*rate
    excess = np.maximum(0., np.average((actual-observed)**2-observed, weights=weight, axis=0))
    denominator = np.average(observed**2, weights=weight, axis=0)
    dispersion = np.divide(excess, denominator, out=np.zeros(12), where=denominator > 0)+1/(total+.5)
    point_rate = float(weight @ values[:, 0])/max(exposure, 1.)
    rate = np.maximum(0., rate + (factor-1.)*point_rate*MATRIX[0])
    role = np.clip(minutes[-8:], 0, 48)
    role_weight = 2. ** (-np.arange(len(role)-1, -1, -1)/3.)
    role_weight /= role_weight.sum()
    expected_minutes = float(role_weight @ role)
    baseline = rate*expected_minutes
    scale = np.maximum(np.std(actual, axis=0, ddof=1), FLOORS)
    mean = np.mean(actual, axis=0)
    common = [np.mean(minutes)/48, expected_minutes/48, np.std(minutes)/20,
              (np.mean(minutes[-3:])-np.mean(minutes))/20,
              np.mean(minutes < 10), factor, np.clip(rest_days, 0, 30)/30]
    x = np.column_stack([mean/scale, np.std(actual, axis=0, ddof=1)/scale,
        np.mean(actual[-3:], axis=0)/scale, np.mean(actual[-8:], axis=0)/scale,
        (np.mean(actual[-3:], axis=0)-mean)/scale, np.min(actual, axis=0)/scale,
        np.max(actual, axis=0)/scale, np.mean(actual == 0, axis=0),
        np.full(12, n/30), baseline/scale, rate*36/scale,
        *[np.full(12, v) for v in common], np.minimum(dispersion, 20),
        np.mean(actual[-5:] == 0, axis=0), np.eye(12)])
    return x, baseline, scale, {'role': role, 'role_weight': role_weight,
        'rate': rate, 'shape': 1/np.maximum(dispersion, 1e-9),
        'expected_minutes': expected_minutes}


@lru_cache(maxsize=1)
def load_model():
    path = Path(__file__).with_name('nba_prop_neural_weights.npz')
    with np.load(path, allow_pickle=False) as archive:
        model = {k: archive[k].copy() for k in archive.files}
    meta = json.loads(str(model['metadata'].item()))
    if meta.get('schema') != 1 or meta.get('features') != FEATURES or meta.get('markets') != MARKETS:
        raise ValueError('NBA neural weights and feature schema do not match')
    widths = [len(FEATURES), 64, 32, 16, 1]
    for i in range(4):
        if model['w'+str(i)].shape != (widths[i], widths[i+1]) or model['b'+str(i)].shape != (widths[i+1],):
            raise ValueError('Invalid NBA neural layer shape')
    if model['x_mean'].shape != (len(FEATURES),) or model['x_scale'].shape != (len(FEATURES),):
        raise ValueError('Invalid NBA feature scaler')
    if any(not np.isfinite(v).all() for k, v in model.items() if k != 'metadata') or (model['x_scale'] <= 0).any():
        raise ValueError('Invalid NBA numerical weights')
    model['meta'] = meta
    return model


def network(x, model):
    x = (np.asarray(x, dtype=float)-model['x_mean'])/model['x_scale']
    for i in range(4):
        x = x @ model['w'+str(i)] + model['b'+str(i)]
        if i < 3:
            x = np.maximum(x, 0.)
    return np.asarray(x).reshape(-1)


def estimate(values, minutes, stat, line, side, factor=1., rest_days=3.):
    components = tuple(stat) if isinstance(stat, (tuple, list)) else (stat,)
    if components not in COMPONENTS or side not in ('Over', 'Under'):
        return None
    if not np.isfinite(float(line)) or float(line) < 0:
        return None
    model = load_model()
    i = COMPONENTS.index(components)
    if not model['meta']['promoted'].get(MARKETS[i], False):
        return None
    x, baseline, scale, aux = feature_batch(values, minutes, factor, rest_days)
    location = max(0., baseline[i]+float(network(x[i], model)[0])*scale[i])
    residuals = model['residual_'+str(i)]
    if len(residuals) < 200:
        return None
    final = np.rint(np.maximum(0., location+scale[i]*residuals))
    above, below = int(np.sum(final > line)), int(np.sum(final < line))
    decisions = above+below
    if decisions < 50:
        return None
    over, under = (above+1)/(decisions+2), (below+1)/(decisions+2)
    meta = model['meta']
    return {'Estimated chance': float(over if side == 'Over' else under),
        'Over chance': float(over), 'Under chance': float(under),
        'Projected stat': float(np.mean(final)), 'Expected minutes': aux['expected_minutes'],
        'Estimated push chance': float(1-decisions/len(final)),
        'NBA prop model version': 3, 'Probability source': 'trained_neural',
        'Estimate method': 'Trained NBA neural model · nba_neural_v1 · chronological residual calibration',
        'neural_model_version': meta['model_id'], 'neural_training_end': meta['training_end'],
        'neural_calibration_end': meta['calibration_end'],
        'Calibration sample size': len(residuals),
        'Probability reliability': 'Experimental · confidence percentages have not been validated against live sportsbook outcomes',
        'Neural evaluation note': 'Historical box-score evaluation at constructed thresholds; live sportsbook accuracy is not established'}
