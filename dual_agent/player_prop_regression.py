"""Saved ridge-regression prop inference. No training, network or sklearn at runtime.

Artifacts are built by train_player_prop_regression.py. Unsupported markets or
failed held-out checks return None so existing models retain control.
"""
from functools import lru_cache
from pathlib import Path
import json
import os
import numpy as np

WINDOW = 8
FEATURES = ['weighted_mean', 'recent_mean', 'mean', 'std', 'trend', 'zero_rate', 'last']


def features(values):
    a = np.asarray(values, dtype=float)[-WINDOW:]
    if len(a) < WINDOW or not np.isfinite(a).all():
        raise ValueError('Eight finite, chronological prior appearances required')
    scale = max(float(np.std(a)), float(np.mean(np.abs(a))) * .2, 1.)
    base = float(np.average(a, weights=np.arange(1, WINDOW+1)))
    x = np.array([base/scale, a[-3:].mean()/scale, a.mean()/scale,
                  a.std()/scale, (a[-3:].mean()-a[:3].mean())/scale,
                  np.mean(a == 0), a[-1]/scale])
    return x, base, scale


def probabilities(samples, line):
    samples = np.rint(np.asarray(samples, dtype=float))
    over = int((samples > line).sum()); under = int((samples < line).sum())
    n = over + under
    if not n: return None
    return (over+1)/(n+2), (under+1)/(n+2), 1-n/len(samples)


@lru_cache(maxsize=8)
def _load(path, stamp):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 1 or data.get('features') != FEATURES:
        raise ValueError('Regression artifact schema mismatch')
    return data


def estimate(values, line, side, league, market, cutoff=None):
    """Conditional non-push probability, using a later residual calibration set."""
    path = Path(os.environ.get('PLAYER_PROP_REGRESSION_PATH',
                               str(Path(__file__).with_name('player_prop_regression_weights.json'))))
    try:
        model = _load(str(path), path.stat().st_mtime_ns)
        group = model['markets'].get(league.upper()+'|'+str(market))
        if not group or not group.get('promoted'): return None
        # ISO UTC dates; live callers must supply the actual forecast cutoff.
        from datetime import datetime, timezone
        day = str(cutoff or datetime.now(timezone.utc).isoformat())[:10]
        if day <= group['evaluation_end']: return None
        if not np.isfinite(float(line)): return None
        x, base, scale = features(values)
        correction = float((x-np.array(group['center']))/np.array(group['spread']) @ np.array(group['coef']) + group['intercept'])
        projection = base + scale * correction
        samples = projection + scale*np.array(group['residuals'])
        if group['nonnegative']: samples = np.maximum(samples, 0)
        p = probabilities(samples, float(line))
        if p is None: return None
        over, under, push = p
        return {'projection': float(np.mean(np.rint(samples))),
                'over_support': over, 'under_support': under,
                'historical_support': over if str(side).upper() in ('OVER','YES') else under,
                'estimated_push_chance': push, 'sample_size': len(values),
                'estimate_method': 'Trained ridge regression; chronological residual calibration',
                'regression_model_id': group['model_id'],
                'regression_evaluation_end': group['evaluation_end']}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def card_estimate(values, line, side, league, market, cutoff):
    result = estimate(values, line, side, league, market, cutoff)
    if result is None: return None
    return {'Projected stat': result['projection'], 'Estimated chance': result['historical_support'],
            'Over chance': result['over_support'], 'Under chance': result['under_support'],
            'Estimated push chance': result['estimated_push_chance'],
            'Estimate method': result['estimate_method'],
            'Regression model ID': result['regression_model_id']}
