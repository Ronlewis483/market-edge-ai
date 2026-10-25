"""Saved deep feedforward NFL prop model. NumPy inference; no web-time training.

The network learns corrections to a history/clock baseline. A separate,
chronologically later calibration set supplies the residual distribution.
Only markets/phases passing the held-out promotion gate use neural estimates.
"""
from functools import lru_cache
from pathlib import Path
import json
import math
import numpy as np

MARKETS=['Passing yards','Rushing yards','Receiving yards','Receptions',
         'Passing completions','Passing attempts','Rushing attempts']
FLOORS=np.array([40.,15.,15.,1.5,3.,4.,3.])
FEATURES=['prior_mean_scaled','prior_sd_scaled','recent_mean_scaled','trend_scaled',
          'min_scaled','max_scaled','zero_fraction','history_count','elapsed_fraction',
          'elapsed_squared','current_scaled','baseline_remaining_scaled']+['market_'+m for m in MARKETS]


def phase(elapsed):
    return 'pregame' if elapsed==0 else 'live'


def features(values,current,market,elapsed=0):
    values=np.asarray(values,dtype=float)[-12:]
    if market not in MARKETS or len(values)<8 or not np.isfinite(values).all():
        raise ValueError('Neural model needs a supported market and eight valid prior games')
    if not np.isfinite([current,elapsed]).all() or not 0<=elapsed<3600:
        raise ValueError('Invalid observed production or game clock')
    i=MARKETS.index(market);scale=max(float(np.std(values,ddof=1)),float(FLOORS[i]))
    prior=float(np.average(values,weights=np.linspace(1,2,len(values))))
    fraction=elapsed/3600
    baseline=((prior+current)/(1+fraction))*(1-fraction)
    recent=float(np.mean(values[-3:]))
    x=[prior/scale,float(np.std(values,ddof=1))/scale,recent/scale,
       (recent-float(np.mean(values)))/scale,float(np.min(values))/scale,
       float(np.max(values))/scale,float(np.mean(values==0)),len(values)/12,
       fraction,fraction*fraction,current/scale,baseline/scale]
    x.extend(float(j==i) for j in range(len(MARKETS)))
    return np.asarray(x,dtype=float),baseline,scale


@lru_cache(maxsize=1)
def load_model():
    path=Path(__file__).with_name('nfl_prop_neural_weights.npz')
    if not path.exists(): raise ValueError('Install nfl_prop_neural_weights.npz beside nfl_prop_neural.py')
    # Weight arrays only; never load executable pickle artifacts.
    with np.load(path,allow_pickle=False) as data:
        result={k:data[k].copy() for k in data.files}
    meta=json.loads(str(result['metadata'].item()))
    if meta.get('schema')!=1 or meta.get('features')!=FEATURES or meta.get('markets')!=MARKETS:
        raise ValueError('Neural weights do not match the inference feature schema')
    result['meta']=meta
    return result


def network(x,model):
    x=(np.asarray(x,dtype=float)-model['x_mean'])/model['x_scale']
    for i in range(4):
        x=x@model['w'+str(i)]+model['b'+str(i)]
        if i<3: x=np.maximum(x,0)
    return np.asarray(x).reshape(-1)


def estimate(values,current,line,market,side,elapsed=0):
    model=load_model();meta=model['meta'];group=market+'|'+phase(elapsed)
    if not meta.get('promoted',{}).get(group,False): return None
    x,baseline,scale=features(values,current,market,elapsed)
    correction=float(network(x,model)[0])*scale
    remaining=baseline+correction
    residuals=model['residual_'+str(MARKETS.index(market))+'_'+phase(elapsed)]
    if len(residuals)<100: return None
    final=current+remaining+scale*residuals
    if 'yards' not in market.lower():
        # Counts cannot decrease; avoid recommending already-attained count lines.
        if elapsed and current>line: return None
        final=np.maximum(final,current)
    final=np.rint(final)
    over=int(np.sum(final>line));under=int(np.sum(final<line));decisions=over+under
    if decisions<20: return None
    p_over=(over+1)/(decisions+2);p_under=(under+1)/(decisions+2)
    return {'historical_support':float(p_over if side.upper() in ('OVER','YES') else p_under),
        'over_support':float(p_over),'under_support':float(p_under),
        'projection':float(np.mean(final)),'sample_size':len(values),
        'estimated_push_chance':float(1-decisions/len(final)),
        'estimate_method':'Trained neural network v1 (64/32/16); chronological residual calibration',
        'neural_model_version':meta['model_id'],'neural_training_end':meta['training_end'],
        'neural_calibration_end':meta['calibration_end'],
        'neural_evaluation_note':'Held-out historical production/constructed thresholds; not verified sportsbook bet accuracy'}
