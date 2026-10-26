"""Earlier NBA learning: a bounded context correction to frozen trained-neural forecasts.
No replacement of neural weights, no sports API calls, and no live promotion.
"""
import base64
import hashlib
import io
import json
from datetime import datetime,timezone
if __package__:
    from .nba_pregame_snapshots import PREFIX, stamp, start_time, number, final, team_name
    from .nba_training_dataset import extract_features, FEATURE_COLUMNS, FEATURE_VERSION
else:
    from nba_pregame_snapshots import PREFIX, stamp, start_time, number, final, team_name
    from nba_training_dataset import extract_features, FEATURE_COLUMNS, FEATURE_VERSION

ROOT='nba/context_models_v1/'
CACHE=ROOT+'early_adjustment/cache.json.gz'
BASE_MODEL='nba_winner_neural_v1'
FIELDS=('elo_difference','margin_difference','pace_difference','rest_difference','reported_out_minutes_advantage','market_logit_gap')
RULES={'version':1,'minimum_games':{'train':80,'validation':25,'test':25},
       'minimum_classes':{'train':10,'validation':5,'test':5},'max_logit_change':.35,
       'penalties':[1.,10.,100.],'base_model_id':BASE_MODEL}


def hashed(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def derived(features,base_probability):
    import numpy as np
    if set(features)!=set(FEATURE_COLUMNS):raise ValueError('Early adjustment feature schema mismatch')
    if any(v is not None and (isinstance(v,(str,bool)) or number(v) is None) for v in features.values()):
        raise ValueError('Invalid early adjustment input')
    if number(base_probability) is None or not 0<float(base_probability)<1:raise ValueError('Invalid frozen neural probability')
    p=float(base_probability);market=number(features['market_home_probability'])
    if market is not None and not 0<market<1:raise ValueError('Invalid sportsbook probability')
    def difference(home,away):
        h,a=number(features[home]),number(features[away])
        return h-a if h is not None and a is not None else np.nan
    return np.array([difference('home_elo','away_elo'),difference('home_form_margin_8','away_form_margin_8'),
        difference('home_matchup_estimated_pace_48','away_matchup_estimated_pace_48'),
        difference('home_form_days_since_last_game','away_form_days_since_last_game'),
        difference('away_rotation_out_prior_minutes','home_rotation_out_prior_minutes'),
        (np.log(market)-np.log1p(-market))-(np.log(p)-np.log1p(-p)) if market is not None else np.nan],dtype=float)


def partition(rows):
    rows=sorted(rows,key=lambda r:(r['start_time'],r['game_id']))
    days=sorted({r['start_time'][:10] for r in rows});groups={k:[] for k in RULES['minimum_games']}
    if len(days)<3:return groups
    n=len(days);a=max(1,min(n-2,int(.6*n)));b=max(a+1,min(n-1,int(.8*n)))
    for row in rows:
        day=row['start_time'][:10];name='train' if day<days[a] else 'validation' if day<days[b] else 'test'
        groups[name].append(row)
    # No outcome used for fitting or selection may become available after later inputs.
    for i,name in enumerate(('train','validation')):
        later=[r for other in ('train','validation','test')[i+1:] for r in groups[other]]
        if later:
            boundary=min(stamp(r['input_time']) for r in later)
            groups[name]=[r for r in groups[name] if stamp(r['label_observed_at'])<boundary]
    return groups


def collect_rows(client,now):
    checkpoint=client.read(PREFIX+'checkpoint.json.gz',compressed=True)
    if not checkpoint or checkpoint.get('schema')!=1:raise ValueError('Collector checkpoint missing for early learning')
    cache=client.read(CACHE,compressed=True)
    if cache is None:cache={'schema':1,'feature_version':FEATURE_VERSION,'rows':{}}
    if cache.get('schema')!=1 or cache.get('feature_version')!=FEATURE_VERSION or not isinstance(cache.get('rows'),dict):raise ValueError('Early-learning cache invalid; existing history retained')
    rows=[];skipped=0;pending=0
    for name,version in checkpoint.get('joined_forecasts',{}).items():
        path=PREFIX+'evaluations/'+name[:-5]+'/'+version+'.json';cached=cache['rows'].get(path)
        if cached is not None:
            rows.append(cached);continue
        evaluation=client.read(path)
        if not isinstance(evaluation,dict):raise ValueError('Joined forecast record missing')
        if evaluation.get('status')!='Final':pending+=1;continue
        forecast=evaluation.get('forecast') or {};gid=str(evaluation.get('game_id'));game=checkpoint['games'].get(gid)
        if forecast.get('learning_model_id')!=BASE_MODEL or game is None or not final(game):skipped+=1;continue
        p=number(forecast.get('learning_home_probability'));frozen=stamp(forecast.get('captured_at'));odds_start=stamp(forecast.get('start_time'));start=start_time(game)
        if (p is None or not 0<p<1 or frozen is None or start is None or odds_start is None
            or not frozen<min(start,odds_start) or abs((start-odds_start).total_seconds())>900
            or team_name(forecast.get('home_team'))!=team_name(game['home_team'].get('full_name'))
            or team_name(forecast.get('away_team'))!=team_name(game['visitor_team'].get('full_name'))):skipped+=1;continue
        snapshot_path=evaluation.get('snapshot_path','')
        if not snapshot_path.startswith(PREFIX+'snapshots/'+gid+'/'):raise ValueError('Unexpected early-learning snapshot path')
        snapshot=client.read(snapshot_path)
        if not isinstance(snapshot,dict):raise ValueError('Early-learning snapshot missing')
        at=stamp(snapshot.get('captured_at'))
        if at is None or at>frozen:skipped+=1;continue
        features,_=extract_features(snapshot,game,checkpoint)
        outcome_hash=checkpoint.get('settled',{}).get(gid)
        if not outcome_hash:pending+=1;continue
        outcome=client.read(PREFIX+'outcomes/'+gid+'/'+outcome_hash+'.json')
        if not isinstance(outcome,dict):raise ValueError('Early-learning outcome missing')
        observed=stamp(outcome.get('observed_at'));hs=number(outcome.get('home_score'));aws=number(outcome.get('away_score'))
