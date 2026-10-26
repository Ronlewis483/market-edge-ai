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
        if (str(outcome.get('game_id'))!=gid or observed is None or not start<observed<=now
            or hs is None or aws is None or min(hs,aws)<=0 or hs==aws
            or hs!=number(game.get('home_team_score')) or aws!=number(game.get('visitor_team_score'))):raise ValueError('Early-learning final result inconsistent')
        row={'game_id':gid,'start_time':start.isoformat(),'input_time':at.isoformat(),'forecast_time':frozen.isoformat(),
             'label_observed_at':observed.isoformat(),'base_probability':p,'base_model_id':BASE_MODEL,
             'features':features,'label':int(hs>aws),'evaluation_path':path}
        cache['rows'][path]=row;rows.append(row)
    # One genuine first forecast per game, even if an older paired-comparison version duplicated it.
    chosen={}
    for row in sorted(rows,key=lambda r:(r['forecast_time'],r['evaluation_path'])):chosen.setdefault(row['game_id'],row)
    client.write(CACHE,cache,replace=True,compressed=True)
    return list(chosen.values()),{'pending_joined_forecasts':pending,'skipped_incompatible_forecasts':skipped}


def transform(raw,mean,scale):
    import numpy as np
    missing=~np.isfinite(raw)
    return np.column_stack([np.ones(len(raw)),(np.where(missing,mean,raw)-mean)/scale,missing.astype(float)])


def probabilities(coef,x,base,max_change=.35):
    import numpy as np
    base=np.asarray(base,dtype=float)
    if not np.isfinite(base).all() or ((base<=0)|(base>=1)).any():raise ValueError('Invalid neural base array')
    offset=np.log(base)-np.log1p(-base)
    correction=max_change*np.tanh((x@coef)/max_change)
    p=1/(1+np.exp(-np.clip(offset+correction,-700,700)))
    return np.clip(p,np.nextafter(0.,1.),np.nextafter(1.,0.))


def load(package):
    import numpy as np
    if package.get('schema')!=1 or package.get('kind')!='nba_neural_context_adjustment':raise ValueError('Unsupported early adjustment')
    meta=package.get('metadata',{})
    if meta.get('feature_version')!=FEATURE_VERSION or meta.get('base_model_id')!=BASE_MODEL or meta.get('fields')!=list(FIELDS):
        raise ValueError('Early adjustment provenance mismatch')
    if meta.get('max_logit_change')!=RULES['max_logit_change']:raise ValueError('Unexpected early adjustment bound')
    raw=base64.b64decode(package['npz_base64'],validate=True)
    if hashlib.sha256(raw).hexdigest()!=package['weights_sha256']:raise ValueError('Early adjustment checksum mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as source:a={k:source[k].copy() for k in source.files}
    if (a['coef'].shape!=(1+2*len(FIELDS),) or a['mean'].shape!=(len(FIELDS),) or a['scale'].shape!=(len(FIELDS),)
        or any(not np.isfinite(v).all() for v in a.values()) or (a['scale']<=0).any()):raise ValueError('Invalid early adjustment arrays')
    return a


def predict(package,features,base_probability,base_model_id,captured_at,scheduled_start):
    captured,start=stamp(captured_at),stamp(scheduled_start);available=stamp(package['metadata'].get('available_after'))
    if available is None or captured is None or start is None or not available<=captured<start:raise ValueError('Early adjustment is not available for this pregame forecast')
    if base_model_id!=package['metadata']['base_model_id']:raise ValueError('Frozen neural base version changed')
    a=load(package);raw=derived(features,base_probability)[None,:];x=transform(raw,a['mean'],a['scale'])
    import numpy as np
    p=float(probabilities(a['coef'],x,np.array([float(base_probability)]),package['metadata']['max_logit_change'])[0])
    return {'home_probability':p,'away_probability':1-p,'model_id':package['metadata']['model_id'],
            'status':'experimental bounded adjustment to trained neural model'}


def fit(groups,trained_at=None):
    import numpy as np
    from scipy.optimize import minimize
    if __package__:
        from .train_nba_context_neural import metrics
    else:from train_nba_context_neural import metrics
    raw={name:np.array([derived(r['features'],r['base_probability']) for r in rows]) for name,rows in groups.items()}
    mean=np.array([float(np.mean(col[np.isfinite(col)])) if np.isfinite(col).any() else 0. for col in raw['train'].T])
    scale=np.array([float(np.std(col[np.isfinite(col)])) if np.isfinite(col).sum()>1 else 1. for col in raw['train'].T]);scale[scale<1e-8]=1.
    x={name:transform(values,mean,scale) for name,values in raw.items()}
    p0={name:np.array([r['base_probability'] for r in rows]) for name,rows in groups.items()}
    y={name:np.array([r['label'] for r in rows]) for name,rows in groups.items()}
    limit=RULES['max_logit_change'];best=None;trials=[]
    for penalty in RULES['penalties']:
        def objective(coef):
            z=x['train']@coef;t=np.tanh(z/limit);p=probabilities(coef,x['train'],p0['train'],limit)
            logloss=-np.mean(y['train']*np.log(np.clip(p,1e-12,1))+(1-y['train'])*np.log(np.clip(1-p,1e-12,1)))
            value=logloss+penalty*np.dot(coef,coef)/(2*len(y['train']))
            gradient=x['train'].T@((p-y['train'])*(1-t*t))/len(p)+penalty*coef/len(p)
            return value,gradient
        result=minimize(objective,np.zeros(x['train'].shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12})
        if not result.success or not np.isfinite(result.x).all():continue
        vp=probabilities(result.x,x['validation'],p0['validation'],limit);validation=metrics(y['validation'],vp)
        trials.append({'penalty':penalty,'validation':validation})
        if best is None or validation['brier']<best[0]:best=(validation['brier'],result.x.copy(),penalty,validation)
    if best is None:raise ValueError('Bounded adjustment did not converge')
    base_validation=metrics(y['validation'],p0['validation'])
    passed=(best[3]['brier']<base_validation['brier']-1e-8 and best[3]['log_loss']<=base_validation['log_loss']
            and best[3]['accuracy']>=base_validation['accuracy'])
    report={'training_performed':True,'validation_passed':passed,'validation':best[3],
        'validation_neural_base':base_validation,'penalty_trials':trials,'selected_penalty':best[2],
        'partition_games':{k:len(v) for k,v in groups.items()},'approved_for_live_use':False,
        'note':'Frozen neural predictions retained as base; only 13 regularized correction parameters fitted. Test outcomes never select the correction.'}
    if not passed:return None,report
    # Inspect the held-out test only after validation passes; it cannot tune parameters.
    report['test']=metrics(y['test'],probabilities(best[1],x['test'],p0['test'],limit))
    report['test_neural_base']=metrics(y['test'],p0['test'])
    training_id=hashed({'rows':groups,'rules':RULES});available=trained_at or datetime.now(timezone.utc)
    available=max([available]+[stamp(r['label_observed_at']) for rows in groups.values() for r in rows])
    meta={'model_id':'nba_neural_adjustment_'+training_id[:20],'base_model_id':BASE_MODEL,'feature_version':FEATURE_VERSION,
          'fields':list(FIELDS),'max_logit_change':limit,'available_after':available.isoformat(),'rules':RULES}
    buf=io.BytesIO();np.savez_compressed(buf,coef=best[1],mean=mean,scale=scale);weights=buf.getvalue()
    package={'schema':1,'kind':'nba_neural_context_adjustment','metadata':meta,'report':report,
             'weights_sha256':hashlib.sha256(weights).hexdigest(),'npz_base64':base64.b64encode(weights).decode()}
    load(package)
    return package,report


def run_early(client,now=None):
    now=now or datetime.now(timezone.utc);rows,diagnostics=collect_rows(client,now)
    for row in rows:
        at,forecast,start,observed=[stamp(row.get(k)) for k in ('input_time','forecast_time','start_time','label_observed_at')]
        if (None in (at,forecast,start,observed) or not at<=forecast<start<observed<=now
            or type(row.get('label')) is not int or row['label'] not in (0,1) or row.get('base_model_id')!=BASE_MODEL):
            raise ValueError('Invalid provenance in early-learning history')
        derived(row['features'],row['base_probability'])
    groups=partition(rows);reasons=[]
    for name,minimum in RULES['minimum_games'].items():
        if len(groups[name])<minimum:reasons.append(f'{name}: {len(groups[name])}/{minimum} usable completed games')
        if min(sum(r['label']==y for r in groups[name]) for y in (0,1))<RULES['minimum_classes'][name]:reasons.append(name+': insufficient examples of both outcomes')
    status={'method':'bounded_context_adjustment_to_frozen_trained_neural_predictions','usable_completed_games':len(rows),
        'partition_games':{k:len(v) for k,v in groups.items()},'reasons':reasons,**diagnostics,
        'live_model_changed':False,'sports_api_requests':0,'minimum_partition_games':RULES['minimum_games']}
    if reasons:status.update(status='waiting_for_early_learning_data',training_performed=False);return status
    identity=hashed({'groups':groups,'rules':RULES});path=ROOT+'candidates/early_'+identity+'.json.gz'
    package=client.read(path,compressed=True);trained=package is None
    if package is None:
        package,report=fit(groups,now)
        if package is None:
            status.update(status='early_adjustment_no_validation_gain',training_performed=True,validation_report=report)
            return status
        client.write(path,package,compressed=True)
    else:load(package);report=package['report']
    current=client.read(ROOT+'latest_candidate.json')
    # A full-context neural candidate takes priority over the earlier adjustment stage.
    if current is None or str(current.get('model_id','')).startswith('nba_neural_adjustment_'):
        client.write(ROOT+'latest_candidate.json',{'schema':1,'candidate_path':path,'model_id':package['metadata']['model_id'],
            'approved_for_live_use':False,'candidate_kind':package['kind']},replace=True)
    status.update(status='early_adjustment_candidate_saved',training_performed=trained,
        candidate_path=path,model_id=package['metadata']['model_id'],validation_report=report)
    return status
