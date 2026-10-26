"""Train a separate NBA winner candidate from saved context, never replace live weights.

Run: python dual_agent/train_nba_context_neural.py
No sports API calls. Training readiness is checked before loading ML dependencies.
"""
import base64
import copy
import hashlib
import io
import json
from datetime import datetime, timezone
if __package__:
    from .nba_pregame_snapshots import Client, stamp, number, safe_issue
    from .nba_training_dataset import DATA_PREFIX, FEATURE_COLUMNS, FEATURE_VERSION
else:
    from nba_pregame_snapshots import Client, stamp, number, safe_issue
    from nba_training_dataset import DATA_PREFIX, FEATURE_COLUMNS, FEATURE_VERSION

MODEL_PREFIX='nba/context_models_v1/'
PARTITIONS=('train','validation','calibration','test')
MIN_GAMES={'train':500,'validation':100,'calibration':100,'test':100}
MIN_CLASS=20
CONFIG={'trainer_version':1,'hidden_layers':[64,32,16],'seed':42,'alpha':.1,
        'learning_rate':.0003,'max_epochs':100,'patience':10,'min_games':MIN_GAMES,
        'min_class':MIN_CLASS,'probability_clip':1e-7,'min_observed_fraction':.1,'min_observed_games':20}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def validate_dataset(data,now=None):
    now=now or datetime.now(timezone.utc)
    if data.get('schema')!=1 or data.get('feature_version')!=FEATURE_VERSION:
        raise ValueError('Unsupported dataset schema or feature version; prepare the current dataset first')
    if data.get('label_column')!='label': raise ValueError('Unexpected label field')
    if data.get('feature_columns')!=list(FEATURE_COLUMNS): raise ValueError('Dataset feature allowlist differs from trainer')
    groups={name:[] for name in PARTITIONS};seen=set()
    for row in data.get('rows',[]):
        if row.get('game_id') is None: raise ValueError('Missing game identity')
        identity=str(row['game_id'])
        if identity in seen: raise ValueError('Duplicate game in dataset')
        seen.add(identity)
        if set(row.get('features',{}))!=set(FEATURE_COLUMNS): raise ValueError('Unexpected or missing predictor fields')
        for value in row['features'].values():
            if value is not None and (isinstance(value,(str,bool)) or number(value) is None):
                raise ValueError('Non-numeric or nonfinite predictor')
        captured=stamp(row.get('captured_at'));start=stamp(row.get('scheduled_start'))
        if captured is None or start is None or captured>now or (start-captured).total_seconds()<3600:
            raise ValueError('Training input violates frozen one-hour pregame policy')
        if row.get('label') is None: continue
        if type(row['label']) is not int or row['label'] not in (0,1): raise ValueError('Invalid final-result label')
        observed=stamp(row.get('label_observed_at'))
        if observed is None or observed<=start or observed>now: raise ValueError('Final result lacks a valid observation timestamp')
        partition=row.get('partition')
        if partition not in PARTITIONS:
            if partition in ('purged','unassigned'): continue
            raise ValueError('Unknown training partition')
        groups[partition].append(row)
    for i,name in enumerate(PARTITIONS[:-1]):
        later=[r for p in PARTITIONS[i+1:] for r in groups[p]]
        if not groups[name] or not later: continue
        if max(stamp(r['scheduled_start']).date() for r in groups[name])>=min(stamp(r['scheduled_start']).date() for r in later):
            raise ValueError('Training partitions overlap or are not chronological date groups')
        boundary=min(stamp(r['captured_at']) for r in later)
        if any(stamp(r['label_observed_at'])>=boundary for r in groups[name]):
            raise ValueError('A training or calibration label was observed after later partition inputs')
    return groups


def readiness(groups):
    reasons=[]
    for name in PARTITIONS:
        rows=groups[name];counts={y:sum(r['label']==y for r in rows) for y in (0,1)}
        if len(rows)<MIN_GAMES[name]: reasons.append(f'{name}: {len(rows)}/{MIN_GAMES[name]} completed games')
        if min(counts.values())<MIN_CLASS: reasons.append(f'{name}: needs at least {MIN_CLASS} examples of each outcome')
    required_groups={
        'recent form':('home_form_margin_8','away_form_margin_8'),
        'strength':('home_elo','away_elo'),
        'matchup':('home_matchup_estimated_pace_48','away_matchup_estimated_pace_48'),
        'rotation':('home_rotation_conditional_top5_minutes','away_rotation_conditional_top5_minutes'),
        'sportsbook context':('market_home_probability',)}
    support=max(CONFIG['min_observed_games'],int(CONFIG['min_observed_fraction']*len(groups['train'])))
    for label,fields in required_groups.items():
        if any(sum(r['features'][f] is not None for r in groups['train'])<support for f in fields):
            reasons.append(f'{label}: insufficient observed training support ({support} per required feature)')
    return reasons


def fit_preprocessing(train):
    import numpy as np
    full=np.array([[r['features'][f] if r['features'][f] is not None else np.nan for f in FEATURE_COLUMNS] for r in train],dtype=float)
    minimum=max(CONFIG['min_observed_games'],int(CONFIG['min_observed_fraction']*len(train)))
    selected=[];medians=[];means=[];scales=[]
    for j,name in enumerate(FEATURE_COLUMNS):
        column=full[:,j];observed=column[np.isfinite(column)]
        if len(observed)<minimum: continue
        middle=float(np.median(observed));filled=np.where(np.isfinite(column),column,middle)
        scale=float(np.std(filled))
        if scale<1e-8: continue
        selected.append(name);medians.append(middle);means.append(float(np.mean(filled)));scales.append(scale)
    if not selected: raise ValueError('No supported varying training inputs')
    return selected,np.array(medians),np.array(means),np.array(scales)


def matrix(rows,features,medians,means,scales):
    import numpy as np
    x=np.array([[r['features'][f] if r['features'][f] is not None else np.nan for f in features] for r in rows],dtype=float)
    return (np.where(np.isfinite(x),x,medians)-means)/scales


def metrics(y,p):
    import numpy as np
    y=np.asarray(y);p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    reliability=[]
    for lo,hi in ((.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.01)):
        confidence=np.maximum(p,1-p);mask=(confidence>=lo)&(confidence<hi)
        if mask.any(): reliability.append({'low':lo,'high':min(1.,hi),'games':int(mask.sum()),
            'average_confidence':float(confidence[mask].mean()),'actual_accuracy':float(((p[mask]>=.5)==y[mask]).mean())})
    return {'games':len(y),'accuracy':float(((p>=.5)==y).mean()),'brier':float(((p-y)**2).mean()),
        'log_loss':float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),'confidence_buckets':reliability}


def load_candidate(package):
    """Safe NumPy-only loader for this candidate format, separate from live .npz files."""
    import numpy as np
    if package.get('kind')!='nba_context_neural_candidate' or package.get('schema')!=1:
        raise ValueError('Unsupported context candidate package')
    raw=base64.b64decode(package['npz_base64'],validate=True)
    if hashlib.sha256(raw).hexdigest()!=package.get('weights_sha256'): raise ValueError('Candidate weights checksum mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as source: arrays={k:source[k].copy() for k in source.files}
    if package['metadata'].get('feature_version')!=FEATURE_VERSION: raise ValueError('Candidate feature version mismatch')
    features=package['metadata']['features'];widths=[len(features),64,32,16,1]
    if len(set(features))!=len(features) or not set(features)<=set(FEATURE_COLUMNS): raise ValueError('Unknown candidate feature')
    for i in range(4):
        if arrays['w'+str(i)].shape!=(widths[i],widths[i+1]) or arrays['b'+str(i)].shape!=(widths[i+1],):
            raise ValueError('Candidate weight dimension mismatch')
    for key in ('median','mean','scale'):
        if arrays[key].shape!=(len(features),): raise ValueError('Candidate preprocessing dimension mismatch')
    if any(not np.isfinite(a).all() for a in arrays.values()) or (arrays['scale']<=0).any():
        raise ValueError('Invalid candidate numerical values')
    if arrays['cal_coef'].shape!=(1,) or arrays['cal_intercept'].shape!=(1,): raise ValueError('Invalid calibrator dimensions')
    return arrays


def forward(arrays,x):
    import numpy as np
    for i in range(4):
        x=x@arrays['w'+str(i)]+arrays['b'+str(i)]
        if i<3:x=np.maximum(x,0)
    limit=np.log((1-CONFIG['probability_clip'])/CONFIG['probability_clip'])
    logits=np.clip(x.ravel(),-limit,limit)*arrays['cal_coef'][0]+arrays['cal_intercept'][0]
    return 1/(1+np.exp(-np.clip(logits,-30,30)))


def predict_candidate(package,features,captured_at,scheduled_start):
    """For a later prospective comparison; not called by the live winner engine."""
    import numpy as np
    captured,start=stamp(captured_at),stamp(scheduled_start)
    available=stamp(package.get('metadata',{}).get('available_after'))
    if captured is None or start is None or available is None or not available<=captured<start:
        raise ValueError('Candidate prediction must be pregame and after model availability')
    if set(features)!=set(FEATURE_COLUMNS): raise ValueError('Candidate requires the exact prepared feature schema')
    if any(v is not None and (isinstance(v,(str,bool)) or number(v) is None) for v in features.values()):
        raise ValueError('Invalid candidate predictor value')
    arrays=load_candidate(package)
    x=matrix([{'features':features}],package['metadata']['features'],arrays['median'],arrays['mean'],arrays['scale'])
    probability=float(forward(arrays,x)[0])
    return {'home_probability':probability,'away_probability':1-probability,
            'model_id':package['metadata']['model_id'],'status':'experimental context candidate'}


def train_package(data,dataset_hash,groups,trained_at=None):
    """Fit one fixed architecture; validation selects epoch, calibration fits probability mapping."""
    import numpy as np
    import sklearn
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from threadpoolctl import threadpool_limits
    reasons=readiness(groups)
    if reasons: raise ValueError('Training attempted before readiness requirements were met')
    features,median_values,mean_values,scales=fit_preprocessing(groups['train'])
    x={name:matrix(rows,features,median_values,mean_values,scales) for name,rows in groups.items()}
    y={name:np.array([r['label'] for r in rows]) for name,rows in groups.items()}
    net=MLPClassifier(hidden_layer_sizes=tuple(CONFIG['hidden_layers']),alpha=CONFIG['alpha'],
        learning_rate_init=CONFIG['learning_rate'],random_state=CONFIG['seed'],batch_size=128)
    best=None;best_loss=float('inf');stale=0;best_epoch=0
    with threadpool_limits(limits=2):
        for epoch in range(CONFIG['max_epochs']):
            net.partial_fit(x['train'],y['train'],classes=np.array([0,1]))
            score=log_loss(y['validation'],net.predict_proba(x['validation'])[:,1],labels=[0,1])
            if score<best_loss-1e-5:
                best=[(w.copy(),b.copy()) for w,b in zip(net.coefs_,net.intercepts_)]
                best_loss=score;stale=0;best_epoch=epoch+1
            else:
                stale+=1
                if stale>=CONFIG['patience']:break
        net.coefs_=[w for w,b in best];net.intercepts_=[b for w,b in best]
        raw=np.clip(net.predict_proba(x['calibration'])[:,1],1e-7,1-1e-7)
        calibration=LogisticRegression(C=1.,random_state=42).fit(np.log(raw/(1-raw))[:,None],y['calibration'])
    arrays={'median':median_values,'mean':mean_values,'scale':scales,
            'cal_coef':calibration.coef_.ravel(),'cal_intercept':calibration.intercept_}
    for i,(w,b) in enumerate(best):arrays['w'+str(i)]=w;arrays['b'+str(i)]=b
    p=forward(arrays,x['test'])
    report={'test':metrics(y['test'],p),'training_prevalence_baseline':metrics(y['test'],np.full(len(p),y['train'].mean())),
        'partition_games':{name:len(rows) for name,rows in groups.items()},'best_epoch':best_epoch,
        'selected_features':len(features),'omitted_features':[f for f in FEATURE_COLUMNS if f not in features],
        'validation_log_loss':float(best_loss),'calibration_slope':float(calibration.coef_[0,0]),
        'training_performed':True,'approved_for_live_use':False,
        'limitations':['Held-out results do not establish improvement over the current live engine.',
            'No automatic promotion. A later prospective comparison is required.',
            'Missing starters and causal replacement impact remain unknown; rotation scenarios are descriptive.',
            'This is a winner candidate, not player-prop or live-game training.']}
    market=[(i,r['features']['market_home_probability']) for i,r in enumerate(groups['test']) if r['features']['market_home_probability'] is not None]
    if market:
        indices,prices=zip(*market)
        report['same_market_subset']={'candidate':metrics(y['test'][list(indices)],p[list(indices)]),
                                      'sportsbook_no_vig':metrics(y['test'][list(indices)],prices)}
    trained_at=trained_at or datetime.now(timezone.utc)
    available=max([trained_at]+[stamp(r['label_observed_at']) for rows in groups.values() for r in rows])
    model_id='nba_context_neural_'+fingerprint({'dataset':dataset_hash,'config':CONFIG})[:20]
    meta={'model_id':model_id,'dataset_hash':dataset_hash,'feature_version':FEATURE_VERSION,'features':features,
          'architecture':[len(features),64,32,16,1],'available_after':available.isoformat(),
          'config':copy.deepcopy(CONFIG),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
          'partition_dates':{name:{'first':min(r['scheduled_start'] for r in rows),'last':max(r['scheduled_start'] for r in rows)} for name,rows in groups.items()}}
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);raw=buffer.getvalue()
    package={'schema':1,'kind':'nba_context_neural_candidate','metadata':meta,'report':report,
             'weights_sha256':hashlib.sha256(raw).hexdigest(),'npz_base64':base64.b64encode(raw).decode()}
    # Verify exported inference matches sklearn + calibrator before publishing.
    restored=load_candidate(package);replayed=forward(restored,x['test'])
    if not np.allclose(replayed,p,rtol=1e-10,atol=1e-10):raise ValueError('Exported inference parity failed')
    raw_test=np.clip(net.predict_proba(x['test'])[:,1],1e-7,1-1e-7)
    expected=calibration.predict_proba(np.log(raw_test/(1-raw_test))[:,None])[:,1]
    if not np.allclose(expected,p,rtol=1e-6,atol=1e-7):raise ValueError('Saved neural inference differs from fitted classifier')
    return package


def run(client=None):
    client=client or Client()
    latest=client.read(DATA_PREFIX+'latest.json')
    if not latest: raise ValueError('Prepared dataset pointer missing; run the collector workflow first')
    path=latest.get('dataset_path','')
    if not path.startswith(DATA_PREFIX+'datasets/') or not path.endswith('.json.gz'): raise ValueError('Unexpected dataset reference')
    data=client.read(path,compressed=True)
    if not data or fingerprint(data)!=latest.get('dataset_hash'): raise ValueError('Dataset checksum mismatch')
    groups=validate_dataset(data);reasons=readiness(groups)
    if reasons:
        status={'status':'waiting_for_completed_training_data','training_performed':False,
            'partition_games':{name:len(rows) for name,rows in groups.items()},'reasons':reasons,
            'dataset_hash':latest['dataset_hash'],'live_model_changed':False,'sports_api_requests':0}
        client.write(MODEL_PREFIX+'training_status.json',status,replace=True)
        print(json.dumps(status),flush=True);return status
    identity=fingerprint({'dataset':latest['dataset_hash'],'config':CONFIG})
    candidate_path=MODEL_PREFIX+'candidates/'+identity+'.json.gz'
    package=client.read(candidate_path,compressed=True)
    trained=False
    if package is None:
        package=train_package(data,latest['dataset_hash'],groups);trained=True
        client.write(candidate_path,package,compressed=True)
    else:
        load_candidate(package)
        if package['metadata']['dataset_hash']!=latest['dataset_hash'] or package['metadata']['config']!=CONFIG:
            raise ValueError('Existing candidate identity mismatch')
    pointer={'schema':1,'candidate_path':candidate_path,'model_id':package['metadata']['model_id'],
             'dataset_hash':latest['dataset_hash'],'approved_for_live_use':False}
    client.write(MODEL_PREFIX+'latest_candidate.json',pointer,replace=True)
    status={'status':'candidate_saved' if trained else 'existing_candidate_reused','training_performed':trained,
        **pointer,'test_metrics':package['report']['test'],'live_model_changed':False,'sports_api_requests':0}
    client.write(MODEL_PREFIX+'training_status.json',status,replace=True)
    print(json.dumps(status),flush=True);return status


if __name__=='__main__':
    try: run()
    except Exception as exc:
        print('NBA context training stopped: '+safe_issue(exc),flush=True)
        raise SystemExit(1)
