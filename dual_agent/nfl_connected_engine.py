"""Fast original NFL inference and background connected prediction finalization.

The original model equation is unchanged. Fitted estimators are reused only
when the exact earlier training rows match. No context/odds/storage requests
occur in predict_original_cached().
"""
from collections import OrderedDict
import hashlib
import threading
import time

import numpy as np
import pandas as pd

FEATURES=['win_pct_diff','avg_point_diff_diff','recent_5_win_pct_diff','recent_5_point_diff_diff','rest_diff']
_MODELS=OrderedDict();_LOCK=threading.RLock()


def predict_original_cached(feature_games,future_features):
    from sklearn.linear_model import LogisticRegression
    if feature_games is None or feature_games.empty or future_features is None or future_features.empty:
        raise ValueError('NFL historical/future features are empty')
    missing=[c for c in FEATURES if c not in feature_games or c not in future_features]
    if missing:raise ValueError('Missing NFL model features: '+str(missing))
    future=future_features.iloc[[0]].copy()
    start=pd.to_datetime(future.iloc[0]['start_time'],utc=True,errors='coerce')
    if pd.isna(start):raise ValueError('Upcoming NFL game time is invalid')
    history=feature_games.copy()
    history['start_time']=pd.to_datetime(history.start_time,utc=True,errors='coerce')
    history=history[(history.start_time<start)&history.home_win.notna()].copy()
    history=history.dropna(subset=FEATURES).sort_values('start_time')
    if history.empty or history.home_win.nunique()<2:raise ValueError('Not enough completed NFL history of both outcomes')
    if future[FEATURES].isna().any().any():raise ValueError('Future NFL matchup contains missing model features')
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(history[FEATURES+['home_win','start_time']],index=False).values.tobytes()).hexdigest()
    with _LOCK:
        cached=_MODELS.get(fingerprint)
        hit=cached is not None and time.monotonic()-cached[0]<86400
        if not hit:
            model=LogisticRegression(max_iter=2000).fit(history[FEATURES],history.home_win.astype(int))
            _MODELS[fingerprint]=(time.monotonic(),model)
            while len(_MODELS)>16:_MODELS.popitem(last=False)
        else:model=cached[1]
        _MODELS.move_to_end(fingerprint)
    p=float(model.predict_proba(future[FEATURES])[0,1]);home=future.iloc[0]['home_team'];away=future.iloc[0]['away_team']
    return dict(home_team=home,away_team=away,start_time=start,home_win_probability=p,away_win_probability=1-p,
                predicted_team=home if p>=.5 else away,confidence=max(p,1-p),training_games=len(history),
                original_model_cache_hit=hit)


def finalize_connected_result(result):
    """Called by the app's existing background worker after context retrieval."""
    from .sports_connected_winner import correct,freeze,nfl_context,digest
    from .nfl_live_engine import build_live_nfl_opportunities
    frame=pd.DataFrame(result.get('predictions')).copy()
    if frame.empty:return result
    base_id=result.get('connected_base_id')
    if not base_id:raise ValueError('NFL original model identity missing')
    frozen=[]
    for i,row in frame.iterrows():
        start=pd.Timestamp(row['commence_time'])
        if start.tzinfo is None:start=start.tz_localize('UTC')
        if start<=pd.Timestamp.now(tz='UTC'):
            frame.at[i,'connected_status']='Game started; saved prediction retained'
            continue
        original=float(row.get('original_home_win_probability',row['home_win_probability']))
        features=row.get('connected_features');features=features if isinstance(features,dict) else {}
        context=row.get('game_context');context=dict(context) if isinstance(context,dict) else {}
        forecast=correct('NFL',base_id,original,nfl_context(features,context),start_time=start)
        p=forecast['connected_probability']
        frame.at[i,'home_win_probability']=p;frame.at[i,'away_win_probability']=1-p
        frame.at[i,'confidence']=max(p,1-p);frame.at[i,'predicted_team']=row['home_team'] if p>=.5 else row['away_team']
        frame.at[i,'connected_model_id']=forecast['model_id'];frame.at[i,'connected_status']=forecast['status']
        frame.at[i,'connected_applied']=forecast['applied'];frame.at[i,'input_status']=forecast['status']
        context['probability_use']='Original probability with learned NFL neural/player-context residual correction' if forecast['applied'] else 'Original NFL probability retained; sport-specific correction not available'
        market=context.get('market')
        if isinstance(market,dict) and market.get('home_probability') is not None:
            market=dict(market);market['gap']=abs(float(market['home_probability'])-p);context['market']=market
        if 'game_context' not in frame:frame['game_context']=[{} for _ in range(len(frame))]
        frame.at[i,'game_context']=context
        frozen.append(dict(event_id=digest([row['home_team'],row['away_team'],start.isoformat()]),start_time=start.isoformat(),
                           home_team=row['home_team'],away_team=row['away_team'],connected=forecast))
    result['predictions']=frame;result['all_predictions']=frame
    result['opportunities']=build_live_nfl_opportunities(model_predictions=frame,best_lines=result.get('best_lines'),
            historical_accuracy=result.get('connected_historical_accuracy'),historical_sample=result.get('connected_historical_sample'))
    issue=freeze('NFL',frozen)
    if issue:result['connected_learning_issue']=issue
    result['connected_context_pending']=False
    return result


def _encoder(rows):
    from .sports_connected_winner import number
    names=sorted({k for r in rows for k in r['context']})
    columns=[k for k in names if sum(number(r['context'].get(k)) is not None for r in rows)>=20]
    values=np.array([[np.nan if number(r['context'].get(k)) is None else number(r['context'][k]) for k in columns] for r in rows],dtype=float)
    median=np.nanmedian(values,axis=0);filled=np.where(np.isfinite(values),values,median)
    mean=filled.mean(axis=0);scale=filled.std(axis=0);scale=np.where(scale<1e-8,1.,scale)
    return dict(columns=columns,median=median.tolist(),mean=mean.tolist(),scale=scale.tolist())


def _encode(model,row):
    from .sports_connected_winner import encode
    return encode(row['context'],model['columns'],np.array(model['median']),np.array(model['mean']),np.array(model['scale']),row['original_probability'])


def _fit_brain(rows):
    """An NFL-only neural context signal, fitted on previously observed results."""
    from scipy.optimize import minimize
    from .sports_connected_winner import probability
    model=_encoder(rows);x=np.array([_encode(model,r) for r in rows]);y=np.array([r['outcome'] for r in rows])
    p=np.array([r['original_probability'] for r in rows]);d=x.shape[1];h=8
    sizes=[d*h,h,h,1];edges=np.cumsum([0]+sizes);shapes=[(d,h),(h,),(h,),()]
    def unpack(theta):return {k:theta[edges[i]:edges[i+1]].reshape(shapes[i]) for i,k in enumerate(('w1','b1','w2','b2'))}
    initial=np.r_[np.random.default_rng(42).normal(0,.05,d*h),np.zeros(2*h+1)]
    def objective(theta):
        v=unpack(theta);hidden=np.tanh(x@v['w1']+v['b1']);t=np.tanh(hidden@v['w2']+v['b2']);q=probability(p,.7*t)
        loss=-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))+10*np.dot(theta,theta)/(2*len(y))
        dz=(q-y)*.7*(1-t*t)/len(y);dh=dz[:,None]*v['w2']*(1-hidden*hidden)
        grad=np.r_[(x.T@dh).ravel(),dh.sum(axis=0),(hidden.T@dz).ravel(),dz.sum()]+10*theta/len(y)
        return loss,grad
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':300})
    if not result.success or not np.isfinite(result.x).all():return None
    model.update({k:v.tolist() for k,v in unpack(result.x).items()});return model


def _brain_delta(model,row):
    from .sports_connected_winner import forward
    return float(forward(_encode(model,row),{k:np.asarray(model[k]) for k in ('w1','b1','w2','b2')}))


def _residual_x(model,row,neural_delta):
    return np.r_[_encode(model,row),neural_delta,row['original_probability']]


def connected_delta(model,original,context):
    """Same connected flow as NBA: original offset + neural/player residual."""
    row={'original_probability':original,'context':context}
    neural=_brain_delta(model['brain'],row)
    x=np.asarray(_residual_x(model['encoder'],row,neural),dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Invalid NFL connected inputs')
    delta=0.
    for tree in model['trees']:
        node=0
        for _ in range(len(tree['left'])+1):
            if tree['left'][node]==-1:
                delta+=model['learning_rate']*tree['value'][node];break
            f=tree['feature'][node]
            if not 0<=f<len(x):raise ValueError('Invalid NFL residual tree')
            node=tree['left'][node] if x[f]<=tree['threshold'][node] else tree['right'][node]
        else:raise ValueError('Invalid NFL residual cycle')
    return float(np.clip(model['correction_gain']*delta,-.7,.7))


def train_connected(sport,base_id,rows,store):
    """Fit NFL neural signal and offset trees using forward-only internal folds."""
    from datetime import datetime,timezone
    from sklearn.base import ClassifierMixin,BaseEstimator
    from sklearn.ensemble import GradientBoostingClassifier
    from .sports_connected_winner import ROOT,partitions,stamp,metrics,probability,digest
    if sport!='NFL':raise ValueError('NFL learner cannot train another sport')
    rows=[r for r in rows if r['base_id']==base_id];groups=partitions(rows)
    report={'sport':'NFL','base_id':base_id,'architecture':'original_offset_neural_context_residual_v2',
            'status':'waiting_for_graded_forecasts','graded_games':len(rows),'trained':False,'activated':False,
            'partition_games':dict(zip(('train','validation','test'),map(len,groups))),
            'minimum_games':{'train':80,'validation':25,'test':25},'paid_sports_api_requests':0}
    if any(len(g)<n or min(sum(r['outcome']==v for r in g) for v in (0,1))<5 for g,n in zip(groups,(80,25,25))):return report
    previous=store.read(ROOT+'nfl/training_report.json.gz')
    if previous and previous.get('architecture')==report['architecture'] and previous.get('base_id')==base_id and previous.get('trained') and len(rows)<previous.get('graded_games',0)+25:
        return dict(previous,status='waiting_for_25_new_results',available_graded_games=len(rows))
    training=sorted(groups[0],key=lambda r:(r['start_time'],r['event_id']));oof=[];cache={}
    for date in sorted({r['start_time'][:10] for r in training}):
        batch=[r for r in training if r['start_time'][:10]==date]
        boundary=min(stamp(r['captured_at']) for r in batch)
        prior=[r for r in training if r['start_time'][:10]<date and stamp(r['label_observed_at'])<boundary]
        if len(prior)<40 or min(sum(r['outcome']==v for r in prior) for v in (0,1))<5:continue
        identity=tuple(r['event_id'] for r in prior)
        if identity not in cache:cache[identity]=_fit_brain(prior)
        brain=cache[identity]
        if brain is not None:
            oof.extend((r,_brain_delta(brain,r)) for r in batch)
    report['residual_training_games']=len(oof)
    if len(oof)<40 or min(sum(r['outcome']==v for r,_ in oof) for v in (0,1))<5:
        return dict(report,status='waiting_for_forward_neural_examples')
    brain=_fit_brain(training)
    if brain is None:return dict(report,status='neural_fit_not_converged')
    encoder=_encoder(training);x=np.array([_residual_x(encoder,r,n) for r,n in oof]);y=np.array([r['outcome'] for r,_ in oof])
    class OriginalOffset(ClassifierMixin,BaseEstimator):
        def fit(self,X,y,sample_weight=None):self.classes_=np.array([0,1]);self.n_features_in_=np.asarray(X).shape[1];return self
        def predict_proba(self,X):
            p=np.clip(np.asarray(X)[:,-1],1e-7,1-1e-7);return np.column_stack([1-p,p])
    residual=GradientBoostingClassifier(n_estimators=80,learning_rate=.025,max_depth=2,min_samples_leaf=10,random_state=42,init=OriginalOffset())
    residual.fit(x,y);trees=[]
    for estimator in residual.estimators_[:,0]:
        t=estimator.tree_;trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),
                                          feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,0,0].tolist()))
    model={'schema':1,'kind':'nfl_offset_tree_v2','sport':'NFL','base_id':base_id,'brain':brain,'encoder':encoder,
           'trees':trees,'learning_rate':float(residual.learning_rate),'correction_gain':1.,'trained_at':datetime.now(timezone.utc).isoformat()}
    def raw_delta(g):
        xx=np.array([_residual_x(encoder,r,_brain_delta(brain,r)) for r in g])
        return sum(residual.learning_rate*t.predict(xx) for t in residual.estimators_[:,0])
    val=groups[1];vy=np.array([r['outcome'] for r in val]);vp=np.array([r['original_probability'] for r in val]);vd=raw_delta(val)
    gain=min([0.,.25,.5,.75,1.],key=lambda v:(-metrics(vy,probability(vp,np.clip(v*vd,-.7,.7)))['accuracy'],metrics(vy,probability(vp,np.clip(v*vd,-.7,.7)))['brier']))
    model['correction_gain']=gain;passed=True
    for label,g in zip(('validation','test'),groups[1:]):
        p=np.array([r['original_probability'] for r in g]);y=np.array([r['outcome'] for r in g]);delta=np.clip(gain*raw_delta(g),-.7,.7)
        exported=np.array([connected_delta(model,r['original_probability'],r['context']) for r in g])
        if not np.allclose(delta,exported,atol=1e-10,rtol=0):raise ValueError('NFL residual export mismatch')
        old,new=metrics(y,p),metrics(y,probability(p,delta));report[label]={'original':old,'connected':new}
        passed=passed and new['accuracy']>old['accuracy'] and new['brier']<=old['brier'] and new['log_loss']<=old['log_loss']
    model['checks_passed']=bool(passed);model['model_id']='nfl_connected_'+digest(model)[:16]
    report.update(trained=True,status='candidate_saved',activated=bool(passed),checks_passed=bool(passed),correction_gain=gain,model_id=model['model_id'],probability_averaging=False)
    store.write(ROOT+'nfl/candidates/'+model['model_id']+'.json.gz',model)
    if passed:store.write(ROOT+'nfl/active.json.gz',model,replace=True)
    return report
