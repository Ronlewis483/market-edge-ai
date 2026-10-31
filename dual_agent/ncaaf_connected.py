"""NCAA-specific original-offset -> neural -> context-residual winner engine.

No NBA/NFL weights and no probability averaging. Context currently consists of
college team history; player-strength/injury inputs are not fabricated.
"""
import hashlib
import json
from pathlib import Path
import math
from datetime import datetime,timezone
import numpy as np
from sklearn.base import ClassifierMixin,BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy.optimize import minimize
from scipy.special import expit
from dual_agent.ncaaf_prediction import build_dataset

ARCHITECTURE='original_offset_neural_context_residual_v1'


class OriginalOffset(ClassifierMixin,BaseEstimator):
    def fit(self,X,y,sample_weight=None):
        self.classes_=np.array([0,1]);return self
    def predict_proba(self,X):
        p=np.clip(np.asarray(X)[:,-1],1e-6,1-1e-6)
        return np.column_stack([1-p,p])


def _base(rows):
    return make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=1000,random_state=42)).fit(
        [r['x'] for r in rows],[r['y'] for r in rows])


def _logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p)-np.log1p(-p)


def _brain_x(x,p):return np.column_stack([np.asarray(x),_logit(np.asarray(p))/3])


def _brain_fit(rows,probabilities):
    x=_brain_x([r['x'] for r in rows],probabilities);y=np.array([r['y'] for r in rows])
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),.1);x=(x-mean)/scale
    n,d=x.shape;hidden=8;rng=np.random.default_rng(42)
    initial=np.r_[rng.normal(0,.04,d*hidden),np.zeros(hidden),rng.normal(0,.04,hidden),0.]
    def unpack(w):return w[:d*hidden].reshape(d,hidden),w[d*hidden:d*hidden+hidden],w[d*hidden+hidden:-1],w[-1]
    def loss(w):
        W,b,v,c=unpack(w);h=np.tanh(x@W+b);a=np.tanh(h@v+c);logits=_logit(probabilities)+.7*a
        penalty=.01;value=np.mean(np.logaddexp(0,logits)-y*logits)+penalty*np.dot(w,w)/2
        dz=(expit(logits)-y)*.7*(1-a*a)/n
        dh=dz[:,None]*v*(1-h*h)
        grad=np.r_[(x.T@dh).ravel(),dh.sum(axis=0),h.T@dz,dz.sum()]+penalty*w
        return value,grad
    result=minimize(loss,initial,jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-9})
    if not result.success or not np.isfinite(result.x).all():raise ValueError('NCAA neural training did not converge; no connected forecast generated.')
    W,b,v,c=unpack(result.x)
    return {'mean':mean.tolist(),'scale':scale.tolist(),'W':W.tolist(),'b':b.tolist(),'v':v.tolist(),'c':float(c)}


def _brain(model,x,p):
    z=(_brain_x(x,p)-np.array(model['mean']))/np.array(model['scale'])
    return .7*np.tanh(np.tanh(z@np.array(model['W'])+np.array(model['b']))@np.array(model['v'])+model['c'])


def _context(x,p,delta):
    return np.column_stack([np.asarray(x),delta,np.asarray(p)])


def _metrics(y,p):
    from sklearn.metrics import accuracy_score,brier_score_loss,log_loss
    return {'games':len(y),'accuracy':float(accuracy_score(y,np.asarray(p)>=.5)),
            'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1]))}


def _tree_delta(artifact,x):
    x = np.asarray(x,dtype=np.float32)
    output=np.zeros(len(x))
    for tree in artifact['trees']:
        for i,row in enumerate(x):
            node=0
            while tree['left'][node]!=-1:
                node=tree['left'][node] if row[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
            output[i]+=artifact['learning_rate']*tree['value'][node]
    return np.clip(output,-.7,.7)*artifact['gain']


def _predict(artifact,x):
    base=artifact['base']
    p=expit(((np.asarray(x)-np.array(base['mean']))/np.array(base['scale']))@np.array(base['coef'])+base['intercept'])
    neural=_brain(artifact['brain'],x,p)
    correction=_tree_delta(artifact,_context(x,p,neural))
    return p,expit(_logit(p)+correction)


def fit_connected(records,now=None):
    now=now or datetime.now(timezone.utc)
    rows,future,unsupported=build_dataset(records,now)
    if len(rows)<400:raise ValueError('Connected NCAA training needs at least 400 supported completed FBS games.')
    identity=hashlib.sha256(json.dumps([[r['id'],r['x'],r['y']] for r in rows],separators=(',',':')).encode()).hexdigest()[:20]
    path=Path('ncaaf_history_cache/connected_models')/(identity+'.json')
    try:
        artifact=json.loads(path.read_text())
        if artifact.get('architecture')!=ARCHITECTURE:raise ValueError('Model architecture mismatch')
    except (OSError,ValueError):
        artifact=_train(rows,identity)
        try:
            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(artifact,allow_nan=False));temp.replace(path)
        except OSError:pass
    games=[]
    if future:
        base,connected=_predict(artifact,[r['x'] for r in future])
        for r,p,original in zip(future,connected,base):
            games.append({'game_id':r['id'],'home_id':r['home'],'away_id':r['away'],
                'home_team':r['home_name'],'away_team':r['away_name'],'commence_time':r['start'].isoformat(),
                'neutral_site':r['neutral'],'model_home_probability':float(p),'model_away_probability':float(1-p),
                'original_home_probability':float(original),'predicted_team':r['home_name'] if p>=.5 else r['away_name'],
                'model_confidence':float(max(p,1-p)),'model_id':artifact['model_id']})
    return {'games':games,'model_id':artifact['model_id'],'architecture':ARCHITECTURE,
        'training_games':len(rows),'evaluation':artifact['test']['connected'],
        'comparison':{'validation':artifact['validation'],'test':artifact['test']},
        'checks_passed':artifact['checks_passed'],'partition_games':artifact['partition_games'],
        'unsupported_upcoming':unsupported,'correction_gain':artifact['gain'],
        'limitations':'Experimental connected NCAA engine. Retrospective evaluation with approximate historical label timing; prospective accuracy and probability calibration are unestablished. College team context only: player-strength, injuries, transfers and weather are not inputs.'}


def _train(rows,identity):
    days=sorted({r['start'].date() for r in rows});n=len(days)
    if n<10:raise ValueError('Insufficient distinct NCAA game dates.')
    val_day=days[int(.6*n)];test_day=days[int(.8*n)]
    def before_label(r,day):return r['label_time'].date()<day
    train=[r for r in rows if r['start'].date()<val_day and before_label(r,val_day)]
    validation=[r for r in rows if val_day<=r['start'].date()<test_day and before_label(r,test_day)]
    test=[r for r in rows if r['start'].date()>=test_day]
    for name,part,minimum in [('train',train,200),('validation',validation,50),('test',test,50)]:
        if len(part)<minimum or min(sum(r['y']==v for r in part) for v in (0,1))<10:
            raise ValueError('Insufficient chronological NCAA '+name+' support (both outcomes required).')
    # Forward base forecasts: each block model uses only earlier known labels.
    oof=[];i=150
    while i<len(train):
        while i<len(train) and i>0 and train[i]['start'].date()==train[i-1]['start'].date():i+=1
        if i>=len(train):break
        end=min(len(train),i+100)
        while end<len(train) and train[end]['start'].date()==train[end-1]['start'].date():end+=1
        block=train[i:end];prior=[r for r in train[:i] if r['label_time']<block[0]['start']]
        if len(prior)>=150 and len({r['y'] for r in prior})==2:
            base=_base(prior);p=base.predict_proba([r['x'] for r in block])[:,1]
            oof.extend(dict(r,base_p=float(v)) for r,v in zip(block,p))
        i=end
    if len(oof)<180:raise ValueError('Insufficient forward NCAA forecasts to train the neural/context layers.')
    # Forward neural signals avoid fitting the residual on in-sample neural outputs.
    residual=[];i=80
    while i<len(oof):
        while i<len(oof) and i>0 and oof[i]['start'].date()==oof[i-1]['start'].date():i+=1
        if i>=len(oof):break
        end=min(len(oof),i+100)
        while end<len(oof) and oof[end]['start'].date()==oof[end-1]['start'].date():end+=1
        block=oof[i:end];prior=[r for r in oof[:i] if r['label_time']<block[0]['start']]
        if len(prior)>=80 and len({r['y'] for r in prior})==2:
            brain=_brain_fit(prior,np.array([r['base_p'] for r in prior]))
            delta=_brain(brain,[r['x'] for r in block],np.array([r['base_p'] for r in block]))
            residual.extend(dict(r,neural_delta=float(d)) for r,d in zip(block,delta))
        i=end
    if len(residual)<80 or len({r['y'] for r in residual})<2:raise ValueError('Insufficient forward neural signals for NCAA residual training.')
    brain=_brain_fit(oof,np.array([r['base_p'] for r in oof]))
    x=_context([r['x'] for r in residual],[r['base_p'] for r in residual],[r['neural_delta'] for r in residual])
    tree=GradientBoostingClassifier(n_estimators=80,learning_rate=.025,max_depth=2,min_samples_leaf=20,
        random_state=42,init=OriginalOffset()).fit(x,[r['y'] for r in residual])
    exported=[]
    for estimator in tree.estimators_[:,0]:
        t=estimator.tree_;exported.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),
            'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,0,0].tolist()})
    base=_base(train)
    def probabilities(part,gain):
        p=base.predict_proba([r['x'] for r in part])[:,1]
        delta=_brain(brain,[r['x'] for r in part],p)
        raw=_context([r['x'] for r in part],p,delta)
        change=_tree_delta({'trees':exported,'learning_rate':.025,'gain':gain},raw)
        return p,expit(_logit(p)+change)
    trials=[]
    for gain in (.25,.5,.75,1.):
        p,c=probabilities(validation,gain);score=_metrics([r['y'] for r in validation],c)
        trials.append((score['log_loss'],score['brier'],-score['accuracy'],gain))
    gain=min(trials)[-1]
    comparisons={}
    for name,part in [('validation',validation),('test',test)]:
        p,c=probabilities(part,gain);y=[r['y'] for r in part]
        comparisons[name]={'original':_metrics(y,p),'connected':_metrics(y,c)}
    passed=all(c['connected']['accuracy']>=c['original']['accuracy'] and c['connected']['log_loss']<c['original']['log_loss']
               and c['connected']['brier']<=c['original']['brier'] for c in comparisons.values())
    # Refit the base on all observed history for current inference. Correction
    # selection stays frozen to the earlier training/validation partitions.
    live_base=_base(rows);scaler=live_base.named_steps['standardscaler'];lr=live_base.named_steps['logisticregression']
    return {'architecture':ARCHITECTURE,'model_id':'ncaaf_connected_'+identity,
        'base':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0])},
        'brain':brain,'trees':exported,'learning_rate':.025,'gain':gain,
        'checks_passed':passed,**comparisons,'residual_training_games':len(residual),
        'partition_games':{'train':len(train),'validation':len(validation),'test':len(test)}}
