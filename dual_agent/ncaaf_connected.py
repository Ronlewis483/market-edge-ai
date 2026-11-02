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
    import os
    if os.environ.get('NCAAF_ACTIONS_RUN') != 'true':
        from dual_agent.ncaaf_storage import runtime_bundle
        bundle=runtime_bundle();artifact=bundle.get('winner')
        if not artifact:raise ValueError('No saved connected NCAA model yet. Run the NCAAF Actions workflow in bootstrap mode.')
        if artifact.get('architecture')!=ARCHITECTURE:raise ValueError('Saved NCAA winner architecture mismatch')
        return _forecast(artifact,rows,future,unsupported)
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
    return _forecast(artifact,rows,future,unsupported)


def _forecast(artifact,rows,future,unsupported):
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


def _train_once(rows,identity,config=None,development_only=False,test_after=None):
    config=config or {'depth':2,'trees':80,'leaf':20,'half_life_days':None,'gain_cap':1.}
    days=sorted({r['start'].date() for r in rows});n=len(days)
    if n<10:raise ValueError('Insufficient distinct NCAA game dates.')
    val_day=days[int(.6*n)];test_day=days[int(.8*n)]
    def before_label(r,day):return r['label_time'].date()<day
    train=[r for r in rows if r['start'].date()<val_day and before_label(r,val_day)]
    validation=[r for r in rows if val_day<=r['start'].date()<test_day and before_label(r,test_day)]
    test=[r for r in rows if r['start'].date()>=test_day and (test_after is None or r['start']>test_after)]
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
    from dual_agent.experiment_controller import weights
    sample_weights=weights([{'start_time':r['start'].isoformat()} for r in residual],config['half_life_days'])
    tree=GradientBoostingClassifier(n_estimators=config['trees'],learning_rate=.025,max_depth=config['depth'],min_samples_leaf=config['leaf'],
        random_state=42,init=OriginalOffset()).fit(x,[r['y'] for r in residual],sample_weight=sample_weights)
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
    for gain in [v for v in (0.,.125,.25,.5,.75,1.) if v<=config['gain_cap']]:
        p,c=probabilities(validation,gain);score=_metrics([r['y'] for r in validation],c)
        trials.append((score['log_loss'],score['brier'],-score['accuracy'],gain))
    gain=min(trials)[-1]
    comparisons={};heldout=[]
    for name,part in ([('validation',validation)] if development_only else [('validation',validation),('test',test)]):
        p,c=probabilities(part,gain);y=[r['y'] for r in part]
        comparisons[name]={'original':_metrics(y,p),'connected':_metrics(y,c)}
        if name=='test':heldout=[{'id':r['id'],'date':r['start'].isoformat(),'y':int(r['y']),'probability':float(v)} for r,v in zip(part,c)]
    passed=all(c['connected']['accuracy']>=c['original']['accuracy'] and c['connected']['log_loss']<c['original']['log_loss']
               and c['connected']['brier']<=c['original']['brier'] for c in comparisons.values())
    # Keep the deployable base on the same training partition used to evaluate
    # the candidate; final-test labels never enter its fitted coefficients.
    live_base=_base(train);scaler=live_base.named_steps['standardscaler'];lr=live_base.named_steps['logisticregression']
    return {'architecture':ARCHITECTURE,'model_id':'ncaaf_connected_'+identity,
        'base':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0])},
        'trained_through':max(r['start'] for r in rows).isoformat(),'training_ids':[r['id'] for r in rows],
        'experiment_config':config,'_groups':[train,validation,test],
        'heldout':heldout,'brain':brain,'trees':exported,'learning_rate':.025,'gain':gain,
        'checks_passed':passed,**comparisons,'residual_training_games':len(residual),
        'partition_games':{'train':len(train),'validation':len(validation),'test':len(test)}}


def _train(rows,identity,experiment_store=None):
    """Validation-only search; final evaluation occurs once on the selected model."""
    from . import experiment_controller as controller
    from . import ncaaf_storage
    from .sports_connected_winner import digest,stamp
    store=experiment_store or ncaaf_storage.Store()
    cohort=ARCHITECTURE+'|'+digest(FEATURES if 'FEATURES' in globals() else ['college_team_context_v1'])[:16]
    def adapt(r):
        return dict(base_id=cohort,event_id=str(r['id']),start_time=r['start'].isoformat(),
            captured_at=r['start'].isoformat(),label_observed_at=r['label_time'].isoformat(),
            context=dict(enumerate(r['x'])),original_probability=None,outcome=int(r['y']))
    ordered=sorted(rows,key=lambda r:(r['start'],str(r['id'])))
    # Prepare splits without fitting or evaluating any candidate.
    days=sorted({r['start'].date() for r in ordered})
    if len(days)<10:raise ValueError('Insufficient NCAA dates')
    a,b=days[int(.6*len(days))],days[int(.8*len(days))]
    groups=[[r for r in ordered if r['start'].date()<a and r['label_time'].date()<a],
        [r for r in ordered if a<=r['start'].date()<b and r['label_time'].date()<b],
        [r for r in ordered if r['start'].date()>=b]]
    converted=[[adapt(r) for r in part] for part in groups]
    experiment,control=controller.prepare('NCAAF',cohort,[adapt(r) for r in ordered],converted,store)
    if experiment is None:return {'controller_wait':control}
    if len(experiment['groups'][2])<50:
        return {'controller_wait':dict(control,status='waiting_for_50_fresh_test_results',fresh_test_games=len(experiment['groups'][2]))}
    cutoff=stamp(experiment['state']['tested_through']).to_pydatetime() if experiment['state'].get('tested_through') else None
    trials=[];fitted=[]
    for config in controller.plan(experiment,'NCAAF'):
        model=_train_once(ordered,identity,config,True,cutoff)
        trial={'config':config,'validation':model['validation']['connected']}
        trials.append(trial);fitted.append((controller.rank(trial['validation']),config))
    _,chosen=min(fitted,key=lambda r:r[0])
    controller.claim_test(experiment)
    model=_train_once(ordered,identity,chosen,False,cutoff)
    native=model.pop('_groups')
    _,q=_predict(dict(model,base=_evaluation_base(native[0])),[r['x'] for r in native[1]])
    y=[r['y'] for r in native[1]];score=model['validation']['connected'];baseline=model['validation']['original']
    diagnosis=controller.diagnose(y,q,None,score,baseline)
    selected=next(t for t in trials if t['config']==chosen)
    model['model_id']='ncaaf_connected_'+digest([identity,chosen,experiment['run_id']])[:24]
    report={'model_id':model['model_id'],'checks_passed':model['checks_passed'],
        'validation':model['validation'],'test':model['test']}
    model['controller']=controller.finish(experiment,trials,selected,diagnosis,report)
    return model


def _evaluation_base(rows):
    base=_base(rows);scaler=base.named_steps['standardscaler'];lr=base.named_steps['logisticregression']
    return {'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
        'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0])}
