"""Independent NFL/MLB connected winner learners and immutable pregame records.

Original probability is a fixed log-odds offset. Sport-specific context enters a
bounded neural correction. Never import NBA weights, props weights or outcomes
as live inputs. Supabase only on the prediction path; free score feeds in job.
"""
import gzip
import hashlib
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

ROOT='sports/connected_winners_v1/'
_CACHE={};_LOCK=threading.RLock();_SAVED=set()


def setting(name,default=None):
    value=os.environ.get(name)
    if value is not None:return value
    try:
        import streamlit as st
        return st.secrets.get(name,default)
    except Exception:return default


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def number(value):
    if isinstance(value,(bool,str)):
        try:value=float(value) if not isinstance(value,bool) else None
        except ValueError:return None
    try:
        v=float(value);return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None


def stamp(value):
    t=pd.Timestamp(value)
    if pd.isna(t):raise ValueError('Missing timestamp')
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def clean_team(value,sport):
    text=re.sub('[^a-z0-9]','',str(value).lower())
    if sport=='NFL':
        codes='ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS'.split()
        names=['Arizona Cardinals','Atlanta Falcons','Baltimore Ravens','Buffalo Bills','Carolina Panthers','Chicago Bears','Cincinnati Bengals','Cleveland Browns','Dallas Cowboys','Denver Broncos','Detroit Lions','Green Bay Packers','Houston Texans','Indianapolis Colts','Jacksonville Jaguars','Kansas City Chiefs','Los Angeles Rams','Los Angeles Chargers','Las Vegas Raiders','Miami Dolphins','Minnesota Vikings','New England Patriots','New Orleans Saints','New York Giants','New York Jets','Philadelphia Eagles','Pittsburgh Steelers','Seattle Seahawks','San Francisco 49ers','Tampa Bay Buccaneers','Tennessee Titans','Washington Commanders']
        aliases={'lar':'la','wsh':'was','jac':'jax','oak':'lv','sd':'lac','stl':'la'}
        for c,n in zip(codes,names):
            if text==re.sub('[^a-z0-9]','',n.lower()):return c.lower()
        return aliases.get(text,text)
    return text


class Store:
    def __init__(self):
        self.url=setting('SUPABASE_URL');self.key=setting('SUPABASE_KEY')
        self.bucket=setting('CONNECTED_WINNER_BUCKET',setting('NBA_SNAPSHOT_BUCKET','market-edge-data'))
        if not self.url or not self.key:raise RuntimeError('Connected winner storage settings missing')
        self.base=self.url.rstrip('/')+'/storage/v1/object/'
        self.http=requests.Session()
    def headers(self):return {'apikey':self.key,'Authorization':'Bearer '+self.key}
    def read(self,path):
        r=self.http.get(self.base+'authenticated/'+quote(self.bucket,safe='')+'/'+quote(path,safe='/'),headers=self.headers(),timeout=(3,8))
        if r.status_code!=200:
            try:code=str(r.json().get('code','')).lower();message=str(r.json().get('message','')).lower()
            except Exception:code=message=''
            if r.status_code==404 or r.status_code==400 and (code in ('nosuchkey','not_found') or 'object not found' in message):return None
            raise RuntimeError('Connected storage read failed (HTTP '+str(r.status_code)+')')
        return json.loads(gzip.decompress(r.content))
    def write(self,path,value,replace=False):
        payload=gzip.compress(json.dumps(value,allow_nan=False,separators=(',',':')).encode())
        r=self.http.post(self.base+quote(self.bucket,safe='')+'/'+quote(path,safe='/'),headers=dict(self.headers(),**{'Content-Type':'application/gzip','x-upsert':str(replace).lower()}),data=payload,timeout=(3,8))
        if r.status_code not in (200,201):raise RuntimeError('Connected storage write failed (HTTP '+str(r.status_code)+')')
    def list(self,prefix):
        names=[];offset=0
        while True:
            r=self.http.post(self.base+'list/'+quote(self.bucket,safe=''),headers=self.headers(),json={'prefix':prefix,'offset':offset,'limit':100,'sortBy':{'column':'name','order':'asc'}},timeout=(3,8))
            if r.status_code!=200:raise RuntimeError('Connected storage listing failed')
            batch=r.json()
            if not isinstance(batch,list):raise ValueError('Invalid storage listing')
            names += [x['name'] for x in batch if str(x.get('name','')).endswith('.json.gz')]
            if len(batch)<100:return names
            offset+=len(batch)


def load_model(sport,base_id):
    cache_key=(sport,base_id)
    with _LOCK:
        cached=_CACHE.get(cache_key)
        if cached and time.monotonic()-cached[0]<(600 if cached[1] else 60):return cached[1]
    artifact=Store().read(ROOT+sport.lower()+'/active.json.gz')
    if artifact and (artifact.get('sport')!=sport or artifact.get('base_id')!=base_id or artifact.get('schema')!=1):artifact=None
    with _LOCK:_CACHE[cache_key]=(time.monotonic(),artifact)
    return artifact


def probability(original,correction):
    p=np.clip(np.asarray(original,dtype=float),1e-7,1-1e-7)
    return 1/(1+np.exp(-np.clip(np.log(p/(1-p))+np.asarray(correction),-35,35)))


def encode(context,columns,median,mean,scale,original):
    raw=np.array([number(context.get(k)) if number(context.get(k)) is not None else np.nan for k in columns])
    missing=~np.isfinite(raw)
    values=(np.where(missing,median,raw)-mean)/scale
    return np.r_[np.log(original/(1-original)),values,missing.astype(float)]


def forward(x,params):
    h=np.tanh(x@params['w1']+params['b1'])
    return .7*np.tanh(h@params['w2']+params['b2'])


def correct(sport,base_id,original,context,at=None,start_time=None):
    """No fitting or paid API requests; missing artifacts retain original pick."""
    original=float(original)
    if not 0<original<1:raise ValueError('Original probability invalid')
    details={'sport':sport,'base_id':base_id,'original_probability':original,
             'connected_probability':original,'context':{str(k):number(v) for k,v in context.items()},
             'applied':False,'model_id':'original:'+base_id,'status':'Collecting sport-specific learning history'}
    now=stamp(at or datetime.now(timezone.utc))
    if start_time is not None and stamp(start_time)<=now:
        details['status']='Original retained; connected correction is pregame only';return details
    try:artifact=load_model(sport,base_id)
    except Exception as exc:
        details['status']='Original retained; learning storage unavailable ('+type(exc).__name__+')';return details
    if artifact is None:return details
    try:
        if stamp(artifact['trained_at'])>now:return details
        if sport == 'NFL' and artifact.get('kind') == 'nfl_offset_tree_v2':
            from .nfl_connected_engine import connected_delta
            delta=connected_delta(artifact,original,context)
        else:
            x=encode(context,artifact['columns'],np.array(artifact['median']),np.array(artifact['mean']),np.array(artifact['scale']),original)
            params={k:np.asarray(artifact[k]) for k in ('w1','b1','w2','b2')}
            if not all(np.isfinite(v).all() for v in params.values()):raise ValueError('Connected parameters invalid')
            delta=float(forward(x,params))
        p=float(probability(original,delta))
        if not 0<p<1:raise ValueError('Connected result invalid')
        details.update(connected_probability=p,raw_correction_logit=delta,applied=True,model_id=artifact['model_id'],status='Sport-specific connected correction applied')
    except Exception as exc:
        details['status']='Original retained; correction unavailable ('+type(exc).__name__+')'
    return details


def freeze(sport,rows):
    """One immutable slate upload; repeated identical slates keep the first copy."""
    now=datetime.now(timezone.utc).isoformat();clean=[]
    for r in rows:
        if stamp(r['start_time'])<=stamp(now):continue
        result=r['connected']
        clean.append(dict(event_id=str(r['event_id']),start_time=stamp(r['start_time']).isoformat(),
             captured_at=now,home_team=r['home_team'],away_team=r['away_team'],**result))
    if not clean:return None
    clean.sort(key=lambda r:(r['event_id'],r['base_id']))
    identity=digest([(r['event_id'],r['start_time'],r['base_id'],r['model_id']) for r in clean])
    path=ROOT+sport.lower()+'/forecasts/'+identity+'.json.gz'
    with _LOCK:
        if path in _SAVED:return None
    try:
        store=Store()
        if store.read(path) is None:store.write(path,{'schema':1,'sport':sport,'rows':clean})
        with _LOCK:_SAVED.add(path)
        return None
    except Exception as exc:return 'Learning forecast save failed ('+type(exc).__name__+')'


def nfl_context(features,context):
    output={'team_'+k:number(features.get(k)) for k in ('win_pct_diff','avg_point_diff_diff','recent_5_win_pct_diff','recent_5_point_diff_diff','rest_diff')}
    context=context or {};fetched=context.get('fetched_at')
    fresh=False
    try:fresh=pd.Timedelta(0)<=pd.Timestamp.now(tz='UTC')-stamp(fetched)<=pd.Timedelta(hours=6)
    except Exception:pass
    for side in ('home','away'):
        team=context.get(side) or {};qb=team.get('qb') or {};status=str(qb.get('injury_status','')).lower()
        verified=fresh and qb.get('report_updated_at') is not None
        try:verified=verified and pd.Timedelta(0)<=pd.Timestamp.now(tz='UTC')-stamp(qb['report_updated_at'])<=pd.Timedelta(hours=72)
        except Exception:verified=False
        output[side+'_qb_out']=int(status in ('out','inactive','suspended')) if verified else None
        output[side+'_qb_uncertain']=int(status in ('doubtful','questionable','gtd')) if verified else None
        output[side+'_reported_out']=sum(str(r.get('status','')).lower()=='out' for r in team.get('injuries',[])) if fresh and team.get('injuries') else None
    weather=context.get('weather') or {}
    outdoor=str(weather.get('roof_type','')).strip().lower() in ('outdoors','outdoor','open')
    for name in ('temperature_f','wind_mph','rain_chance'):
        output['weather_'+name]=number(weather.get(name)) if fresh and outdoor else None
    return output


def metrics(y,p):
    from sklearn.metrics import accuracy_score,brier_score_loss,log_loss
    return {'games':len(y),'accuracy':float(accuracy_score(y,np.asarray(p)>=.5)),
            'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1]))}


def partitions(rows):
    rows=sorted(rows,key=lambda r:(r['start_time'],r['event_id']))
    dates=sorted({r['start_time'][:10] for r in rows});best=None
    for i,a in enumerate(dates[1:-1],1):
        n1=sum(r['start_time'][:10]<a for r in rows)
        for b in dates[i+1:]:
            n2=sum(a<=r['start_time'][:10]<b for r in rows);n3=len(rows)-n1-n2
            if min(n1-80,n2-25,n3-25)<0:continue
            rank=((n1-.6*len(rows))**2+(n2-.2*len(rows))**2+(n3-.2*len(rows))**2,a,b)
            if best is None or rank<best:best=rank
    if best is None:return [[],[],[]]
    _,a,b=best;groups=[[r for r in rows if r['start_time'][:10]<a],[r for r in rows if a<=r['start_time'][:10]<b],[r for r in rows if r['start_time'][:10]>=b]]
    for i in (0,1):
        boundary=min(stamp(r['captured_at']) for g in groups[i+1:] for r in g)
        groups[i]=[r for r in groups[i] if stamp(r['label_observed_at'])<boundary]
    return groups


def train(sport,base_id,rows,store):
    """Fit a small offset neural correction on frozen forecasts of this base only."""
    if sport == 'NFL':
        from .nfl_connected_engine import train_connected
        return train_connected(sport,base_id,rows,store)
    from scipy.optimize import minimize
    rows=[r for r in rows if r['base_id']==base_id]
    groups=partitions(rows)
    report={'sport':sport,'base_id':base_id,'status':'waiting_for_graded_forecasts','graded_games':len(rows),
            'partition_games':dict(zip(('train','validation','test'),map(len,groups))),
            'minimum_games':{'train':80,'validation':25,'test':25},'trained':False,'activated':False,'paid_sports_api_requests':0}
    if any(len(g)<n or min(sum(r['outcome']==v for r in g) for v in (0,1))<5 for g,n in zip(groups,(80,25,25))):return report
    prior=store.read(ROOT+sport.lower()+'/training_report.json.gz')
    if prior and prior.get('base_id')==base_id and prior.get('trained') and len(rows)<prior.get('graded_games',0)+25:
        return dict(prior,status='waiting_for_25_new_results',available_graded_games=len(rows))
    # Never activate a context field that has no observed training support.
    names=sorted({k for r in groups[0] for k in r['context']})
    columns=[k for k in names if sum(number(r['context'].get(k)) is not None for r in groups[0])>=20]
    raw=np.array([[np.nan if number(r['context'].get(k)) is None else number(r['context'][k]) for k in columns] for r in groups[0]],dtype=float)
    median=np.nanmedian(raw,axis=0);filled=np.where(np.isfinite(raw),raw,median)
    mean=filled.mean(axis=0);scale=filled.std(axis=0);scale=np.where(scale<1e-8,1.,scale)
    def arrays(g):
        x=np.array([encode(r['context'],columns,median,mean,scale,r['original_probability']) for r in g])
        return x,np.array([r['outcome'] for r in g]),np.array([r['original_probability'] for r in g])
    x,y,p=arrays(groups[0]);d=x.shape[1];h=8
    shapes=[(d,h),(h,),(h,),()];sizes=[d*h,h,h,1];edges=np.cumsum([0]+sizes)
    def unpack(theta):return {k:theta[edges[i]:edges[i+1]].reshape(shapes[i]) for i,k in enumerate(('w1','b1','w2','b2'))}
    initial=np.r_[np.random.default_rng(42).normal(0,.05,d*h),np.zeros(2*h+1)]
    fitted=[]
    for penalty in (1.,10.,100.):
        def objective(theta):
            v=unpack(theta);hidden=np.tanh(x@v['w1']+v['b1']);t=np.tanh(hidden@v['w2']+v['b2'])
            q=probability(p,.7*t);loss=-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))+penalty*np.dot(theta,theta)/(2*len(y))
            dz=(q-y)*.7*(1-t*t)/len(y);dh=dz[:,None]*v['w2']*(1-hidden*hidden)
            grad=np.r_[(x.T@dh).ravel(),dh.sum(axis=0),(hidden.T@dz).ravel(),dz.sum()]+penalty*theta/len(y)
            return loss,grad
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':300})
        if not result.success or not np.isfinite(result.x).all():continue
        params=unpack(result.x);vx,vy,vp=arrays(groups[1]);q=probability(vp,forward(vx,params));m=metrics(vy,q)
        fitted.append(((-m['accuracy'],m['brier'],m['log_loss']),penalty,params))
    if not fitted:return dict(report,status='numerical_fit_failed')
    _,penalty,params=min(fitted,key=lambda v:v[0]);passed=True
    for label,g in zip(('validation','test'),groups[1:]):
        tx,ty,tp=arrays(g);q=probability(tp,forward(tx,params));old,new=metrics(ty,tp),metrics(ty,q)
        report[label]={'original':old,'connected':new}
        passed=passed and new['accuracy']>old['accuracy'] and new['brier']<=old['brier'] and new['log_loss']<=old['log_loss']
    model={'schema':1,'sport':sport,'base_id':base_id,'trained_at':datetime.now(timezone.utc).isoformat(),
           'columns':columns,'median':median.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
           **{k:v.tolist() for k,v in params.items()},'checks_passed':bool(passed)}
    model['model_id']=sport.lower()+'_connected_'+digest(model)[:16]
    report.update(trained=True,status='candidate_saved',activated=bool(passed),penalty=penalty,model_id=model['model_id'],checks_passed=bool(passed))
    store.write(ROOT+sport.lower()+'/candidates/'+model['model_id']+'.json.gz',model)
    if passed:store.write(ROOT+sport.lower()+'/active.json.gz',model,replace=True)
    return report


def fetch_finals(sport,rows,budget):
    """Free final-score lookups only, capped per scheduled job."""
    finals=[];now=pd.Timestamp.now(tz='UTC')
    pending=[r for r in rows if stamp(r['start_time'])<now]
    if sport=='MLB':
        ids=sorted({str(r['event_id']) for r in pending if str(r['event_id']).isdigit()})
        for index in range(0,len(ids),100):
            if budget['remaining']<=0:break
            budget['remaining']-=1
            response=requests.get('https://statsapi.mlb.com/api/v1/schedule',params={'sportId':1,'gamePks':','.join(ids[index:index+100])},timeout=(5,20))
            response.raise_for_status()
            for date in response.json().get('dates',[]):
                for g in date.get('games',[]):
                    if g.get('status',{}).get('abstractGameState')!='Final':continue
                    teams=g.get('teams',{})
                    h,a=teams.get('home',{}),teams.get('away',{})
                    finals.append({'event_id':str(g['gamePk']),'start_time':g['gameDate'],'home_team':h.get('team',{}).get('name'),
                                   'away_team':a.get('team',{}).get('name'),'home_score':number(h.get('score')),'away_score':number(a.get('score'))})
    else:
        dates=sorted({stamp(r['start_time']).tz_convert('America/Chicago').strftime('%Y%m%d') for r in pending})
        for date in dates:
            if budget['remaining']<=0:break
            budget['remaining']-=1
            response=requests.get('https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',params={'dates':date,'limit':100},timeout=(5,20))
            response.raise_for_status()
            for event in response.json().get('events',[]):
                for game in event.get('competitions',[]):
                    if game.get('status',{}).get('type',{}).get('completed') is not True:continue
                    teams={r.get('homeAway'):r for r in game.get('competitors',[])}
                    if not {'home','away'}.issubset(teams):continue
                    h,a=teams['home'],teams['away']
                    finals.append({'event_id':str(event['id']),'start_time':game.get('date',event.get('date')),
                        'home_team':h.get('team',{}).get('abbreviation'),'away_team':a.get('team',{}).get('abbreviation'),
                        'home_score':number(h.get('score')),'away_score':number(a.get('score'))})
    return finals


def run(sport,store=None,budget=None):
    if sport not in ('NFL','MLB'):raise ValueError('Sport must be NFL or MLB')
    store=store or Store();budget=budget or {'remaining':12};prefix=ROOT+sport.lower()+'/'
    state=store.read(prefix+'checkpoint.json.gz') or {'schema':1,'slates':{},'graded':{}}
    for name in store.list(prefix+'forecasts'):
        if name not in state['slates']:
            payload=store.read(prefix+'forecasts/'+name)
            if not isinstance(payload,dict) or payload.get('sport')!=sport:raise ValueError('Invalid saved forecast slate')
            state['slates'][name]=payload['rows']
    earliest={}
    for rows in state['slates'].values():
        for r in rows:
            if r.get('sport')!=sport or not stamp(r['captured_at'])<stamp(r['start_time']):continue
            identity=r['base_id']+'|'+str(r['event_id'])+'|'+r['start_time']
            if identity not in earliest or stamp(r['captured_at'])<stamp(earliest[identity]['captured_at']):earliest[identity]=r
    pending=[r for k,r in earliest.items() if k not in state['graded']]
    finals=fetch_finals(sport,pending,budget)
    for identity,r in earliest.items():
        if identity in state['graded']:continue
        matches=[g for g in finals if (sport!='MLB' or g['event_id']==str(r['event_id']))
                 and clean_team(g['home_team'],sport)==clean_team(r['home_team'],sport)
                 and clean_team(g['away_team'],sport)==clean_team(r['away_team'],sport)
                 and abs((stamp(g['start_time'])-stamp(r['start_time'])).total_seconds())<=900]
        if len(matches)!=1:continue
        g=matches[0];h,a=g['home_score'],g['away_score']
        if h is None or a is None or min(h,a)<0 or stamp(g['start_time'])>=pd.Timestamp.now(tz='UTC'):continue
        state['graded'][identity]=dict(r,home_score=h,away_score=a,outcome=None if h==a else int(h>a),
              label_observed_at=datetime.now(timezone.utc).isoformat(),grading_status='tie_excluded' if h==a else 'Final')
    store.write(prefix+'checkpoint.json.gz',state,replace=True)
    valid=[r for r in state['graded'].values() if r.get('outcome') in (0,1)]
    reports=[]
    active_base=max(earliest.values(),key=lambda r:stamp(r['captured_at']))['base_id'] if earliest else None
    for base_id in ([active_base] if active_base else []):
        report=train(sport,base_id,valid,store);reports.append(report)
        # Independent base cohorts never silently activate a correction for another model.
        store.write(prefix+'training_report.json.gz',report,replace=True)
    summary={'sport':sport,'frozen_unique_games':len(earliest),'graded_games':len(valid),'pending':len(earliest)-len(state['graded']),
             'paid_sports_api_requests':0,'training':reports,'score_lookup_budget_remaining':budget['remaining']}
    store.write(prefix+'latest_report.json.gz',summary,replace=True)
    print(json.dumps(summary));return summary


if __name__=='__main__':
    import sys
    league=sys.argv[1].upper() if len(sys.argv)>1 else 'ALL'
    budget={'remaining':12}
    failures=[]
    for sport in (('NFL','MLB') if league=='ALL' else (league,)):
        try:run(sport,budget=budget)
        except Exception as exc:
            failures.append(sport);print(json.dumps({'sport':sport,'status':'learning_job_failed','error_type':type(exc).__name__}))
    if failures:raise SystemExit(1)
