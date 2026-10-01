"""Elo, chronological win layer and immutable prospective recommendation records."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

WIN_COLUMNS = ['matchup_log_odds', 'pregame_elo_difference', 'projected_run_difference']

def attach_elo(games, schedules):
    """Process prior results with a 48h buffer; no current-game outcome in ratings.

    Fixed K=20, home advantage=35 points and offseason retention=2/3.
    Start at 1500 in first available season; these are not tuned on test results.
    """
    history=[]
    seen=set()
    for g in schedules:
        if g['gamePk'] in seen: continue
        seen.add(g['gamePk'])
        teams=g.get('teams', {})
        if (g.get('gameType')!='R' or g.get('resumeDate') or
            g.get('status',{}).get('abstractGameState')!='Final'): continue
        if any(teams.get(s,{}).get('score') is None for s in ['home','away']): continue
        if teams['home']['score']==teams['away']['score']: continue
        history.append((pd.to_datetime(g['gameDate'],utc=True)+pd.Timedelta(hours=48),g))
    history.sort(key=lambda pair:(pair[0],pair[1]['gamePk']))
    ratings={};season=None;index=0
    def advance(year):
        nonlocal season
        if season is not None and year>season:
            for key in ratings: ratings[key]=1500+(ratings[key]-1500)*(2/3)**(year-season)
        season=year if season is None else max(year,season)
    for target in sorted(games,key=lambda g:g['timecode']):
        cutoff=pd.to_datetime(target['timecode'],format='%Y%m%d_%H%M%S',utc=True)
        while index<len(history) and history[index][0]<cutoff:
            _,g=history[index];index+=1;advance(int(g['season']))
            h=g['teams']['home']['team']['id'];a=g['teams']['away']['team']['id']
            rh=ratings.get(h,1500.);ra=ratings.get(a,1500.)
            expected=1/(1+10**(-(rh-ra+35)/400))
            delta=20*(int(g['teams']['home']['score']>g['teams']['away']['score'])-expected)
            ratings[h]=rh+delta;ratings[a]=ra-delta
        advance(int(target['season']))
        h=target['recorded_identities']['home'].get('team_id')
        a=target['recorded_identities']['away'].get('team_id')
        if h is None or a is None:
            raise ValueError('Archived team IDs required for Elo; update historical archive module and rebuild rows')
        target['elo_home']=ratings.get(int(h),1500.)
        target['elo_away']=ratings.get(int(a),1500.)
        target['elo_difference']=target['elo_home']-target['elo_away']
    return {'K':20,'home_advantage_points':35,'offseason_retention':2/3,'result_delay_hours':48,'prior_results':len(history)}

def win_rows(games,rates,probabilities):
    p=np.clip(probabilities,1e-6,1-1e-6)
    return np.column_stack([np.log(p/(1-p)),[g['elo_difference'] for g in games],rates[:,0]-rates[:,1]])

def fit_win_layer(X,y,C):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=42))
    model.fit(X,y)
    return model

def chronological_win_layer(train24,dev25,test26,alpha,flags,fit_runs,predict_runs,final_rates,final_p):
    """Train on earlier-block forecasts, not fitted predictions of training games."""
    ordered=sorted(train24,key=lambda g:g['start_time'])
    split_date=ordered[int(len(ordered)*.6)]['start_time'][:10]
    early=[g for g in ordered if g['start_time'][:10]<split_date]
    late=[g for g in ordered if g['start_time'][:10]>=split_date]
    # Exclude labels within 48h of the first forecast in each block.
    def eligible(rows,forecast):
        cap=pd.to_datetime(min(g['timecode'] for g in forecast),format='%Y%m%d_%H%M%S',utc=True)-pd.Timedelta(hours=48)
        return [g for g in rows if pd.to_datetime(g['start_time'],utc=True)<cap]
    early=eligible(early,late)
    if len(early)<40 or len(late)<40: raise ValueError('Need at least 40 early and 40 later 2024 games for chronological win layer')
    m=fit_runs(early,alpha,*flags);r,p=predict_runs(m,late)
    X=win_rows(late,r,p);y=[g['home_win'] for g in late]
    m=fit_runs(eligible(train24,dev25),alpha,*flags);dr,dp=predict_runs(m,dev25)
    DX=win_rows(dev25,dr,dp);dy=[g['home_win'] for g in dev25]
    candidates=[]
    for C in [.01,.1,1.]:
        layer=fit_win_layer(X,y,C);pred=layer.predict_proba(DX)[:,1]
        candidates.append({'C':C,'Log loss':float(log_loss(dy,pred,labels=[0,1]))})
    selected=min(candidates,key=lambda row:row['Log loss'])['C']
    development=fit_win_layer(X,y,selected).predict_proba(DX)[:,1]
    layer=fit_win_layer(np.vstack([X,DX]),y+dy,selected)
    probabilities=layer.predict_proba(win_rows(test26,final_rates,final_p))[:,1]
    scaler,logistic=layer.steps[0][1],layer.steps[1][1]
    bundle={'columns':WIN_COLUMNS,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'weights':logistic.coef_[0].tolist(),'intercept':float(logistic.intercept_[0]),'C':selected}
    return probabilities,development,bundle,candidates

def select_recommendations(probabilities,outcomes):
    """Choose on development only; show research picks even below the target."""
    probabilities=np.asarray(probabilities,dtype=float)
    rows=[]
    for cutoff in [.55,.60,.65,.70,.75,.80]:
        mask=np.maximum(probabilities,1-probabilities)>=cutoff
        n=int(mask.sum())
        if n>=50:
            correct=int(((probabilities[mask]>=.5)==np.asarray(outcomes)[mask]).sum())
            rows.append({'threshold':cutoff,'games':n,'correct':correct,'accuracy':correct/n})
    meets=[r for r in rows if r['accuracy']>=.70]
    chosen=max(meets,key=lambda r:r['games']) if meets else (max(rows,key=lambda r:(r['accuracy'],r['games'])) if rows else {'threshold':.5})
    return chosen['threshold'],rows

def recommendation_metrics(probabilities,outcomes,threshold):
    mask=np.maximum(probabilities,1-probabilities)>=threshold if threshold is not None else np.zeros(len(probabilities),dtype=bool)
    n=int(mask.sum());correct=int(((probabilities[mask]>=.5)==np.asarray(outcomes)[mask]).sum())
    rate=correct/n if n else None
    interval=None
    if n:
        z=1.96;den=1+z*z/n;center=(rate+z*z/(2*n))/den
        half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
        interval=[max(0,center-half),min(1,center+half)]
    return mask,{'games':n,'correct':correct,'accuracy':rate,'coverage':n/len(probabilities) if len(probabilities) else 0,
                 'target':.70,'wilson_95_interval':interval,'target_demonstrated':bool(n>=100 and interval and interval[0]>=.70)}

def freeze_recommendation(game,home_probability,threshold,model_id,now=None):
    """Create prospective record only; historical rows cannot enter live ledger."""
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.to_datetime(now,utc=True)
    start=pd.to_datetime(game['start_time'],utc=True)
    capture=pd.to_datetime(game['timecode'],format='%Y%m%d_%H%M%S',utc=True)
    if not model_id or now>=start or capture>now or now-capture>pd.Timedelta(minutes=15):
        raise ValueError('Recommendation requires a model ID and a fresh pregame capture')
    p=float(home_probability)
    if not np.isfinite(p) or not 0<=p<=1: raise ValueError('Invalid probability')
    if threshold is None or not .5<=threshold<=1 or max(p,1-p)<threshold:
        raise ValueError('Game does not qualify under the frozen recommendation rule')
    record={'schema_version':1,'game_id':int(game['game_id']),'start_time':start.isoformat(),'recorded_at':now.isoformat(),
            'capture_time':capture.isoformat(),'model_id':str(model_id),'threshold':float(threshold),
            'home_team':game['home_team'],'away_team':game['away_team'],'home_probability':p,
            'pick':game['home_team'] if p>=.5 else game['away_team'],'picked_home':p>=.5,'status':'pending'}
    record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return record

def save_recommendation(record):
    """Immutable per-game Supabase upload. Duplicates fail; never overwrite picks."""
    from dual_agent import supabase_db as storage
    # Recheck clock at upload, verify content hash and pending state.
    payload=dict(record);digest=payload.pop('sha256',None)
    if hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=digest:
        raise ValueError('Recommendation content changed after freezing')
    if payload.get('status')!='pending' or pd.Timestamp.now(tz='UTC')>=pd.to_datetime(payload['start_time'],utc=True):
        raise ValueError('Only future pending recommendations may be saved')
    ready=storage.ensure_market_edge_storage_bucket()
    if not ready.get('success'): raise RuntimeError(ready.get('error','Storage unavailable'))
    path=f"mlb/matchup_recommendations/picks/{int(record['game_id'])}.json"
    storage.get_supabase_client().storage.from_(storage.MLB_STORAGE_BUCKET).upload(
        path,json.dumps(record,allow_nan=False).encode(),{'content-type':'application/json','upsert':'false'})
    return {'success':True,'archive_file':path}

def predict_win_bundle(games,rates,matchup_probabilities,bundle):
    """Use saved win-layer coefficients without retraining or outcome fields."""
    if bundle['columns']!=WIN_COLUMNS: raise ValueError('Unsupported win-layer columns')
    X=win_rows(games,np.asarray(rates),np.asarray(matchup_probabilities))
    z=((X-np.asarray(bundle['mean']))/np.asarray(bundle['scale']))@np.asarray(bundle['weights'])+bundle['intercept']
    return 1/(1+np.exp(-np.clip(z,-40,40)))

def settle_recommendation(game_id):
    """Fetch official final score; save grading separately from original pick."""
    import urllib.request
    from dual_agent import supabase_db as storage
    bucket=storage.get_supabase_client().storage.from_(storage.MLB_STORAGE_BUCKET)
    pick_path=f'mlb/matchup_recommendations/picks/{int(game_id)}.json'
    record=json.loads(bucket.download(pick_path))
    payload=dict(record);digest=payload.pop('sha256',None)
    if hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=digest:
        raise ValueError('Original pick integrity check failed')
    with urllib.request.urlopen(f'https://statsapi.mlb.com/api/v1/schedule?gamePk={int(game_id)}',timeout=25) as response:
        schedule=json.load(response)
    matches=[g for day in schedule.get('dates',[]) for g in day.get('games',[]) if g['gamePk']==int(game_id)]
    if len(matches)!=1: raise ValueError('Official game not found')
    game=matches[0]
    if game['status'].get('abstractGameState')!='Final': return {'status':'pending','game_id':int(game_id)}
    home=game['teams']['home']['score'];away=game['teams']['away']['score']
    if home==away: return {'status':'pending','game_id':int(game_id),'note':'No winner established'}
    grade={'game_id':int(game_id),'pick_sha256':digest,'graded_at':pd.Timestamp.now(tz='UTC').isoformat(),
           'home_runs':home,'away_runs':away,'correct':bool(record['picked_home']==(home>away)),
           'status':'final','source':'MLB Stats API'}
    path=f'mlb/matchup_recommendations/results/{int(game_id)}.json'
    bucket.upload(path,json.dumps(grade).encode(),{'content-type':'application/json','upsert':'false'})
    return grade


def confidence_diagnostics(probabilities,outcomes):
    """Report ranking quality without choosing a rule on evaluated outcomes."""
    p=np.asarray(probabilities);y=np.asarray(outcomes);confidence=np.maximum(p,1-p)
    correct=(p>=.5)==y
    bands=[]
    for low,high in [(.5,.55),(.55,.6),(.6,.65),(.65,.7),(.7,.8),(.8,1.00001)]:
        mask=(confidence>=low)&(confidence<high);n=int(mask.sum())
        bands.append({'Confidence band':f'{low:.0%}–{min(high,1):.0%}','Games':n,
                      'Correct':int(correct[mask].sum()),'Accuracy':float(correct[mask].mean()) if n else None,
                      'Mean confidence':float(confidence[mask].mean()) if n else None})
    order=np.argsort(-confidence,kind='stable');ranked=[]
    for count in [10,25,50,100]:
        if len(order)<count:continue
        chosen=order[:count]
        ranked.append({'Top ranked games':count,'Correct':int(correct[chosen].sum()),
                       'Accuracy':float(correct[chosen].mean()),'Mean confidence':float(confidence[chosen].mean())})
    return {'bands':bands,'ranked_groups':ranked,'note':'Diagnostic groups only. Never select a cutoff from these evaluation outcomes; top groups are across the whole evaluated period, not a daily quota.'}
