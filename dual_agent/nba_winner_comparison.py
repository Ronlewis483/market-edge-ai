"""Pregame NBA winner challenger and frozen paired results. No paid API calls."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
from urllib.parse import quote
import threading
import numpy as np
import pandas as pd

LOCK = threading.RLock()
ROOT = Path('nba_player_history_cache')
LEDGER = ROOT/'winner_comparison_v1.json'


@lru_cache(maxsize=1)
def load_model():
    with np.load(Path(__file__).with_name('nba_winner_neural_weights.npz'), allow_pickle=False) as source:
        data = {k:source[k].copy() for k in source.files}
    data['meta'] = json.loads(str(data.pop('metadata').item()))
    if data['meta'].get('schema') != 1:
        raise ValueError('Unsupported NBA winner feature schema')
    widths=[len(data['meta']['features']),64,32,16,1]
    for i in range(4):
        if data['w'+str(i)].shape!=(widths[i],widths[i+1]) or data['b'+str(i)].shape!=(widths[i+1],):
            raise ValueError('Invalid NBA winner weight dimensions')
    if (data['scale']<=0).any() or any(not np.isfinite(v).all() for k,v in data.items() if k!='meta'):
        raise ValueError('Invalid NBA winner numerical weights')
    return data


def predict(features, start):
    model = load_model(); meta = model['meta']
    if pd.Timestamp(start).date() <= pd.Timestamp(meta['calibration_end']).date():
        raise ValueError('Matchup predates the saved learning model cutoff')
    x = features[meta['features']].to_numpy(dtype=float)
    if x.shape != (1,len(meta['features'])) or not np.isfinite(x).all():
        raise ValueError('Incomplete NBA winner features')
    x = (x-model['mean'])/model['scale']
    for i in range(4):
        x = x @ model['w'+str(i)]+model['b'+str(i)]
        if i<3: x = np.maximum(0.,x)
    raw = float(np.clip(x.item(),-30,30))
    logit = float(model['cal_coef'].item())*raw+float(model['cal_intercept'].item())
    p = float(1/(1+np.exp(-np.clip(logit,-30,30))))
    return {'learning_home_probability':p,'learning_confidence':max(p,1-p),
            'learning_model_id':meta['model_id']}


def read_ledger():
    try:
        data = json.loads(LEDGER.read_text())
        if data.get('schema') != 1 or not isinstance(data.get('records'),dict):
            raise ValueError('Unexpected comparison ledger schema')
        return data
    except FileNotFoundError:
        return {'schema':1,'records':{}}


def persist_forecasts(ledger):
    """One immutable object per frozen forecast; reuse successful local upload marks."""
    import requests
    def setting(name,default=None):
        value=os.environ.get(name)
        if value: return value
        try:
            import streamlit as st
            return st.secrets.get(name,default)
        except Exception: return default
    url=setting('SUPABASE_URL');key=setting('SUPABASE_KEY')
    if not url or not key: raise RuntimeError('Frozen forecasts saved locally; Supabase settings missing for durable snapshot joins')
    bucket=setting('NBA_SNAPSHOT_BUCKET','market-edge-data')
    for identity,row in ledger['records'].items():
        if row.get('durable_forecast_saved'): continue
        # Outcomes are deliberately excluded from the immutable forecast object.
        fields=('event_id','start_time','captured_at','home_team','away_team','home_code','away_code',
                'current_home_probability','learning_home_probability','current_model_version','learning_model_id')
        payload={k:row[k] for k in fields}
        path='nba/pregame_v1/forecasts/'+identity+'.json'
        response=requests.post(url.rstrip('/')+'/storage/v1/object/'+quote(bucket,safe='')+'/'+path,
            headers={'apikey':key,'Authorization':'Bearer '+key,'Content-Type':'application/json','x-upsert':'false'},
            data=json.dumps(payload,allow_nan=False),timeout=(3,8))
        if response.status_code not in (200,201):
            # Retry after a server restart is safe only when bytes represent the same forecast.
            if response.status_code in (400,409):
                old=requests.get(url.rstrip('/')+'/storage/v1/object/authenticated/'+quote(bucket,safe='')+'/'+path,
                    headers={'apikey':key,'Authorization':'Bearer '+key},timeout=(3,8))
                if old.status_code==200:
                    original=old.json()
                    identity_fields=('event_id','start_time','home_team','away_team','current_model_version','learning_model_id')
                    if (all(original.get(k)==payload[k] for k in identity_fields)
                        and pd.Timestamp(original['captured_at'])<pd.Timestamp(original['start_time'])
                        and all(0<float(original[k])<1 for k in ('current_home_probability','learning_home_probability'))):
                        # Restore the original pair after a Streamlit disk reset.
                        row.update({k:original[k] for k in fields})
                        if row.get('status')=='Final':
                            row['current_correct']=bool((row['current_home_probability']>=.5)==row['home_win'])
                            row['learning_correct']=bool((row['learning_home_probability']>=.5)==row['home_win'])
                        row['durable_forecast_saved']=True; continue
            raise RuntimeError('Frozen forecast persistence failed (HTTP '+str(response.status_code)+'); local forecasts retained')
        row['durable_forecast_saved']=True


def freeze_and_grade(predictions, historical_games, now=None):
    """First paired pregame forecast is immutable; scores grade exact team/date matches."""
    from dual_agent.nba_research import normalize_nba_team_name
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    with LOCK:
        ledger = read_ledger(); records = ledger['records']
        for row in predictions.to_dict('records'):
            start = pd.to_datetime(row['commence_time'],utc=True,errors='coerce')
            if pd.isna(start) or not now < start or not row.get('learning_model_id'):
                continue
            cp,lp = float(row['home_win_probability']),float(row['learning_home_probability'])
            if not np.isfinite([cp,lp]).all() or not (0<cp<1 and 0<lp<1): continue
            identity = '|'.join([str(row['event_id']),str(row.get('probability_version',3)),row['learning_model_id']])
            key = hashlib.sha256(identity.encode()).hexdigest()
            if key in records: continue
            records[key] = {'event_id':str(row['event_id']),'start_time':start.isoformat(),
                'captured_at':now.isoformat(),'home_team':row['home_team'],'away_team':row['away_team'],
                'home_code':normalize_nba_team_name(row['home_team']),
                'away_code':normalize_nba_team_name(row['away_team']),
                'current_home_probability':cp,'learning_home_probability':lp,
                'current_model_version':row.get('probability_version',3),
                'learning_model_id':row['learning_model_id'],'status':'Pending'}
        finals = {}
        for row in historical_games.to_dict('records'):
            date = pd.to_datetime(row.get('game_date'),utc=True,errors='coerce')
            if pd.isna(date): continue
            try: hp,ap = float(row['home_points']),float(row['away_points'])
            except (KeyError,TypeError,ValueError): continue
            if not np.isfinite([hp,ap]).all() or min(hp,ap)<=0 or hp==ap: continue
            identity = (str(row['home_team']),str(row['away_team']),date.date())
            finals.setdefault(identity, {})[str(row['game_id'])]=(hp,ap)
        for row in records.values():
            if row['status'] != 'Pending': continue
            start = pd.Timestamp(row['start_time'])
            if start>=now: continue
            # Provider game dates are local schedule dates; never guess adjacent days.
            date = start.tz_convert('America/Chicago').date()
            matched = finals.get((row['home_code'],row['away_code'],date),{})
            if len(matched)!=1: continue
            hp,ap = next(iter(matched.values())); outcome = int(hp>ap)
            row.update(status='Final',home_score=hp,away_score=ap,home_win=outcome,
                settled_at=now.isoformat(),current_correct=bool((row['current_home_probability']>=.5)==outcome),
                learning_correct=bool((row['learning_home_probability']>=.5)==outcome))
        # Save locally first, so a network failure cannot erase the frozen pair.
        ROOT.mkdir(parents=True,exist_ok=True)
        temporary = LEDGER.with_suffix('.tmp')
        temporary.write_text(json.dumps(ledger,allow_nan=False)); temporary.replace(LEDGER)
        try:
            persist_forecasts(ledger)
            ledger.pop('persistence_issue',None)
        except Exception as exc:
            ledger['persistence_issue']=str(exc) if type(exc) is RuntimeError else type(exc).__name__
        temporary.write_text(json.dumps(ledger,allow_nan=False)); temporary.replace(LEDGER)
        return ledger


def render_comparison(game):
    import streamlit as st
    if not game.get('learning_model_id'):
        st.caption('Learning-model comparison unavailable: '+str(game.get('learning_model_issue','generate a new NBA forecast')))
        return
    saved=None
    try:
        saved=next((r for r in read_ledger()['records'].values() if str(r['event_id'])==str(game.get('event_id'))
                    and r['learning_model_id']==game['learning_model_id']),None)
    except (ValueError,OSError): pass
    current=float(saved['current_home_probability']) if saved else float(game['home_win_probability'])
    learning=float(saved['learning_home_probability']) if saved else float(game['learning_home_probability'])
    st.caption('Frozen pregame comparison · captured '+saved['captured_at'] if saved else 'Preview only · this pair has not been saved for grading')
    st.caption('Learning model is experimental; the main recommendation still uses the current model')
    for col,label,p in zip(st.columns(2),['Current model','Learning model'],
                           [current,learning]):
        with col:
            st.markdown('**'+label+'**')
            st.write(game['home_team'] if p>=.5 else game['away_team'])
            st.write(f'Estimated win chance: {max(p,1-p):.1%}')
    st.caption('Team-form inputs only. Neither comparison includes verified current injuries or starting lineups.')


def render_results():
    import streamlit as st
    with st.expander('NBA winner models · saved predictions and results',expanded=False):
        try: records = list(read_ledger()['records'].values())
        except (ValueError,OSError):
            st.warning('Saved comparison results could not be read; they have not been reset.'); return
        final = [r for r in records if r['status']=='Final']
        st.caption(f'{len(records)} paired pregame forecasts saved · {len(final)} graded · {len(records)-len(final)} pending. Grading runs when NBA winner predictions refresh.')
        if final:
            rows=[]
            for label,key,correct in [('Current model','current_home_probability','current_correct'),
                                      ('Learning model','learning_home_probability','learning_correct')]:
                p=np.array([r[key] for r in final]); y=np.array([r['home_win'] for r in final])
                rows.append({'Model':label,'Games':len(final),'Wins':sum(r[correct] for r in final),
                    'Accuracy':float(np.mean((p>=.5)==y)), 'Brier score (lower is better)':float(np.mean((p-y)**2))})
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.caption('Same graded games for both models. Small samples do not establish an improvement.')
            reliability=[]
            for label,key,correct in [('Current model','current_home_probability','current_correct'),
                                      ('Learning model','learning_home_probability','learning_correct')]:
                for low,high in [(0.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.01)]:
                    bucket=[r for r in final if low<=max(r[key],1-r[key])<high]
                    if bucket:
                        reliability.append({'Model':label,'Confidence range':f'{low:.0%}–{min(high,1):.0%}',
                            'Games':len(bucket),'Average predicted chance':float(np.mean([max(r[key],1-r[key]) for r in bucket])),
                            'Actual win rate':float(np.mean([r[correct] for r in bucket]))})
            st.dataframe(pd.DataFrame(reliability),hide_index=True,width='stretch')
        if records:
            display=[]
            for r in sorted(records,key=lambda x:x['start_time'],reverse=True):
                display.append({'Game':r['away_team']+' @ '+r['home_team'],'Start':r['start_time'],
                    'Frozen at':r['captured_at'],'Current home chance':r['current_home_probability'],
                    'Learning home chance':r['learning_home_probability'],'Status':r['status'],
                    'Current correct':r.get('current_correct'),'Learning correct':r.get('learning_correct')})
            st.dataframe(pd.DataFrame(display),hide_index=True,width='stretch')
        issue=read_ledger().get('persistence_issue')
        if issue: st.warning(issue)
        st.caption('Frozen forecasts are copied to Supabase for scheduled snapshot joins and grading. This table uses the app’s local comparison ledger.')
        if records:
            st.download_button('Download frozen comparison results',json.dumps({'schema':1,'records':records},indent=2),
                               'nba_winner_comparison_results.json','application/json')
