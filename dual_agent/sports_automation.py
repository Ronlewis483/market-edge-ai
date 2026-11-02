"""Scheduled NFL/MLB lifecycle. No Odds API calls; app equations stay unchanged.

cycle: collect current finals/context, freeze forecasts, grade and readiness-train.
backfill: add bounded history, reconstruct prior-only forecasts, train candidates.
All durable state is stored in the existing private Supabase bucket.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from dual_agent.sports_connected_winner import ROOT, Store, digest, clean_team


def stamp(value):
    at=pd.Timestamp(value)
    return at.tz_localize('UTC') if at.tzinfo is None else at.tz_convert('UTC')


def mlb_context(game):
    """Exactly the context names used by the MLB live pipeline."""
    from dual_agent.sports_connected_winner import number
    context={}
    for source in ('team_sides','player_sides','bullpen_sides','hand_sides','form_sides'):
        for side,values in (game.get(source) or {}).items():
            if isinstance(values,(list,tuple,np.ndarray)):
                context.update({source+'_'+side+'_'+str(i):number(v) for i,v in enumerate(values)})
    context.update({'environment_'+str(i):number(v) for i,v in enumerate(game.get('environment') or [])})
    context['elo_difference']=number(game.get('elo_difference'))
    # Archived lineup lists do not prove confirmed lineups or starters.
    return context


def collect_finals(sport,store,now):
    """Store today's and yesterday's explicit finals, even without a forecast."""
    import requests
    prefix=ROOT+sport.lower()+'/results/'
    total=0;fresh=[]
    for day in (now.date()-timedelta(days=1),now.date()):
        path=prefix+str(day)+'.json.gz'
        old=store.read(path) or {'games':{}}
        if sport=='NFL':
            from dual_agent.sports_connected_winner import fetch_finals
            # fetch_finals groups scoreboard requests by local event date.
            dummy=[{'start_time':pd.Timestamp(str(day)+'T12:00:00Z').isoformat()}]
            games=fetch_finals('NFL',dummy,{'remaining':1})
        else:
            response=requests.get('https://statsapi.mlb.com/api/v1/schedule',params={
                'sportId':1,'date':str(day)},timeout=(5,25));response.raise_for_status()
            games=[]
            for date in response.json().get('dates',[]):
                for game in date.get('games',[]):
                    if game.get('status',{}).get('abstractGameState')!='Final':continue
                    home=game['teams']['home'];away=game['teams']['away']
                    if home.get('score') is None or away.get('score') is None:continue
                    games.append(dict(event_id=str(game['gamePk']),start_time=game['gameDate'],
                        home_team=home['team']['name'],away_team=away['team']['name'],
                        home_score=home['score'],away_score=away['score']))
        for game in games:
            if stamp(game['start_time'])>=stamp(now):continue
            identity=game['event_id']
            previous=old['games'].get(identity)
            observed=previous.get('first_observed_at') if previous else now.isoformat()
            old['games'][identity]=dict(game,first_observed_at=observed,last_observed_at=now.isoformat(),provider_final=True)
            fresh.append(old['games'][identity]);total+=int(previous is None)
        store.write(path,old,replace=True)
    return {'new_finals':total,'games':fresh,'paid_sports_api_requests':0}


def nfl_forecasts(store,now,finals):
    from dual_agent.nfl_live_context import fresh_schedule, completed_history, TEAMS, team_code, attach_context
    from dual_agent.nfl_research import build_nfl_pregame_features, build_nfl_future_matchup_features
    from dual_agent.nfl_connected_engine import predict_original_cached, FEATURES
    from dual_agent.nfl_historical_learning import base_identity
    from dual_agent.sports_connected_winner import correct,freeze,nfl_context
    schedule,_=fresh_schedule();schedule=schedule.copy()
    # Explicit scoreboard finals can precede nflverse's archive update.
    for game in finals:
        matches=schedule[(schedule.home_team.map(team_code)==team_code(game['home_team'])) &
            (schedule.away_team.map(team_code)==team_code(game['away_team'])) &
            ((pd.to_datetime(schedule.start_time,utc=True)-stamp(game['start_time'])).abs()<=pd.Timedelta(minutes=15))]
        if len(matches)!=1:continue
        i=matches.index[0]
        schedule.loc[i,['home_score','away_score','result']]=[game['home_score'],game['away_score'],game['home_score']-game['away_score']]
    features=build_nfl_pregame_features(completed_history(schedule,now))
    starts=pd.to_datetime(schedule.start_time,utc=True)
    future=schedule[(starts>stamp(now))&(starts<=stamp(now)+pd.Timedelta(days=7)) & schedule.game_type.isin(['REG','WC','DIV','CON','SB','POST'])]
    # Missing kickoff times must never be fabricated into forecast identities.
    future=future[future.gametime.notna()]
    rows=[];base_id=base_identity()
    for game in future.sort_values('start_time').itertuples():
        home,away=TEAMS.get(team_code(game.home_team)),TEAMS.get(team_code(game.away_team))
        if not home or not away:continue
        x=build_nfl_future_matchup_features(features,home,away,game.start_time)
        original=predict_original_cached(features,x)
        rows.append(dict(home_team=home,away_team=away,commence_time=game.start_time,
            event_id=str(game.game_id),original_probability=original['home_win_probability'],
            features={key:float(x.iloc[0][key]) for key in FEATURES}))
    if not rows:return {'status':'no_upcoming_games','frozen_forecasts':0}
    frame=attach_context(pd.DataFrame(rows),pd.DataFrame())
    frozen=[]
    for row in frame.to_dict('records'):
        context=nfl_context(row['features'],row.get('game_context') or {})
        forecast=correct('NFL',base_id,row['original_probability'],context,start_time=row['commence_time'])
        frozen.append(dict(event_id=row['event_id'],start_time=stamp(row['commence_time']).isoformat(),
            home_team=row['home_team'],away_team=row['away_team'],connected=forecast))
    snapshot={'schema':1,'captured_at':now.isoformat(),'rows':json.loads(frame.to_json(orient='records',date_format='iso'))}
    store.write(ROOT+'nfl/snapshots/'+digest(snapshot)+'.json.gz',snapshot)
    issue=freeze('NFL',frozen)
    if issue:raise RuntimeError(issue)
    return {'status':'forecasts_saved','frozen_forecasts':len(frozen),'base_id':base_id}


def mlb_forecasts(now):
    import requests
    # Offseason succeeds without requiring a model or launching player scans.
    response=requests.get('https://statsapi.mlb.com/api/v1/schedule',params={'sportId':1,
        'startDate':str(now.date()),'endDate':str((now+timedelta(days=7)).date())},timeout=(5,25))
    response.raise_for_status()
    upcoming=[g for day in response.json().get('dates',[]) for g in day.get('games',[])
        if g.get('status',{}).get('abstractGameState')=='Preview' and stamp(g['gameDate'])>stamp(now)]
    if not upcoming:return {'status':'offseason_or_no_upcoming_games','frozen_forecasts':0}
    from dual_agent.mlb_prediction_pipeline import run_mlb_prediction_pipeline
    result=run_mlb_prediction_pipeline(api_key=None,include_props=False)
    issues=[r.get('Connected learning issue') for r in result.get('all_predictions',[]) if r.get('Connected learning issue')]
    if issues:raise RuntimeError(issues[0])
    return {'status':result.get('message'),'frozen_forecasts':len(result.get('all_predictions',[])),
        'input_errors':result.get('errors',[]),'paid_sports_api_requests':0}


class HistoricalStore:
    def __init__(self,store,sport):self.store=store;self.prefix=ROOT+sport.lower()+'/';self.candidate=None
    def read(self,path):
        if path==self.prefix+'active.json.gz':return None
        return self.store.read(self.redirect(path))
    def redirect(self,path):
        if not path.startswith(self.prefix):raise ValueError('Cross-sport historical write refused')
        return self.prefix+'historical_v1/'+path[len(self.prefix):]
    def write(self,path,value,replace=False):
        if path==self.prefix+'active.json.gz':self.candidate=value;return
        self.store.write(self.redirect(path),value,replace=replace)


def mlb_historical(store,now):
    """Forward refit the existing Poisson/Elo equation on its saved pregame archive.

    No current full-history model is used to score its own training games.
    Hyperparameters come from the existing model, so this is exploratory replay.
    """
    from dual_agent.mlb_prediction_pipeline import load_active_model
    from dual_agent.mlb_matchup_model import fit_run_model,predict_matchups
    from dual_agent.mlb_matchup_elo import fit_win_layer,win_rows
    from dual_agent.sports_connected_winner import train
    model=load_active_model();base_id='mlb_matchup:'+digest(model)[:20]
    archive=model
    if not archive.get('retained_feature_rows'):
        # Probability-upgrade bundles reference the full archive they came from.
        from dual_agent.mlb_prediction_pipeline import _bucket
        bucket=_bucket()
        for entry in bucket.list('mlb/matchup_model',{'limit':1000,'sortBy':{'column':'name','order':'desc'}}):
            if not entry.get('name','').endswith('.json'):continue
            candidate=json.loads(bucket.download('mlb/matchup_model/'+entry['name']))
            if candidate.get('retained_feature_rows') and (not model.get('source_archive') or candidate.get('created_at')==model['source_archive']):
                archive=candidate;break
    games=sorted(archive.get('retained_feature_rows',[]),key=lambda g:(g['start_time'],str(g['game_id'])))
    prefix=ROOT+'mlb/historical_v1/'
    from dual_agent.experiment_controller import VERSION
    fingerprint=digest([VERSION,base_id,games])
    prior=store.read(prefix+'training_report.json.gz') or {}
    if prior.get('fingerprint')==fingerprint and prior.get('trained'):return dict(prior,status='unchanged_history_skipped',activated=False)
    if len(games)<330:return {'status':'waiting_for_archived_pregame_features','archived_games':len(games),
        'required':330,'action':'Build/save the existing MLB matchup model with historical feature rows first.'}
    if model.get('market',{}).get('selected_on_late_2025') in ('Market consensus','Matchup + market'):
        return {'status':'waiting_for_historical_market_snapshots','reason':'Selected live base needs archived market quotes; no synthetic market probabilities are inserted.'}
    from dual_agent.mlb_matchup_model import BULLPEN_FEATURES
    from dual_agent.mlb_handedness import HAND_FEATURES
    from dual_agent.mlb_matchup_context import FORM_FEATURES, ENVIRONMENT_FEATURES
    columns=model['model_bundle']['columns']
    flags=[bool(set(columns)&set(group)) for group in (BULLPEN_FEATURES,HAND_FEATURES,FORM_FEATURES+ENVIRONMENT_FEATURES)]
    from dual_agent.mlb_prediction_pipeline import CORE
    expected=CORE+(BULLPEN_FEATURES if flags[0] else [])+(HAND_FEATURES if flags[1] else [])+(FORM_FEATURES+ENVIRONMENT_FEATURES if flags[2] else [])
    if not set(columns)<=set(expected):raise ValueError('Unsupported historical base columns')
    alpha=float(model.get('selected_alpha',model.get('selected_run_configuration',{}).get('alpha',archive.get('selected_alpha'))))
    layer_bundle=(model.get('layer_bundles',{}).get(model['selected_layer']) if 'selected_layer' in model else model.get('win_layer_bundle') if model.get('win_layer_selected',True) else None)
    layer_columns=(layer_bundle or {}).get('columns',[])
    if layer_columns and layer_columns not in (['matchup_log_odds'],['matchup_log_odds','pregame_elo_difference','projected_run_difference']):
        return {'status':'waiting_for_historical_layer_inputs'}
    layer_C=.1 if 'selected_layer' in model else float((layer_bundle or {}).get('C',1))
    from dual_agent.mlb_matchup_model import matchup_rows
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import PoissonRegressor
    from scipy.stats import skellam
    indexes=[expected.index(c) for c in columns]
    def fit_base(prior):
        x,y=matchup_rows(prior,*flags)
        return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PoissonRegressor(alpha=alpha,max_iter=2000,tol=1e-7)).fit(x[:,indexes],y)
    def predict_base(base,batch):
        x,_=matchup_rows(batch,*flags);rates=base.predict(x[:,indexes]).reshape(-1,2)
        p=skellam.sf(0,rates[:,0],rates[:,1])+.5*skellam.pmf(0,rates[:,0],rates[:,1])
        return rates,np.clip(p,1e-6,1-1e-6)
    rows=[];fits=0
    days=sorted({g['start_time'][:10] for g in games})
    for day in days:
        batch=[g for g in games if g['start_time'][:10]==day]
        captured=min(pd.to_datetime(g['timecode'],format='%Y%m%d_%H%M%S',utc=True) for g in batch)
        prior_games=[g for g in games if stamp(g['start_time'])+pd.Timedelta(hours=48)<captured]
        if len(prior_games)<200:continue
        base=fit_base(prior_games);rates,p=predict_base(base,batch);fits+=1
        if layer_bundle:
            # Fit the extra winner layer using only forward forecasts before this batch.
            earlier=[r for r in rows if stamp(r['label_observed_at'])<captured]
            if len(earlier)<80 or len({r['outcome'] for r in earlier})<2:
                # Still save Poisson forward rows for this layer's later training.
                p_final=None
            else:
                layer=fit_win_layer(np.array([r['_win_x'][:len(layer_columns)] for r in earlier]),[r['outcome'] for r in earlier],layer_C)
                p_final=layer.predict_proba(win_rows(batch,rates,p)[:,:len(layer_columns)])[:,1]
        else:p_final=p
        for i,g in enumerate(batch):
            if g.get('home_win') not in (0,1):continue
            rows.append(dict(sport='MLB',base_id=base_id,event_id=str(g['game_id']),start_time=g['start_time'],
                captured_at=captured.isoformat(),label_observed_at=(stamp(g['start_time'])+pd.Timedelta(hours=48)).isoformat(),
                home_team=g['home_team'],away_team=g['away_team'],original_probability=float(p[i] if p_final is None else p_final[i]),
                context=mlb_context(g),outcome=int(g['home_win']),source='historical_reconstruction',
                _win_x=win_rows([g],rates[i:i+1],p[i:i+1])[0].tolist(),_eligible=p_final is not None))
    usable=[{k:v for k,v in r.items() if not k.startswith('_')} for r in rows if r['_eligible']]
    store.write(prefix+'reconstructed_rows.json.gz',{'schema':1,'rows':usable,'source':'historical_reconstruction'},replace=True)
    isolated=HistoricalStore(store,'MLB');report=train('MLB',base_id,usable,isolated)
    report.update(fingerprint=fingerprint,forward_base_fits=fits,historical_rows=len(usable),activated=False,
        limitations=['Retrospective pregame archive; current archive corrections and existing hyperparameter selection are not an untouched benchmark.',
            'Historical publication timing approximated at kickoff plus 48 hours; no invented injury or confirmed-lineup fields.'])
    candidate=isolated.candidate
    if candidate and not store.read(ROOT+'mlb/active.json.gz'):
        candidate=dict(candidate,source='historical_reconstruction',limitations=report['limitations'])
        store.write(ROOT+'mlb/candidates/'+candidate['model_id']+'.json.gz',candidate,replace=True)
        store.write(ROOT+'mlb/active.json.gz',candidate,replace=True);report['activated']=True
    store.write(prefix+'training_report.json.gz',report,replace=True)
    return report


def player_history(sport,store,now,train_models=False,backfill=False):
    """Separate bounded history collection and numeric prop regression training."""
    prefix=ROOT+sport.lower()+'/props/'
    from dual_agent.train_player_prop_regression import read_history,dataset,train
    from dual_agent.player_prop_regression import FEATURES
    if sport=='NFL':
        from dual_agent.nfl_player_history import load_player_history
        season=now.year if now.month>=8 else now.year-1
        pieces=[]
        for year in range(season-2,season+1):
            cache_path=prefix+'seasons/'+str(year)+'.json.gz'
            saved=store.read(cache_path)
            fresh=saved and (year<season or stamp(now)-stamp(saved['collected_at'])<pd.Timedelta(minutes=15))
            if fresh:part=pd.DataFrame(saved['rows'])
            else:
                part=load_player_history([year])
                store.write(cache_path,{'rows':json.loads(part.to_json(orient='records',date_format='iso')),'collected_at':now.isoformat()},replace=True)
            pieces.append(part)
        frame=pd.concat(pieces,ignore_index=True)
    else:
        from dual_agent import mlb_research as research,supabase_db as storage
        print('MLB player history: loading saved warehouse',flush=True)
        old=storage.load_mlb_player_history()
        # Recent finals first; missing historical boxes remain resumable.
        start=f'{now.year-2}-01-01' if backfill else str((now-timedelta(days=3)).date())
        print('MLB player history: checking completed games since '+start,flush=True)
        games=research.fetch_mlb_games(start,str(now.date()))
        if games.empty:
            return {'status':'no_completed_games_in_collection_window','saved_stat_rows':len(old),'training_performed':False}
        games=games.sort_values('start_time',ascending=False)
        print('MLB player history: collecting at most 25 missing boxes',flush=True)
        batch=research.collect_mlb_player_logs_batch(games,old,batch_size=25,max_workers=4)
        frame=batch['combined_logs']
        if not frame.empty:
            saved=storage.save_mlb_player_history(frame)
            if isinstance(saved,dict) and saved.get('success') is False:raise RuntimeError('MLB player warehouse save failed')
    path=Path('automation_cache')/(sport.lower()+'_player_history.csv');path.parent.mkdir(exist_ok=True)
    frame.to_csv(path,index=False)
    canonical=read_history(path,sport)
    payload=json.loads(canonical.to_json(orient='records',date_format='iso'))
    store.write(prefix+'history.json.gz',{'schema':1,'rows':payload,'collected_at':now.isoformat()},replace=True)
    report={'saved_stat_rows':len(canonical),'training_performed':False}
    if not train_models or canonical.empty:
        report['status']='recent_history_saved_training_runs_on_daily_backfill' if sport=='MLB' else 'history_saved'
        return report
    fingerprint=digest(payload);last=store.read(prefix+'training_report.json.gz') or {}
    if last.get('fingerprint')==fingerprint:return dict(report,status='unchanged_history_skipped')
    active=store.read(prefix+'active.json.gz') or {'schema':1,'features':FEATURES,'markets':{}}
    print(sport+' player history: training changed numeric prop history',flush=True)
    from dual_agent.ncaaf_learning_job import train_props
    updated,checks=train_props(payload,active,80,league=sport)
    changed=digest(updated)!=digest(active)
    if changed:
        version=digest(updated);store.write(prefix+'models/'+version+'.json.gz',updated)
        store.write(prefix+'active.json.gz',updated,replace=True)
    report.update(status='trained',training_performed=True,fingerprint=fingerprint,
        published=changed,market_reports=checks,
        limitation='Ridge residual calibration compared to weighted-mean baseline and later incumbent examples; existing promoted neural models retain priority.')
    store.write(prefix+'training_report.json.gz',report,replace=True)
    return report


def bounded_mlb_player_history(mode,now):
    """Run blocking warehouse/provider calls in a killable worker, never a thread.

    Frequent cycles collect recent boxes only. Full-history numeric prop fitting
    remains automatic on the daily backfill. Worker logs stream into Actions.
    """
    destination=Path('mlb_player_history_stage.json')
    destination.unlink(missing_ok=True)
    limit=int(os.environ.get('MLB_PLAYER_STAGE_SECONDS','1200' if mode=='backfill' else '240'))
    if not 1<=limit<=3600:raise ValueError('Invalid MLB player-stage timeout')
    command=[sys.executable,'-u','-m','dual_agent.sports_automation','MLB','--mode',mode,
             '--player-history-only','--observed-at',now.isoformat()]
    started=time.monotonic()
    print('MLB player history: starting '+mode+' worker (maximum '+str(limit)+' seconds)',flush=True)
    try:
        result=subprocess.run(command,timeout=limit,check=False)
    except subprocess.TimeoutExpired:
        report={'status':'deferred_timeout','training_performed':False,'retry_on_next_backfill':mode=='backfill',
                'duration_seconds':round(time.monotonic()-started,2),'timeout_seconds':limit}
        destination.write_text(json.dumps(report,indent=2))
        return report
    if not destination.exists():raise RuntimeError('MLB player worker exited without a diagnostic report')
    report=json.loads(destination.read_text())
    if result.returncode:
        if report.get('error_type') in ('ReadTimeout','ConnectTimeout','Timeout'):
            report['status']='deferred_source_timeout'
            return report
        raise RuntimeError('MLB player worker failed: '+str(report.get('error_type','unknown')))
    return report


def run(sport,mode='cycle',store=None,now=None):
    if sport not in ('NFL','MLB') or mode not in ('cycle','backfill'):raise ValueError('Invalid lifecycle request')
    store=store or Store();now=now or datetime.now(timezone.utc)
    report={'sport':sport,'mode':mode,'checked_at':now.isoformat(),'stages':{},'paid_sports_api_requests':0}
    failures=[]
    def checkpoint():
        Path('sports_automation_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    def stage(name,action):
        started=time.monotonic()
        print(sport+' stage starting: '+name,flush=True)
        report['stages'][name]={'status':'running'};checkpoint()
        try:report['stages'][name]=action()
        except Exception as exc:
            failures.append(name);report['stages'][name]={'status':'failed','error_type':type(exc).__name__}
        report['stages'][name]['duration_seconds']=round(time.monotonic()-started,2)
        checkpoint()
        print(sport+' stage finished: '+name+' ('+str(report['stages'][name]['duration_seconds'])+' seconds)',flush=True)
    # Independent work continues after a feed failure; workflow still reports failure.
    stage('final_results',lambda:collect_finals(sport,store,now))
    finals=report['stages']['final_results'].get('games',[])
    if mode=='backfill':
        if sport=='NFL':
            from dual_agent.nfl_historical_learning import run as historical
            stage('historical_training',lambda:historical(store=store,now=now,activate=True))
        else:stage('historical_training',lambda:mlb_historical(store,now))
    from dual_agent.sports_connected_winner import run as learn
    stage('grade_and_train',lambda:learn(sport,store=store,budget={'remaining':12},train_models=True))
    stage('pregame_forecasts',lambda:nfl_forecasts(store,now,finals) if sport=='NFL' else mlb_forecasts(now))
    # Routine MLB cycles never replay or train full player history.
    pointer=store.read(ROOT+sport.lower()+'/props/collection_clock.json.gz') or {}
    due=not pointer.get('at') or stamp(now)-stamp(pointer['at'])>=pd.Timedelta(minutes=15)
    if due or mode=='backfill':
        if sport=='MLB':
            upcoming=report['stages'].get('pregame_forecasts',{})
            if mode=='cycle' and not report['stages']['final_results'].get('new_finals') and upcoming.get('status')=='offseason_or_no_upcoming_games':
                report['stages']['player_history']={'status':'skipped_no_new_finals_or_upcoming_games','training_performed':False,
                    'historical_training':'automatic_daily_backfill'}
            else:stage('player_history',lambda:bounded_mlb_player_history(mode,now))
        else:stage('player_history',lambda:player_history(sport,store,now,train_models=True))
        if 'player_history' not in failures and not report['stages']['player_history'].get('status','').startswith('deferred'):store.write(ROOT+sport.lower()+'/props/collection_clock.json.gz',{'at':now.isoformat()},replace=True)
    deferred=[name for name,value in report['stages'].items() if str(value.get('status','')).startswith('deferred')]
    report['status']='failed' if failures else 'completed_with_deferred_work' if deferred else 'completed'
    report['failed_stages']=failures;report['deferred_stages']=deferred
    # Raw current finals already persisted, so keep the workflow report compact.
    report['stages']['final_results'].pop('games',None)
    store.write(ROOT+sport.lower()+'/automation_latest.json.gz',report,replace=True)
    Path('sports_automation_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,allow_nan=False),flush=True)
    if failures:raise RuntimeError('Lifecycle stages failed: '+', '.join(failures))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sport',choices=['NFL','MLB'])
    parser.add_argument('--mode',choices=['cycle','backfill'],default='cycle')
    parser.add_argument('--player-history-only',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--observed-at',default=None,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.player_history_only:
        if args.sport!='MLB':parser.error('Player worker is MLB-only')
        try:
            result=player_history('MLB',Store(),datetime.fromisoformat(args.observed_at) if args.observed_at else datetime.now(timezone.utc),
                train_models=args.mode=='backfill',backfill=args.mode=='backfill')
        except Exception as exc:
            Path('mlb_player_history_stage.json').write_text(json.dumps({'status':'failed','error_type':type(exc).__name__}))
            print('MLB player history failed: '+type(exc).__name__,flush=True)
            raise SystemExit(1)
        Path('mlb_player_history_stage.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    else:run(args.sport,args.mode)
