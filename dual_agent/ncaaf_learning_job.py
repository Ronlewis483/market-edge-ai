"""Daily NCAA collection/grading and weekly offline connected/prop training."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

from dual_agent import ncaaf_storage as storage
from dual_agent.ncaaf_prediction import load_history, build_dataset, stamp, update_forecast_ledger


def read(name,default):
    try:return json.loads((storage.ROOT/name).read_text())
    except (OSError,ValueError):return default


def write(name,value):
    path=storage.ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False));tmp.replace(path)


def calibration(records,prop=False):
    import math
    groups=defaultdict(list)
    for row in records:
        if prop:
            if row.get('status') not in ('Won','Lost'):continue
            model=row['regression_model_id'];confidence=float(row['historical_support']);correct=int(row['status']=='Won')
        else:
            if row.get('status')!='Graded':continue
            model=row['model_id'];p=float(row['model_home_probability']);confidence=max(p,1-p);correct=int(row['correct'])
        groups[model].append((confidence,correct))
    report={}
    for model,rows in groups.items():
        bands=[]
        for low,high in ((0,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.0000001)):
            selected=[(p,y) for p,y in rows if low<=p<high]
            if selected:
                bands.append({'low':low,'high':min(high,1.),'forecasts':len(selected),
                    'average_confidence':sum(p for p,y in selected)/len(selected),
                    'actual_accuracy':sum(y for p,y in selected)/len(selected)})
        report[model]={'graded_forecasts':len(rows),'accuracy':sum(y for p,y in rows)/len(rows),
            'brier':sum((p-y)**2 for p,y in rows)/len(rows),
            'log_loss':-sum(y*math.log(max(min(p,1-1e-9),1e-9))+(1-y)*math.log(max(min(1-p,1-1e-9),1e-9)) for p,y in rows)/len(rows),
            'confidence_bands':bands}
    return report


def grade(store,records,now):
    archive=read('forecast_results_v1.json',{'schema':1,'records':{}})
    ledger=archive['records'];seen=read('remote_forecasts_seen.json',[]);known=set(seen)
    for entry in store.list(storage.PREFIX+'forecasts'):
        name=entry['name']
        if name in known:continue
        row=store.read(storage.PREFIX+'forecasts/'+name)
        if not row:continue
        start,captured,created=stamp(row.get('commence_time')),stamp(row.get('captured_at')),stamp(entry.get('created_at'))
        if not start or not captured or not created or not captured<=created<start:
            known.add(name);continue
        identity=str(row['game_id'])+'|'+row['model_id']
        ledger.setdefault(identity,dict(row,status='Pending',storage_created_at=created.isoformat()))
        known.add(name)
    write('forecast_results_v1.json',{'schema':1,'records':ledger})
    counts=update_forecast_ledger([],records,now)
    write('remote_forecasts_seen.json',sorted(known))
    saved_results=list(read('forecast_results_v1.json',{'records':{}})['records'].values())
    try:
        from .shared_experience import publish
        experience=publish('NCAAF',saved_results,store,now)
    except Exception as exc:experience={'status':'experience_unavailable','error_type':type(exc).__name__}
    write('shared_experience_report.json',experience)
    measured=calibration(saved_results)
    for model,metrics in measured.items():counts.setdefault(model,{}).update(calibration=metrics)
    return counts


def grade_props(store,history,now):
    state=read('prop_results.json',{'records':{},'seen':[]});ledger=state['records'];seen=set(state['seen'])
    for entry in store.list(storage.PREFIX+'prop_forecasts'):
        name=entry['name']
        if name in seen:continue
        row=store.read(storage.PREFIX+'prop_forecasts/'+name)
        if not row:continue
        start,captured,created=stamp(row.get('commence_time')),stamp(row.get('captured_at')),stamp(entry.get('created_at'))
        if start and captured and created and captured<=created<start:
            ledger.setdefault(name,dict(row,status='Pending'))
        seen.add(name)
    actual={(str(r['game_id']),str(r['player_id']),r['market']):r for r in history}
    for row in ledger.values():
        if row['status']!='Pending':continue
        result=actual.get((str(row['game_id']),str(row['player_id']),row['market']))
        if not result or str(result['team_id'])!=str(row['team_id']):continue
        started=stamp(row['commence_time']);observed=stamp(result['date'])
        if not observed or abs((started-observed).total_seconds())>900:continue
        # The canonical collector accepts only explicit final games and actual stats.
        if result.get('label_observed_at') and stamp(result['label_observed_at'])>now:continue
        value=float(result['value']);line=float(row['line'])
        won=(value>line) if row['side']=='Over' else (value<line)
        row.update(status='Push' if value==line else 'Won' if won else 'Lost',actual=value,graded_at=now.isoformat())
    counts=defaultdict(lambda:{'won':0,'lost':0,'push':0,'pending':0})
    for row in ledger.values():counts[row['regression_model_id']][row['status'].lower()]+=1
    write('prop_results.json',{'records':ledger,'seen':sorted(seen)})
    for model,metrics in calibration(ledger.values(),prop=True).items():counts[model]['calibration']=metrics
    return dict(counts)


def better(candidate,incumbent):
    return (candidate['accuracy']>=incumbent['accuracy'] and candidate['brier']<=incumbent['brier']
            and candidate['log_loss']<incumbent['log_loss'])


def train_winner(records,active,now,min_new,store=None):
    from dual_agent.ncaaf_connected import _train,_predict,_metrics
    rows,_,_=build_dataset(records,now)
    current_ids={str(r['id']) for r in rows};previous=set((active or {}).get('training_ids',[]))
    if active and len(current_ids-previous)<min_new:return active,{'status':'waiting_for_new_games','new_games':len(current_ids-previous)}
    if len(rows)<400:return active,{'status':'waiting_for_history','usable_games':len(rows)}
    identity=hashlib.sha256(json.dumps([[r['id'],r['x'],r['y']] for r in rows],separators=(',',':')).encode()).hexdigest()[:20]
    candidate=_train(rows,identity,experiment_store=store)
    if candidate.get('controller_wait'):return active,candidate['controller_wait']
    write('candidates/winner_'+identity+'.json',candidate)
    report={'candidate_id':candidate['model_id'],'comparison':{'validation':candidate['validation'],'test':candidate['test']},'controller':candidate.get('controller')}
    if active is None:
        if not candidate['checks_passed']:
            report['status']='candidate_rejected_baseline_checks';return active,report
        candidate['deployment_status']='bootstrap_passed_base_checks'
        report['status']=candidate['deployment_status'];return candidate,report
    cutoff=stamp(active.get('trained_through'))
    eligible={r['id']:r for r in rows if cutoff and r['start']>cutoff+timedelta(days=2)}
    held=[r for r in candidate['heldout'] if r['id'] in eligible]
    if len(held)<min_new:
        report.update(status='incumbent_retained_insufficient_later_comparison',eligible_games=len(held));return active,report
    ordered=[eligible[r['id']] for r in held]
    _,p=_predict(active,[r['x'] for r in ordered])
    old=_metrics([r['y'] for r in held],p);new=_metrics([r['y'] for r in held],[r['probability'] for r in held])
    report['incumbent_comparison']={'candidate':new,'incumbent':old}
    if candidate['checks_passed'] and better(new,old):
        candidate['deployment_status']='passed_later_incumbent_comparison';report['status']='candidate_activated';return candidate,report
    report['status']='incumbent_retained';return active,report


def train_props(history,active,min_new,league='NCAAF'):
    import numpy as np
    import pandas as pd
    from dual_agent.train_player_prop_regression import dataset,train,metric
    from dual_agent.player_prop_regression import FEATURES
    rows=dataset(pd.DataFrame(history))
    result=dict(active or {'schema':1,'features':FEATURES,'markets':{}});result['markets']=dict(result['markets'])
    reports={};ready=[]
    for market in sorted({r[0] for r in rows}):
        key=league+'|'+market;previous=result['markets'].get(key)
        group=[r for r in rows if r[0]==market]
        if previous and previous.get('promoted'):
            count=sum((r[1]-pd.Timestamp(previous['evaluation_end'],tz='UTC')).days>2 for r in group)
            if count<min_new:
                reports[key]={'status':'waiting_for_new_examples','examples':count};continue
        ready.extend(group)
    candidates=train(ready,league) if ready else {}
    for key,candidate in candidates.items():
        previous=result['markets'].get(key)
        if not candidate['promoted']:reports[key]={'status':'failed_baseline_checks','checks':candidate['checks']};continue
        if previous and previous.get('promoted'):
            market=key.split('|',1)[1]
            group=sorted([r for r in rows if r[0]==market],key=lambda r:r[1])
            test=group[-candidate['partition_games'][-1]:]
            later=[r for r in test if r[1].date().isoformat()>previous['evaluation_end'] and
                   (r[1]-pd.Timestamp(previous['evaluation_end'],tz='UTC')).days>2]
            if len(later)<min_new:reports[key]={'status':'incumbent_retained_insufficient_later_examples','examples':len(later)};continue
            def evaluate(model):
                x=np.array([r[2] for r in later]);scale=np.array([r[4] for r in later]);base=np.array([r[3] for r in later])
                point=base+scale*(((x-np.array(model['center']))/np.array(model['spread']))@np.array(model['coef'])+model['intercept'])
                samples=point[:,None]+scale[:,None]*np.array(model['residuals'])
                if model['nonnegative']:samples=np.maximum(samples,0)
                return metric(later,samples)
            new,old=evaluate(candidate),evaluate(previous)
            if not (new['mae']<=old['mae'] and new['brier']<=old['brier'] and new['log_loss']<old['log_loss']):
                reports[key]={'status':'incumbent_retained','candidate':new,'incumbent':old};continue
        result['markets'][key]=candidate;reports[key]={'status':'eligible_candidate_saved','checks':candidate['checks']}
    return result,reports


def runtime_history(history,now):
    groups=defaultdict(list)
    for row in history:
        at=stamp(row.get('date'))
        if at and at>now-timedelta(days=365):groups[(row['player_id'],row['market'])].append(row)
    return [row for group in groups.values() for row in sorted(group,key=lambda r:r['date'])[-12:]]


def run(mode='collect',max_weeks=8,store=None,now=None):
    if mode not in ('collect','train','bootstrap','cycle') or not 1<=int(max_weeks)<=100:raise ValueError('Invalid NCAA mode or week limit')
    os.environ['NCAAF_ACTIONS_RUN']='true'
    now=now or datetime.now(timezone.utc);store=store or storage.Store()
    previous=storage.restore_state(store)
    previous_runtime=store.read(storage.PREFIX+'runtime/latest.json.gz')
    active_bundle=store.read(previous_runtime['object']) if previous_runtime else {}
    active_bundle=active_bundle or {}
    report={'schema':1,'mode':mode,'checked_at':now.isoformat(),'issues':[],'training_performed':False}
    # Capture a collection snapshot even if a later fitting step fails.
    manifest=previous
    try:
        records,teams,issues=load_history(os.environ.get('CFBD_API_KEY'),now)
        report['issues'].extend(issues)
        from dual_agent.ncaaf_player_props import prepare_history
        collection=prepare_history(os.environ.get('CFBD_API_KEY'),max_weeks,now,fit=False)
        report['collection']=collection;report['issues'].extend(collection.get('issues',[]))
        history=read('player_props/history.json',[])
        winner=active_bundle.get('winner');props=active_bundle.get('props')
        if mode in ('train','bootstrap','cycle'):
            rows,_,_=build_dataset(records,now)
            from dual_agent.experiment_controller import VERSION
            winner_input=storage.digest([VERSION,[[r['id'],r['x'],r['y']] for r in rows]])
            prop_input=storage.digest(history)
            attempted=read('last_training_inputs.json',{})
            if attempted.get('winner')!=winner_input:
                winner,report['winner_training']=train_winner(records,winner,now,25,store=store)
                report['training_performed']=True
            else:report['winner_training']={'status':'unchanged_history_skipped'}
            if history and attempted.get('props')!=prop_input:
                props,report['prop_training']=train_props(history,props,80);report['training_performed']=True
            else:report['prop_training']={'status':'unchanged_or_missing_history_skipped'}
            write('last_training_inputs.json',{'winner':winner_input,'props':prop_input})
        counts=grade(store,records,now)
        report['prospective_results']=counts
        report['shared_experience']=read('shared_experience_report.json',{'status':'not_published'})
        report['prop_results']=grade_props(store,history,now)
        # Generate daily pregame forecasts using saved weights only, no odds calls.
        if winner:
            from dual_agent.ncaaf_connected import _forecast
            rows,future,unsupported=build_dataset(records,now)
            forecasts=_forecast(winner,rows,future,unsupported)['games']
            report['new_frozen_forecasts']=storage.freeze_games(forecasts,store,now)
        report['status']='completed' if winner else 'collected_waiting_for_initial_training'
        write('automation_report.json',report)
        manifest=storage.save_state(store,previous)
        bundle={'schema':1,'checked_at':now.isoformat(),'records':records,'teams':teams,'issues':report['issues'],
                'winner':winner,'props':props,'player_history':runtime_history(history,now),
                'training_report':report,'prospective_results':counts,'prop_results':report['prop_results']}
        storage.publish_runtime(store,bundle)
    except Exception as exc:
        report['status']='failed';report['issues'].append(type(exc).__name__+': '+str(exc))
        write('automation_report.json',report)
        storage.save_state(store,manifest)
        store.write(storage.PREFIX+'reports/latest_failure.json.gz',report,replace=True)
        raise
    finally:
        output=Path('ncaaf_automation_report.json');output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(report,allow_nan=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['collect','train','bootstrap','cycle'],default='collect')
    parser.add_argument('--max-weeks',type=int,default=8)
    args=parser.parse_args();run(args.mode,args.max_weeks)
