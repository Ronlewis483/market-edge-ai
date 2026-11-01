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
    measured=calibration(read('forecast_results_v1.json',{'records':{}})['records'].values())
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
        if not observed or abs((started-observed).total_seconds())>900 or now<observed+timedelta(days=2):continue
        value=float(result['value']);line=float(row['line'])
        won=(value>line) if row['side']=='Over' else (value<line)
        row.update(status='Push' if value==line else 'Won' if won else 'Lost',actual=value,graded_at=now.isoformat())
    counts=defaultdict(lambda:{'won':0,'lost':0,'push':0,'pending':0})
    for row in ledger.values():counts[row['regression_model_id']][row['status'].lower()]+=1
