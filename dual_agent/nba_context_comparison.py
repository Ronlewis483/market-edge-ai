"""Prospective comparison of the expanded NBA candidate with frozen current picks.

Run after collection and dataset preparation. No sports API calls, model training,
or live model promotion. Forecasts made after tipoff are never backfilled.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
if __package__:
    from .nba_pregame_snapshots import Client, PREFIX, stamp, start_time, team_name, number, final, scheduled as is_scheduled, safe_issue
    from .nba_training_dataset import extract_features
    from .train_nba_context_neural import MODEL_PREFIX, predict_candidate, load_candidate
else:
    from nba_pregame_snapshots import Client, PREFIX, stamp, start_time, team_name, number, final, scheduled as is_scheduled, safe_issue
    from nba_training_dataset import extract_features
    from train_nba_context_neural import MODEL_PREFIX, predict_candidate, load_candidate

COMPARE_PREFIX=MODEL_PREFIX+'prospective/'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_immutable(client,path,value):
    existing=client.read(path)
    if existing is None: client.write(path,value)
    elif existing!=value: raise ValueError('Immutable comparison object conflict; existing forecast preserved')


def score_pairs(records):
    """The same final games, separately for each candidate and baseline version."""
    groups={}
    for record in records:
        if not record.get('result'): continue
        key=(record['forecast']['candidate_model_id'],str(record['forecast']['current_model_version']))
        groups.setdefault(key,[]).append(record)
    summaries=[]
    for (model_id,version),rows in sorted(groups.items()):
        summary={'candidate_model_id':model_id,'current_model_version':version,'paired_games':len(rows),
                 'evidence_status':'needs_more_completed_pairs' if len(rows)<100 else 'review_required_no_automatic_promotion'}
        models=[('current','current_home_probability'),('candidate','candidate_home_probability')]
        if all(number(r['forecast'].get('neural_base_probability')) is not None for r in rows):
            models.append(('neural_base','neural_base_probability'))
        for label,field in models:
            probabilities=[r['forecast'][field] for r in rows];outcomes=[r['result']['home_win'] for r in rows]
            wins=sum((p>=.5)==y for p,y in zip(probabilities,outcomes));n=len(rows);rate=wins/n;z=1.96
            center=(rate+z*z/(2*n))/(1+z*z/n)
            radius=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            reliability=[]
            for lo,hi in ((.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.01)):
                bucket=[(p,y) for p,y in zip(probabilities,outcomes) if lo<=max(p,1-p)<hi]
                if bucket: reliability.append({'low':lo,'high':min(hi,1.),'games':len(bucket),
                    'average_confidence':sum(max(p,1-p) for p,_ in bucket)/len(bucket),
                    'actual_accuracy':sum((p>=.5)==y for p,y in bucket)/len(bucket)})
            summary[label]={'accuracy':rate,'wins':wins,'accuracy_interval_95':[max(0.,center-radius),min(1.,center+radius)],
                'brier':sum((p-y)**2 for p,y in zip(probabilities,outcomes))/n,
                'log_loss':-sum(y*math.log(max(1e-7,p))+(1-y)*math.log(max(1e-7,1-p)) for p,y in zip(probabilities,outcomes))/n,
                'confidence_buckets':reliability}
        if 'neural_base' in summary:
            summary['candidate_minus_neural_base_accuracy']=summary['candidate']['accuracy']-summary['neural_base']['accuracy']
            summary['candidate_minus_neural_base_brier']=summary['candidate']['brier']-summary['neural_base']['brier']
        summary['candidate_minus_current_accuracy']=summary['candidate']['accuracy']-summary['current']['accuracy']
        summary['candidate_minus_current_brier']=summary['candidate']['brier']-summary['current']['brier']
        summaries.append(summary)
    return summaries


def grade_existing(client,checkpoint,state,now):
    count=0
    for record in state['pairs'].values():
        frozen=record['forecast'];gid=str(frozen['game_id']);game=checkpoint['games'].get(gid)
        outcome_hash=checkpoint.get('settled',{}).get(gid)
        if not game or not final(game) or not outcome_hash or record.get('outcome_hash')==outcome_hash: continue
        home=team_name(game['home_team'].get('full_name'));away=team_name(game['visitor_team'].get('full_name'))
        if home!=team_name(frozen['home_team']) or away!=team_name(frozen['away_team']):
            raise ValueError('Final result team identity differs from frozen pair')
        scheduled=start_time(game);saved_start=stamp(frozen['schedule_start'])
        if scheduled!=saved_start: continue # A rescheduled event must not be silently graded.
        path=PREFIX+'outcomes/'+gid+'/'+outcome_hash+'.json';outcome=client.read(path)
        if not isinstance(outcome,dict): raise ValueError('Referenced outcome sidecar missing')
        observed=stamp(outcome.get('observed_at'));hs=number(outcome.get('home_score'));aws=number(outcome.get('away_score'))
        if (str(outcome.get('game_id'))!=gid or observed is None or not saved_start<observed<=now
            or hs is None or aws is None or min(hs,aws)<=0 or hs==aws
            or hs!=number(game.get('home_team_score')) or aws!=number(game.get('visitor_team_score'))):
            raise ValueError('Final result sidecar is inconsistent')
        result={'schema':1,'forecast_path':record['forecast_path'],'game_id':game['id'],
                'home_win':int(hs>aws),'home_score':hs,'away_score':aws,'outcome_path':path,
                'result_observed_at':observed.isoformat()}
        result_path=COMPARE_PREFIX+'results/'+digest(result)+'.json'
        write_immutable(client,result_path,result)
        record.update(result=result,outcome_hash=outcome_hash,result_path=result_path);count+=1
    return count


def run(client=None,now=None):
    client=client or Client();now=now or datetime.now(timezone.utc)
    checkpoint=client.read(PREFIX+'checkpoint.json.gz',compressed=True)
    if not checkpoint or checkpoint.get('schema')!=1: raise ValueError('Collector checkpoint missing or incompatible')
    state=client.read(COMPARE_PREFIX+'checkpoint.json.gz',compressed=True)
    if state is None: state={'schema':1,'pairs':{}}
    if state.get('schema')!=1 or not isinstance(state.get('pairs'),dict): raise ValueError('Comparison checkpoint invalid; existing results not reset')
    graded=grade_existing(client,checkpoint,state,now)
    diagnostics={k:0 for k in ('baseline_forecasts_seen','invalid_baseline','no_unique_game','already_frozen',
        'game_started','baseline_neural_mismatch','baseline_predates_candidate','no_prior_snapshot','invalid_snapshot','pairs_saved')}
    pointer=client.read(MODEL_PREFIX+'latest_candidate.json');model_id=None
    if pointer is not None:
        path=pointer.get('candidate_path','')
        if not path.startswith(MODEL_PREFIX+'candidates/') or not path.endswith('.json.gz'):
            raise ValueError('Unexpected candidate package path')
        package=client.read(path,compressed=True)
        if not isinstance(package,dict): raise ValueError('Candidate package missing')
        load_candidate(package)
        model_id=package['metadata']['model_id'];available=stamp(package['metadata'].get('available_after'))
        if model_id!=pointer.get('model_id') or available is None: raise ValueError('Candidate metadata differs from saved pointer')
        # Earliest eligible baseline wins; each game is counted once per candidate/version.
        baselines=sorted(checkpoint.get('frozen_forecasts',{}).items(),key=lambda x:(str(x[1].get('captured_at','')),x[0]))
        for name,baseline in baselines:
            diagnostics['baseline_forecasts_seen']+=1
            adjustment=package.get('kind')=='nba_neural_context_adjustment'
            base_neural_probability=number(baseline.get('learning_home_probability'))
            if adjustment and (baseline.get('learning_model_id')!=package['metadata']['base_model_id']
                or base_neural_probability is None or not 0<base_neural_probability<1):
                diagnostics['baseline_neural_mismatch']+=1;continue
            frozen=stamp(baseline.get('captured_at'));start=stamp(baseline.get('start_time'))
            probability=number(baseline.get('current_home_probability'))
            if frozen is None or start is None or not frozen<=now or frozen>=start or probability is None or not 0<probability<1:
                diagnostics['invalid_baseline']+=1;continue
            matches=[g for g in checkpoint['games'].values() if start_time(g) is not None
                and abs((start_time(g)-start).total_seconds())<=900
                and team_name(g['home_team'].get('full_name'))==team_name(baseline.get('home_team'))
                and team_name(g['visitor_team'].get('full_name'))==team_name(baseline.get('away_team'))]
            if len(matches)!=1:diagnostics['no_unique_game']+=1;continue
            game=matches[0];gid=str(game['id']);scheduled=start_time(game)
            version=baseline.get('current_model_version');identity=digest([gid,model_id,version])
            if identity in state['pairs']:diagnostics['already_frozen']+=1;continue
            forecast_path=COMPARE_PREFIX+'forecasts/'+identity+'.json'
            previous=client.read(forecast_path)
            if previous is not None:
                at=stamp(previous.get('candidate_captured_at'));base_at=stamp(previous.get('baseline_captured_at'))
                if (str(previous.get('game_id'))!=gid or previous.get('candidate_model_id')!=model_id
                    or previous.get('current_model_version')!=version or at is None or base_at is None
                    or not available<=base_at<=at<min(start,scheduled)
                    or previous.get('candidate_weights_sha256')!=package['weights_sha256']):
                    raise ValueError('Saved comparison identity or pregame cutoff conflict')
                state['pairs'][identity]={'forecast_path':forecast_path,'forecast':previous}
                diagnostics['pairs_saved']+=1;continue
            if not is_scheduled(game) or now>=min(start,scheduled):diagnostics['game_started']+=1;continue
            if available>frozen:diagnostics['baseline_predates_candidate']+=1;continue
            candidates=[]
            for snapshot_path in checkpoint.get('captures',{}).get(gid,{}).get('paths',[]):
                if not snapshot_path.startswith(PREFIX+'snapshots/'+gid+'/'):raise ValueError('Unexpected snapshot reference')
                snapshot=client.read(snapshot_path)
                if not isinstance(snapshot,dict):raise ValueError('Referenced snapshot missing')
                at=stamp(snapshot.get('captured_at'))
                if at and at<=frozen and stamp(snapshot.get('scheduled_start'))==scheduled:
                    candidates.append((at,snapshot_path,snapshot))
            if not candidates:diagnostics['no_prior_snapshot']+=1;continue
            _,snapshot_path,snapshot=max(candidates,key=lambda x:(x[0],x[1]))
            try:features,gaps=extract_features(snapshot,game,checkpoint)
            except ValueError:diagnostics['invalid_snapshot']+=1;continue
            # Stamp the actual inference time; never pretend it ran at the old baseline time.
            if adjustment:
                prediction=predict_candidate(package,features,now.isoformat(),min(start,scheduled).isoformat(),
                    baseline_home_probability=base_neural_probability,baseline_model_id=baseline['learning_model_id'])
            else:
                prediction=predict_candidate(package,features,now.isoformat(),min(start,scheduled).isoformat())
            candidate_probability=number(prediction.get('home_probability'))
            if candidate_probability is None or not 0<candidate_probability<1:raise ValueError('Invalid candidate probability')
            forecast={'schema':1,'game_id':game['id'],'candidate_model_id':model_id,'candidate_package_path':path,
                'candidate_weights_sha256':package['weights_sha256'],'current_model_version':version,
                'home_team':baseline['home_team'],'away_team':baseline['away_team'],
                'schedule_start':scheduled.isoformat(),'odds_start':start.isoformat(),
                'baseline_forecast_path':PREFIX+'forecasts/'+name,'baseline_captured_at':frozen.isoformat(),
                'candidate_captured_at':now.isoformat(),'snapshot_path':snapshot_path,
                'input_snapshot_captured_at':snapshot['captured_at'],'coverage_gaps':gaps,
                'current_home_probability':probability,'candidate_home_probability':candidate_probability,
                'candidate_kind':package.get('kind','nba_context_neural_candidate'),
                'neural_base_probability':base_neural_probability if adjustment else None,
                'neural_base_model_id':baseline.get('learning_model_id') if adjustment else None,
                'comparison_policy':'Same game; candidate uses a snapshot already saved when baseline was frozen; both forecasts before earlier provider tipoff.',
                'approved_for_live_use':False}
            client.write(forecast_path,forecast)
            state['pairs'][identity]={'forecast_path':forecast_path,'forecast':forecast}
            diagnostics['pairs_saved']+=1
            client.write(COMPARE_PREFIX+'checkpoint.json.gz',state,replace=True,compressed=True)
    # Includes any immutable forecasts recovered after an interrupted checkpoint write.
    graded+=grade_existing(client,checkpoint,state,now)
    client.write(COMPARE_PREFIX+'checkpoint.json.gz',state,replace=True,compressed=True)
    report={'schema':1,'status':'waiting_for_trained_candidate' if pointer is None else 'prospective_comparison_active',
        'checked_at':now.isoformat(),'latest_candidate_model_id':model_id,'frozen_pairs':len(state['pairs']),
        'graded_pairs':sum(bool(r.get('result')) for r in state['pairs'].values()),'new_results':graded,
        'diagnostics':diagnostics,'model_comparisons':score_pairs(state['pairs'].values()),
        'live_model_changed':False,'sports_api_requests':0,'automatic_promotion':False,
        'note':'Only genuinely prospective paired forecasts are scored. An accuracy target is not an established result. Historical backfill remains deferred.'}
    client.write(COMPARE_PREFIX+'status.json',report,replace=True)
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    try:run()
    except Exception as exc:
        print('NBA prospective comparison stopped: '+safe_issue(exc),flush=True)
        raise SystemExit(1)
