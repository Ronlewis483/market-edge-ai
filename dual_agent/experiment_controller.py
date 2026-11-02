"""Bounded, validation-guided experiments; durable state per sport/base cohort.

Final-test metrics are audit-only, never planner input. Each search consumes a
later test window once. No self-written code or sports API calls are performed.
"""
from datetime import datetime,timezone
import copy
import math
import numpy as np

VERSION=1
MAX_TRIALS=4


def fingerprint(rows):
    from .sports_connected_winner import digest
    return digest([{k:r.get(k) for k in ('base_id','event_id','start_time','captured_at',
        'label_observed_at','original_probability','context','outcome')} for r in sorted(rows,key=lambda r:(r['start_time'],str(r['event_id'])))])


def prepare(sport,base_id,rows,groups,store):
    from .sports_connected_winner import ROOT,digest,stamp
    prefix=ROOT+sport.lower()+'/experiments_v1/'+digest(base_id)[:24]+'/'
    state=store.read(prefix+'state.json.gz') or {'schema':VERSION,'rounds':0}
    if state.get('schema')!=VERSION:raise ValueError('Experiment state schema mismatch')
    identity=fingerprint(rows)
    report={'controller_version':VERSION,'dataset_fingerprint':identity,'experiment_prefix':prefix,
        'maximum_trials':MAX_TRIALS,'planner_uses_final_test_metrics':False}
    if state.get('last_dataset')==identity:
        return None,dict(report,status='unchanged_experiment_data_skipped',last_model_id=state.get('last_model_id'))
    groups=[list(g) for g in groups]
    previous_end=state.get('tested_through')
    if previous_end:
        groups[2]=[r for r in groups[2] if stamp(r['start_time'])>stamp(previous_end)]
    if len(groups[2])<25 or min(sum(r['outcome']==v for r in groups[2]) for v in (0,1))<5:
        return None,dict(report,status='waiting_for_fresh_test_results',fresh_test_games=len(groups[2]),minimum_fresh_test_games=25)
    if previous_end and len(rows)<int(state.get('last_rows',0))+25:
        return None,dict(report,status='waiting_for_25_new_results',new_rows=len(rows)-int(state.get('last_rows',0)))
    from .sports_connected_winner import digest
    run_id=digest([VERSION,base_id,identity])[:24]
    # Claim final evaluation before reading test labels for scoring. A failed job
    # conservatively burns this window rather than repeatedly mining its scores.
    claim=prefix+'claims/'+run_id+'.json.gz'
    if store.read(claim):
        return None,dict(report,status='interrupted_experiment_requires_fresh_results')
    ctx={'sport':sport,'base_id':base_id,'prefix':prefix,'state':state,'groups':groups,'run_id':run_id,
         'dataset':identity,'rows':len(rows),'store':store,'report':report}
    return ctx,report


def plan(context,kind):
    """Only development summaries/configuration are allowed into the planner."""
    state=context['state'];diagnosis=state.get('development_diagnosis',{})
    conservative=bool(diagnosis.get('probability_regression') or diagnosis.get('overconfidence'))
    if kind in ('NFL','NBA','NCAAF'):
        baseline={'depth':2,'trees':80,'leaf':{'NFL':10,'NBA':30,'NCAAF':20}[kind],'half_life_days':None,'gain_cap':1.}
        candidates=[baseline,
            {'depth':1,'trees':60,'leaf':30,'half_life_days':None,'gain_cap':.5},
            {'depth':2,'trees':80,'leaf':30,'half_life_days':365,'gain_cap':.5},
            {'depth':2 if conservative else 3,'trees':60 if conservative else 100,
             'leaf':40 if conservative else 20,'half_life_days':None,'gain_cap':.25 if conservative else 1.}]
    elif kind=='MLB':
        candidates=[{'penalty':10.,'half_life_days':None,'scale':1.},
                    {'penalty':100.,'half_life_days':None,'scale':.5},
                    {'penalty':30.,'half_life_days':365,'scale':.5},
                    {'penalty':100. if conservative else 1.,'half_life_days':None,'scale':.25 if conservative else 1.}]
    else:raise ValueError('Unsupported experiment adapter')
    previous=state.get('best_development_config')
    if previous and previous not in candidates:candidates=[previous]+candidates
    unique=[]
    for value in candidates:
        if value not in unique:unique.append(value)
    return unique[:MAX_TRIALS]


def weights(rows,half_life):
    from .sports_connected_winner import stamp
    if half_life is None:return np.ones(len(rows))
    end=max(stamp(r['start_time']) for r in rows)
    ages=np.array([(end-stamp(r['start_time'])).total_seconds()/86400 for r in rows])
    values=np.exp2(-ages/half_life)
    return values/values.mean()


def rank(score):
    return (-score['accuracy'],score['brier'],score['log_loss'])


def diagnose(y,probabilities,baseline,score,base_score):
    p=np.asarray(probabilities);correct=(p>=.5)==np.asarray(y)
    confidence=np.maximum(p,1-p)
    return {'probability_regression':bool(score['brier']>base_score['brier'] or score['log_loss']>base_score['log_loss']),
        'overconfidence':bool(float(confidence.mean())>float(correct.mean())+.05),
        'accuracy_regression':bool(score['accuracy']<base_score['accuracy']),
        'mean_confidence':float(confidence.mean()),'actual_accuracy':float(correct.mean()),
        'validation_only':True}


def claim_test(context):
    from .sports_connected_winner import stamp
    store=context['store'];state=copy.deepcopy(context['state']);test=context['groups'][2]
    state.update(tested_through=max(stamp(r['start_time']) for r in test).isoformat(),
        last_dataset=context['dataset'],last_rows=context['rows'],rounds=int(state.get('rounds',0))+1)
    claim={'run_id':context['run_id'],'claimed_at':datetime.now(timezone.utc).isoformat(),
        'tested_through':state['tested_through'],'test_games':len(test)}
    store.write(context['prefix']+'claims/'+context['run_id']+'.json.gz',claim)
    store.write(context['prefix']+'state.json.gz',state,replace=True)
    context['state']=state


def finish(context,trials,selected,diagnosis,report):
    baseline=report.get('validation',{}).get('original') or report.get('validation',{}).get('original_engine')
    for trial in trials:
        trial['selected']=trial['config']==selected['config']
        if baseline and trial.get('validation'):
            score=trial['validation'];reasons=[]
            if score['accuracy']<=baseline['accuracy']:reasons.append('no_validation_accuracy_gain')
            if score['brier']>baseline['brier']:reasons.append('validation_brier_worse')
            if score['log_loss']>baseline['log_loss']:reasons.append('validation_log_loss_worse')
            trial['validation_rejections']=reasons
            trial['status']='selected_for_final_evaluation' if trial['selected'] else 'rejected_on_validation' if reasons else 'lower_validation_rank'
    state=dict(context['state'],best_development_config=selected['config'],development_diagnosis=diagnosis,
        last_model_id=report.get('model_id'))
    # Reports may contain test scores; only sanitized state above informs planning.
    audit={'schema':VERSION,'sport':context['sport'],'base_id':context['base_id'],
        'run_id':context['run_id'],'checked_at':datetime.now(timezone.utc).isoformat(),
        'dataset_fingerprint':context['dataset'],'trials':trials,'selected_config':selected['config'],
        'development_diagnosis':diagnosis,'evaluation':report,'test_metrics_used_for_search':False}
    context['store'].write(context['prefix']+'runs/'+context['run_id']+'.json.gz',audit)
    context['store'].write(context['prefix']+'state.json.gz',state,replace=True)
    return dict(context['report'],run_id=context['run_id'],trials=trials,selected_config=selected['config'],
        development_diagnosis=diagnosis,fresh_test_games=len(context['groups'][2]))
