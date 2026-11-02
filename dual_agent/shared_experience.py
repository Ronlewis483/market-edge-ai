"""Shared reliability memory; sport-specific evidence, cross-sport hypotheses.

No fitting or sports API requests. Only timestamped prospective results count.
Win probabilities are never changed here. Concurrent writers own separate sport
objects; daily snapshots allow training to use only earlier-known lessons.
"""
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from threading import RLock
import hashlib
import json
import math
import time

ROOT='sports/shared_experience_v1/'
SPORTS=('NBA','NFL','NCAAF','MLB')
_CACHE={};_LOCK=RLock()


def at(value):
    try:
        result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (ValueError,TypeError):return None


def numeric(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError):return None


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def shape(sport,row):
    """Adapt only fields actually saved by each current engine."""
    if sport not in SPORTS:raise ValueError('Unknown sport')
    nested=row.get('unified_forecast')
    extra=nested if isinstance(nested,dict) else row
    def first(*values):
        return next((v for v in values if v is not None and not (isinstance(v,float) and math.isnan(v))),None)
    p=numeric(first(extra.get('connected_probability'),extra.get('connected_home_probability'),
        row.get('model_home_probability'),row.get('Home win probability'),row.get('home_win_probability')))
    original=numeric(first(extra.get('original_probability'),extra.get('original_home_probability'),
        row.get('Original home win probability'),row.get('original_home_win_probability')))
    model=first(extra.get('model_id'),row.get('connected_model_id'),row.get('Connected model ID'),row.get('winner_model_id'),row.get('probability_version'))
    captured=at(row.get('captured_at'));start=at(first(row.get('start_time'),row.get('commence_time')))
    observed=at(row.get('label_observed_at'))
    context=extra.get('experience_context') or extra.get('context') or row.get('experience_context')
    context=context if isinstance(context,dict) else {}
    tags=[]
    if p is not None and original is not None:
        tags+=['pick_changed' if (p>=.5)!=(original>=.5) else 'pick_retained']
        tags+=['large_adjustment' if abs(p-original)>=.1 else 'small_adjustment']
    else:tags+=['correction_unknown']
    if context:
        values=list(context.values());known=sum(numeric(v) is not None for v in values)
        ratio=known/len(values)
        tags+=['sparse_inputs' if ratio<.5 else 'partial_inputs' if ratio<.9 else 'numeric_inputs_present']
    else:tags+=['input_coverage_unknown']
    return {'sport':sport,'model_id':str(model) if model is not None else None,'p':p,'original':original,
        'captured':captured,'start':start,'observed':observed,'outcome':row.get('outcome',row.get('home_won')),
        'event':str(first(row.get('game_id'),row.get('event_id'),'')),
        'home':str(first(row.get('home_team'),row.get('Home'),'')),
        'away':str(first(row.get('away_team'),row.get('Away'),'')),
        'tags':tags,'source':row.get('source')}


def band(p):
    return str(min(9,int(max(p,1-p)*10)))


def accumulator():
    return {'games':0,'wins':0,'confidence_sum':0.,'brier_sum':0.,'log_loss_sum':0.,'dates':[]}


def add(group,row):
    p=row['p'];y=row['outcome'];group['games']+=1
    group['wins']+=int((p>=.5)==bool(y));group['confidence_sum']+=max(p,1-p)
    group['brier_sum']+=(p-y)**2;group['log_loss_sum']-=y*math.log(p)+(1-y)*math.log1p(-p)
    day=row['start'].date().isoformat()
    if day not in group['dates']:group['dates'].append(day)


def interval(wins,n):
    if not n:return None
    z=1.96;p=wins/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def summarize(group):
    n=group['games']
    return {'games':n,'wins':group['wins'],'accuracy':group['wins']/n if n else None,
        'average_estimated_win_chance':group['confidence_sum']/n if n else None,
        'brier':group['brier_sum']/n if n else None,'log_loss':group['log_loss_sum']/n if n else None,
        'distinct_dates':len(group['dates']),'accuracy_interval':interval(group['wins'],n)}


def build(sport,records,now=None):
    now=at(now) if now is not None else datetime.now(timezone.utc)
    if now is None:raise ValueError('Invalid experience timestamp')
    from .sports_connected_winner import clean_team
    earliest={};rejected=0
    for raw in records:
        r=shape(sport,raw)
        valid=(not str(r['source'] or '').startswith(('historical','retrospective')) and r['model_id'] and r['p'] is not None and 0<r['p']<1
            and (r['event'] or r['home'] and r['away']) and r['outcome'] in (0,1) and r['captured'] and r['start'] and r['observed']
            and r['captured']<r['start']<r['observed']<=now and r['start']>=now-timedelta(days=365))
        if not valid:rejected+=1;continue
        event=(clean_team(r['home'],sport),clean_team(r['away'],sport),r['start'].isoformat()) if r['home'] and r['away'] else (r['event'],r['start'].isoformat())
        key=(r['model_id'],event)
        if key not in earliest or r['captured']<earliest[key]['captured']:earliest[key]=r
    models=defaultdict(accumulator);buckets=defaultdict(accumulator);patterns=defaultdict(accumulator)
    independent={}
    for (model,event),r in earliest.items():
        add(models[model],r);add(buckets[identity([model,band(r['p'])])],r)
        add(patterns[identity([model,band(r['p']),r['tags']])],r)
        if event not in independent or r['captured']<independent[event]['captured']:independent[event]=r
    total=accumulator();conditions=defaultdict(accumulator)
    for r in independent.values():
        add(total,r)
        for tag in r['tags']:add(conditions[tag],r)
    evidence={'schema':1,'sport':sport,'window_days':365,'unique_games':len(independent),
        'models':{k:summarize(v) for k,v in models.items()},'buckets':{k:summarize(v) for k,v in buckets.items()},
        'patterns':{k:summarize(v) for k,v in patterns.items()},'overall':summarize(total),
        'conditions':{k:summarize(v) for k,v in conditions.items()}}
    evidence['input_fingerprint']=identity(evidence)
    evidence.update(published_at=now.isoformat(),rejected_records=rejected)
    return evidence


def publish(sport,records,store,now=None):
    report=build(sport,records,now);path=ROOT+sport.lower()+'/latest.json.gz'
    previous=store.read(path)
    if previous and previous.get('input_fingerprint')==report['input_fingerprint']:
        return {'status':'unchanged_experience_skipped','unique_games':report['unique_games']}
    # Own-sport daily objects avoid cross-sport read/modify/write collisions.
    day=at(report['published_at']).date().isoformat()
    store.write(ROOT+sport.lower()+'/daily/'+day+'.json.gz',report,replace=True)
    store.write(path,report,replace=True)
    with _LOCK:_CACHE.pop(sport,None)
    return {'status':'experience_published','unique_games':report['unique_games'],'model_groups':len(report['models'])}


def profile(sport):
    with _LOCK:
        cached=_CACHE.get(sport)
        if cached and time.monotonic()-cached[0]<600:return cached[1]
    value=None
    try:
        from .sports_connected_winner import Store
        value=Store().read(ROOT+sport.lower()+'/latest.json.gz')
        if value and (value.get('schema')!=1 or value.get('sport')!=sport):value=None
    except Exception:pass
    with _LOCK:_CACHE[sport]=(time.monotonic(),value)
    return value


def describe(sport,row,now=None,saved_profile=None):
    now=at(now) if now is not None else datetime.now(timezone.utc)
    r=shape(sport,row);p=r['p']
    report={'status':'limited_evidence','sport':sport,'model_id':r['model_id'],
        'estimated_win_chance':max(p,1-p) if p is not None and 0<p<1 else None,
        'evidence_as_of':None,'matching_games':0,'current_model_games':0,'conditions':r['tags'],
        'cross_sport_results_used_as_confidence':False,'probability_changed':False}
    evidence=saved_profile if saved_profile is not None else profile(sport)
    if not evidence or p is None or not 0<p<1 or not r['model_id']:return report
    published=at(evidence.get('published_at'))
    if not published or published>now or now-published>timedelta(days=14):
        return dict(report,status='stale_or_unavailable_evidence')
    if evidence.get('sport')!=sport or evidence.get('schema')!=1:return report
    model=evidence['models'].get(r['model_id'])
    if not model:return dict(report,prior_model_sport_games=evidence['unique_games'],evidence_as_of=published.isoformat())
    exact=evidence['patterns'].get(identity([r['model_id'],band(p),r['tags']]))
    group=evidence['buckets'].get(identity([r['model_id'],band(p)]))
    use=exact if exact and exact['games']>=30 else group
    report.update(current_model_games=model['games'],matching_situation_games=(exact or {}).get('games',0),evidence_as_of=published.isoformat())
    if not use:return report
    report.update(matching_games=use['games'],observed_accuracy=use['accuracy'],
        observed_accuracy_interval=use['accuracy_interval'],average_past_estimate=use['average_estimated_win_chance'],
        matching_scope='same_model_probability_band_and_conditions' if use is exact else 'same_model_probability_band',
        interval_note='Descriptive Wilson interval; games can be correlated, so this is not a guarantee.')
    n=use['games'];overconfident=use['accuracy_interval'][1]<use['average_estimated_win_chance']-.03
    report['status']='needs_review' if n>=30 and overconfident else 'supported' if n>=100 and use['distinct_dates']>=20 and use is exact else 'developing' if n>=30 else 'limited_evidence'
    if any(tag in r['tags'] for tag in ('sparse_inputs','input_coverage_unknown')) and report['status']=='supported':report['status']='developing'
    return report


def attach(sport,row,now=None):
    try:return describe(sport,row,now)
    except Exception:return {'status':'evidence_unavailable','probability_changed':False,'cross_sport_results_used_as_confidence':False}


def lessons(sport,store,cutoff):
    """Only other-sport snapshots known before this search's test window."""
    cutoff=at(cutoff)
    if not cutoff:return {'status':'unavailable','hypotheses':[]}
    supporting=defaultdict(list);inspected=[]
    for other in SPORTS:
        if other==sport:continue
        selected=None
        try:
            latest=store.read(ROOT+other.lower()+'/latest.json.gz')
            if latest and at(latest.get('published_at')) and at(latest['published_at'])<cutoff:selected=latest
            if selected is None:
                for days in (1,2,3):
                    day=(cutoff.date()-timedelta(days=days)).isoformat()
                    candidate=store.read(ROOT+other.lower()+'/daily/'+day+'.json.gz')
                    if candidate and at(candidate.get('published_at')) and at(candidate['published_at'])<cutoff:
                        selected=candidate;break
        except Exception:continue
        if not selected or selected.get('sport')!=other or selected.get('schema')!=1:continue
        published=at(selected.get('published_at'))
        if cutoff-published>timedelta(days=30):continue
        inspected.append(other)
        for tag,g in selected.get('conditions',{}).items():
            if tag not in ('large_adjustment','pick_changed','sparse_inputs'):continue
            bounds=g.get('accuracy_interval')
            if g.get('games',0)>=30 and bounds and bounds[1]<g['average_estimated_win_chance']-.03:
                supporting[tag].append({'sport':other,'games':g['games'],'snapshot_at':selected['published_at']})
    hypotheses=[{'condition':tag,'suggestion':'test_smaller_corrections','supporting_sports':values,
        'causality_established':False,'requires_local_validation':True}
        for tag,values in supporting.items() if len(values)>=2]
    return {'status':'hypotheses_available' if hypotheses else 'insufficient_cross_sport_evidence',
        'cutoff':cutoff.isoformat(),'inspected_sports':inspected,'hypotheses':hypotheses}


def render(sport,row):
    """Display evidence support separately from the existing win-probability badge."""
    import streamlit as st
    frozen=row.get('evidence_confidence')
    if not isinstance(frozen,dict):
        nested=row.get('unified_forecast')
        frozen=nested.get('evidence_confidence') if isinstance(nested,dict) else None
    evidence=frozen if isinstance(frozen,dict) else attach(sport,row)
    label=str(evidence.get('status','limited_evidence')).replace('_',' ')
    st.caption(('Evidence support when saved: ' if isinstance(frozen,dict) else 'Evidence support: ')+label+' · separate from estimated win chance')
    with st.expander('Experience behind this estimate'):
        st.write('Same-model comparable games:',evidence.get('matching_games',0))
        st.write('Matching situation examples:',evidence.get('matching_situation_games',0))
        rate=evidence.get('observed_accuracy')
        if rate is not None:
            st.write('Observed win rate in those saved forecasts:',f'{rate:.1%}')
            bounds=evidence.get('observed_accuracy_interval')
            if bounds:st.caption(f'Descriptive interval: {bounds[0]:.1%}–{bounds[1]:.1%}; game outcomes may be correlated.')
        if evidence.get('evidence_as_of'):st.caption('Evidence updated: '+evidence['evidence_as_of'])
        st.caption('Only this sport and model determine evidence support. Other sports may suggest experiments; they do not raise this rating.')
        st.caption('These observations do not establish why a game was won or lost. No probability was changed by this evidence display.')
