"""Persistent outcome audit and development-only experiment lessons.

Outcome audits may contain held-out results; candidate planning NEVER reads
them. Planning evidence is rebuilt from the caller's permitted development
partitions. All observations are associations, not established loss causes.
"""
from collections import defaultdict
from datetime import datetime, timezone
import copy
import hashlib
import json
import math

ROOT = 'sports/learning_memory_v1/'
SPORTS = ('NBA', 'NFL', 'MLB', 'NCAAF')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def at(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (TypeError, ValueError):
        return None


def number(value):
    if value is None or isinstance(value, (bool, str)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def record_outcomes(sport, records, store, now=None):
    """Idempotent prospective ledger, including wins AND losses, per model."""
    from .shared_experience import shape
    if sport not in SPORTS:
        raise ValueError('Unknown sport')
    now = at(now) if now is not None else datetime.now(timezone.utc)
    if now is None:
        raise ValueError('Invalid memory timestamp')
    path = ROOT+sport.lower()+'/outcomes.json.gz'
    prior = store.read(path) or {}
    ledger = copy.deepcopy(prior.get('records', {}))
    added = 0
    for raw in records:
        r = shape(sport, raw)
        p = number(r['p'])
        if (not r['model_id'] or not r['event'] or p is None or not 0 < p < 1
                or r['outcome'] not in (0, 1) or not all((r['captured'], r['start'], r['observed']))
                or not r['captured'] < r['start'] < r['observed'] <= now
                or raw.get('historical_reconstruction')
                or str(r['source'] or '').startswith(('historical', 'retrospective'))):
            continue
        event = [r['home'], r['away'], r['start'].isoformat()] if r['home'] and r['away'] else [r['event'], r['start'].isoformat()]
        key = digest([r['model_id'], event])
        entry = dict(model_id=r['model_id'], event_id=r['event'],
            start_time=r['start'].isoformat(), captured_at=r['captured'].isoformat(),
            label_observed_at=r['observed'].isoformat(), probability=p,
            outcome=int(r['outcome']), correct=bool((p >= .5) == r['outcome']),
            conditions=r['tags'])
        if key not in ledger or entry['captured_at'] < ledger[key]['captured_at']:
            ledger[key] = entry
            added += 1
    if added:
        store.write(path, dict(schema=1, sport=sport, records=ledger,
            updated_at=now.isoformat(), planner_must_not_read=True), replace=True)
    return dict(status='outcomes_retained' if added else 'unchanged_outcomes',
        unique_model_games=len(ledger), new_or_earlier_records=added,
        wins=sum(r['correct'] for r in ledger.values()),
        losses=sum(not r['correct'] for r in ledger.values()),
        planner_uses_outcome_audit=False)


def summary(rows):
    """Summaries accept only rows already admitted to development by the caller."""
    usable = [r for r in rows if number(r.get('original_probability')) is not None
              and 0 < r['original_probability'] < 1]
    n = len(usable)
    if not n:
        return dict(games=0, wins=0, losses=0, distinct_dates=0)
    wins = sum((r['original_probability'] >= .5) == r['outcome'] for r in usable)
    confidence = sum(max(r['original_probability'], 1-r['original_probability']) for r in usable)/n
    z = 1.96
    p = wins/n
    upper = (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    return dict(games=n, wins=wins, losses=n-wins, accuracy=p,
        mean_confidence=confidence, accuracy_upper_95=min(1., upper),
        distinct_dates=len({r['start_time'][:10] for r in usable}))


def development(sport, base_id, groups, store, prefix):
    """Accumulate development evidence even when the fresh-test gate is closed."""
    cutoff = min((at(r['captured_at']) for r in groups[2]), default=None)
    eligible = {}
    for row in groups[0]+groups[1]:
        observed, start = at(row.get('label_observed_at')), at(row.get('start_time'))
        if (row.get('base_id') != base_id or cutoff is None or observed is None
                or start is None or not start < observed < cutoff or row.get('outcome') not in (0, 1)):
            continue
        key = digest([base_id, str(row['event_id']), row['start_time']])
        if key not in eligible or row['captured_at'] < eligible[key]['captured_at']:
            eligible[key] = row
    rows = sorted(eligible.values(), key=lambda r:(r['start_time'], str(r['event_id'])))
    buckets = defaultdict(list)
    for row in rows:
        p = number(row.get('original_probability'))
        if p is not None and 0 < p < 1:
            buckets['confidence_'+str(min(9, int(max(p, 1-p)*10)))].append(row)
            buckets['home_pick' if p >= .5 else 'away_pick'].append(row)
        context = row.get('context') or {}
        known = sum(number(value) is not None for value in context.values())
        tag = 'input_coverage_unknown' if not context else 'sparse_inputs' if known/len(context) < .5 else 'inputs_present'
        buckets[tag].append(row)
    totals = summary(rows)
    segments = {name:summary(values) for name, values in buckets.items()}
    hypotheses = []
    for name, segment in dict(overall=totals, **segments).items():
        if (segment['games'] >= 30 and segment['distinct_dates'] >= 5
                and segment['accuracy_upper_95'] < segment['mean_confidence']-.03):
            hypotheses.append(dict(condition=name, action='test_smaller_corrections',
                games=segment['games'], causality_established=False))
    sparse, present = segments.get('sparse_inputs', {}), segments.get('inputs_present', {})
    if (min(sparse.get('games', 0), present.get('games', 0)) >= 30
            and min(sparse['distinct_dates'], present['distinct_dates']) >= 5
            and sparse['accuracy']+.1 < present['accuracy']):
        hypotheses.append(dict(condition='sparse_inputs', action='test_stronger_regularization',
            games=sparse['games'], causality_established=False))
    recent, earlier = summary(rows[-30:]), summary(rows[-60:-30])
    if (min(recent['games'], earlier['games']) >= 30
            and min(recent['distinct_dates'], earlier['distinct_dates']) >= 5
            and recent['accuracy']+.1 < earlier['accuracy']):
        hypotheses.append(dict(condition='recent_development_decline', action='test_shorter_recency_window',
            games=60, causality_established=False))
    report = dict(schema=1, sport=sport, base_id=base_id, development_only=True,
        uses_current_test_labels=False, cutoff=cutoff.isoformat() if cutoff else None,
        games_retained=len(rows), baseline=totals, segments=segments, hypotheses=hypotheses,
        data_fingerprint=digest(rows))
    path = prefix+'memory/development_latest.json.gz'
    previous = store.read(path)
    if not previous or previous.get('data_fingerprint') != report['data_fingerprint']:
        # A compact immutable chapter retains each changed development set.
        store.write(prefix+'memory/development/'+report['data_fingerprint']+'.json.gz',
            dict(report, records=rows))
        store.write(path, report, replace=True)
    journal = store.read(prefix+'memory/validation_experiments.json.gz') or {'runs': {}}
    experiments = [r for r in journal['runs'].values() if r.get('sport') == sport
        and r.get('base_id') == base_id and cutoff is not None
        and at(r.get('validation_labels_known_at')) is not None
        and at(r['validation_labels_known_at']) < cutoff]
    failures = defaultdict(int)
    for run in experiments:
        for trial in run['trials']:
            if trial.get('validation_rejections') or trial.get('status') == 'numerical_fit_failed':
                failures[digest(trial['config'])] += 1
    report['prior_validation_experiments'] = len(experiments)
    report['repeated_validation_failures'] = dict(failures)
    return report


def remember_experiment(context, trials, diagnosis):
    """Sanitize explicitly: no final-test result, score, label or activation field."""
    path = context['prefix']+'memory/validation_experiments.json.gz'
    store = context['store']
    journal = store.read(path) or {'schema':1, 'runs':{}}
    end = max((r['label_observed_at'] for r in context['groups'][1]), default=None)
    allowed_metrics = ('games', 'accuracy', 'brier', 'log_loss')
    safe = []
    for trial in trials:
        safe.append(dict(config=copy.deepcopy(trial['config']),
            validation={k:trial.get('validation', {}).get(k) for k in allowed_metrics
                        if k in trial.get('validation', {})},
            status=trial.get('status'), validation_rejections=trial.get('validation_rejections', [])))
    journal['runs'][context['run_id']] = dict(sport=context['sport'], base_id=context['base_id'],
        validation_labels_known_at=end, trials=safe,
        diagnosis={k:bool(diagnosis.get(k)) for k in ('overconfidence', 'accuracy_regression', 'probability_regression')},
        final_test_metrics_retained_here=False)
    store.write(path, journal, replace=True)
