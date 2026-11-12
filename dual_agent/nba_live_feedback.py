"""Learn bounded NBA corrections from immutable, graded pregame forecasts.

No score fetching or training occurs on the prediction path. Error summaries
are associations in recorded inputs, never explanations of what caused a loss.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import math

import numpy as np
import pandas as pd

ROOT = 'nba/connected_winner_v2/'
TEAM_FIELDS = ('win_pct_diff', 'avg_point_diff_diff', 'recent_5_win_pct_diff',
               'recent_5_point_diff_diff', 'rest_diff')


def timestamp(value):
    at = pd.Timestamp(value)
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError('An explicit timezone is required')
    return at.tz_convert('UTC')


def finite(value):
    if isinstance(value, (str, bool)) or value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def identity(artifact):
    return 'nba_live_feedback_v1:' + artifact.get('family_id', artifact['model_id'])


def frozen_context(extra, features=None):
    # Explicit allowlist: never accidentally include scores, IDs or outcomes.
    from .nba_unified_winner import FEATURES
    result = {name: finite(extra.get(name)) for name in FEATURES}
    result['availability_scenario_known'] = finite(extra.get('availability_scenario_known'))
    for name in TEAM_FIELDS:
        result['team_' + name] = finite((features or {}).get(name))
    return result


def normalize(artifact, state, now=None):
    from .nba_unified_winner import bounded_probability
    now = timestamp(now or datetime.now(timezone.utc))
    family = artifact.get('family_id', artifact['model_id'])
    earliest, rejected = {}, Counter()
    for row in state.get('results', {}).values():
        try:
            if (row.get('source_player_model_id') != family or
                    row.get('architecture') != 'original_offset_neural_player_residual'):
                rejected['other_model_family'] += 1
                continue
            if row.get('source') not in (None, 'live', 'live_pregame') or row.get('historical_reconstruction'):
                raise ValueError('Retrospective record')
            capture, start, observed = map(timestamp, (row['captured_at'], row['start_time'], row['label_observed_at']))
            if not capture < start < observed <= now:
                raise ValueError('Invalid time ordering')
            if row['outcome'] not in (0, 1) or not row.get('game_id') or not row.get('event_id'):
                raise ValueError('Missing graded identity')
            if not row.get('home_team') or not row.get('away_team') or row['home_team'] == row['away_team']:
                raise ValueError('Invalid matchup')
            baseline = finite(row.get('base_connected_home_probability'))
            if baseline is None:
                original = finite(row.get('original_home_probability'))
                delta, gain = finite(row.get('raw_correction_logit')), finite(row.get('correction_gain'))
                if original is None or not 0 < original < 1 or delta is None or gain is None:
                    raise ValueError('Missing frozen baseline')
                baseline = float(bounded_probability(original, delta, gain))
            if not 0 < baseline < 1:
                raise ValueError('Invalid probability')
            extra = dict(row.get('experience_context') or {})
            saved = row.get('feedback_context') or {}
            extra.update(saved)
            context = frozen_context(extra, {k: saved.get('team_' + k) for k in TEAM_FIELDS})
            record = dict(base_id=identity(artifact), event_id=str(row['game_id']),
                          start_time=start.isoformat(), captured_at=capture.isoformat(),
                          label_observed_at=observed.isoformat(), original_probability=baseline,
                          context=context, outcome=int(row['outcome']))
            event = record['event_id']
            if event not in earliest or capture < timestamp(earliest[event]['captured_at']):
                earliest[event] = record
        except (KeyError, TypeError, ValueError, OverflowError):
            rejected['invalid_or_nonprospective_record'] += 1
    return sorted(earliest.values(), key=lambda r: (r['start_time'], r['event_id'])), dict(rejected)


class FeedbackStore:
    """Isolate live experiments from historical models and other NBA cohorts."""
    def __init__(self, client, artifact):
        self.client = client
        cohort = hashlib.sha256(identity(artifact).encode()).hexdigest()[:24]
        self.prefix = ROOT + 'live_feedback/' + cohort + '/'

    def path(self, path):
        from .sports_connected_winner import ROOT as SPORTS_ROOT
        source = SPORTS_ROOT + 'nba/'
        return self.prefix + path[len(source):] if path.startswith(source) else path

    def read(self, path):
        return self.client.read(self.path(path), compressed=True)

    def write(self, path, value, replace=False):
        return self.client.write(self.path(path), value, compressed=True, replace=replace)


def error_review(rows):
    from .sports_connected_winner import metrics
    if not rows:
        return {'games': 0, 'causality_established': False, 'misses': []}
    p = np.array([r['original_probability'] for r in rows])
    y = np.array([r['outcome'] for r in rows])
    correct = (p >= .5) == y
    bands = []
    for low, high in ((.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)):
        mask = (np.maximum(p, 1-p) >= low) & (np.maximum(p, 1-p) < high)
        if mask.any():
            bands.append(dict(low=low, high=min(high, 1.), games=int(mask.sum()),
                              mean_confidence=float(np.maximum(p, 1-p)[mask].mean()),
                              actual_accuracy=float(correct[mask].mean())))
    misses = [dict(event_id=r['event_id'], start_time=r['start_time'],
                   probability=r['original_probability'], outcome=r['outcome'],
                   frozen_context=r['context']) for r, hit in zip(rows, correct) if not hit]
    return dict(metrics(y, p), correct=int(correct.sum()), wrong=int((~correct).sum()),
                confidence_bands=bands, misses=misses[-50:], causality_established=False,
                interpretation='Recorded pregame associations; no individual loss cause is established')


def learn(client, artifact, state):
    from .sports_connected_winner import ROOT as SPORTS_ROOT, train
    rows, rejected = normalize(artifact, state)
    store = FeedbackStore(client, artifact)
    report = train('NBA', identity(artifact), rows, store)
    report.update(graded_unique_games=len(rows), rejected_records=rejected,
                  baseline_description='Frozen connected forecasts before live feedback',
                  error_review=error_review(rows), sports_api_requests=0)
    active = store.read(SPORTS_ROOT + 'nba/active.json.gz')
    changed = bool(active and active.get('base_id') == identity(artifact) and active.get('checks_passed')
                   and active != artifact.get('live_feedback'))
    if changed:
        artifact['live_feedback'] = active
        client.write(ROOT + 'latest.json.gz', artifact, compressed=True, replace=True)
    report['correction_changed'] = changed
    # Also persist waiting/unchanged status; the generic learner returns early.
    store.write(SPORTS_ROOT + 'nba/training_report.json.gz', report, replace=True)
    client.write(ROOT + 'learning_report.json', report, replace=True)
    return report


def apply(artifact, baseline, context, start, now=None):
    """Apply only the approved, matching feedback artifact. No external reads."""
    model = artifact.get('live_feedback')
    if not model:
        return baseline, None, 'collecting_graded_live_forecasts'
    try:
        now, start = timestamp(now or datetime.now(timezone.utc)), timestamp(start)
        if (model.get('schema') != 1 or model.get('base_id') != identity(artifact) or
                model.get('sport') != 'NBA' or not model.get('checks_passed') or
                not timestamp(model['trained_at']) <= now < start):
            raise ValueError('Unapproved or incompatible live feedback')
        from .sports_connected_winner import encode, forward, probability
        gain = float(model['correction_scale'])
        if not 0 <= gain <= 1:
            raise ValueError('Unbounded correction scale')
        x = encode(context, model['columns'], np.array(model['median']), np.array(model['mean']),
                   np.array(model['scale']), baseline)
        delta = float(gain * forward(x, {k: np.array(model[k]) for k in ('w1', 'b1', 'w2', 'b2')}))
        if not math.isfinite(delta):
            raise ValueError('Invalid correction')
        return float(probability(baseline, np.clip(delta, -.7, .7))), model['model_id'], 'applied'
    except (KeyError, TypeError, ValueError, OverflowError):
        return baseline, None, 'live_feedback_unavailable'
