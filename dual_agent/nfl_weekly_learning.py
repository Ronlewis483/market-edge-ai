"""Weekly NFL fitting with frequent grading and separate live evidence reports."""
from datetime import datetime, timezone
import pandas as pd

from .sports_connected_winner import ROOT, digest, stamp, calibration


def weekly_attempt(store, base_id, action, now=None, historical=False):
    now = stamp(now or datetime.now(timezone.utc))
    local = now.tz_convert('America/Chicago')
    monday = local.normalize() - pd.Timedelta(days=local.weekday())
    prefix = ROOT + 'nfl/' + ('historical_v1/' if historical else '')
    path = prefix + 'weekly_learning/' + digest(base_id)[:24] + '.json.gz'
    previous = store.read(path) or {}
    if previous and previous.get('schema') != 1:
        raise ValueError('Invalid NFL weekly learning state')
    week = monday.date().isoformat()
    if previous.get('week_start') == week:
        report = dict(status='weekly_training_not_due', trained=False, activated=False,
                      last_training=previous.get('last_training'))
    else:
        report = action()
        # Readiness waits and failed fits remain retryable as more finals arrive.
        # Only completed fitting consumes this week's opportunity.
        if report.get('trained'):
            summary = {k: report.get(k) for k in ('status', 'model_id', 'activated',
                       'checks_passed', 'partition_games', 'validation', 'test')}
            store.write(path, dict(schema=1, week_start=week,
                        attempted_at=now.isoformat(), last_training=summary), replace=True)
    report.update(learning_cadence='weekly', learning_timezone='America/Chicago',
                  learning_week_start=week,
                  next_week_start=(monday + pd.DateOffset(weeks=1)).isoformat(),
                  evidence_source='historical_reconstruction' if historical else 'historical_and_live_training')
    return report


def review(rows, now=None):
    """Report real frozen forecasts separately from retrospective fitting rows."""
    now = stamp(now or datetime.now(timezone.utc))
    live = [r for r in rows if r.get('source') != 'historical_reconstruction'
            and not r.get('historical_reconstruction')]
    recent = [r for r in live if now-pd.Timedelta(days=7) <= stamp(r['label_observed_at']) <= now]
    misses = [dict(event_id=r['event_id'], start_time=r['start_time'],
                   home_team=r['home_team'], away_team=r['away_team'],
                   probability=r.get('connected_probability', r['original_probability']),
                   outcome=r['outcome'], frozen_context=r.get('context', {}))
              for r in recent if (r.get('connected_probability', r['original_probability']) >= .5) != bool(r['outcome'])]
    return dict(prospective_graded_games=len(live), weekly_graded_games=len(recent),
                weekly_calibration={'original': calibration(recent, 'original_probability'),
                                    'connected': calibration(recent, 'connected_probability')},
                weekly_misses=misses, causality_established=False)
