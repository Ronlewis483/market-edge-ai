"""Offline checks for frozen live NBA feedback, promotion and serving."""
import copy
import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from dual_agent import nba_live_feedback as feedback
from dual_agent import nba_unified_winner as unified
from dual_agent import sports_connected_winner as learner


class MemoryClient:
    def __init__(self):
        self.data = {}

    def read(self, path, **kwargs):
        return copy.deepcopy(self.data.get(path))

    def write(self, path, value, **kwargs):
        self.data[path] = copy.deepcopy(value)


def artifact():
    return {'model_id': 'base-tree', 'family_id': 'base-tree'}


def forecast(i=0):
    start = pd.Timestamp('2023-01-01T20:00:00Z') + pd.Timedelta(days=i)
    return dict(event_id=str(i), game_id=str(i), home_team='Home', away_team='Away',
                source_player_model_id='base-tree', model_id='base-tree',
                architecture='original_offset_neural_player_residual',
                start_time=start.isoformat(), captured_at=(start-pd.Timedelta(hours=2)).isoformat(),
                label_observed_at=(start+pd.Timedelta(hours=3)).isoformat(),
                outcome=i % 2, original_home_probability=.52,
                raw_correction_logit=0., correction_gain=1., connected_home_probability=.52,
                experience_context={'home_projected_pts': 100. + i % 2},
                feedback_context={'team_rest_diff': 1. if i % 2 else -1.})


def state(n):
    return {'results': {str(i): forecast(i) for i in range(n)}}


def simple_model():
    return dict(schema=1, sport='NBA', base_id=feedback.identity(artifact()), checks_passed=True,
                trained_at='2022-01-01T00:00:00Z', model_id='approved-feedback',
                columns=['team_rest_diff'], median=[0.], mean=[0.], scale=[1.],
                w1=[[0.], [1.], [0.]], b1=[0.], w2=[1.], b2=0., correction_scale=1.)


class LiveFeedbackTests(unittest.TestCase):
    def test_legacy_baseline_and_dedup_without_postgame_features(self):
        first = forecast()
        first.update(original_home_probability=.6, raw_correction_logit=.2,
                     connected_home_probability=.95)
        first['feedback_context'].update(outcome=1, home_score=130, team_rest_diff=3.)
        later = dict(first, captured_at='2023-01-01T19:00:00Z', model_id='feedback-version')
        rows, _ = feedback.normalize(artifact(), {'results': {'later': later, 'first': first}})
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['original_probability'], float(unified.bounded_probability(.6, .2)))
        self.assertEqual(rows[0]['context']['team_rest_diff'], 3.)
        self.assertNotIn('outcome', rows[0]['context'])
        self.assertNotIn('home_score', rows[0]['context'])
        self.assertEqual(rows[0]['captured_at'], first['captured_at'])

    def test_reject_nonprospective_or_invalid_records(self):
        changes = [dict(captured_at='2023-01-01T20:00:00Z'),
                   dict(captured_at='2023-01-01T18:00:00'),
                   dict(label_observed_at='2099-01-01T00:00:00Z'),
                   dict(source='historical_reconstruction'), dict(source_player_model_id='other'),
                   dict(original_home_probability=float('nan')), dict(outcome=2),
                   dict(label_observed_at=None), dict(base_connected_home_probability=1.)]
        for change in changes:
            with self.subTest(change=change):
                rows, reasons = feedback.normalize(artifact(), {'results': {'x': dict(forecast(), **change)}})
                self.assertEqual(rows, [])
                self.assertEqual(sum(reasons.values()), 1)

    def test_waits_for_130_and_reports_real_misses_without_requests(self):
        client = MemoryClient()
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No HTTP')):
            report = feedback.learn(client, artifact(), state(90))
        self.assertFalse(report['trained'])
        self.assertFalse(report['activated'])
        self.assertEqual(report['graded_unique_games'], 90)
        self.assertEqual(report['error_review']['wrong'], 45)
        self.assertFalse(report['error_review']['causality_established'])
        self.assertNotIn(unified.ARTIFACT, client.data)
        self.assertIn(unified.ROOT+'learning_report.json', client.data)

    def test_fits_real_saved_results_and_does_not_reuse_test_window(self):
        client, base = MemoryClient(), artifact()
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No HTTP')):
            report = feedback.learn(client, base, state(180))
            self.assertTrue(report['trained'])
            self.assertTrue(report['activated'], report)
            self.assertTrue(report['checks_passed'])
            self.assertIn('live_feedback', client.data[unified.ARTIFACT])
            active_before = copy.deepcopy(base['live_feedback'])
            again = feedback.learn(client, base, state(180))
        self.assertFalse(again['trained'])
        self.assertEqual(again['status'], 'unchanged_experiment_data_skipped')
        self.assertEqual(base['live_feedback'], active_before)
        self.assertFalse(report['controller']['planner_uses_final_test_metrics'])
        # Only genuinely later holdout outcomes can authorize another experiment.
        expanded = feedback.learn(client, base, state(190))
        self.assertFalse(expanded['trained'])
        self.assertIn(expanded['status'], ('waiting_for_25_new_results', 'waiting_for_fresh_test_results'))

    def test_failed_candidate_never_overwrites_active(self):
        client, base = MemoryClient(), artifact()
        store = feedback.FeedbackStore(client, base)
        active = simple_model()
        store.write(learner.ROOT+'nba/active.json.gz', active)
        base['live_feedback'] = copy.deepcopy(active)
        data = state(180)
        for row in data['results'].values():
            row['base_connected_home_probability'] = .99 if row['outcome'] else .01
        report = feedback.learn(client, base, data)
        self.assertTrue(report['trained'])
        self.assertFalse(report['checks_passed'])
        self.assertFalse(report['activated'])
        self.assertFalse(report['correction_changed'])
        self.assertEqual(store.read(learner.ROOT+'nba/active.json.gz'), active)
        self.assertEqual(base['live_feedback'], active)

    def test_store_isolated_from_historical_and_other_family(self):
        client = MemoryClient()
        one = feedback.FeedbackStore(client, artifact())
        two = feedback.FeedbackStore(client, {'model_id': 'different'})
        path = learner.ROOT+'nba/experiments_v1/state.json.gz'
        one.write(path, {'live': True})
        self.assertIsNone(two.read(path))
        self.assertIsNone(client.read(path))
        self.assertEqual(one.path('sports/shared_experience_v1/nfl/latest.json.gz'),
                         'sports/shared_experience_v1/nfl/latest.json.gz')

    def test_apply_guards_and_bounded_no_network_prediction(self):
        base = dict(artifact(), live_feedback=simple_model())
        now, start = '2024-01-01T00:00:00Z', '2024-01-02T00:00:00Z'
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No HTTP')):
            p, mid, status = feedback.apply(base, .52, {'team_rest_diff': 1.}, start, now)
        self.assertEqual(status, 'applied')
        self.assertEqual(mid, 'approved-feedback')
        self.assertGreater(p, .52)
        self.assertLessEqual(abs(np.log(p/(1-p))-np.log(.52/.48)), .7+1e-10)
        for change in ({'checks_passed': False}, {'base_id': 'other'},
                       {'trained_at': '2025-01-01T00:00:00Z'}, {'correction_scale': 2.}):
            bad = dict(base, live_feedback=dict(base['live_feedback'], **change))
            self.assertEqual(feedback.apply(bad, .52, {}, start, now)[0], .52)

    def test_refresh_invokes_live_learning_even_if_historical_training_waits(self):
        from dual_agent.nba_pregame_snapshots import PREFIX
        client = MemoryClient()
        client.data[unified.ARTIFACT] = dict(artifact(), games=[], aliases={})
        client.data[PREFIX+'checkpoint.json.gz'] = {'schema': 1}
        with patch.object(unified, 'compact_games', return_value=([], {})), \
             patch.object(unified, 'grade', return_value=state(90)), \
             patch.object(unified, 'learn_from_results', wraps=unified.learn_from_results) as learning, \
             redirect_stdout(io.StringIO()):
            unified.refresh(client)
        learning.assert_called_once()
        self.assertEqual(client.data[unified.ROOT+'learning_report.json']['graded_unique_games'], 90)

    def test_served_feedback_and_frozen_inputs_are_saved_in_forecast(self):
        now = pd.Timestamp.now(tz='UTC')
        start = now+pd.Timedelta(days=1)
        base = dict(artifact(), schema=2, trained_at=(now-pd.Timedelta(days=2)).isoformat(),
                    context_refreshed_at=now.isoformat(), aliases={'home': 'H', 'away': 'A'},
                    games=[], columns=['original_probability'], tree_model={}, correction_gain=1.,
                    checks_passed=True, live_feedback=simple_model())
        extra = {k: 1. for k in unified.FEATURES}
        with patch.object(unified, 'player_features', return_value=extra), \
             patch.object(unified, 'tree_probability', return_value=.1), \
             patch('dual_agent.shared_experience.attach', return_value={}):
            result = unified.predict(base, pd.DataFrame([{'rest_diff': 1., 'home_score': 999}]),
                                     .52, .55, 'Home', 'Away', start)
        self.assertEqual(result['model_id'], 'approved-feedback')
        self.assertEqual(result['combined_home_probability'], result['connected_home_probability'])
        self.assertNotEqual(result['base_connected_home_probability'], result['connected_home_probability'])
        self.assertNotIn('home_score', result['feedback_context'])
        client = MemoryClient()
        unified.save_forecasts(client, pd.DataFrame([dict(event_id='1', home_team='Home', away_team='Away',
                                                        commence_time=start, unified_forecast=result)]))
        saved = next(iter(client.data.values()))['rows'][0]
        self.assertEqual(saved['source_player_model_id'], base['family_id'])
        self.assertEqual(saved['feedback_context']['team_rest_diff'], 1.)
        self.assertIn('base_connected_home_probability', saved)


if __name__ == '__main__':
    unittest.main()
