import unittest
from unittest.mock import patch
from dual_agent.nfl_weekly_learning import weekly_attempt, review
from dual_agent.sports_connected_winner import load_model, _CACHE

class Store:
    def __init__(self): self.values = {}
    def read(self, path): return self.values.get(path)
    def write(self, path, value, replace=False): self.values[path] = value

class WeeklyTests(unittest.TestCase):
    def test_once_per_local_week_and_next_week(self):
        store = Store(); calls = []
        def fit():
            calls.append(1)
            return dict(trained=True, activated=False, status='candidate_saved')
        weekly_attempt(store, 'base', fit, '2026-11-15T23:00:00Z')
        result = weekly_attempt(store, 'base', fit, '2026-11-16T05:59:00Z')
        self.assertEqual(result['status'], 'weekly_training_not_due')
        weekly_attempt(store, 'base', fit, '2026-11-16T06:00:00Z')
        self.assertEqual(len(calls), 2)

    def test_waiting_and_exceptions_do_not_consume_week(self):
        store = Store()
        weekly_attempt(store, 'base', lambda:dict(trained=False), '2026-11-14T18:00:00Z')
        self.assertFalse(store.values)
        def fail(): raise RuntimeError('storage unavailable')
        with self.assertRaises(RuntimeError):
            weekly_attempt(store, 'base', fail, '2026-11-14T18:00:00Z')
        self.assertFalse(store.values)
        self.assertTrue(weekly_attempt(store, 'base', lambda:dict(trained=True), '2026-11-14T18:00:00Z')['trained'])

    def test_cohorts_and_historical_gate_are_independent(self):
        store = Store(); fit=lambda:dict(trained=True)
        for base, historical in [('a',False), ('b',False), ('a',True)]:
            self.assertTrue(weekly_attempt(store,base,fit,'2026-11-14T18:00:00Z',historical)['trained'])
        self.assertEqual(len(store.values),3)

    def test_review_excludes_history_and_reports_frozen_misses(self):
        row=dict(event_id='1',start_time='2026-11-12T18:00:00Z', captured_at='2026-11-12T17:00:00Z',
                 label_observed_at='2026-11-13T00:00:00Z',home_team='DAL',away_team='PHI',
                 original_probability=.7,connected_probability=.6,outcome=0,base_id='b',context={'home_qb_out':1})
        result=review([row,dict(row,source='historical_reconstruction')], '2026-11-14T18:00:00Z')
        self.assertEqual(result['prospective_graded_games'],1)
        self.assertEqual(result['weekly_graded_games'],1)
        self.assertEqual(result['weekly_misses'][0]['frozen_context'],row['context'])

    def test_unapproved_correction_is_not_loaded(self):
        _CACHE.clear()
        with patch('dual_agent.sports_connected_winner.Store') as factory:
            factory.return_value.read.return_value=dict(sport='NFL',base_id='b',schema=1,checks_passed=False)
            self.assertIsNone(load_model('NFL','b'))
        _CACHE.clear()

if __name__ == '__main__': unittest.main()
