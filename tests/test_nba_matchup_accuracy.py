"""Verify evaluation cannot mutate production storage or report activation."""
import copy
import os
import unittest
from unittest.mock import Mock, patch

from dual_agent import nba_matchup_accuracy as accuracy
from dual_agent import nba_unified_winner as winner
from dual_agent.nba_winner_box_context import scores


class Source:
    env = {'SUPABASE_URL':'https://example.supabase.co'}
    bucket = 'market-edge-data'
    def __init__(self):
        self.http = Mock()
        self.objects = {winner.ARTIFACT:{'model_id':'incumbent'}}
    def headers(self):return {}
    def read(self, path, **kwargs):return copy.deepcopy(self.objects.get(path))
    def write(self, *args, **kwargs):raise AssertionError('Production writes forbidden')


class AccuracyTests(unittest.TestCase):
    def test_evaluation_diverts_training_writes_and_restores_source(self):
        source=Source();original=source.http;before=copy.deepcopy(source.objects)
        def fit(store):
            self.assertEqual(store.read(winner.ARTIFACT)['model_id'],'incumbent')
            store.write(winner.ARTIFACT,{'model_id':'candidate'})
            self.assertEqual(store.read(winner.ARTIFACT)['model_id'],'candidate')
            return {'status':'candidate_saved','activated':True,'live_model_changed':True,
                    'test':{'connected_engine':{'accuracy':.65}}}
        with patch.object(winner,'train',side_effect=fit):report=accuracy.evaluate(source)
        self.assertEqual(source.objects,before)
        self.assertIs(source.http,original)
        self.assertFalse(report['activated'])
        self.assertFalse(report['live_model_changed'])
        self.assertTrue(report['would_activate_under_training_rules'])
        self.assertEqual(report['storage_writes'],0)
        self.assertEqual(report['test']['connected_engine']['accuracy'],.65)

    def test_storage_guard_blocks_mutations_foreign_hosts_and_providers(self):
        session=Mock();guard=accuracy.StorageReadsOnly(session,'https://example.supabase.co','market-edge-data')
        cases=[('DELETE','https://example.supabase.co/storage/v1/object/market-edge-data/nba/latest'),
               ('POST','https://example.supabase.co/storage/v1/object/market-edge-data/nba/latest'),
               ('GET','https://api.balldontlie.io/v1/games'),
               ('GET','https://foreign.supabase.co/storage/v1/object/authenticated/market-edge-data/nba/latest'),
               ('GET','https://example.supabase.co/storage/v1/object/authenticated/market-edge-data/nfl/latest')]
        for method,url in cases:
            with self.subTest(method=method,url=url),self.assertRaises(RuntimeError):guard.request(method,url)
        session.request.assert_not_called()
        guard.get('https://example.supabase.co/storage/v1/object/authenticated/market-edge-data/nba/latest')
        self.assertFalse(session.request.call_args.kwargs['allow_redirects'])
        guard.post('https://example.supabase.co/storage/v1/object/list/market-edge-data',json={'prefix':'nba/historical_player_sample_v1/games'})
        self.assertEqual(session.request.call_count,2)

    def test_exception_never_leaves_a_changed_client_or_promoted_model(self):
        source=Source();original=source.http
        with patch.object(winner,'train',side_effect=ValueError('Invalid sample')):
            with self.assertRaises(ValueError):accuracy.evaluate(source)
        self.assertIs(source.http,original)
        self.assertEqual(source.objects[winner.ARTIFACT]['model_id'],'incumbent')

    def test_incumbent_scoring_accepts_python_lists_and_rejects_invalid_values(self):
        result=scores([0,1,1,0],[.2,.8,.6,.4])
        self.assertEqual(result['accuracy'],1.)
        self.assertEqual(result['games'],4)
        for probabilities in ([.2], [.2,float('nan'),.6,.4], [.2,1.5,.6,.4]):
            with self.assertRaises(ValueError):scores([0,1,1,0],probabilities)

    def test_replay_never_removes_production_claims_or_skips_incumbent(self):
        source=Source()
        prefix='sports/connected_winners_v1/nba/experiments_v1/existing/'
        source.objects[prefix+'state.json.gz']={'last_dataset':'old','tested_through':'2026-11-01'}
        source.objects[prefix+'claims/old.json.gz']={'claimed_at':'2026-11-01'}
        before=copy.deepcopy(source.objects)
        store=accuracy.EvaluationStore(source)
        self.assertIsNone(store.read(prefix+'state.json.gz'))
        self.assertIsNone(store.read(prefix+'claims/old.json.gz'))
        self.assertEqual(store.read(winner.ARTIFACT)['model_id'],'incumbent')
        store.write(prefix+'state.json.gz',{'last_dataset':'diagnostic'})
        self.assertEqual(store.read(prefix+'state.json.gz')['last_dataset'],'diagnostic')
        self.assertEqual(source.objects,before)

    def test_failure_report_keeps_trace_but_redacts_credentials_and_urls(self):
        with patch.dict(os.environ, {'SUPABASE_KEY':'private-key-value'}):
            try:
                raise TypeError('Invalid value private-key-value at https://example.supabase.co/storage?token=hidden')
            except TypeError as exc:
                report=accuracy.failure_report(exc)
        self.assertEqual(report['error_type'],'TypeError')
        self.assertNotIn('private-key-value',report['error_message'])
        self.assertNotIn('token=hidden',report['error_message'])
        self.assertEqual(report['error_trace'][-1]['function'],
                         'test_failure_report_keeps_trace_but_redacts_credentials_and_urls')
        self.assertFalse(report['activated'])


if __name__=='__main__':unittest.main()
