"""Exercise the actual app helpers without executing the app's page routing."""
import ast
import copy
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
import tempfile
import os
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from dual_agent import ticket_sources


APP = Path(__file__).resolve().parents[1]/'app.py'
NAMES = {'ticket_market_label', 'ticket_is_standard', 'ticket_prop_candidates',
         'build_suggested_ticket', 'ticket_cached_sources', 'ticket_offer_fingerprint',
         'ticket_quote_status', 'ticket_offer_inputs', 'prepare_ticket_offers',
         '_render_ai_ticket_results', 'drain_ticket_prop_updates', 'merge_nba_display_props', 'render_ai_ticket_results'}


def app_helpers():
    parsed = ast.parse(APP.read_text())
    nodes = []
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TICKET_STANDARD_MARKETS' for t in node.targets):
            nodes.append(node)
    namespace = dict(np=np, pd=pd, st=types.SimpleNamespace(session_state={}))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), 'exec'), namespace)
    return namespace


@contextmanager
def workspace():
    before = os.getcwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try: yield
        finally: os.chdir(before)


def prop(player='Player One', book='underdog', now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else now
    return {'Player': player, 'Market': 'Points', 'Pick': 'Over', 'Line': 20.5,
            'Estimated chance': .7, 'Game': 'Boston Celtics @ Milwaukee Bucks',
            'start_time': (now+pd.Timedelta(days=1)).isoformat(),
            'Sportsbook offers': [{'book': book, 'decimal_odds': 1.91, 'updated_at': now.isoformat()}]}


def history(pid, now, count=30):
    rows = []
    for day in range(count):
        when = now-pd.Timedelta(days=day+1)
        rows.append(dict(player={'id':pid}, team={'id':pid}, min='30:00',
            pts=18+day%7, reb=5, ast=4, fg3m=2, blk=1, stl=1, turnover=2,
            source_date=when.isoformat(), history_provider='balldontlie',
            game={'id':pid*1000+day,'status':'Final','date':when.isoformat()}))
    return rows


def quote(name, now, start=None):
    return dict(player=name, line=20.5, market='player_points', book='prizepicks',
        event_id='nba-event', home_team='Milwaukee Bucks', away_team='Boston Celtics',
        start_time=start or (now+pd.Timedelta(hours=5)).isoformat(),
        updated_at=now.isoformat(), capture_time=now.isoformat(),
        over_probability=.5, over_decimal_odds=1.91, under_decimal_odds=1.91)


class TicketBuilderTests(unittest.TestCase):
    def setUp(self):
        self.app = app_helpers()

    def candidates(self, rows):
        return self.app['ticket_prop_candidates']({'NBA': {'player_props': {'game_picks': rows}}})

    def test_empty_game_picks_falls_back_to_saved_picks(self):
        rows = self.app['ticket_prop_candidates']({'NBA': {'player_props': {'game_picks': [], 'picks': [prop()]}}})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['book'], 'underdog')

    def test_ticket_scan_available_without_any_winner_results(self):
        with workspace():
            source = {'player_props': {'game_picks': [prop()]}}
            ticket_sources.save_ticket_source('NBA', source)
            restored = self.app['ticket_cached_sources'](['NBA'], session={})
            self.assertEqual(len(self.app['ticket_prop_candidates'](restored)), 1)
            self.assertFalse(restored['NBA'].get('predictions'))

    def test_collector_prepares_history_then_scores_fresh_lines_without_reloading(self):
        from dual_agent.sports_player_prop_predictions import nba_score_cached_quotes
        now = pd.Timestamp.now(tz='UTC')
        event = dict(id='nba-event', home_team='Milwaukee Bucks', away_team='Boston Celtics',
                     commence_time=(now+pd.Timedelta(hours=5)).isoformat())
        names = ['Player One','Player Two','Player Three','Player Four','Player Five']
        discovery = [quote(name, now=now, start=event['commence_time']) for name in names]
        refreshed = [dict(q,line=25.5,over_decimal_odds=2.1,under_decimal_odds=1.8) for q in discovery]
        histories = {name:history(i+1,now) for i,name in enumerate(names)}
        calls = []; published = []
        def fetch(*args):
            calls.append('quotes')
            return (discovery if calls.count('quotes') == 1 else refreshed), []
        def prepare(wanted, key, **kwargs):
            self.assertEqual(kwargs['cold_batch_size'], 5)
            self.assertEqual(calls.count('quotes'), 1)
            # Cached estimates have already been published before downloading.
            self.assertTrue(any(ticket_sources.prop_records(r['player_props']) for r in published))
            calls.append('history')
            kwargs['status']('Loading recent player history')
            for name in wanted:
                yield name, histories[name], None
        def score(*args, **kwargs):
            calls.append('score')
            return nba_score_cached_quotes(*args, **kwargs)
        with workspace(), patch('dual_agent.sports_player_prop_predictions._get', return_value=[event,event]), \
             patch('dual_agent.sports_player_prop_predictions.fetch_prop_quotes', side_effect=fetch), \
             patch('dual_agent.sports_player_prop_predictions.nba_cached_histories', return_value={names[0]:histories[names[0]]}), \
             patch('dual_agent.sports_player_prop_predictions._nba_history_batches', side_effect=prepare), \
             patch('dual_agent.sports_player_prop_predictions.nba_score_cached_quotes', side_effect=score), \
             patch('dual_agent.sports_player_prop_predictions.load_nba_matchup_context', return_value={}), \
             patch('dual_agent.odds_request_cache.refresh_game_props') as refresh, \
             patch('requests.sessions.Session.request', side_effect=AssertionError('No live HTTP in offline checks')):
            result = ticket_sources.collect_nba_ticket_props('odds','nba',published.append)
            self.assertEqual(refresh.call_count, 2)
            self.assertTrue(result['ticket_scan_complete'])
            self.assertEqual(result['ticket_quote_passes'], 2)
            self.assertEqual(calls.count('history'), 1)
            self.assertEqual(calls[-2:], ['quotes','score'])
            rows = ticket_sources.prop_records(result['player_props'])
            self.assertTrue(rows)
            self.assertTrue(all(r['Line'] == 25.5 for r in rows))
            candidates = self.candidates(rows)
            picks = self.app['build_suggested_ticket'](candidates,'prizepicks',5)
            self.assertEqual(len(picks), 5)
            self.assertEqual(len({r['player'] for r in picks}), 5)
            self.assertTrue(all(r['book'] == 'prizepicks' for r in picks))

    def test_old_events_are_not_requested_or_faked(self):
        event = dict(id='past', home_team='A', away_team='B', commence_time='2020-01-01T00:00:00Z')
        with workspace(), patch('dual_agent.sports_player_prop_predictions._get', return_value=[event]), \
             patch('dual_agent.sports_player_prop_predictions.generate_player_prop_picks') as generate:
            result = ticket_sources.collect_nba_ticket_props('odds', 'nba')
        generate.assert_not_called()
        self.assertTrue(result['ticket_scan_complete'])
        self.assertEqual(result['all_predictions'], [])

    def test_wrong_side_expired_or_mismatched_line_still_rejected(self):
        rows = []
        for changes in ({'side': 'UNDER'}, {'line': 99}, {'updated_at': '2020-01-01T00:00:00Z'}):
            row = prop()
            row['Sportsbook offers'][0].update(changes)
            rows.append(row)
        self.assertEqual(self.candidates(rows), [])

    def test_two_leg_complete_entry_and_no_cross_platform_padding(self):
        rows = [prop('One'), prop('Two'), prop('Three', 'draftkings')]
        candidates = self.candidates(rows)
        picks = self.app['build_suggested_ticket'](candidates, 'underdog', 2)
        self.assertEqual(len(picks), 2)
        self.assertTrue(all(r['book'] == 'underdog' for r in picks))
        self.assertEqual(len(self.app['build_suggested_ticket'](candidates, 'underdog', 3)), 2)

    def test_price_changes_invalidate_but_new_unrelated_offers_do_not(self):
        candidates = self.candidates([prop('One'), prop('Two')])
        old = self.app['ticket_offer_fingerprint'](candidates)
        extra = candidates+self.candidates([prop('Three')])
        current = ticket_sources.current_ticket_legs(candidates, extra)
        self.assertEqual(self.app['ticket_offer_fingerprint'](current), old)
        changed = copy.deepcopy(candidates)
        changed[0]['decimal_odds'] = 1.8
        self.assertNotEqual(self.app['ticket_offer_fingerprint'](changed), old)
        self.assertNotEqual(self.app['ticket_offer_fingerprint'](ticket_sources.current_ticket_legs(candidates, [])), old)

    def test_freshness_rechecked_while_new_preparation_pending(self):
        now = pd.Timestamp.now(tz='UTC')
        candidates = self.candidates([prop(now=now-pd.Timedelta(minutes=29))])
        self.assertEqual(len(ticket_sources.fresh_candidates(candidates, ['NBA'], now)), 1)
        self.assertEqual(ticket_sources.fresh_candidates(candidates, ['NBA'], now+pd.Timedelta(minutes=2)), [])
        self.assertEqual(ticket_sources.fresh_candidates(candidates, ['NFL'], now), [])
        pending = Future()
        self.app['st'].session_state.update(ai_ticket_prepare_future=pending,
            ai_ticket_prepared=('older-signature', time.monotonic()-31, ({}, {}, candidates, 0.)))
        self.app['ticket_prepare_pool'] = MagicMock()
        ready = self.app['ticket_offer_inputs'](['NBA'])
        self.assertEqual(len(ready[2]), 1)
        self.app['ticket_prepare_pool'].assert_not_called()

    def fake_ui(self, click=None):
        ui = MagicMock()
        ui.session_state = dict(ai_ticket_sport='NBA', ai_ticket_legs=3, ai_ticket_specialty=False)
        ui.button.side_effect = lambda label, **kwargs: label == click and not kwargs.get('disabled', False)
        self.app.update(st=ui, drain_ticket_prop_updates=lambda: None, ticket_offer_inputs=lambda sports: None)
        return ui

    def test_initial_preparation_does_not_hide_build_or_scan_controls(self):
        ui = self.fake_ui()
        self.app['_render_ai_ticket_results']()
        labels = [call.args[0] for call in ui.button.call_args_list]
        self.assertIn('Build my ticket', labels)
        self.assertIn('Scan all sportsbooks / refresh props', labels)
        self.assertFalse(ui.rerun.called)

    def test_empty_nba_page_can_start_direct_scan(self):
        ui = self.fake_ui('Scan all sportsbooks / refresh props')
        start = MagicMock()
        self.app.update(start_nba_ticket_refresh=start, prediction_api_key=lambda *args: 'configured')
        self.app['_render_ai_ticket_results']()
        start.assert_called_once()
        self.assertIn('winner predictions are not required', ui.session_state['ai_ticket_refresh_notice'])

    def test_game_page_scan_does_not_block_dedicated_ticket_collector(self):
        ui = self.fake_ui('Scan all sportsbooks / refresh props')
        old_job = Future()
        ui.session_state['nba_enrichment_future'] = old_job
        start = MagicMock()
        self.app.update(start_nba_ticket_refresh=start, prediction_api_key=lambda *args:'configured')
        self.app['_render_ai_ticket_results']()
        start.assert_called_once()
        self.assertIs(ui.session_state['nba_enrichment_future'],old_job)
        self.assertFalse(old_job.cancelled())
        self.assertIn('collecting listed props directly',ui.session_state['ai_ticket_refresh_notice'])

    def test_pending_ticket_collector_still_blocks_duplicate_scan(self):
        ui = self.fake_ui('Scan all sportsbooks / refresh props')
        ui.session_state['ai_ticket_nba_future'] = Future()
        start = MagicMock()
        self.app.update(start_nba_ticket_refresh=start, prediction_api_key=lambda *args:'configured')
        self.app['_render_ai_ticket_results']()
        start.assert_not_called()
        button = next(c for c in ui.button.call_args_list if c.args[0]=='Scan all sportsbooks / refresh props')
        self.assertTrue(button.kwargs['disabled'])

    def test_partial_shortlist_is_not_stored_as_completed_ticket(self):
        ui = self.fake_ui('Build my ticket')
        candidates = self.candidates([prop('One'), prop('Two')])
        self.app['ticket_offer_inputs'] = lambda sports: ({}, {}, candidates, 0.)
        self.app['_render_ai_ticket_results']()
        self.assertNotIn('ai_suggested_ticket', ui.session_state)
        build = next(call for call in ui.button.call_args_list if call.args[0] == 'Build my ticket')
        self.assertTrue(build.kwargs['disabled'])

    def test_empty_scan_progress_does_not_hide_previous_completed_estimates(self):
        from queue import Queue
        queue = Queue()
        queue.put({'player_props': {'game_picks': [], 'picks': []}, 'ticket_scan_complete': False})
        self.app['st'].session_state.update(ai_ticket_nba_updates=queue,
            ai_ticket_nba_result={'player_props': {'game_picks': [prop()]}})
        self.app['drain_ticket_prop_updates']()
        self.assertEqual(len(self.app['st'].session_state['ai_ticket_nba_result']['player_props']['game_picks']), 1)

    def test_explicit_checks_consume_results_without_automatic_redraw_or_network(self):
        from queue import Queue
        ui = self.fake_ui()
        # Restore the actual app functions replaced by fake_ui.
        actual = app_helpers()
        self.app.update({name:actual[name] for name in ('drain_ticket_prop_updates','ticket_offer_inputs')})
        # Imported function globals must use this test's UI/session and helpers.
        actual.update(self.app)
        self.app = actual
        queue = Queue()
        queue.put({'player_props':{'game_picks':[prop('One'),prop('Two'),prop('Three')]},
                   'ticket_scan_complete':False})
        ui.session_state['ai_ticket_nba_updates'] = queue
        running = Future()
        ui.session_state['ai_ticket_nba_future'] = running
        class ImmediatePool:
            def submit(self, function, *args):
                completed = Future()
                completed.set_result(function(*args))
                return completed
        self.app['ticket_prepare_pool'] = lambda: ImmediatePool()
        def fragment(**kwargs):
            self.assertIsNone(kwargs['run_every'])
            return lambda function: function
        ui.fragment.side_effect = fragment
        with workspace(), patch('requests.sessions.Session.request', side_effect=AssertionError('No provider requests')):
            self.app['render_ai_ticket_results']()  # Explicit render schedules saved-offer prep.
            self.app['render_ai_ticket_results']()  # Next explicit check consumes completed work.
        self.assertEqual(len(ui.session_state['ai_ticket_prepared'][2][2]), 3)
        build = [call for call in ui.button.call_args_list if call.args[0] == 'Build my ticket'][-1]
        self.assertFalse(build.kwargs['disabled'])
        ui.rerun.assert_not_called()

    def test_no_quotes_skip_all_player_history_and_matchup_fetching(self):
        from dual_agent.sports_player_prop_predictions import generate_player_prop_picks
        with patch('dual_agent.sports_player_prop_predictions.fetch_prop_quotes', return_value=([], ['No quotes'])), \
             patch('dual_agent.sports_player_prop_predictions.load_nba_matchup_context') as context, \
             patch('dual_agent.sports_player_prop_predictions._nba_history_batches') as history:
            result = generate_player_prop_picks('NBA', [], 'odds', nba_key='nba', quoted_players_only=True)
        history.assert_not_called()
        context.assert_not_called()
        self.assertEqual(result['game_picks'], [])
        self.assertTrue(result['scan_complete'])

    def test_scan_errors_report_http_status_without_credentials(self):
        error = RuntimeError('HTTP 429 https://provider/odds?apiKey=SECRET')
        text = ticket_sources.safe_ticket_error(error)
        self.assertIn('429', text)
        self.assertNotIn('SECRET', text)
        self.assertNotIn('https', text)

    def test_warm_scan_does_not_fetch_quotes_twice(self):
        now = pd.Timestamp.now(tz='UTC')
        event = dict(id='nba-event',home_team='Milwaukee Bucks',away_team='Boston Celtics',
                     commence_time=(now+pd.Timedelta(hours=5)).isoformat())
        quotes = [quote('Player One',now)]
        with workspace(), patch('dual_agent.sports_player_prop_predictions._get',return_value=[event]), \
             patch('dual_agent.sports_player_prop_predictions.fetch_prop_quotes',return_value=(quotes,[])) as fetch, \
             patch('dual_agent.sports_player_prop_predictions.nba_cached_histories',return_value={'Player One':history(1,now)}), \
             patch('dual_agent.sports_player_prop_predictions._nba_history_batches',return_value=iter(())), \
             patch('dual_agent.sports_player_prop_predictions.load_nba_matchup_context',return_value={}), \
             patch('dual_agent.odds_request_cache.refresh_game_props'), \
             patch('requests.sessions.Session.request',side_effect=AssertionError('No live HTTP')):
            result = ticket_sources.collect_nba_ticket_props('odds','nba')
        fetch.assert_called_once()
        self.assertEqual(result['ticket_quote_passes'],1)
        self.assertTrue(result['ticket_scan_complete'])

    def test_prepared_history_scorer_rejects_expired_quotes_without_requests(self):
        from dual_agent.sports_player_prop_predictions import nba_score_cached_quotes
        now = pd.Timestamp.now(tz='UTC')
        q = quote('Player One',now-pd.Timedelta(minutes=31))
        with patch('requests.sessions.Session.request', side_effect=AssertionError('No HTTP')):
            result = nba_score_cached_quotes([], [q], {'Player One':history(1,now)}, now=now)
        self.assertEqual(result['game_picks'], [])

    def test_completed_empty_scan_does_not_resurrect_old_fresh_offers(self):
        session = {'nba_prediction_pipeline_result':{'player_props':{'game_picks':[prop()]}},
                   'ai_ticket_nba_result':{'ticket_scan_complete':True,'player_props':{'game_picks':[],'picks':[]}}}
        with workspace():
            result = self.app['ticket_cached_sources'](['NBA'],session=session)
        self.assertEqual(self.app['ticket_prop_candidates'](result), [])

    def test_cache_reader_validates_id_and_does_not_download_player_history(self):
        from dual_agent.sports_player_prop_predictions import nba_cached_histories
        now = pd.Timestamp.now(tz='UTC')
        archive = {'version':3,'players':{'Player One':{'player_id':1,'checked_at':now.isoformat(),
                                                     'rows':history(1,now)+history(99,now)}}}
        with workspace(), patch('dual_agent.sports_player_prop_predictions._nba_load_history_archive', return_value=archive), \
             patch('dual_agent.sports_player_prop_predictions._get', side_effect=AssertionError('No player-data downloads')):
            rows = nba_cached_histories(['Player One','Missing'],now)
        self.assertEqual(len(rows['Player One']),30)
        self.assertEqual({r['player']['id'] for r in rows['Player One']},{1})
        self.assertEqual(rows['Missing'],[])

    def test_cold_histories_share_complete_paginated_batches(self):
        from dual_agent.sports_player_prop_predictions import _nba_history_batches
        now = pd.Timestamp.now(tz='UTC')
        names = ['Player One','Player Two','Player Three','Player Four','Player Five']
        catalog = [dict(id=i+1,first_name='Player',last_name=name.split()[-1]) for i,name in enumerate(names)]
        samples = [r for i in range(5) for r in history(i+1,now)]
        stat_requests = []
        def provider(url,key,params,bdl):
            if url.endswith('/players/active'):
                return {'data':catalog,'meta':{}}
            self.assertTrue(url.endswith('/stats'))
            stat_requests.append(params)
            self.assertEqual(params['player_ids[]'],[1,2,3,4,5])
            return {'data':samples[100:] if params.get('cursor') else samples[:100],
                    'meta':{} if params.get('cursor') else {'next_cursor':'next'}}
        with workspace(), patch('dual_agent.sports_player_prop_predictions._get',side_effect=provider), \
             patch('dual_agent.sports_player_prop_predictions._nba_load_history_archive',return_value={'version':3,'players':{}}), \
             patch('dual_agent.sports_player_prop_predictions._nba_save_remote_history_archive'):
            ready = list(_nba_history_batches(names,'nba',cold_batch_size=5))
        self.assertEqual(len(stat_requests),2)
        self.assertEqual(len(ready),5)
        self.assertTrue(all(len(rows)==30 and error is None for _,rows,error in ready))

    def test_partial_history_pagination_never_becomes_training_or_scoring_data(self):
        from dual_agent.sports_player_prop_predictions import _nba_history_batches
        now = pd.Timestamp.now(tz='UTC')
        catalog = [dict(id=1,first_name='Player',last_name='One')]
        def provider(url,key,params,bdl):
            if url.endswith('/players/active'): return {'data':catalog,'meta':{}}
            if params.get('cursor'): raise RuntimeError('HTTP 503')
            return {'data':history(1,now),'meta':{'next_cursor':'next'}}
        with workspace(), patch('dual_agent.sports_player_prop_predictions._get',side_effect=provider), \
             patch('dual_agent.sports_player_prop_predictions._nba_load_history_archive',return_value={'version':3,'players':{}}), \
             patch('dual_agent.sports_player_prop_predictions._nba_save_remote_history_archive'):
            ready = list(_nba_history_batches(['Player One'],'nba',cold_batch_size=5))
        self.assertEqual(len(ready),1)
        self.assertEqual(ready[0][1],[])
        self.assertIn('503',ready[0][2])

    def test_scan_failure_preserves_partial_rows(self):
        from queue import Queue
        queue = Queue()
        queue.put({'player_props': {'game_picks': [prop()]}})
        failed = Future()
        failed.set_exception(ValueError('Provider unavailable'))
        self.app['st'].session_state.update(ai_ticket_nba_updates=queue, ai_ticket_nba_future=failed)
        self.app['drain_ticket_prop_updates']()
        saved = self.app['st'].session_state['ai_ticket_nba_result']
        self.assertEqual(len(ticket_sources.prop_records(saved['player_props'])), 1)
        self.assertIn('Completed estimates remain', self.app['st'].session_state['ai_ticket_refresh_issue'])


if __name__ == '__main__':
    unittest.main()
