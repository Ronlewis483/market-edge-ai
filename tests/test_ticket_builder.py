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
         '_render_ai_ticket_results', 'drain_ticket_prop_updates', 'merge_nba_display_props'}


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

    def test_collector_discovers_events_and_publishes_before_finishing(self):
        now = pd.Timestamp.now(tz='UTC')
        event = dict(id='nba-event', home_team='Milwaukee Bucks', away_team='Boston Celtics',
                     commence_time=(now+pd.Timedelta(hours=5)).isoformat())
        published = []
        def generate(league, games, odds_key, **kwargs):
            self.assertEqual(games[0]['event_id'], 'nba-event')
            self.assertNotIn('confidence', games[0])
            self.assertTrue(kwargs['nba_history_enabled'])
            self.assertTrue(kwargs['foreground'])
            kwargs['progress']({'game_picks': [prop()], 'picks': [prop()]})
            self.assertTrue(ticket_sources.prop_records(published[-1]['player_props']))
            self.assertFalse(published[-1]['ticket_scan_complete'])
            # A final status-only result must not erase the partial forecast.
            return {'game_picks': [], 'message': 'Complete'}
        with workspace(), patch('dual_agent.sports_player_prop_predictions._get', return_value=[event, event]), \
             patch('dual_agent.sports_player_prop_predictions.generate_player_prop_picks', side_effect=generate):
            result = ticket_sources.collect_nba_ticket_props('odds', 'nba', published.append)
            self.assertTrue(result['ticket_scan_complete'])
            self.assertEqual(len(result['all_predictions']), 1)
            self.assertEqual(len(ticket_sources.prop_records(result['player_props'])), 1)

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
        ui.button.side_effect = lambda label, **kwargs: label == click
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

    def test_partial_shortlist_is_not_stored_as_completed_ticket(self):
        ui = self.fake_ui('Build my ticket')
        candidates = self.candidates([prop('One'), prop('Two')])
        self.app['ticket_offer_inputs'] = lambda sports: ({}, {}, candidates, 0.)
        self.app['_render_ai_ticket_results']()
        self.assertNotIn('ai_suggested_ticket', ui.session_state)
        self.assertIn('Found 2 eligible legs; 3 were requested', ui.session_state['ai_ticket_build_issue'])

    def test_empty_scan_progress_does_not_hide_previous_completed_estimates(self):
        from queue import Queue
        queue = Queue()
        queue.put({'player_props': {'game_picks': [], 'picks': []}, 'ticket_scan_complete': False})
        self.app['st'].session_state.update(ai_ticket_nba_updates=queue,
            ai_ticket_nba_result={'player_props': {'game_picks': [prop()]}})
        self.app['drain_ticket_prop_updates']()
        self.assertEqual(len(self.app['st'].session_state['ai_ticket_nba_result']['player_props']['game_picks']), 1)

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
