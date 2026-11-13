"""Verify raw saved-history reuse and category controls without live requests."""
import ast
import importlib.util
from pathlib import Path
from concurrent.futures import Future
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def history_module():
    spec=importlib.util.spec_from_file_location('nfl_history_test',ROOT/'dual_agent/nfl_player_history.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules',{'nflreadpy':types.SimpleNamespace()}):spec.loader.exec_module(module)
    return module


def extract(path,names,**scope):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    for node in nodes:node.decorator_list=[]
    namespace=dict(pd=pd,np=np,**scope)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.module=history_module()
        self.now=pd.Timestamp('2026-11-13T00:00Z')
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.module._PROP_HISTORY_ROOT=Path(self.directory.name)/'cache'
        self.store=MagicMock()

    def payload(self,year,age=5):
        return {'collected_at':(self.now-pd.Timedelta(minutes=age)).isoformat(),
            'rows':[dict(player='Example QB',market='Passing yards',game_time=f'{year}-09-11T00:00:00Z',
                value=250.,team='NYG',opponent='WAS',is_home=True,provider_player_id='123',
                provider_player_aliases=['Example QB','E. QB'])]}

    def test_saved_automation_raw_seasons_skip_provider_and_preserve_units_and_ids(self):
        self.store.read.side_effect=lambda path:self.payload(2025 if '/2025.' in path else 2026)
        with patch.object(self.module,'load_player_history',side_effect=AssertionError('No bulk download')):
            frame=self.module.load_cached_player_history([2025,2026],now=self.now,store=self.store)
        self.assertEqual(len(frame),2)
        self.assertEqual(frame.value.tolist(),[250.,250.])
        self.assertEqual(frame.provider_player_id.tolist(),['123','123'])
        self.assertEqual(frame.provider_player_aliases.iloc[0],['Example QB','E. QB'])
        for call in self.store.read.call_args_list:
            self.assertIn('/props/seasons/',call.args[0])
            self.assertNotIn('/props/history.json',call.args[0])

    def test_next_scan_uses_local_rows_without_storage_or_provider_calls(self):
        self.store.read.return_value=self.payload(2026)
        self.module.load_cached_player_history([2026],now=self.now,store=self.store)
        self.store.read.side_effect=AssertionError('No storage download')
        with patch.object(self.module,'load_player_history',side_effect=AssertionError('No bulk download')):
            frame=self.module.load_cached_player_history([2026],now=self.now+pd.Timedelta(minutes=2),store=self.store)
        self.assertEqual(frame.attrs['history_sources'],{'2026':'local raw history'})

    def test_stale_current_season_refreshes_only_that_season(self):
        self.store.read.side_effect=lambda path:self.payload(2025 if '/2025.' in path else 2026,age=60)
        updated=pd.DataFrame(self.payload(2026)['rows']).assign(value=275.)
        with patch.object(self.module,'load_player_history',return_value=updated) as provider:
            frame=self.module.load_cached_player_history([2025,2026],now=self.now,store=self.store)
        provider.assert_called_once_with([2026])
        self.assertEqual(frame.value.tolist(),[250.,275.])

    def test_future_rows_and_wrong_season_cache_do_not_become_history(self):
        bad=self.payload(2026)
        bad['rows'][0]['game_time']='2026-11-14T00:00:00Z'
        self.store.read.return_value=bad
        with patch.object(self.module,'load_player_history',return_value=pd.DataFrame(self.payload(2026)['rows'])) as provider:
            frame=self.module.load_cached_player_history([2026],now=self.now,store=self.store)
        provider.assert_called_once_with([2026])
        self.assertLess(pd.Timestamp(frame.game_time.iloc[0]),self.now)

    def test_ui_cache_rechecks_in_three_minutes_and_does_not_call_bulk_loader(self):
        tree=ast.parse((ROOT/'dual_agent/nfl_player_props_ui.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cached_player_history')
        self.assertEqual(next(k.value.value for k in node.decorator_list[0].keywords if k.arg=='ttl'),180)
        saved=MagicMock(return_value=pd.DataFrame({'value':[250.]}))
        scope=extract(ROOT/'dual_agent/nfl_player_props_ui.py',{'cached_player_history'},
                      load_cached_player_history=saved,validate_history=lambda frame:frame)
        result=scope['cached_player_history']((2025,2026))
        saved.assert_called_once_with((2025,2026))
        self.assertEqual(result.value.iloc[0],250.)


class CategoryTests(unittest.TestCase):
    def test_switching_categories_preserves_worker_and_uses_its_loading_flag(self):
        now=pd.Timestamp.now(tz='UTC')
        game=dict(home_team='New York Giants',away_team='Washington Commanders',commence_time=now+pd.Timedelta(hours=4))
        rows=[dict(game,game_time=game['commence_time'],player='Example QB',market=market,
            model_pick='OVER',line=1.5,historical_support=.7,best_sportsbook='draftkings',
            availability_checked_at=now.isoformat()) for market in ['Passing yards','Receptions']]
        base=dict(nfl_player_props=pd.DataFrame(rows),workspace_pending=False,enrichment_pending=False)
        scoped=dict(nfl_player_props=pd.DataFrame(rows),game_prop_scan_pending=True,game_prop_stage='Loading saved player history')
        ui=MagicMock();worker=Future()
        ui.session_state={'nfl_prediction_pipeline_result':base,'nfl_game_prop_jobs':{'key':{'future':worker}}}
        ui.selectbox.side_effect=['Passing yards','Receptions','Receptions']
        compare=MagicMock()
        addons=extract(ROOT/'dual_agent/game_prediction_addons.py',{'records','game_fields','matching_game_rows'})
        namespace=extract(ROOT/'app.py',{'_render_cross_prop_cards','merge_nfl_game_props'},st=ui,
            nba_prop_game_key=lambda game:'key',drain_cross_updates=lambda league:None,
            cross_sport_state=lambda league:'nfl_prediction_pipeline_result',
            collect_nfl_game_prop_updates=lambda game:scoped,render_prop_category_buttons=MagicMock(),
            render_prop_comparisons=compare)
        modules={'dual_agent.game_prediction_addons':types.SimpleNamespace(matching_game_rows=addons['matching_game_rows']),
                 'dual_agent.nfl_player_props':types.SimpleNamespace(MARKETS={'Passing yards':'player_pass_yds','Receptions':'player_receptions'})}
        with patch.dict('sys.modules',modules),patch('requests.sessions.Session.request',side_effect=AssertionError('Category change must not fetch')):
            namespace['_render_cross_prop_cards']('NFL',game,base)
            namespace['_render_cross_prop_cards']('NFL',game,base)
            self.assertEqual(compare.call_args_list[0].args[0][0]['market'],'Passing yards')
            self.assertEqual(compare.call_args_list[1].args[0][0]['market'],'Receptions')
            self.assertTrue(all(call.kwargs['pending'] for call in compare.call_args_list))
            # A separate full-slate scan must not keep a finished selected-game scan "loading".
            scoped['game_prop_scan_pending']=False;base['workspace_pending']=True
            namespace['_render_cross_prop_cards']('NFL',game,base)
            self.assertFalse(compare.call_args.kwargs['pending'])
        self.assertIs(ui.session_state['nfl_game_prop_jobs']['key']['future'],worker)
        self.assertFalse(worker.cancelled())
        ui.rerun.assert_not_called()


if __name__=='__main__':unittest.main()
