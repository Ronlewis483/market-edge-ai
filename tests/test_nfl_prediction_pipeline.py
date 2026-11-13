"""Offline NFL delivery checks. No sports API keys or live model activation."""
import ast
import copy
import importlib.util
from pathlib import Path
from queue import Queue
import tempfile
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from dual_agent import nfl_prediction_jobs as jobs
from dual_agent import sports_prop_workspace as workspace

ROOT = Path(__file__).resolve().parents[1]


def forecast():
    return pd.DataFrame([dict(home_team='New York Giants',away_team='Washington Commanders',
        commence_time=pd.Timestamp.now(tz='UTC')+pd.Timedelta(hours=4),
        predicted_team='New York Giants',confidence=.6,home_win_probability=.6,
        away_win_probability=.4,input_status='Initial team forecast · current context loading in background')])


def functions(path, names, **globals):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    for node in nodes: node.decorator_list=[]
    scope = dict(pd=pd,**globals)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope


def context_module():
    # Keep the production functions intact; only Streamlit's cache decorators
    # are replaced so tests do not need a running app or persistent caches.
    streamlit = types.SimpleNamespace(cache_data=lambda **kwargs:lambda function:function)
    spec = importlib.util.spec_from_file_location('nfl_context_test',ROOT/'dual_agent/nfl_live_context.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules',{'streamlit':streamlit}):spec.loader.exec_module(module)
    return module


class NflPipelineTests(unittest.TestCase):
    def test_initial_forecast_published_and_saved_before_context_or_props(self):
        updates,progress = Queue(),Queue()
        result = dict(predictions=forecast(),connected_context_pending=True)
        load_history=MagicMock(return_value=('history','features','checked'))
        load_odds=MagicMock(return_value=pd.DataFrame())
        pipeline=MagicMock(return_value=result)
        def enrich(incoming,*args):
            self.assertEqual(updates.qsize(),1)
            snapshot=updates.queue[0]
            self.assertTrue(snapshot['connected_context_pending'])
            self.assertIn('Initial',snapshot['nfl_generation_stage'])
            self.assertEqual(len(snapshot['all_predictions']),1)
            save.assert_called_once()
            incoming['predictions'].loc[0,'confidence']=.7
            # Queue consumers must not see later mutations of an older stage.
            self.assertEqual(snapshot['predictions'].iloc[0].confidence,.6)
            return incoming
        with patch.object(workspace,'checkpoint') as save:
            jobs.generate({},None,None,'odds',updates,progress,'revision',
                load_history=load_history,load_odds=load_odds,run_pipeline=pipeline,enrich=enrich)
        load_history.assert_called_once();load_odds.assert_called_once();pipeline.assert_called_once()
        self.assertEqual(pipeline.call_args.kwargs['feature_games'],'features')

    def run_enrichment(self, fail_props=False, fail_connected=False):
        updates=Queue(); order=[]
        incoming=dict(predictions=forecast(),connected_context_pending=True,
                      enrichment_pending=True,workspace_pending=True)
        def attach(predictions,odds):
            order.append('context')
            result=predictions.copy()
            result['game_context']=[{'home':{'qb':{'name':'Verified feed starter'}}}]
            return result
        def finalize(result):
            order.append('connected')
            self.assertIn('game_context',result['predictions'])
            if fail_connected:raise RuntimeError('correction store unavailable')
            result['predictions']=result['predictions'].assign(confidence=.65)
            result['connected_context_pending']=False
            return result
        def enhance(league,result,key,features,on_props):
            order.append('props')
            self.assertEqual(updates.qsize(),1)
            self.assertEqual(updates.queue[0]['predictions'].iloc[0].confidence,.6 if fail_connected else .65)
            self.assertEqual(updates.queue[0]['connected_context_pending'],fail_connected)
            self.assertGreaterEqual(save.call_count,1)
            if fail_props:raise RuntimeError('provider failed')
            on_props(pd.DataFrame())
            return result
        modules={
            'dual_agent.nfl_live_context':types.SimpleNamespace(attach_context=attach),
            'dual_agent.nfl_connected_engine':types.SimpleNamespace(finalize_connected_result=finalize),
            'dual_agent.game_prediction_addons':types.SimpleNamespace(enhance_result=enhance),
            'dual_agent.nfl_player_props_ui':types.SimpleNamespace(cached_player_history=lambda years:pd.DataFrame()),
        }
        with patch.dict('sys.modules',modules),patch.object(workspace,'checkpoint') as save, \
             patch.object(workspace,'track'),patch.object(workspace,'settle_nfl'):
            result=jobs.enrich_result(incoming,'features','odds',updates)
        self.assertEqual(order,['context','connected','props'])
        self.assertEqual(result['all_predictions'].iloc[0].confidence,.6 if fail_connected else .65)
        self.assertFalse(result['workspace_pending'])
        self.assertFalse(result['enrichment_pending'])
        return result

    def test_connected_winners_reach_queue_and_disk_before_props(self):
        self.run_enrichment()

    def test_failed_props_preserve_connected_winners(self):
        result=self.run_enrichment(fail_props=True)
        self.assertIn('RuntimeError',result['prop_issues'][0])
        self.assertEqual(len(result['predictions']),1)

    def test_failed_connected_update_keeps_initial_forecasts_explicitly_labeled(self):
        result=self.run_enrichment(fail_connected=True)
        self.assertTrue(result['connected_context_pending'])
        self.assertIn('initial forecasts retained',result['nfl_generation_stage'])
        self.assertIn('RuntimeError',result['connected_learning_issue'])

    def test_saved_initial_status_survives_restart_without_claiming_worker_is_running(self):
        result=dict(predictions=forecast(),all_predictions=forecast(),connected_context_pending=True,
                    nfl_generation_stage='Initial forecasts ready',workspace_pending=True,enrichment_pending=True)
        with tempfile.TemporaryDirectory() as directory,patch.object(workspace,'_ROOT',Path(directory)), \
             patch.object(workspace,'_remote',side_effect=AssertionError('No remote calls')):
            workspace.checkpoint('NFL',result)
            restored=workspace.restore_result('NFL')
        self.assertTrue(restored['connected_context_pending'])
        self.assertFalse(restored['workspace_pending'])
        self.assertFalse(restored['enrichment_pending'])
        self.assertEqual(restored['nfl_generation_stage'],'Initial forecasts ready')

    def test_tonight_ranks_above_stronger_sunday_games_using_central_date(self):
        tonight=forecast().assign(commence_time=pd.Timestamp('2026-11-13T01:15Z'),confidence=.55)
        sunday=pd.concat([forecast().assign(commence_time=pd.Timestamp('2026-11-15T18:00Z'),
                            confidence=.9,home_team='Sunday '+str(i)) for i in range(12)])
        result=jobs.prioritize_today(pd.concat([sunday,tonight]),now='2026-11-12T23:00Z')
        self.assertEqual(result.iloc[0].home_team,'New York Giants')
        self.assertEqual(int(result['_nfl_today'].sum()),1)
        self.assertEqual(len(result),13)

    def test_tonights_schedule_generates_slate_without_sportsbook_prices(self):
        context=context_module()
        schedule=pd.DataFrame([dict(home_team='NYG',away_team='WAS',
            start_time=pd.Timestamp('2026-11-13T01:15Z'),gametime='20:15',
            game_type='REG',game_id='2026_10_WAS_NYG',home_score=None,away_score=None)])
        scope=functions(ROOT/'dual_agent/nfl_prediction_pipeline.py',{'scheduled_slate','nfl_week_rows'})
        with patch.object(context,'fresh_schedule',return_value=(schedule,'checked')), \
             patch.dict('sys.modules',{'dual_agent.nfl_live_context':context}):
            result=scope['scheduled_slate'](pd.DataFrame(),pd.Timestamp('2026-11-12T23:00Z'))
            after_kickoff=scope['scheduled_slate'](pd.DataFrame(),pd.Timestamp('2026-11-13T01:16Z'))
        self.assertEqual(result.iloc[0].away_team,'Washington Commanders')
        self.assertEqual(result.iloc[0].home_team,'New York Giants')
        self.assertFalse(result.iloc[0].moneyline_available)
        self.assertTrue(after_kickoff.empty)

    def test_weather_starts_before_player_feeds_finish(self):
        context=context_module()
        game=forecast()
        when=game.iloc[0].commence_time
        schedule=pd.DataFrame([dict(home_team='NYG',away_team='WAS',start_time=when,
            season=2026,week=10,game_type='REG',stadium_id='NYC')])
        weather_started=threading.Event()
        def feed(*args):
            self.assertTrue(weather_started.wait(timeout=2),'Weather was blocked behind player feeds')
            return pd.DataFrame(),None
        def weather(*args):
            weather_started.set()
            return {'source':'test weather'}
        with patch.object(context,'fresh_schedule',return_value=(schedule,'checked')), \
             patch.object(context,'player_feed',side_effect=feed) as feeds, \
             patch.object(context,'kickoff_weather',side_effect=weather) as weather_call:
            result=context.attach_context(game,pd.DataFrame())
        self.assertEqual(feeds.call_count,2)
        weather_call.assert_called_once()
        self.assertEqual(result.iloc[0].game_context['weather']['source'],'test weather')
        self.assertIn('Expected starting quarterback not resolved',result.iloc[0].game_context['issues'][0])

    def test_manual_check_updates_same_revision_without_auto_redraw_or_new_scan(self):
        current=dict(predictions=forecast().assign(confidence=.7),nfl_winner_revision='same')
        ui=MagicMock()
        ui.session_state={'nfl_displayed_winner_revision':'same'}
        def drain(league):ui.session_state['nfl_prediction_pipeline_result']=current
        retain=MagicMock()
        scope=functions(ROOT/'app.py',{'_render_nfl_generation_progress','render_nfl_generation_progress'},
                        st=ui,drain_cross_updates=drain,retain_nfl_picks=retain)
        scope['render_nfl_generation_progress']()
        self.assertEqual(ui.session_state['live_nfl_predictions'].iloc[0].confidence,.7)
        ui.button.assert_called_once()
        self.assertIn('Check for updated NFL results',ui.button.call_args.args)
        ui.fragment.assert_not_called()  # Manual check must update the cards outside the progress control.
        ui.rerun.assert_not_called()


if __name__ == '__main__':unittest.main()
