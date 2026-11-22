"""Offline source-to-panel regression checks using real collection/model/UI code.

Only external HTTP, raw-history storage and UI worker/session boundaries are
substituted. These tests do not establish current provider or deployed health.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import requests
from dual_agent import nfl_player_props as source
from dual_agent import nfl_player_prop_prediction_pipeline as pipeline
from dual_agent import game_prediction_addons as addons
from dual_agent import odds_request_cache as cache
from dual_agent import sports_prop_workspace as workspace
from test_nfl_prop_loading import extract, history_module
from test_ticket_builder import app_helpers

ROOT = Path(__file__).resolve().parents[1]


class SourceToPanelTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp.now(tz='UTC')
        self.game = dict(id='fixture-event', home_team='New York Giants',
            away_team='Washington Commanders', commence_time=(self.now+pd.Timedelta(hours=4)).isoformat())
        self.history = pd.DataFrame([dict(player='Receiver '+str(player),market='Receptions',
            game_time=self.now-pd.Timedelta(days=7*(i+1)),value=4+i%3,
            team='NYG',opponent='WAS',is_home=True)
            for player in range(3) for i in range(6)])
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with cache._guard:
            cache._entries.clear();cache._locks.clear()
        self.calls = []
        self.raw = dict(self.game,bookmakers=[dict(key='draftkings',title='DraftKings',markets=[
            dict(key='player_receptions',last_update=self.now.isoformat(),outcomes=[
                dict(description='Receiver '+str(i),name=side,point=3.5,price=-110)
                for i in range(3) for side in ('Over','Under')])])])

    def response(self, payload, status=200):
        response = requests.Response();response.status_code=status
        response._content=json.dumps(payload).encode();return response

    def http(self, url, params=None, **kwargs):
        self.calls.append((url,params))
        if url.endswith('/events'):return self.response([self.game,self.game])
        if url.endswith('/markets'):
            markets=['player_receptions']+[f'player_unsupported_{i}_q1' for i in range(20)]
            return self.response({'bookmakers':[{'markets':[{'key':key} for key in markets]}]})
        if url.endswith('/odds'):return self.response(self.raw)
        raise AssertionError('Unexpected request '+url)

    def collect(self, http=None):
        history = history_module();store=MagicMock()
        season=self.now.year if self.now.month>=8 else self.now.year-1
        def saved(path):
            year=season-1 if f'/{season-1}.' in path else season
            rows=self.history.copy()
            if year!=season:
                rows['game_time']=[pd.Timestamp(f'{year}-09-01',tz='UTC')+pd.Timedelta(days=i)
                                    for i in range(len(rows))]
            return dict(collected_at=self.now.isoformat(),rows=json.loads(rows.to_json(orient='records',date_format='iso')))
        store.read.side_effect=saved
        def inputs(years):
            frame=source.validate_history(history.load_cached_player_history(years,now=self.now,store=store))
            return frame,pipeline.build_player_history_index(frame)
        upcoming=extract(ROOT/'dual_agent/nfl_player_props_ui.py',{'upcoming_week_events'})['upcoming_week_events']
        ui=types.SimpleNamespace(cached_events=source.events,cached_props=source.event_props,
            cached_player_history=lambda years:inputs(years)[0],cached_player_inputs=MagicMock(side_effect=inputs),
            upcoming_week_events=upcoming)
        streamlit=types.SimpleNamespace(cache_data=lambda **kw:lambda fn:fn)
        updates=[]
        with patch.dict('sys.modules',{'dual_agent.nfl_player_props_ui':ui,'streamlit':streamlit}), \
             patch.object(history,'_PROP_HISTORY_ROOT',self.root/'history'), \
             patch.object(history,'load_player_history',side_effect=AssertionError('Must reuse raw saved history')), \
             patch('requests.get',side_effect=http or self.http), \
             patch.object(workspace,'settle_nfl'), \
             patch.object(pipeline,'_estimate_prop',wraps=pipeline._estimate_prop) as score:
            result,issues=addons.automatic_nfl_props(pd.DataFrame([self.game]),'fixture-key',on_props=updates.append)
        ui.cached_player_inputs.assert_called_once()
        return result,issues,updates,score.call_count

    def test_raw_response_to_saved_restored_panel_and_ticket_keeps_exact_offers(self):
        self.raw['bookmakers'][0]['markets'][0]['outcomes'].insert(0,
            dict(description='Malformed Neighbor',name='Over',point='invalid',price=-110))
        result,issues,updates,scoring=self.collect()
        self.assertEqual(len(result),6);self.assertTrue(updates)
        self.assertEqual(scoring,3)  # one calculation per player/market/line, not per side
        self.assertEqual(len(self.calls),3)  # event list, discovery, supported markets only
        self.assertTrue(all('q1' not in str(p.get('markets','')) for _,p in self.calls))
        diagnostic=result.attrs['pipeline_diagnostics'][0]
        self.assertEqual(diagnostic['source_listings'],7)
        self.assertEqual(diagnostic['rejections'],{'invalid_line':1})
        self.assertEqual(diagnostic['quote_checks']['accepted_listings'],6)
        payload=dict(predictions=pd.DataFrame([self.game]),nfl_player_props=result,
            prop_pipeline_diagnostics=result.attrs['pipeline_diagnostics'])
        with patch.object(workspace,'_ROOT',self.root/'workspace'),patch.object(workspace,'_remote',side_effect=AssertionError('Offline')):
            workspace.checkpoint('NFL',payload);restored=workspace.restore_result('NFL')
            workspace.track('NFL',restored)
            self.assertEqual(len(workspace.tracking_records('NFL')),3)
        self.assertEqual(restored['prop_pipeline_diagnostics'],payload['prop_pipeline_diagnostics'])
        app=app_helpers()
        candidates=app['ticket_prop_candidates']({'NFL':restored})
        self.assertEqual(len(candidates),6)
        ticket=app['build_suggested_ticket'](candidates,'draftkings',3)
        self.assertEqual(len(ticket),3)
        self.assertEqual(len({r['player'] for r in ticket}),3)
        self.assertTrue(all(r['side']=='OVER' and r['line']==3.5 and r['book']=='draftkings' for r in ticket))
        self.assertTrue(all(pd.Timestamp(r['quote_time'])==self.now for r in ticket))
        self.render_panel(restored)

    def render_panel(self, result):
        """Run the real game panel in Streamlit's UI test runtime, not a mock renderer."""
        if not importlib.util.find_spec('streamlit'):
            self.skipTest('Streamlit is required for the rendered panel check')
        from streamlit.testing.v1 import AppTest
        names={'_render_cross_prop_cards','merge_nfl_game_props','cross_sport_state','nba_prop_game_key',
            'prop_comparisons','render_prop_comparisons','render_prop_category_buttons','select_prop_category'}
        tree=ast.parse((ROOT/'app.py').read_text())
        functions='\n\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
        raw=json.loads(result['nfl_player_props'].to_json(orient='records',date_format='iso'))
        script='import pandas as pd\nimport numpy as np\nimport streamlit as st\n'+functions+'''
def drain_cross_updates(league): pass
def collect_nfl_game_prop_updates(game): return None
'''+f'game={self.game!r}\nprops={raw!r}\n'+'''
st.session_state['prop_platform']='DraftKings'
st.session_state['prop_ranking']='Highest probability'
st.session_state['nfl_prediction_pipeline_result']={'nfl_player_props':pd.DataFrame(props)}
_render_cross_prop_cards('NFL',game,{})
'''
        with patch('requests.sessions.Session.request',side_effect=AssertionError('Panel must not fetch')):
            app=AppTest.from_string(script).run(timeout=20)
            self.assertEqual(len(app.exception),0)
            table=app.dataframe[0].value
            self.assertEqual(len(table),6)
            self.assertEqual(set(table.Player),{'Receiver 0','Receiver 1','Receiver 2'})
            self.assertEqual(set(table.Sportsbook),{'draftkings'})
            self.assertEqual(set(table.Line),{3.5})
            app.selectbox[0].set_value('Receptions').run()
            self.assertEqual(len(app.exception),0);self.assertEqual(len(app.dataframe[0].value),6)
            app.selectbox[0].set_value('Passing yards').run()
            self.assertEqual(len(app.exception),0);self.assertEqual(len(app.dataframe),0)

    def test_quote_exclusions_explain_empty_result_without_fabrication(self):
        for name,change in [('expired_quote',{'last_update':(self.now-pd.Timedelta(minutes=31)).isoformat()}),
            ('missing_update',{'last_update':None}),('invalid_price',{'american_odds':0}),
            ('invalid_side',{'side':'invalid'}),('unsupported_bookmaker',{'bookmaker_key':'other'})]:
            line=source.normalize_props(self.raw).assign(**change)
            rejected=pipeline.run_nfl_player_prop_prediction_pipeline(line,self.history,horizon_days=7)
            self.assertTrue(rejected.empty)
            self.assertEqual(rejected.attrs['quote_checks'][name],6)
            self.assertIn(name,rejected.attrs['reason'])

    def test_pickem_without_prices_survives_source_to_ticket(self):
        self.raw['bookmakers'][0]['key']='underdog'
        for row in self.raw['bookmakers'][0]['markets'][0]['outcomes']:row.pop('price')
        result,_,_,_=self.collect()
        self.assertEqual(len(result),6)
        app=app_helpers();candidates=app['ticket_prop_candidates']({'NFL':{'nfl_player_props':result}})
        ticket=app['build_suggested_ticket'](candidates,'underdog',3)
        self.assertEqual(len(ticket),3);self.assertTrue(all(r['decimal_odds'] is None for r in ticket))

    def test_failed_batch_retains_successful_game_batches(self):
        def http(url,params=None,**kwargs):
            if url.endswith('/markets'):
                self.calls.append((url,params))
                return self.response({'bookmakers':[{'markets':[{'key':key} for key in source.MARKETS.values()]}]})
            if url.endswith('/odds') and 'player_receptions' not in params['markets'].split(','):
                self.calls.append((url,params));return self.response({},status=503)
            return self.http(url,params,**kwargs)
        result,issues,_,_=self.collect(http)
        self.assertEqual(len(result),6)
        self.assertTrue(any('other returned offers retained' in reason for reason in issues))
        self.assertGreater(result.attrs['pipeline_diagnostics'][0]['failed_batches'],0)

    def test_mismatched_event_is_not_scored(self):
        self.raw['id']='wrong-game'
        result,issues,_,scoring=self.collect()
        self.assertTrue(result.empty);self.assertEqual(scoring,0)
        self.assertEqual(result.attrs['pipeline_diagnostics'][0]['failed_batches'],1)

    def test_same_line_model_values_match_previous_per_side_inference(self):
        values=np.array([0,3,5,7,4,8])
        for market,line in [('Receptions',3.5),('Anytime touchdown',.5)]:
            base=pipeline._estimate_prop(values,line,market)
            for side in ('YES','NO','OVER','UNDER'):
                expected=pipeline._estimate_prop(values,line,market,side=side)
                actual=pipeline._estimate_for_side(base,side)
                for field in ('historical_support','projection','prediction_score','over_support','under_support'):
                    self.assertAlmostEqual(actual[field],expected[field])

    def test_regression_push_estimate_survives_to_saved_result(self):
        regression=dict(projection=5.,over_support=.7,under_support=.3,estimated_push_chance=.25,
            estimate_method='fixture regression',regression_model_id='test',regression_evaluation_end='2020-01-01')
        history=pd.concat([self.history,self.history.assign(game_time=self.history.game_time-pd.Timedelta(days=100))])
        with patch('dual_agent.player_prop_regression.estimate',return_value=regression):
            result=pipeline.run_nfl_player_prop_prediction_pipeline(source.normalize_props(self.raw).assign(line=3.),history,
                horizon_days=7,retain_all_qualified=True,include_value_candidates=True)
        self.assertEqual(len(result),6)
        self.assertTrue(result.estimated_push_chance.eq(.25).all())


if __name__=='__main__':unittest.main()
