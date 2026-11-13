"""Product pages route to current pipelines without legacy diagnostic work."""
import ast
import builtins
from pathlib import Path
import symtable
import types
import unittest
from unittest.mock import MagicMock,patch

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'app.py').read_text()
TREE=ast.parse(SOURCE)


def helpers(names,**scope):
    nodes=[]
    for original in ast.parse(SOURCE).body:
        if isinstance(original,ast.FunctionDef) and original.name in names:
            original.decorator_list=[];nodes.append(original)
        elif isinstance(original,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAVIGATION_GROUPS' for t in original.targets):
            nodes.append(original)
    namespace=dict(pd=pd,**scope)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'app.py','exec'),namespace)
    return namespace


def ui():
    value=MagicMock()
    value.session_state={}
    value.button.return_value=False
    value.created_columns=[]
    def columns(count,**kwargs):
        widgets=[MagicMock() for _ in range(count if isinstance(count,int) else len(count))]
        value.created_columns.extend(widgets)
        return widgets
    value.columns.side_effect=columns
    return value


class NavigationTests(unittest.TestCase):
    def test_home_sidebar_contains_only_navigation_actions(self):
        screen=ui();screen.session_state['main_navigation']='🏠 Home'
        scope=helpers({'select_main_page'},st=screen,drain_ticket_prop_updates=MagicMock())
        sidebar=next(n for n in TREE.body if isinstance(n,ast.With) and any(ast.unparse(item.context_expr)=='st.sidebar' for item in n.items))
        with patch('requests.sessions.Session.request',side_effect=AssertionError('No data work on Home')):
            exec(compile(ast.Module(body=[sidebar],type_ignores=[]),'app.py','exec'),scope)
        expected={destination for _,destinations in scope['NAVIGATION_GROUPS'] for destination in destinations}
        self.assertEqual({call.args[0] for call in screen.button.call_args_list},expected)
        scope['drain_ticket_prop_updates'].assert_not_called()

    def test_home_only_changes_pages_and_never_starts_prediction_or_api_work(self):
        screen=ui()
        scope=helpers({'render_home_navigation','select_main_page'},st=screen)
        with patch('requests.sessions.Session.request',side_effect=AssertionError('Home must not fetch data')):
            scope['render_home_navigation']()
            for call in screen.columns.call_args_list:self.assertEqual(call.args,(3,))
            self.assertEqual(screen.columns.call_count,3)
            buttons=[call for column in screen.created_columns for call in column.button.call_args_list]
            expected={destination for group,destinations in scope['NAVIGATION_GROUPS'] if group!='START' for destination in destinations}
            self.assertEqual({call.args[0] for call in buttons},expected)
            for call in buttons:
                self.assertIs(call.kwargs['on_click'],scope['select_main_page'])
                call.kwargs['on_click'](*call.kwargs['args'])
                self.assertEqual(screen.session_state['main_navigation'],call.args[0])

    def test_each_sport_has_one_live_entry_point(self):
        hub=next(n for n in TREE.body if isinstance(n,ast.If) and ast.unparse(n.test)=="page in ['🏈 NFL', '🏀 NBA', '⚾ MLB']")
        for sport in ('NFL','NBA','MLB'):
            screen=ui();start=MagicMock();render=MagicMock();mlb=MagicMock()
            scope=dict(st=screen,page={'NFL':'🏈 NFL','NBA':'🏀 NBA','MLB':'⚾ MLB'}[sport],
                start_nfl_prediction_request=start,render_nfl_generation_progress=MagicMock(),
                render_league_prediction_results=render,render_mlb_prediction_center=mlb,
                render_mlb_update_notice=MagicMock())
            with patch('requests.sessions.Session.request',side_effect=AssertionError('No scan without a click')):
                exec(compile(ast.Module(body=[hub],type_ignores=[]),'app.py','exec'),scope)
            if sport=='MLB':mlb.assert_called_once_with('mlb_sidebar')
            else:
                self.assertEqual(screen.button.call_count,1)
                self.assertIn('Generate '+sport+' Predictions',screen.button.call_args.args[0])
                self.assertEqual(render.call_count,1)
            start.assert_not_called()

    def test_nfl_generate_still_launches_the_current_background_worker(self):
        hub=next(n for n in TREE.body if isinstance(n,ast.If) and ast.unparse(n.test).startswith('page in '))
        screen=ui();screen.button.return_value=True;start=MagicMock()
        scope=dict(st=screen,page='🏈 NFL',start_nfl_prediction_request=start,
            render_nfl_generation_progress=MagicMock(),render_league_prediction_results=MagicMock())
        exec(compile(ast.Module(body=[hub],type_ignores=[]),'app.py','exec'),scope)
        start.assert_called_once()

    def test_old_benchmarks_and_duplicate_home_generators_are_not_executable_routes(self):
        top_conditions=[ast.unparse(n.test) for n in TREE.body if isinstance(n,ast.If)]
        self.assertEqual(top_conditions.count("page == '🏠 Home'"),1)
        self.assertFalse(any(condition.startswith("page == '🏈 NFL'") for condition in top_conditions))
        self.assertFalse(any('show_bets' in condition or 'benchmark' in condition or 'accuracy_audit' in condition for condition in top_conditions))
        forbidden={'render_research_lab','render_stock_connection','run_nfl_accuracy_audit',
                   'test_connection','test_nba_connections','compare_v1_v2_walkforward'}
        loads={n.id for n in ast.walk(TREE) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        self.assertFalse(loads & forbidden)
        self.assertIn("page == '🎫 AI Ticket Builder'",top_conditions)
        self.assertIn("page == '📐 Model Accuracy'",top_conditions)

    def test_cleanup_did_not_leave_dangling_global_dependencies(self):
        table=symtable.symtable(SOURCE,'app.py','exec')
        bound={symbol.get_name() for symbol in table.get_symbols() if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}|set(dir(builtins))
        def missing(scope):
            found=[(scope.get_name(),symbol.get_name()) for symbol in scope.get_symbols()
                   if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in bound]
            return found+[item for child in scope.get_children() for item in missing(child)]
        self.assertEqual(missing(table),[])

    def test_accuracy_page_only_loads_saved_reports_on_request(self):
        screen=ui();screen.selectbox.return_value='NFL';load=MagicMock(return_value={'Report':{'graded_games':12}})
        scope=helpers({'render_model_accuracy'},st=screen,load_model_accuracy_reports=load)
        scope['render_model_accuracy']()
        load.assert_not_called()
        screen.button.return_value=True
        scope['render_model_accuracy']()
        load.assert_called_once_with('NFL')
        self.assertEqual(screen.session_state['accuracy_reports_NFL']['Report']['graded_games'],12)

    def test_nfl_accuracy_reads_current_grading_and_training_reports_without_writing(self):
        store=MagicMock();store.read.return_value={'status':'completed'}
        module=types.SimpleNamespace(Store=MagicMock(return_value=store),ROOT='sports/connected_winners_v1/')
        scope=helpers({'load_model_accuracy_reports'})
        with patch.dict('sys.modules',{'dual_agent.sports_connected_winner':module}):
            result=scope['load_model_accuracy_reports']('NFL')
        self.assertEqual([call.args[0] for call in store.read.call_args_list],
            ['sports/connected_winners_v1/nfl/latest_report.json.gz','sports/connected_winners_v1/nfl/training_report.json.gz'])
        store.write.assert_not_called()
        self.assertEqual(len(result),2)

    def test_opening_ncaa_does_not_load_training_reports_or_player_history(self):
        source=(ROOT/'dual_agent/ncaaf_station.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='render_station')
        screen=ui();screen.cache_resource.side_effect=lambda **kwargs:lambda function:function
        namespace={'_player_ui':MagicMock(side_effect=AssertionError('No player-report load on page open'))}
        with patch.dict('sys.modules',{'streamlit':screen}), \
             patch('requests.sessions.Session.request',side_effect=AssertionError('No request on page open')):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'ncaaf_station.py','exec'),namespace)
            namespace['render_station']()
        namespace['_player_ui'].assert_not_called()


if __name__=='__main__':unittest.main()
