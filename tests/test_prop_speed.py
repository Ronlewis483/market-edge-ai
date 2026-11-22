"""Bulk display and scoring parity without live API calls."""
import ast
import copy
import importlib.util
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from dual_agent import sports_player_prop_predictions as props

ROOT=Path(__file__).resolve().parents[1]

def fixture(players=8, game_count=120):
    now=pd.Timestamp.now(tz='UTC'); start=now+pd.Timedelta(hours=4)
    teams={1:{'id':1,'full_name':'Home'},2:{'id':2,'full_name':'Away'}}
    games=[dict(id=i,date=(now-pd.Timedelta(days=i+1)).isoformat(),status='Final',
        home_team=teams[1],visitor_team=teams[2],home_team_score=110,visitor_team_score=108) for i in range(game_count)]
    context=dict(teams=teams,games=games,injuries={},player_teams={},injury_complete=True)
    histories={};quotes=[]
    for pid in range(players):
        name=f'Player {pid}';context['player_teams'][pid]=1
        histories[name]=[dict(player={'id':pid},team=teams[1],game=dict(games[i],season_type='regular'),
            source_date=games[i]['date'],history_provider='balldontlie',history_checked_at=now.isoformat(),
            min='30:00',pts=20+i%4,reb=5+i%3,ast=4+i%2,fg3m=2,blk=1,stl=1,turnover=2) for i in reversed(range(30))]
        for market in props.MARKETS['NBA']:
            quotes.append(dict(event_id='event',player=name,market=market,line=3.5,book='draftkings',
                start_time=start.isoformat(),capture_time=now.isoformat(),updated_at=now.isoformat(),
                home_team='Home',away_team='Away',over_probability=.5,over_decimal_odds=1.9,under_decimal_odds=1.9))
    game=dict(home_team='Home',away_team='Away',commence_time=start.isoformat())
    groups={ (q['event_id'],q['player'],q['market'],q['line']):[q] for q in quotes}
    return [game],quotes,histories,context,groups

class SpeedTests(unittest.TestCase):
    def test_defense_cache_reuses_scan_but_preserves_result(self):
        _,quotes,histories,context,_=fixture()
        rows=next(iter(histories.values()));cutoff=pd.Timestamp(quotes[0]['capture_time'])
        expected=props._nba_points_defense(rows,quotes[0],context,cutoff)
        with patch.object(props,'_nba_points_defense',wraps=props._nba_points_defense) as scan:
            for _ in range(24):self.assertEqual(props.nba_points_defense(rows,quotes[0],context,cutoff),expected)
        self.assertEqual(scan.call_count,1)
        self.assertEqual(props.nba_points_defense(rows,quotes[0],dict(context,_points_defense_cache={}),cutoff),expected)

    def test_all_cached_players_reach_first_scored_publication(self):
        games,quotes,histories,context,_=fixture(12)
        updates=[]
        with patch.object(props,'fetch_prop_quotes',return_value=(quotes,[])), \
             patch.object(props,'load_nba_matchup_context',return_value=context), \
             patch.object(props,'nba_cached_histories',return_value=histories), \
             patch.object(props,'_nba_history_batches',return_value=iter(())), \
             patch('dual_agent.nba_star_spotlight.collect_star_rosters',return_value={}):
            result=props.generate_player_prop_picks('NBA',games,'odds',nba_key='nba',progress=updates.append)
        first=next(u for u in updates if u.get('game_picks'))
        self.assertEqual(len({p['Player'] for p in first['game_picks']}),12)
        self.assertEqual(len({p['Player'] for p in result['game_picks']}),12)

    def test_scoring_cache_does_not_mutate_shared_context(self):
        games,quotes,histories,context,groups=fixture(2)
        original=copy.deepcopy(context)
        props._rank_prop_candidates('NBA',games,groups,histories,[],{},{},{},nba_context=context)
        self.assertEqual(context,original)

    def test_fragment_polling_does_not_start_network_or_workers(self):
        tree=ast.parse((ROOT/'app.py').read_text())
        for name in ('render_cross_prop_cards','render_nba_live_prop_cards'):
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            ui=MagicMock();render=MagicMock();ui.fragment.return_value=render
            ns={'st':ui,'_render_cross_prop_cards':MagicMock(),'_render_nba_live_prop_cards':MagicMock()}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'app.py','exec'),ns)
            ns[name](*(['NFL',{},{}] if name=='render_cross_prop_cards' else [{},{}]))
            ui.fragment.assert_not_called()
            ns['_render_cross_prop_cards' if name=='render_cross_prop_cards' else '_render_nba_live_prop_cards'].assert_called_once()

    def test_nfl_alternate_lines_reuse_one_historical_sample(self):
        from dual_agent import nfl_player_prop_prediction_pipeline as pipeline
        now=pd.Timestamp.now(tz='UTC')
        lines=pd.DataFrame([dict(event_id='game',home_team='Home',away_team='Away',
            game_time=now+pd.Timedelta(hours=4),player='Receiver',market_key='player_receptions',
            line=line,side='Over',bookmaker_key='draftkings',american_odds=-110,last_update=now.isoformat())
            for line in (1.5,2.5,3.5,4.5,5.5)])
        history=pd.DataFrame([dict(player='Receiver',market='Receptions',
            game_time=now-pd.Timedelta(days=7*(i+1)),value=4+i%3) for i in range(6)])
        with patch.object(pipeline,'_historical_player_games',wraps=pipeline._historical_player_games) as lookup:
            result=pipeline.run_nfl_player_prop_prediction_pipeline(lines,history,
                horizon_days=7,retain_all_qualified=True,include_value_candidates=True)
        self.assertEqual(len(result),5)
        self.assertEqual(lookup.call_count,1)

    def test_table_default_has_no_top_five_truncation(self):
        tree=ast.parse((ROOT/'app.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render_prop_comparisons')
        ui=MagicMock();ui.session_state={};ui.checkbox.return_value=False
        pairs=[dict(player=f'Player {i}',market='Points',line=20.5,binary=False,source='player_history',sides={'Over':{'_offered':True,'Estimated chance':.7,'Best sportsbook':'draftkings','Best decimal odds':1.9}}) for i in range(80)]
        ns={'st':ui,'pd':pd,'prop_comparisons':lambda *args:pairs}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'app.py','exec'),ns)
        ns['render_prop_comparisons']([],scope='test')
        self.assertEqual(len(ui.dataframe.call_args.args[0]),80)
        ui.container.assert_not_called()

if __name__=='__main__':unittest.main()
