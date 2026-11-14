"""Offline matchup checks: temporal integrity, opponent effects and serving parity."""
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from dual_agent import nba_matchup_context as matchup
from dual_agent import nba_unified_winner as winner
from dual_agent import nba_live_feedback as feedback


def history():
    games = []
    for i in range(6):
        for team, opponent in [('h', 'x'), ('a', 'y')]:
            box = dict(fgm=40., fg3m=10., fga=85., fta=20., turnover=12., reb=45., poss=100.)
            games.append(dict(id=f'{team}{i}', date=f'2026-01-{i+1:02}', season=2025,
                              teams=[team, opponent], scores={team:110., opponent:100.},
                              boxes={team:copy.deepcopy(box), opponent:copy.deepcopy(box)}))
    return games


class MatchupTests(unittest.TestCase):
    def test_target_day_and_future_results_cannot_change_features(self):
        games = history()
        expected = matchup.features(games, 'h', 'a', '2026-01-07T20:00Z')
        future = copy.deepcopy(games[-2]);future.update(id='future', date='2026-01-07')
        future['scores']['h'] = 200.
        self.assertEqual(expected, matchup.features(games+[future], 'h', 'a', '2026-01-07T20:00Z'))
        future['date'] = '2026-01-08'
        self.assertEqual(expected, matchup.features(games+[future], 'h', 'a', '2026-01-07T20:00Z'))

    def test_same_offense_faces_a_different_defense(self):
        games = history()
        baseline = matchup.features(games, 'h', 'a', '2026-01-07')
        for game in games:
            if 'a' in game['teams']:
                game['boxes']['y']['fgm'] = 45.
                game['boxes']['y']['fta'] = 30.
                game['boxes']['y']['turnover'] = 20.
                game['scores']['y'] = 120.
        changed = matchup.features(games, 'h', 'a', '2026-01-07')
        self.assertGreater(changed['home_matchup_shooting'], baseline['home_matchup_shooting'])
        self.assertGreater(changed['home_matchup_free_throws'], baseline['home_matchup_free_throws'])
        self.assertGreater(changed['home_matchup_turnovers'], baseline['home_matchup_turnovers'])
        self.assertGreater(changed['home_matchup_scoring'], baseline['home_matchup_scoring'])

    def test_swapping_teams_reverses_every_advantage(self):
        games=history();games[0]['boxes']['h']['reb']=60.
        h=matchup.features(games,'h','a','2026-01-07')
        a=matchup.features(games,'a','h','2026-01-07')
        for metric in matchup.METRICS:
            self.assertAlmostEqual(h['difference_matchup_'+metric], -a['difference_matchup_'+metric])
            self.assertEqual(h['home_matchup_'+metric], a['away_matchup_'+metric])

    def test_incomplete_stale_or_previous_season_boxes_are_not_imputed(self):
        games=history()
        self.assertIsNone(matchup.features(games,'h','a','2026-02-01'))
        for g in games:
            if 'h' in g['teams'] and g['date']<'2026-01-04':g['boxes']=None
        self.assertIsNone(matchup.features(games,'h','a','2026-01-07'))
        games=history()
        for g in games:
            if g['date']<'2026-01-04':g['season']=2024
        self.assertIsNone(matchup.features(games,'h','a','2026-01-07'))

    def test_invalid_box_is_missing_not_a_zero_stat(self):
        games=history()
        for g in games:
            if 'h' in g['teams']:g['boxes']['h']['fga']=float('nan')
        self.assertIsNone(matchup.features(games,'h','a','2026-01-07'))

    def test_inputs_support_old_artifacts_without_fabricating_new_columns(self):
        old={k:1. for k in winner.PLAYER_FEATURES};old['neural_probability']=.6
        columns=winner.PLAYER_FEATURES+['neural_logit_gap','original_probability']
        values=winner.correction_inputs(pd.DataFrame([old]),[.55],columns)
        self.assertEqual(list(values.columns),columns)
        with self.assertRaises(KeyError):winner.correction_inputs(pd.DataFrame([old]),[.55])

    def test_training_and_live_use_identical_matchup_values(self):
        games=history()
        now=pd.Timestamp.now(tz='UTC');start=now+pd.Timedelta(days=1)
        # Move history to immediately before tomorrow's prediction.
        for g in games:g['date']=(start-pd.Timedelta(days=7-int(g['id'][1:]))).date().isoformat()
        expected=matchup.features(games,'h','a',start)
        target=dict(games[-2],id='target',date=start.date().isoformat(),
                    teams=['h','a'],scores={'h':110.,'a':100.})
        def base_rows(records):
            return pd.DataFrame([{'game_id':'target','game_date':start.date().isoformat(),
                                  'home_win':1}])
        historical=SimpleNamespace(research_functions=lambda:{
            'build_balldontlie_pregame_features':base_rows,'NBA_LIVE_FEATURE_COLUMNS':[]})
        neural=SimpleNamespace(load_model=lambda:{'meta':{'calibration_end':'2020-01-01'}},
                               predict=lambda *args:{'learning_home_probability':.6})
        with patch.dict(sys.modules,{'dual_agent.nba_historical_pregame_learning':historical,
                                     'dual_agent.nba_winner_comparison':neural}), patch.object(
                                         winner,'player_features',return_value={k:1. for k in winner.PLAYER_FEATURES}):
            training,_,_=winner.training_rows(games+[target])
        for k,v in expected.items():self.assertEqual(training.iloc[0][k],v)
        extra={k:1. for k in winner.PLAYER_FEATURES}
        columns=winner.FEATURES+['neural_logit_gap','original_probability']
        model=dict(initial_logit=0.,learning_rate=1.,uses_original_offset=True,trees=[])
        artifact=dict(schema=2,trained_at=(now-pd.Timedelta(days=1)).isoformat(),
                      context_refreshed_at=now.isoformat(),aliases={'home':'h','away':'a'},
                      games=games,columns=columns,tree_model=model,correction_gain=1.,
                      model_id='matchup',checks_passed=True,matchup_version=1)
        with patch.object(winner,'player_features',return_value=extra), patch(
                'dual_agent.shared_experience.attach',return_value={}):
            result=winner.predict(artifact,pd.DataFrame([{'rest_days_diff':2.}]),.6,.6,'Home','Away',start)
        for k,v in expected.items():self.assertEqual(result['experience_context'][k],v)
        self.assertEqual(result['combined_home_probability'],.6)
        self.assertEqual(result['feedback_context']['team_rest_days_diff'],2.)

    def test_frozen_learning_records_matchup_and_real_nba_team_fields(self):
        context=feedback.frozen_context({'home_matchup_shooting':.3,'outcome':1},
                                        {'margin_diff':4.,'rest_days_diff':2.,'home_score':130.})
        self.assertEqual(context['home_matchup_shooting'],.3)
        self.assertEqual(context['team_margin_diff'],4.)
        self.assertNotIn('outcome',context)
        self.assertNotIn('home_score',context)


if __name__=='__main__':unittest.main()
