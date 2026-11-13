"""Broader listed-prop eligibility without fabricated odds or outcomes."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from dual_agent import nfl_player_prop_prediction_pipeline as pipeline


def quotes(book='prizepicks',price=None):
    now=pd.Timestamp.now(tz='UTC')
    return pd.DataFrame([dict(event_id='game',home_team='New York Giants',away_team='Washington Commanders',
        game_time=now+pd.Timedelta(hours=4),player='Example Receiver',market_key='player_receptions',
        line=3.5,side='Over',bookmaker_key=book,american_odds=price,last_update=now.isoformat())])


def history(count=6):
    now=pd.Timestamp.now(tz='UTC')
    return pd.DataFrame([dict(player='Example Receiver',market='Receptions',
        game_time=now-pd.Timedelta(days=7*(i+1)),value=4+i%3) for i in range(count)])


class PropEligibilityTests(unittest.TestCase):
    def test_six_games_generate_a_labeled_prop_without_per_leg_pickem_odds(self):
        with patch('dual_agent.player_prop_regression.estimate',side_effect=AssertionError('Short sample must not invoke eight-game regression')) as trained, \
             patch('requests.sessions.Session.request',side_effect=AssertionError('Offline check')):
            result=pipeline.run_nfl_player_prop_prediction_pipeline(quotes(),history(),
                horizon_days=7,retain_all_qualified=True,include_value_candidates=True)
        self.assertEqual(len(result),1)
        row=result.iloc[0]
        self.assertEqual(row.sample_size,6)
        self.assertEqual(row.non_push_games,6)
        self.assertIn('Limited history',row.history_reliability)
        self.assertEqual(row.confidence_group,'LIMITED HISTORY')
        self.assertEqual(row.best_sportsbook,'prizepicks')
        self.assertTrue(pd.isna(row.best_decimal_odds))
        self.assertTrue(pd.isna(row.estimated_return_per_unit))
        self.assertTrue(pd.isna(row.break_even_cover_chance))
        self.assertAlmostEqual(row.historical_support,7/8)
        self.assertTrue(row.outcome_comparison[0]['listed'])
        self.assertIsNone(row.outcome_comparison[0]['positive_value'])
        trained.assert_not_called()

    def test_underdog_accepts_missing_price_column_but_sportsbook_does_not(self):
        frame=quotes('underdog').drop(columns='american_odds')
        self.assertEqual(len(pipeline._select_consensus_props(frame)),1)
        frame['bookmaker_key']='draftkings'
        self.assertTrue(pipeline._select_consensus_props(frame).empty)

    def test_invalid_supplied_prices_do_not_become_unpriced_listings(self):
        for bad in (0,50,'bad'):
            self.assertTrue(pipeline._select_consensus_props(quotes(price=bad)).empty)

    def test_five_total_games_and_insufficient_nonpush_evidence_remain_excluded(self):
        self.assertIsNone(pipeline._estimate_prop(np.arange(5),3.5,'Receptions'))
        self.assertIsNone(pipeline._estimate_prop([3,3,3,3,2,4],3,'Receptions'))
        self.assertIsNone(pipeline._estimate_prop([3]*12,3,'Receptions'))

    def test_twelve_appearances_with_seven_decisions_no_longer_get_discarded(self):
        estimate=pipeline._estimate_prop([3]*5+[0,1,2,4,5,6,7],3,'Receptions',side='OVER')
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate['non_push_games'],7)
        self.assertIn('Limited history',estimate['history_reliability'])
        self.assertAlmostEqual(estimate['historical_support'],5/9)

    def test_existing_full_sample_and_actual_prices_keep_the_same_calculations(self):
        estimate=pipeline._estimate_prop(np.arange(12),5.5,'Receptions',side='OVER')
        self.assertEqual(estimate['sample_size'],12)
        self.assertAlmostEqual(estimate['historical_support'],.5)
        outcomes=pipeline._compare_prop_outcomes(np.arange(12),5.5,'Receptions',
            {'OVER':[{'bookmaker_key':'draftkings','american_odds':-110,'last_update':'unused'}]})
        self.assertTrue(outcomes[0]['qualifies'])
        self.assertFalse(outcomes[0]['positive_value'])

    def test_expired_quotes_unlisted_sides_and_disallowed_platforms_stay_excluded(self):
        frame=quotes()
        frame['last_update']=(pd.Timestamp.now(tz='UTC')-pd.Timedelta(minutes=31)).isoformat()
        self.assertTrue(pipeline._select_consensus_props(frame).empty)
        for change in ({'side':'PASS'},{'bookmaker_key':'unconfigured'}):
            self.assertTrue(pipeline._select_consensus_props(quotes().assign(**change)).empty)

    def test_new_unpriced_props_can_reach_a_complete_distinct_player_ticket(self):
        from test_ticket_builder import app_helpers
        lines=pd.concat([quotes().assign(player='Receiver '+str(i)) for i in range(3)],ignore_index=True)
        samples=pd.concat([history().assign(player='Receiver '+str(i)) for i in range(3)],ignore_index=True)
        with patch('requests.sessions.Session.request',side_effect=AssertionError('No live provider requests')):
            result=pipeline.run_nfl_player_prop_prediction_pipeline(lines,samples,
                horizon_days=7,retain_all_qualified=True,include_value_candidates=True)
            app=app_helpers()
            candidates=app['ticket_prop_candidates']({'NFL':{'nfl_player_props':result}})
            ticket=app['build_suggested_ticket'](candidates,'prizepicks',3)
        self.assertEqual(len(ticket),3)
        self.assertEqual(len({leg['player'] for leg in ticket}),3)
        self.assertTrue(all(leg['decimal_odds'] is None for leg in ticket))
        self.assertTrue(all('Limited history' in leg['history_reliability'] for leg in ticket))


if __name__=='__main__':unittest.main()
