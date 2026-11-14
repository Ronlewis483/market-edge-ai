import unittest
from unittest.mock import patch
from dual_agent.nfl_selection import decision, summarize, RULE
from dual_agent.sports_connected_winner import nfl_context
import pandas as pd

class SelectionTests(unittest.TestCase):
    def row(self,p=.7,outcome=1):
        return dict(event_id='1',base_id='b',home_team='DAL',away_team='PHI',
            captured_at='2026-11-15T17:00:00Z',start_time='2026-11-15T18:00:00Z',
            original_probability=p,connected_probability=p,outcome=outcome,
            winner_selection=decision(p),context={'market_home_probability':.6})

    def test_fixed_threshold_both_sides(self):
        self.assertTrue(decision(.65)['selected'])
        self.assertTrue(decision(.35)['selected'])
        self.assertFalse(decision(.64)['selected'])
        self.assertFalse(decision(.5)['selected'])

    def test_no_retroactive_tagging_or_duplicate_counts(self):
        row=self.row(); old=dict(row);old.pop('winner_selection')
        invalid=dict(row,captured_at='2026-11-15T19:00:00Z')
        historical=dict(row,source='historical_reconstruction')
        result=summarize([row,row,old,invalid,historical])
        self.assertEqual(result['selected']['games'],1)
        self.assertEqual(result['selected']['correct'],1)

    def test_same_games_baseline_and_missing_market(self):
        a=self.row(.7,1);b=dict(self.row(.3,0),event_id='2',away_team='NYG',context={})
        c=dict(self.row(.55,0),event_id='3',away_team='BUF')
        r=summarize([a,b,c])
        self.assertEqual(r['all_tagged']['games'],3)
        self.assertEqual(r['selected']['games'],2)
        self.assertEqual(r['matched_market']['model']['games'],1)
        self.assertEqual(r['matched_market']['favorite_accuracy'],1)

    def test_panel_displays_both_sides_and_excludes_lower_confidence(self):
        from dual_agent.nfl_selection import render_panel
        class UI:
            def markdown(self,*a,**k): pass
            def caption(self,*a,**k): pass
            def info(self,*a,**k): pass
            def dataframe(self,frame,**kwargs): self.frame=frame
        ui=UI()
        render_panel(ui,[dict(home_win_probability=p,predicted_team=str(p)) for p in [.7,.3,.55]])
        self.assertEqual(set(ui.frame.predicted_team),{'0.7','0.3'})

    def test_altered_rule_is_excluded(self):
        row=self.row();row['winner_selection']['threshold']=.6
        self.assertEqual(summarize([row])['selected']['games'],0)

    def test_fresh_market_reaches_context_and_stale_does_not(self):
        now=pd.Timestamp.now(tz='UTC')
        context=dict(fetched_at=now.isoformat(),market=dict(home_probability=.6,
                     fetched_at=now.isoformat(),books=[{}]))
        self.assertEqual(nfl_context({},context)['market_home_probability'],.6)
        context['market']['fetched_at']=(now-pd.Timedelta(days=1)).isoformat()
        self.assertIsNone(nfl_context({},context)['market_home_probability'])

if __name__=='__main__':unittest.main()
