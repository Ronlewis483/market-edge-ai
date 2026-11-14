import unittest
from datetime import datetime, timezone
from dual_agent.ncaaf_selection import decision,tag,summarize,weekly_winner,render_panel
from dual_agent.ncaaf_storage import freeze_games

class Store:
    def __init__(self):self.values={}
    def write(self,path,value,replace=False):
        if path in self.values and not replace:return False
        self.values[path]=value;return True

class SelectionTests(unittest.TestCase):
    def game(self,p=.7,model='m'):
        return dict(game_id='1',model_id=model,home_id='a',away_id='b',home_team='A',away_team='B',
            commence_time='2026-11-21T18:00:00Z',model_home_probability=p,model_confidence=max(p,1-p),
            predicted_team='A' if p>=.5 else 'B',home_probability=.6,moneylines=[{'book':'test'}])
    def graded(self,p=.7,model='m'):
        return dict(tag(self.game(p,model)),captured_at='2026-11-21T17:00:00Z',status='Graded',home_won=1)
    def test_fixed_rule_both_sides(self):
        self.assertTrue(decision(.65)['selected']);self.assertTrue(decision(.35)['selected'])
        self.assertFalse(decision(.64)['selected'])
        for p in [None,float('nan'),0,1]:
            with self.assertRaises(ValueError):decision(p)
    def test_forecast_freezing_tags_once_before_kickoff(self):
        store=Store();now=datetime(2026,11,21,17,tzinfo=timezone.utc)
        self.assertEqual(freeze_games([self.game()],store,now),1)
        self.assertEqual(freeze_games([self.game()],store,now),0)
        self.assertTrue(next(iter(store.values.values()))['winner_selection']['selected'])
        self.assertEqual(freeze_games([self.game(model='later')],store,datetime(2026,11,21,19,tzinfo=timezone.utc)),0)
    def test_no_old_forecasts_duplicates_or_historical_counts(self):
        a=self.graded();old=dict(a);old.pop('winner_selection')
        bad=dict(a,captured_at='2026-11-21T19:00:00Z')
        historical=dict(a,historical_reconstruction=True)
        result=summarize([a,a,old,bad,historical])
        self.assertEqual(result['by_model']['m']['selected']['games'],1)
    def test_cohorts_separate_and_market_comparison_same_games(self):
        a=self.graded();b=dict(self.graded(.3),game_id='2',home_won=0,selection_market_home_probability=None)
        c=self.graded(model='other')
        result=summarize([a,b,c]);self.assertEqual(len(result['by_model']),2)
        r=result['by_model']['m'];self.assertEqual(r['selected']['games'],2)
        self.assertEqual(r['matched_market']['model']['games'],1)
        self.assertEqual(r['matched_market']['favorite_accuracy'],1)
        self.assertEqual(r['missing_market_games'],1)
    def test_missing_quote_is_not_fabricated(self):
        self.assertIsNone(tag(dict(self.game(),moneylines=[]))['selection_market_home_probability'])
    def test_weekly_gate_and_wait_retry(self):
        memory={};read=lambda key,default:memory.get(key,default)
        write=lambda key,value:memory.update({key:value})
        at=datetime(2026,11,15,18,tzinfo=timezone.utc)
        _,r=weekly_winner(read,write,lambda:('old',{'status':'waiting'}),'old',at)
        self.assertFalse(memory)
        _,r=weekly_winner(read,write,lambda:('new',{'candidate_id':'c'}),'old',at)
        def fail():raise AssertionError('must not retrain this week')
        winner,r=weekly_winner(read,write,fail,'new',datetime(2026,11,16,5,59,tzinfo=timezone.utc))
        self.assertEqual(winner,'new');self.assertEqual(r['status'],'weekly_training_not_due')
        winner,r=weekly_winner(read,write,lambda:('next',{'candidate_id':'d'}),'new',datetime(2026,11,16,6,tzinfo=timezone.utc))
        self.assertEqual(winner,'next')
    def test_tagged_forecast_is_graded_and_included_in_selection_report(self):
        import os,tempfile
        from dual_agent.ncaaf_prediction import update_forecast_ledger
        from pathlib import Path
        import json
        original=os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                game=tag(self.game())
                update_forecast_ledger([game],[],datetime(2026,11,21,17,tzinfo=timezone.utc))
                final=dict(id='1',completed=True,startDate=game['commence_time'],homeId='a',awayId='b',homePoints=28,awayPoints=14)
                update_forecast_ledger([],[final],datetime(2026,11,21,23,tzinfo=timezone.utc))
                rows=json.loads(Path('ncaaf_history_cache/forecast_results_v1.json').read_text())['records'].values()
                r=summarize(rows)['by_model']['m']
                self.assertEqual(r['selected']['correct'],1)
                self.assertEqual(r['matched_market']['favorite_correct'],1)
            finally:os.chdir(original)

    def test_panel_filters_both_sides(self):
        class UI:
            def markdown(self,*a,**k):pass
            def caption(self,*a,**k):pass
            def info(self,*a,**k):pass
            def dataframe(self,frame,**kwargs):self.frame=frame
        ui=UI();render_panel(ui,[self.game(.7),self.game(.3),self.game(.55)])
        self.assertEqual(len(ui.frame),2)

if __name__=='__main__':unittest.main()
