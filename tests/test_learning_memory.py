"""All-sport memory integration, validation feedback and holdout isolation."""
import copy
from datetime import datetime,timedelta,timezone
import unittest
from unittest.mock import patch
from dual_agent import learning_memory as memory, experiment_controller as controller
from dual_agent import shared_experience

class Store:
    def __init__(self):self.data={};self.writes=[]
    def read(self,path):return copy.deepcopy(self.data.get(path))
    def write(self,path,value,replace=False):
        self.data[path]=copy.deepcopy(value);self.writes.append(path)

BASE=datetime(2026,1,1,18,tzinfo=timezone.utc)

def row(i,sport='NBA',p=.9,outcome=0):
    start=BASE+timedelta(days=i)
    return dict(base_id=sport+'-base',event_id=str(i),game_id=str(i),model_id=sport+'-base',
        start_time=start.isoformat(),captured_at=(start-timedelta(hours=1)).isoformat(),
        label_observed_at=(start+timedelta(hours=4)).isoformat(),outcome=outcome,
        original_probability=p,connected_probability=p,
        context={'rest_diff':1.,'injury':None},home_team='Home',away_team='Away',source='live')

def groups(sport='NBA',test_games=25):
    return [[row(i,sport) for i in range(80)], [row(i,sport) for i in range(80,110)],
            [row(i,sport,outcome=i%2) for i in range(110,110+test_games)]]

class MemoryTests(unittest.TestCase):
    def test_every_sport_retains_wins_losses_duplicates_and_earliest_capture(self):
        for sport in memory.SPORTS:
            with self.subTest(sport=sport):
                store=Store();records=[row(i,sport,p=.7,outcome=i%2) for i in range(8)]
                records.append(dict(records[0],event_id='alternate-provider-id',game_id='alternate-provider-id',
                    captured_at=(BASE-timedelta(minutes=30)).isoformat(),connected_probability=.8))
                with patch('requests.sessions.Session.request',side_effect=AssertionError('No provider calls')):
                    first=shared_experience.publish(sport,iter(records),store,'2026-11-26T18:00:00Z')
                    again=shared_experience.publish(sport,iter(records),store,'2026-11-26T19:00:00Z')
                self.assertEqual(first['learning_memory']['unique_model_games'],8)
                self.assertEqual(first['learning_memory']['wins'],4)
                self.assertEqual(first['learning_memory']['losses'],4)
                self.assertEqual(again['learning_memory']['new_or_earlier_records'],0)
                ledger=store.data[memory.ROOT+sport.lower()+'/outcomes.json.gz']
                self.assertEqual(len(ledger['records']),8)
                self.assertTrue(ledger['planner_must_not_read'])
                self.assertEqual(len([p for p in store.writes if p.endswith('outcomes.json.gz')]),1)

    def test_no_retrospective_future_or_ungrounded_outcomes_are_added(self):
        store=Store();valid=row(1)
        bad=[dict(valid,source='historical_reconstruction'),dict(valid,historical_reconstruction=True),
            dict(valid,connected_probability=float('nan')),dict(valid,outcome=2),
            dict(valid,captured_at=valid['start_time']),dict(valid,label_observed_at='2099-01-01T00:00:00Z')]
        result=memory.record_outcomes('NBA',bad,store,'2026-11-26T18:00:00Z')
        self.assertEqual(result['unique_model_games'],0)

    def test_lessons_update_while_training_waits_for_fresh_test_results_all_sports(self):
        for sport in memory.SPORTS:
            with self.subTest(sport=sport):
                store=Store();g=groups(sport,19);rows=sum(g,[])
                ctx,report=controller.prepare(sport,sport+'-base',rows,g,store)
                self.assertIsNone(ctx)
                self.assertEqual(report['status'],'waiting_for_fresh_test_results')
                learned=report['learning_memory']
                self.assertEqual(learned['games_retained'],110)
                self.assertTrue(learned['hypotheses'])
                self.assertFalse(learned['uses_current_test_labels'])
                self.assertTrue(any(p.endswith('development_latest.json.gz') for p in store.writes))

    def test_development_patterns_change_real_candidate_plan_for_every_sport(self):
        for sport in memory.SPORTS:
            with self.subTest(sport=sport):
                store=Store();g=groups(sport)
                context,report=controller.prepare(sport,sport+'-base',sum(g,[]),g,store)
                plain=dict(context,learning_memory={})
                adapter='MLB' if sport in ('NBA','MLB') else sport
                baseline=controller.plan(plain,adapter)
                planned=controller.plan(context,adapter)
                self.assertNotEqual(planned,baseline)
                self.assertEqual(len(planned),controller.MAX_TRIALS)
                self.assertEqual(planned[0],baseline[0])
                if adapter=='MLB':self.assertEqual(planned[-1]['scale'],.25)
                else:self.assertEqual(planned[-1]['gain_cap'],.25)
                self.assertNotIn('active',store.data)

    def test_current_heldout_results_cannot_change_lessons_or_candidates(self):
        store=Store();g=groups();ctx,_=controller.prepare('NBA','NBA-base',sum(g,[]),g,store)
        before=copy.deepcopy(ctx['learning_memory']);planned=controller.plan(ctx,'MLB')
        changed=copy.deepcopy(g)
        for r in changed[2]:
            r.update(outcome=1-r['outcome'],original_probability=.01,context={'postgame_score':999})
        after,_=controller.prepare('NBA','NBA-base',sum(changed,[]),changed,Store())
        self.assertEqual(before,after['learning_memory'])
        self.assertEqual(planned,controller.plan(after,'MLB'))
        entries=[v for k,v in store.data.items() if '/memory/development/' in k]
        self.assertEqual(len(entries[0]['records']),110)
        self.assertTrue(all(int(r['event_id'])<110 for r in entries[0]['records']))

    def test_rejected_experiments_survive_and_change_next_plan_without_test_metrics(self):
        store=Store();g=groups();ctx,_=controller.prepare('NBA','NBA-base',sum(g,[]),g,store)
        ctx['learning_memory']={}
        baseline=controller.plan(ctx,'MLB')
        config=baseline[-1]
        for i in range(2):
            context=dict(ctx,run_id='failed-'+str(i))
            trials=[dict(config=config,validation={'games':30,'accuracy':.4,'brier':.3,'log_loss':.8},
                validation_rejections=['no_validation_accuracy_gain'],status='rejected_on_validation',
                test={'accuracy':1.,'secret':'DO_NOT_PLAN_FROM_THIS'})]
            memory.remember_experiment(context,trials,{'accuracy_regression':True,'test_secret':123})
        stored=store.data[ctx['prefix']+'memory/validation_experiments.json.gz']
        self.assertEqual(len(stored['runs']),2)
        self.assertNotIn('DO_NOT_PLAN_FROM_THIS',str(stored))
        self.assertNotIn('test_secret',str(stored))
        memory_report=memory.development('NBA','NBA-base',g,store,ctx['prefix'])
        memory_report['hypotheses']=[]  # Isolate rejected-experiment feedback.
        next_context=dict(ctx,learning_memory=memory_report)
        self.assertNotEqual(controller.plan(next_context,'MLB'),baseline)
        self.assertEqual(memory_report['prior_validation_experiments'],2)

    def test_audit_and_other_cohort_data_cannot_influence_planning(self):
        g=groups();store=Store()
        store.data[memory.ROOT+'nba/outcomes.json.gz']={'records':{'tainted':{'test_accuracy':1.}}}
        ctx,_=controller.prepare('NBA','NBA-base',sum(g,[]),g,store)
        other=Store();ctx2,_=controller.prepare('NBA','NBA-base',sum(g,[]),g,other)
        self.assertEqual(controller.plan(ctx,'MLB'),controller.plan(ctx2,'MLB'))
        contaminated=copy.deepcopy(g)
        for r in contaminated[0]+contaminated[1]:r['base_id']='other-model'
        isolated=memory.development('NBA','NBA-base',contaminated,store,'isolated/')
        self.assertEqual(isolated['games_retained'],0)
        self.assertEqual(isolated['hypotheses'],[])

    def test_fresh_test_claim_cannot_be_reused_by_new_memory(self):
        store=Store();g=groups();ctx,_=controller.prepare('NBA','NBA-base',sum(g,[]),g,store)
        controller.claim_test(ctx)
        prior_state=copy.deepcopy(store.data[ctx['prefix']+'state.json.gz'])
        again,report=controller.prepare('NBA','NBA-base',sum(g,[]),g,store)
        self.assertIsNone(again)
        self.assertEqual(report['status'],'unchanged_experiment_data_skipped')
        self.assertEqual(prior_state,store.data[ctx['prefix']+'state.json.gz'])


class EarlyContextMemoryTests(unittest.TestCase):
    def test_rejected_early_adjustment_is_retained_and_guides_next_attempt(self):
        from dual_agent import nba_context_residual as early
        class Client(Store):
            def read(self,path,compressed=False):return super().read(path)
            def write(self,path,value,compressed=False,replace=False):return super().write(path,value,replace)
        client=Client()
        g={name:[dict(game_id=r['event_id'],forecast_time=r['captured_at'],
            input_time=r['captured_at'],start_time=r['start_time'],label_observed_at=r['label_observed_at'],
            base_model_id=early.BASE_MODEL,features={},base_probability=.6,label=int(r['event_id'])%2)
            for r in rows] for name,rows in zip(('train','validation','test'),groups())}
        baseline=dict(games=30,accuracy=.5,brier=.26,log_loss=.72)
        report=dict(validation_neural_base=baseline,penalty_trials=[dict(penalty=1.,validation=baseline)])
        with patch.object(early,'collect_rows',return_value=(sum(g.values(),[]),{})), \
             patch.object(early,'partition',return_value=g),patch.object(early,'derived'), \
             patch.object(early,'fit',return_value=(None,report)):
            result=early.run_early(client,datetime(2026,11,26,tzinfo=timezone.utc))
        self.assertEqual(result['status'],'early_adjustment_no_validation_gain')
        context,lessons,penalties=early.early_memory(client,g)
        self.assertEqual(lessons['prior_validation_experiments'],1)
        memory.remember_experiment(dict(context,run_id='second-distinct-run'),
            [dict(config={'penalty':1.},validation=baseline,validation_rejections=['brier_not_improved'])],{})
        context,lessons,penalties=early.early_memory(client,g)
        self.assertEqual(penalties,[1.,10.,100.,300.])
        changed=copy.deepcopy(g)
        for r in changed['test']:r['label']=1-r['label'];r['base_probability']=.99
        self.assertEqual(early.early_memory(client,changed)[2],penalties)
        self.assertFalse(result['live_model_changed'])

if __name__=='__main__':unittest.main()
