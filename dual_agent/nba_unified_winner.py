"""Connected NBA winner: original probability -> context -> learned correction.

The original forecast is a fixed log-odds offset. Existing neural information
and prior-player projections jointly enter a residual learning layer. There
are no independent player votes or probability averages. Saved data only.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd

from .nba_winner_box_context import compact_games, partition_by_game_count, scores, key, day

ROOT = 'nba/connected_winner_v2/'
ARTIFACT = ROOT + 'latest.json.gz'
STATS = ('pts', 'reb', 'ast', 'stl', 'blk', 'turnover')
HALF_LIVES = dict(zip(STATS, (21., 45., 30., 60., 60., 45.)))
PLAYER_FIELDS = tuple('projected_' + s for s in STATS) + ('top2_production', 'role_support', 'minute_uncertainty')
FEATURES = [side + '_' + f for side in ('home', 'away', 'difference') for f in PLAYER_FIELDS]


def add_players(games, samples):
    from .nba_pregame_snapshots import minutes, number
    for g in games:
        rows = samples.get(g['id'])
        if rows is None or not g.get('boxes'):
            g['players'] = None
            continue
        players = []
        for r in rows:
            minute = minutes(r.get('min'))
            if minute is None or minute <= 0:
                continue
            pid, tid = (r.get('player') or {}).get('id'), str((r.get('team') or {}).get('id'))
            if pid is None or tid not in g['teams'] or any(number(r.get(s)) is None for s in STATS):
                continue
            players.append(dict(id=str(pid), team=tid, minutes=float(minute),
                                **{s: float(r[s]) for s in STATS}))
        g['players'] = players or None
    return games


def capped_minutes(roles):
    """Conditional replacement allocation: 240 minutes, no player over 48."""
    if len(roles) < 5 or sum(roles) <= 0:
        return None
    result = np.zeros(len(roles))
    remaining = list(range(len(roles)))
    while remaining:
        budget = 240. - float(result.sum())
        mass = sum(roles[i] for i in remaining)
        proposed = {i: budget * roles[i] / mass for i in remaining}
        capped = [i for i in remaining if proposed[i] > 48.]
        if not capped:
            for i in remaining:
                result[i] = proposed[i]
            break
        for i in capped:
            result[i] = 48.
        remaining = [i for i in remaining if i not in capped]
    return result


def player_projection(games, team, cutoff, availability=None):
    """Identical historical/live projections; never use target-day player boxes.

    Roles react faster than production. Production estimates shrink toward a
    longer current-team history when the recent sample is small. Half-lives are
    fixed design choices, not claimed to be statistically optimized.
    """
    team, cutoff = str(team), pd.Timestamp(day(cutoff))
    prior = [g for g in games if team in g['teams'] and g['date'] < cutoff.date().isoformat()]
    prior.sort(key=lambda g: (g['date'], g['id']))
    if not prior or (cutoff - pd.Timestamp(prior[-1]['date'])).days > 10:
        return None
    segment = []
    newer = prior[-1]
    for g in reversed(prior):
        if g['season'] != prior[-1]['season'] or (pd.Timestamp(newer['date'])-pd.Timestamp(g['date'])).days > 45:
            break
        segment.append(g); newer = g
        if len(segment) >= 30:
            break
    if sum(bool(g.get('players')) for g in segment[:8]) < 4:
        return None
    complete = [g for g in reversed(segment) if g.get('players')]
    # Roster is inferred only from prior appearances; no future actual lineup.
    roster = {r['id'] for g in complete[-3:] for r in g['players'] if r['team'] == team}
    if availability is not None:
        roster &= set(availability['roster'])
        roster -= set(availability['out'])
    projected = []
    for pid in sorted(roster):
        observations = []
        roles = []
        for g in complete:
            r = next((r for r in g['players'] if r['team'] == team and r['id'] == pid), None)
            age = (cutoff - pd.Timestamp(g['date'])).days
            # Complete box absence informs a role estimate, never an injury label.
            roles.append((0. if r is None else r['minutes'], 2. ** (-age/7.)))
            if r is not None:
                observations.append((r, age))
        total_minutes = sum(r['minutes'] for r, _ in observations)
        if len(observations) < 2 or total_minutes < 40:
            continue
        mass = sum(w for _, w in roles)
        role = sum(v*w for v, w in roles)/mass
        if role <= 0:
            continue
        rates = {}
        for stat in STATS:
            half = HALF_LIVES[stat]
            denominator = sum(r['minutes'] * 2.**(-age/half) for r, age in observations)
            fast = sum(r[stat] * 2.**(-age/half) for r, age in observations)/denominator
            slow = sum(r[stat] for r, _ in observations)/total_minutes
            support = denominator/(denominator+200.)
            rates[stat] = support*fast+(1-support)*slow
        variance = sum(w*(v-role)**2 for v, w in roles)/mass
        projected.append((pid, role, rates, math.sqrt(variance), len(observations)))
    allocation = capped_minutes([r[1] for r in projected])
    if allocation is None:
        return None
    values = {'projected_'+s: sum(m*r[2][s] for m,r in zip(allocation,projected)) for s in STATS}
    # Production proxy, not a causal plus-minus or learned player-impact rating.
    production = sorted((r[2]['pts']+r[2]['reb']+r[2]['ast']-r[2]['turnover'])*m for m,r in zip(allocation,projected))
    values.update(top2_production=sum(production[-2:]),
                  role_support=float(np.mean([r[4] for r in projected])),
                  minute_uncertainty=sum(m*r[3] for m,r in zip(allocation,projected))/240.)
    return values


def player_features(games, home, away, cutoff, availability=None):
    h = player_projection(games, home, cutoff, None if availability is None else availability.get(str(home)))
    a = player_projection(games, away, cutoff, None if availability is None else availability.get(str(away)))
    if h is None or a is None:
        return None
    return {**{'home_'+k:h[k] for k in PLAYER_FIELDS}, **{'away_'+k:a[k] for k in PLAYER_FIELDS},
            **{'difference_'+k:h[k]-a[k] for k in PLAYER_FIELDS}}


def tree_probability(model, values, gain=1., raw=False):
    offset_probability = float(np.asarray(values)[-1])
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Invalid unified winner features')
    logit = model['initial_logit']
    for tree in model['trees']:
        node = 0
        for _ in range(len(tree['left'])+1):
            if tree['left'][node] == -1:
                logit += model['learning_rate']*tree['value'][node]
                break
            f = tree['feature'][node]
            if not 0 <= f < len(x):
                raise ValueError('Invalid serialized tree feature')
            node = tree['left'][node] if x[f] <= tree['threshold'][node] else tree['right'][node]
        else:
            raise ValueError('Invalid serialized tree cycle')
    if model.get('uses_original_offset'):
        if raw:return float(logit)
        return float(bounded_probability(offset_probability,logit,gain))
    return 1/(1+math.exp(-max(-35., min(35., logit))))


def original_functions():
    """Reuse actual live fitting/calibration functions without API/UI imports."""
    import ast
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    names = {'_nba_model_pipeline', '_nba_home_probabilities', '_nba_fit_calibrator',
             '_nba_chronological_tail', '_nba_probability_metrics', '_nba_recent_holdout',
             'train_balldontlie_live_model'}
    constants = {'NBA_LIVE_FEATURE_COLUMNS','NBA_FEATURE_SCHEMA_VERSION','NBA_PROBABILITY_VERSION'}
    path = Path(__file__).with_name('nba_research.py')
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names
             or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)]
    namespace = dict(np=np,pd=pd,LogisticRegression=LogisticRegression,
                     StandardScaler=StandardScaler,Pipeline=Pipeline)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    if not names.issubset(namespace):raise ValueError('Live NBA fitting functions missing')
    return namespace


def original_probabilities(functions, model, frame, columns):
    return functions['_nba_home_probabilities'](model['model'],frame[columns],model.get('calibrator'))


def prior_original_forecasts(targets, history, columns, block_size=50):
    """Expanding-date out-of-fold originals, never in-sample training predictions."""
    functions = original_functions()
    pending = targets.sort_values(['game_date','game_id']).copy()
    output = []
    while len(pending):
        cutoff = pending.iloc[0]['game_date']
        boundary = pending.iloc[min(block_size,len(pending))-1]['game_date']
        batch = pending[pending.game_date<=boundary].copy()
        pending = pending[pending.game_date>boundary]
        prior = history[history.game_date<cutoff]
        if len(prior)<500 or prior.home_win.nunique()!=2:
            continue
        model = functions['train_balldontlie_live_model'](prior,prediction_date=cutoff)
        batch['original_probability'] = original_probabilities(functions,model,batch,columns)
        batch['original_training_end'] = day(prior.game_date.max())
        output.append(batch)
    return pd.concat(output,ignore_index=True) if output else targets.iloc[:0].copy()


def correction_inputs(frame, original_probabilities):
    p = np.clip(np.asarray(original_probabilities,dtype=float),1e-7,1-1e-7)
    neural = np.clip(frame.neural_probability.to_numpy(dtype=float),1e-7,1-1e-7)
    x = frame[FEATURES].copy()
    x['neural_logit_gap'] = np.log(neural/(1-neural))-np.log(p/(1-p))
    # Last column is the unmodified original probability used by the offset.
    x['original_probability'] = p
    return x


def bounded_probability(original, raw_correction, gain=1.):
    p = np.clip(np.asarray(original,dtype=float),1e-7,1-1e-7)
    logit = np.log(p/(1-p))+np.clip(gain*np.asarray(raw_correction),-.7,.7)
    return 1/(1+np.exp(-np.clip(logit,-35,35)))


def training_rows(games):
    from .nba_historical_pregame_learning import research_functions
    from .nba_winner_comparison import load_model, predict
    research = research_functions()
    records = [{'game_id':g['id'],'game_date':g['date'],'season':g['season'],
                'home_team':g['teams'][0],'away_team':g['teams'][1],
                'home_points':g['scores'][g['teams'][0]],'away_points':g['scores'][g['teams'][1]],
                'home_win':int(g['scores'][g['teams'][0]]>g['scores'][g['teams'][1]])} for g in games]
    base = research['build_balldontlie_pregame_features'](pd.DataFrame(records))
    lookup = {g['id']:g for g in games}
    calibration_end = pd.Timestamp(load_model()['meta']['calibration_end']).date()
    rows = []
    for r in base.to_dict('records'):
        g = lookup[str(r['game_id'])]
        if pd.Timestamp(g['date']).date() <= calibration_end:
            continue
        extra = player_features(games,*g['teams'],g['date'])
        if extra is None:
            continue
        neural = predict(pd.DataFrame([r]),pd.Timestamp(g['date'],tz='UTC'))['learning_home_probability']
        rows.append(dict(r,neural_probability=neural,**extra))
    return pd.DataFrame(rows), research['NBA_LIVE_FEATURE_COLUMNS'], base


def train(client=None):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from .nba_pregame_snapshots import Client, PREFIX
    from .nba_historical_player_collection import valid_box
    from .nba_historical_pregame_learning import list_samples, HISTORY_ROOT
    client = client or Client()
    checkpoint = client.read(PREFIX+'checkpoint.json.gz',compressed=True)
    if not isinstance(checkpoint,dict) or checkpoint.get('schema') != 1:
        raise ValueError('Saved NBA checkpoint missing')
    samples = {}
    for name in list_samples(client):
        saved = client.read(HISTORY_ROOT+'games/'+name,compressed=True)
        if not isinstance(saved,dict) or not valid_box(saved.get('game',{}),saved.get('player_stats')):
            raise ValueError('Invalid saved player box')
        samples[str(saved['game_id'])] = saved['player_stats']
    games, aliases = compact_games(checkpoint,samples)
    samples.update({k:v for k,v in checkpoint.get('stats',{}).items() if k not in samples})
    games = add_players(games,samples)
    frame, base_columns, original_history = training_rows(games)
    groups = partition_by_game_count(frame)
    report = {'schema':2,'architecture':'original_offset_neural_player_residual','status':'waiting_for_history','training_performed':False,
              'usable_games':len(frame),'partition_games':dict(zip(('train','validation','test'),map(len,groups))),
              'sports_api_requests':0,'live_model_changed':False,'reasons':[],
              'limitations':['Retrospective reconstruction, not an untouched benchmark.',
                            'Historical availability unknown; prior appearances are not confirmed lineups.',
                            'Player strength features are box-production projections, not RAPM.',
                            'Fixed statistic-specific decay with support shrinkage; half-lives not optimized.',
                            'Historical original component reconstructs the team-form algorithm, not archived app forecasts.']}
    for label,g,n in zip(('train','validation','test'),groups,(500,50,50)):
        if len(g)<n:report['reasons'].append(f'{label}: {len(g)}/{n} games')
        if len(g)==0 or g.home_win.nunique()!=2 or g.home_win.value_counts().min()<10:
            report['reasons'].append(label+': needs 10 examples of each outcome')
    if report['reasons']:
        client.write(ROOT+'latest_report.json',report,replace=True)
        print(json.dumps(report));return report
    # Learn only from original forecasts generated using strictly earlier dates.
    correction_train = prior_original_forecasts(groups[0],original_history,base_columns)
    report['correction_training_games'] = len(correction_train)
    report['correction_training_method'] = 'expanding original engine; previous dates; 500-game minimum'
    if len(correction_train)<200 or correction_train.home_win.nunique()!=2 or correction_train.home_win.value_counts().min()<20:
        report['reasons'].append('Correction layer needs 200 out-of-fold forecasts and 20 of each outcome')
        client.write(ROOT+'latest_report.json',report,replace=True)
        print(json.dumps(report));return report
    functions = original_functions()
    cutoff = groups[1].game_date.min()
    prior = original_history[original_history.game_date<cutoff]
    original = functions['train_balldontlie_live_model'](prior,prediction_date=cutoff)
    x = correction_inputs(correction_train,correction_train.original_probability)
    columns = list(x.columns)
    from sklearn.base import ClassifierMixin, BaseEstimator
    class OriginalOffset(ClassifierMixin, BaseEstimator):
        def fit(self,X,y,sample_weight=None):
            self.classes_=np.array([0,1]);self.n_features_in_=np.asarray(X).shape[1]
            return self
        def predict_proba(self,X):
            p=np.clip(np.asarray(X)[:,-1],1e-7,1-1e-7)
            return np.column_stack([1-p,p])
    residual = GradientBoostingClassifier(n_estimators=80,learning_rate=.025,max_depth=2,
                min_samples_leaf=30,random_state=42,init=OriginalOffset())
    residual.fit(x,correction_train.home_win)
    trees=[]
    for estimator in residual.estimators_[:,0]:
        t=estimator.tree_
        trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),
                      'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,0,0].tolist()})
    exported={'initial_logit':0.,'learning_rate':float(residual.learning_rate),'trees':trees,
              'uses_original_offset':True}
    report['learned_context_usage'] = {
        'player_features':float(residual.feature_importances_[:-2].sum()),
        'existing_neural_gap':float(residual.feature_importances_[-2]),
        'original_probability':float(residual.feature_importances_[-1])}
    # Validation selects how far to apply the same learned correction, never votes.
    vp=original_probabilities(functions,original,groups[1],base_columns)
    vx=correction_inputs(groups[1],vp)
    raw=sum(residual.learning_rate * t.predict(vx.to_numpy()) for t in residual.estimators_[:,0])
    gains=[0.,.25,.5,.75,1.]
    gain=min(gains,key=lambda g:(-scores(groups[1].home_win,bounded_probability(vp,raw,g))['accuracy'],
                               scores(groups[1].home_win,bounded_probability(vp,raw,g))['brier']))
    passed=True
    for label,g in zip(('validation','test'),groups[1:]):
        p=original_probabilities(functions,original,g,base_columns)
        inputs=correction_inputs(g,p)
        delta=sum(residual.learning_rate * t.predict(inputs.to_numpy()) for t in residual.estimators_[:,0])
        connected=bounded_probability(p,delta,gain)
        restored=np.array([tree_probability(exported,r,gain=gain) for r in inputs.to_numpy()])
        if not np.allclose(connected,restored,atol=1e-10,rtol=0):
            raise ValueError('Connected export differs from trained correction')
        old,new=scores(g.home_win,p),scores(g.home_win,connected)
        report[label]={'original_engine':old,'connected_engine':new,
                       'changed_picks':int(np.sum((p>=.5)!=(connected>=.5)))}
        passed=passed and new['accuracy']>old['accuracy'] and new['brier']<=old['brier'] and new['log_loss']<=old['log_loss']
    now=datetime.now(timezone.utc).isoformat()
    artifact={'schema':2,'architecture':'original_offset_neural_player_residual',
              'trained_at':now,'context_refreshed_at':now,'columns':columns,'base_columns':base_columns,
              'tree_model':exported,'correction_gain':gain,'checks_passed':bool(passed),
              'games':games,'aliases':aliases,'availability':{},'training_games':len(correction_train),
              'source_neural_model':'nba_winner_neural_v1'}
    artifact['model_id']='nba_connected_'+hashlib.sha256(json.dumps(artifact,sort_keys=True).encode()).hexdigest()[:16]
    report.update(status='candidate_saved',training_performed=True,checks_passed=bool(passed),
                  correction_gain=gain,model_id=artifact['model_id'],probability_averaging=False)
    artifact['family_id']=artifact['model_id'];artifact['report']=report
    client.write(ROOT+'candidates/'+artifact['model_id']+'.json.gz',artifact,compressed=True)
    client.write(ARTIFACT,artifact,compressed=True,replace=True)
    client.write(ROOT+'latest_report.json',report,replace=True)
    print(json.dumps(report));return report


def fresh_availability(snapshot, now):
    """Fresh reports adjust conditional projections, never invent participation."""
    if not isinstance(snapshot,dict):return None
    now = pd.Timestamp(now)
    try:
        at = pd.Timestamp(snapshot['captured_at'])
        reliable = snapshot['confidence_reliability']
        times = snapshot['source_times']
        if not (reliable['injury_feed_complete'] and reliable['active_roster_complete']):return None
        if any(not pd.Timedelta(0)<=now-pd.Timestamp(times[k])<=pd.Timedelta(hours=6) for k in ('injuries','active_roster')):return None
        if not pd.Timedelta(0)<=now-at<=pd.Timedelta(hours=6):return None
        result = {}
        for side in ('home','visitor'):
            context = snapshot['teams'][side]
            rotation = context['availability_and_rotation']
            players = rotation['players']
            tid = str(context['team']['id'])
            if not rotation.get('availability_feed_fresh'):return None
            result[tid] = {'roster':[str(p['player_id']) for p in players],
                           'out':[str(p['player_id']) for p in players if p.get('firm_out_reported')],
                           'uncertain':[str(p['player_id']) for p in players if p.get('availability_uncertain') or p.get('minutes_restriction_reported')]}
        return {'captured_at':snapshot['captured_at'],'scheduled_start':snapshot['scheduled_start'],'teams':result}
    except (KeyError,TypeError,ValueError):return None


def predict(artifact, base_features, original_probability, neural_probability, home, away, start):
    """One original forecast, context inputs, then a jointly learned correction."""
    if not artifact or artifact.get('schema')!=2:return None
    now,start=pd.Timestamp.now(tz='UTC'),pd.Timestamp(start)
    if start.tzinfo is None:start=start.tz_localize('UTC')
    if (not now<start or pd.Timestamp(artifact['trained_at'])>=start or
            not pd.Timedelta(0)<=now-pd.Timestamp(artifact['context_refreshed_at'])<=pd.Timedelta(hours=24)):return None
    probs=np.asarray([original_probability,neural_probability],dtype=float)
    if not np.isfinite(probs).all() or not ((probs>0)&(probs<1)).all():return None
    h,a=artifact['aliases'].get(key(home)),artifact['aliases'].get(key(away))
    if not h or not a or h==a:return None
    extra=player_features(artifact['games'],h,a,start)
    if extra is None:return None
    row=dict(extra,neural_probability=float(probs[1]))
    inputs=correction_inputs(pd.DataFrame([row]),[probs[0]])
    values=inputs[artifact['columns']].iloc[0].to_numpy()
    delta=tree_probability(artifact['tree_model'],values,raw=True)
    connected=float(bounded_probability(probs[0],delta,artifact['correction_gain']))
    conditional=None;status='Prior-player context; verified availability unavailable'
    availability=artifact.get('availability',{}).get(h+'|'+a)
    if availability and pd.Timedelta(0)<=now-pd.Timestamp(availability['captured_at'])<=pd.Timedelta(hours=6) and abs((start-pd.Timestamp(availability['scheduled_start'])).total_seconds())<=900:
        scenario=player_features(artifact['games'],h,a,start,availability['teams'])
        if scenario is not None:
            sx=correction_inputs(pd.DataFrame([dict(scenario,neural_probability=float(probs[1]))]),[probs[0]])
            conditional=tree_probability(artifact['tree_model'],sx[artifact['columns']].iloc[0].to_numpy(),gain=artifact['correction_gain'])
            status='Fresh availability scenario separate; historical injury effect not validated'
    return {'model_id':artifact['model_id'],'source_player_model_id':artifact.get('family_id',artifact['model_id']),
            'architecture':'original_offset_neural_player_residual','original_home_probability':float(probs[0]),
            'neural_home_probability':float(probs[1]),'raw_correction_logit':delta,
            'connected_home_probability':connected,'combined_home_probability':connected,
            'correction_gain':artifact['correction_gain'],'probability_averaging':False,
            'availability_scenario_home_probability':conditional,'availability_status':status,
            'checks_passed':bool(artifact['checks_passed'])}


def refresh(client=None):
    """Reuse the snapshot collector; refresh context and grade frozen forecasts."""
    from .nba_pregame_snapshots import Client, PREFIX
    client = client or Client()
    artifact = client.read(ARTIFACT,compressed=True)
    if artifact is None:
        print(json.dumps({'status':'train_unified_model_first','sports_api_requests':0}));return
    checkpoint = client.read(PREFIX+'checkpoint.json.gz',compressed=True)
    if not isinstance(checkpoint,dict) or checkpoint.get('schema')!=1:raise ValueError('Missing checkpoint')
    updates,aliases = compact_games(checkpoint,checkpoint.get('stats',{}))
    updates = add_players(updates,checkpoint.get('stats',{}))
    merged = {g['id']:g for g in artifact['games']}
    for g in updates:
        old = merged.get(g['id'])
        if old and g['teams']==old['teams'] and g['scores']==old['scores']:
            if not g.get('boxes'):g['boxes']=old.get('boxes')
            if not g.get('players'):g['players']=old.get('players')
        merged[g['id']]=g
    artifact['games']=sorted(merged.values(),key=lambda g:(g['date'],g['id']))
    artifact['aliases'].update(aliases)
    now = datetime.now(timezone.utc)
    availability = {}
    for gid,capture in checkpoint.get('captures',{}).items():
        if not capture.get('paths'):continue
        try:
            at = pd.Timestamp(capture['captured_at'])
            start = pd.Timestamp(capture['scheduled_start'])
            if not pd.Timestamp(now)<start or not pd.Timedelta(0)<=pd.Timestamp(now)-at<=pd.Timedelta(hours=6):continue
        except (KeyError,ValueError,TypeError):continue
        snapshot=client.read(capture['paths'][-1])
        context=fresh_availability(snapshot,now)
        if context:
            ids=list(context['teams'])
            availability['|'.join(ids)]=context
    artifact['availability']=availability
    artifact['context_refreshed_at']=now.isoformat()
    client.write(ARTIFACT,artifact,compressed=True,replace=True)
    comparison = grade(client,checkpoint)
    learn_from_results(client,artifact,comparison)
    print(json.dumps({'status':'unified_context_refreshed','tree_weights_changed':False,'sports_api_requests':0,
                      'fresh_matchup_reports':len(availability)}))


def save_forecasts(client, predictions):
    """One immutable slate object, not one request per player or component."""
    now = datetime.now(timezone.utc)
    rows = []
    for r in predictions.to_dict('records'):
        result = r.get('unified_forecast')
        if not isinstance(result,dict) or not pd.Timestamp(now)<pd.Timestamp(r['commence_time']):continue
        rows.append({'event_id':str(r['event_id']),'home_team':r['home_team'],'away_team':r['away_team'],
                     'start_time':str(r['commence_time']),'captured_at':now.isoformat(),**result})
    if not rows:return
    rows.sort(key=lambda r:(r['model_id'],r['event_id']))
    identity = hashlib.sha256(json.dumps([(r['model_id'],r['event_id'],r['start_time']) for r in rows]).encode()).hexdigest()
    path = ROOT+'forecasts/'+identity+'.json.gz'
    if client.read(path,compressed=True) is None:
        client.write(path,{'schema':1,'rows':rows},compressed=True)


def grade(client, checkpoint):
    """Evaluate exact matched final games; keep models and original picks fixed."""
    from .nba_pregame_snapshots import final, start_time
    state = client.read(ROOT+'comparison.json.gz',compressed=True) or {'schema':1,'forecasts':{},'results':{}}
    offset=0
    while True:
        response=client.http.post(client.env['SUPABASE_URL'].rstrip('/')+'/storage/v1/object/list/'+quote(client.bucket,safe=''),
            headers=client.headers(),json={'prefix':ROOT+'forecasts','limit':100,'offset':offset,'sortBy':{'column':'name','order':'asc'}},timeout=(5,20))
        if response.status_code!=200:raise RuntimeError('Unified forecast listing failed')
        batch=response.json()
        if not isinstance(batch,list):raise ValueError('Invalid unified forecast listing')
        for obj in batch:
            name=obj.get('name','')
            if not name.endswith('.json.gz') or name in state['forecasts']:continue
            payload=client.read(ROOT+'forecasts/'+name,compressed=True)
            state['forecasts'][name]=payload['rows']
        if len(batch)<100:break
        offset+=len(batch)
    earliest={}
    for rows in state['forecasts'].values():
        for r in rows:
            identity=r['model_id']+'|'+r['event_id']
            if identity not in earliest or pd.Timestamp(r['captured_at'])<pd.Timestamp(earliest[identity]['captured_at']):earliest[identity]=r
    for identity,r in earliest.items():
        if identity in state['results']:continue
        matched=[g for g in checkpoint.get('games',{}).values()
                 if key((g.get('home_team') or {}).get('full_name'))==key(r['home_team'])
                 and key((g.get('visitor_team') or {}).get('full_name'))==key(r['away_team'])
                 and start_time(g) is not None and abs((pd.Timestamp(start_time(g))-pd.Timestamp(r['start_time'])).total_seconds())<=900]
        if len(matched)!=1 or not final(matched[0]):continue
        g=matched[0]
        if pd.Timestamp(start_time(g)) >= pd.Timestamp.now(tz='UTC'):continue
        h,a=g.get('home_team_score'),g.get('visitor_team_score')
        if h is None or a is None or min(h,a)<=0 or h==a:continue
        if not pd.Timestamp(r['captured_at'])<pd.Timestamp(start_time(g)):continue
        observed = datetime.now(timezone.utc).isoformat()
        outcome_hash = checkpoint.get('settled',{}).get(str(g['id']))
        if outcome_hash:
            from .nba_pregame_snapshots import PREFIX
            outcome = client.read(PREFIX+'outcomes/'+str(g['id'])+'/'+outcome_hash+'.json')
            if (isinstance(outcome,dict) and outcome.get('observed_at')
                    and outcome.get('home_score')==h and outcome.get('away_score')==a
                    and pd.Timestamp(start_time(g)) < pd.Timestamp(outcome['observed_at']) <= pd.Timestamp.now(tz='UTC')):
                observed = outcome['observed_at']
        state['results'][identity]=dict(r,outcome=int(h>a),game_id=str(g['id']),label_observed_at=observed)
    summaries=[]
    for model in sorted({r['model_id'] for r in state['results'].values()}):
        rows=[r for r in state['results'].values() if r['model_id']==model]
        y=np.array([r['outcome'] for r in rows])
        summaries.append({'model_id':model,'graded_games':len(rows),**{name:scores(y,np.array([r[field] for r in rows])) for name,field in
                         [('original','original_home_probability'),('connected','connected_home_probability')]}})
    state['summaries']=summaries
    client.write(ROOT+'comparison.json.gz',state,compressed=True,replace=True)
    print(json.dumps({'status':'unified_comparison','graded_games':len(state['results']),'summaries':summaries,'sports_api_requests':0}))
    return state


def learn_from_results(client, artifact, state):
    """Prospective calibration of correction strength, never probability weights."""
    family=artifact.get('family_id',artifact['model_id'])
    earliest={}
    for r in state.get('results',{}).values():
        if r.get('source_player_model_id')!=family or r.get('architecture')!='original_offset_neural_player_residual' or not r.get('label_observed_at'):continue
        event=r['event_id']
        if event not in earliest or pd.Timestamp(r['captured_at'])<pd.Timestamp(earliest[event]['captured_at']):earliest[event]=r
    rows=sorted(earliest.values(),key=lambda r:(r['start_time'],r['event_id']))
    report={'status':'waiting_for_connected_results','graded_unique_games':len(rows),
            'correction_changed':False,'sports_api_requests':0,'minimum_games':{'train':80,'validation':25,'test':25}}
    if len(rows)<130 or len(rows)<artifact.get('last_learning_count',0)+30:
        client.write(ROOT+'learning_report.json',report,replace=True);return report
    dates=sorted({r['start_time'][:10] for r in rows})
    candidates=[]
    for i,a in enumerate(dates[1:-1],1):
        n1=sum(r['start_time'][:10]<a for r in rows)
        for b in dates[i+1:]:
            n2=sum(a<=r['start_time'][:10]<b for r in rows);n3=len(rows)-n1-n2
            if n1>=80 and n2>=25 and n3>=25:candidates.append(((n1-.6*len(rows))**2+(n2-.2*len(rows))**2+(n3-.2*len(rows))**2,a,b))
    if not candidates:
        client.write(ROOT+'learning_report.json',report,replace=True);return report
    _,a,b=min(candidates)
    groups=[[r for r in rows if r['start_time'][:10]<a],[r for r in rows if a<=r['start_time'][:10]<b],[r for r in rows if r['start_time'][:10]>=b]]
    for i in (0,1):
        boundary=min(pd.Timestamp(r['captured_at']) for g in groups[i+1:] for r in g)
        groups[i]=[r for r in groups[i] if pd.Timestamp(r['label_observed_at'])<boundary]
    report['partition_games']=dict(zip(('train','validation','test'),map(len,groups)))
    if any(len(g)<n or min(sum(r['outcome']==v for r in g) for v in (0,1))<5 for g,n in zip(groups,(80,25,25))):
        client.write(ROOT+'learning_report.json',report,replace=True);return report
    def metric(g,gain):
        return scores(np.array([r['outcome'] for r in g]),bounded_probability(
            [r['original_home_probability'] for r in g],[r['raw_correction_logit'] for r in g],gain))
    def rank(g,gain):
        m=metric(g,gain);return -m['accuracy'],m['brier'],m['log_loss']
    fitted=min([0.,.25,.5,.75,1.],key=lambda gain:rank(groups[0],gain))
    gain=min([artifact['correction_gain'],0.,fitted],key=lambda v:rank(groups[1],v))
    passed=True
    for label,g in zip(('validation','test'),groups[1:]):
        old,original,new=metric(g,artifact['correction_gain']),metric(g,0.),metric(g,gain)
        report[label]={'previous_connected':old,'original':original,'proposed_connected':new}
        passed=passed and all(new['accuracy']>m['accuracy'] and new['brier']<=m['brier'] and new['log_loss']<=m['log_loss'] for m in (old,original))
    report.update(status='connected_correction_reviewed',proposed_gain=gain,checks_passed=bool(passed))
    artifact['last_learning_count']=len(rows)
    if passed:
        artifact['correction_gain']=gain;artifact['checks_passed']=True
        artifact['model_id']='nba_connected_'+hashlib.sha256(json.dumps([family,gain,[r['event_id'] for r in rows]]).encode()).hexdigest()[:16]
        report.update(correction_changed=True,model_id=artifact['model_id'])
        client.write(ROOT+'candidates/'+artifact['model_id']+'.json.gz',artifact,compressed=True)
    client.write(ARTIFACT,artifact,compressed=True,replace=True)
    client.write(ROOT+'learning_report.json',report,replace=True)
    print(json.dumps(report));return report


if __name__=='__main__':
    import sys
    refresh() if '--refresh' in sys.argv else train()
