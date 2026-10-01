"""Chronological offense-versus-pitching run model; no automatic live promotion."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error

FEATURES = ['home_field', 'offense_runs_per_game', 'opponent_runs_allowed',
            'offense_recent_run_difference', 'lineup_ops', 'lineup_obp', 'lineup_slg',
            'opposing_starter_era', 'opposing_starter_whip', 'opposing_starter_k9',
            'opposing_starter_bb9', 'lineup_ops_x_opposing_starter_whip']


BULLPEN_FEATURES = ['opponent_listed_relief_count', 'opponent_relief_pitches_prior_3d',
                    'opponent_relief_outs_prior_3d', 'opponent_repeat_use_fraction_prior_3d',
                    'opponent_relief_era_prior_30d', 'opponent_relief_whip_prior_30d']


def attach_bullpen_history(games, player_logs, completion_times):
    """Frozen bullpen membership plus strictly prior warehouse appearances.

    Recent workload requires verified completed-game play timestamps.
    Season quality still uses start+48h as conservative eligibility assumption.
    No same-game observations or present-day roster statuses are admitted.
    """
    required={'game_id','player_id','team_id','start_time','games_started',
              'pitching_outs','pitching_games','pitches','earned_runs','pitcher_hits','pitcher_walks'}
    if player_logs is None or player_logs.empty or not required.issubset(player_logs.columns):
        raise ValueError('Load the historical player warehouse; bullpen modeling requires: ' + ', '.join(sorted(required)))
    logs=player_logs.copy()
    logs['start_time']=pd.to_datetime(logs.start_time,utc=True,errors='coerce')
    logs=logs.dropna(subset=['start_time','team_id','player_id']).drop_duplicates(['game_id','team_id','player_id'])
    # Relief identity is established by an actual appearance with zero starts.
    logs=logs[(pd.to_numeric(logs.games_started,errors='coerce')==0) & (pd.to_numeric(logs.pitching_games,errors='coerce')>0)]
    for key in ['pitches','pitching_outs','earned_runs','pitcher_hits','pitcher_walks']:
        logs[key]=pd.to_numeric(logs[key],errors='coerce')
    logs['verified_end']=pd.to_datetime(logs.game_id.map(lambda gid: completion_times.get(int(gid))),utc=True,errors='coerce')
    groups={int(team):frame.sort_values('start_time') for team,frame in logs.groupby('team_id')}
    coverage=[]
    for game in games:
        capture=pd.to_datetime(game['timecode'],format='%Y%m%d_%H%M%S',utc=True)
        cutoff=capture-pd.Timedelta(hours=48)
        game['bullpen_sides']={}
        for side in ['home','away']:
            identity=game['recorded_identities'][side]
            listed=set(int(pid) for pid in identity.get('bullpen_ids',[]))-{int(identity['pitcher_id'])}
            history=groups.get(int(identity['team_id']))
            values=[None]*len(BULLPEN_FEATURES)
            if history is not None and listed:
                prior=history[(history.start_time<cutoff) & (history.start_time>=cutoff-pd.Timedelta(days=30)) & (pd.to_numeric(history.player_id,errors='coerce').isin(listed)) & (pd.to_numeric(history.game_id,errors='coerce')!=game['game_id'])]
                recent=history[(history.start_time>=capture-pd.Timedelta(days=3)) & (history.start_time<capture) & (history.verified_end<capture) & pd.to_numeric(history.player_id,errors='coerce').isin(listed) & (pd.to_numeric(history.game_id,errors='coerce')!=game['game_id'])]
                if not prior.empty or not recent.empty:
                    cohort=set(int(pid) for pid in prior.player_id.unique()) | set(int(pid) for pid in recent.player_id.unique())
                    counts=recent.groupby('player_id').size()
                    def total(frame,key):
                        return float(frame[key].sum()) if frame[key].notna().all() else None
                    outs=total(prior,'pitching_outs')
                    er,hits,walks=(total(prior,key) for key in ['earned_runs','pitcher_hits','pitcher_walks'])
                    values=[len(cohort),total(recent,'pitches'),total(recent,'pitching_outs'),float((counts>=2).sum()/len(cohort)),
                            er*27/outs if outs and er is not None else None,
                            (hits+walks)*3/outs if outs and hits is not None and walks is not None else None]
            game['bullpen_sides'][side]=values
            coverage.append({'game_id':game['game_id'],'season':game['season'],'side':side,'prior_relief_history':values[0] is not None})
    return coverage


def matchup_rows(games, include_bullpen=False, include_handedness=False, include_context=False):
    """Each game produces two scoring rows from frozen pregame observations."""
    rows, targets = [], []
    for game in games:
        for side, other in [('home', 'away'), ('away', 'home')]:
            team = game['team_sides'][side]
            opponent = game['team_sides'][other]
            players = game['player_sides'][side]
            pitcher = game['player_sides'][other]
            if len(players) != 7 or len(pitcher) != 7:
                raise ValueError('Update mlb_historical_accuracy_test.py: extended player rates are required')
            interaction = players[2] * pitcher[1]
            rows.append([int(side == 'home'), team[2], opponent[3], team[7],
                         players[2], players[5], players[6], pitcher[0], pitcher[1],
                         pitcher[3], pitcher[4], interaction])
            if include_bullpen:
                rows[-1].extend(game['bullpen_sides'][other])
            if include_handedness:
                rows[-1].extend(game['hand_sides'][side])
            if include_context:
                rows[-1].extend(game['form_sides'][side]+game['environment'])
            targets.append(game[side + '_runs'])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def fit_run_model(games, alpha, include_bullpen=False, include_handedness=False, include_context=False):
    X, y = matchup_rows(games, include_bullpen, include_handedness, include_context)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          PoissonRegressor(alpha=alpha, max_iter=2000, tol=1e-7))
    model.fit(X, y)
    model.matchup_has_bullpen = include_bullpen
    model.matchup_has_handedness = include_handedness
    model.matchup_has_context = include_context
    return model


def predict_matchups(model, games):
    X, _ = matchup_rows(games, getattr(model, "matchup_has_bullpen", False), getattr(model, "matchup_has_handedness", False), getattr(model, "matchup_has_context", False))
    rates = model.predict(X).reshape(-1, 2)
    # Independent Poisson scoring; tied regulation games split equally.
    probabilities = skellam.sf(0, rates[:, 0], rates[:, 1]) + .5 * skellam.pmf(0, rates[:, 0], rates[:, 1])
    return rates, np.clip(probabilities, 1e-6, 1-1e-6)


def score(label, actual, probabilities):
    return {'Model': label, 'Games': len(actual), 'Accuracy': float(np.mean((probabilities >= .5) == actual)),
            'Brier': float(brier_score_loss(actual, probabilities)),
            'Log loss': float(log_loss(actual, probabilities, labels=[0, 1]))}


def run_mlb_matchup_model(cache_root='mlb_accuracy_results', progress=None, cohort='seasonwide', games_per_season=600, player_logs=None, include_handedness=False, include_context=True, context_dataset=None):
    from dual_agent.mlb_historical_accuracy_test import run_historical_accuracy_test
    from threadpoolctl import threadpool_limits
    run_historical_accuracy_test(cache_root, progress=progress, expanded=True,
                                 games_per_season=games_per_season, cohort=cohort)
    games = json.loads((Path(cache_root)/'mlb_matchup_training_rows.json').read_text())
    schedule_games=[]
    for year in [2024,2025,2026]:
        schedule=json.loads((Path(cache_root)/'historical_pilot_cache'/f'schedule_{year}.json').read_text())
        schedule_games.extend(g for day in schedule['dates'] for g in day['games'] if g.get('gameType')=='R')
    completion_times={};bullpen_source={}
    if player_logs is not None:
        from dual_agent.mlb_bullpen_freshness import collect_bullpen_completion_times
        completion_times,bullpen_source=collect_bullpen_completion_times(games,schedule_games,cache_root,progress)
    if player_logs is not None:
        missing_games=set(completion_times)-set(pd.to_numeric(player_logs.game_id,errors='coerce').dropna().astype(int))
        if missing_games:raise ValueError(f'Recent bullpen warehouse incomplete: {len(missing_games)} verified games have no player logs. Refresh the warehouse before building.')
    bullpen_coverage=attach_bullpen_history(games, player_logs,completion_times) if player_logs is not None else []
    if player_logs is not None and not any(row['prior_relief_history'] for row in bullpen_coverage):
        raise ValueError('No eligible prior relief history matched the archived bullpen identities')
    handedness_coverage=[];handedness_source={}
    if include_handedness:
        from dual_agent.mlb_handedness import collect_prior_appearances, attach_handedness
        records,handedness_source=collect_prior_appearances(games,cache_root,progress)
        handedness_coverage=attach_handedness(games,records)
        if not any(row['lineup_players_with_splits']>=8 for row in handedness_coverage):
            raise ValueError('No lineups have sufficient prior handedness coverage')
    from dual_agent.mlb_matchup_elo import attach_elo, chronological_win_layer, select_recommendations, recommendation_metrics, confidence_diagnostics
    elo_source=attach_elo(games,schedule_games)
    context_coverage=[]
    if include_context:
        from dual_agent.mlb_matchup_context import attach_matchup_context
        context_coverage=attach_matchup_context(games,player_logs,schedule_games,context_dataset)
    development_train = [g for g in games if g['season'] == 2024]
    development_test = [g for g in games if g['season'] == 2025]
    train = [g for g in games if g['season'] < 2026]
    test = [g for g in games if g['season'] == 2026]
    if min(len(development_train), len(development_test), len(test)) < 100:
        raise ValueError('Need at least 100 usable games per season for matchup evaluation')
    y = np.asarray([g['home_win'] for g in test])
    development_scores = []
    with threadpool_limits(limits=2):
        for include_bullpen in ([False,True] if bullpen_coverage else [False]):
            for alpha in [.1, 1., 10.]:
                model = fit_run_model(development_train, alpha, include_bullpen, include_handedness, include_context)
                _, p = predict_matchups(model, development_test)
                development_scores.append({'alpha': alpha, 'bullpen':include_bullpen, 'Log loss': float(log_loss([g['home_win'] for g in development_test], p, labels=[0, 1]))})
        chosen_config = min(development_scores, key=lambda r: r['Log loss'])
        selected = chosen_config['alpha']
        model = fit_run_model(train, selected, chosen_config['bullpen'], include_handedness, include_context)
        rates, p = predict_matchups(model, test)
        baseline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=42))
        baseline.fit(np.asarray([g['features'][:8] for g in train]), [g['home_win'] for g in train])
        control = baseline.predict_proba(np.asarray([g['features'][:8] for g in test]))[:, 1]
    raw_matchup_p=p.copy()
    with threadpool_limits(limits=2):
        p, dp, win_bundle, win_development = chronological_win_layer(
            development_train, development_test, test, selected,
            (chosen_config['bullpen'],include_handedness,include_context),
            fit_run_model,predict_matchups,rates,raw_matchup_p)
    raw_development_model=fit_run_model(development_train,selected,chosen_config['bullpen'],include_handedness,include_context)
    _,raw_dp=predict_matchups(raw_development_model,development_test)
    dy=np.asarray([g['home_win'] for g in development_test])
    win_layer_selected=log_loss(dy,dp,labels=[0,1])<log_loss(dy,raw_dp,labels=[0,1])
    hybrid_p=p.copy();hybrid_dp=dp.copy()
    if not win_layer_selected:p=raw_matchup_p;dp=raw_dp
    differences = ((p >= .5) == y).astype(int) - ((control >= .5) == y).astype(int)
    daily = pd.DataFrame({'date': [g['start_time'][:10] for g in test], 'difference': differences}).groupby('date').difference.agg(['sum', 'count'])
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        draw = daily.iloc[rng.integers(0, len(daily), len(daily))]
        boot.append(float(draw['sum'].sum()/draw['count'].sum()))
    # Recommendation rule uses 2025 outcomes only; never choose cutoff on test outcomes.
    dy=np.asarray([g['home_win'] for g in development_test])
    chosen, thresholds=select_recommendations(dp,dy)
    mask, recommended_metrics=recommendation_metrics(p,y,chosen)
    predictions = [{'game_id': g['game_id'], 'Start UTC': g['start_time'], 'Home': g['home_team'], 'Away': g['away_team'],
                    'Projected home runs': float(rate[0]), 'Projected away runs': float(rate[1]),
                    'Home win probability': float(probability), 'Actual home runs': g['home_runs'], 'Actual away runs': g['away_runs'],
                    'Elo home':g['elo_home'], 'Elo away':g['elo_away'],
                    'Recommended':bool(mask[index]), 'Pick':g['home_team'] if probability>=.5 else g['away_team'],
                    'Record type':'historical evaluation', 'Correct': bool((probability >= .5) == g['home_win'])} for index,(g, rate, probability) in enumerate(zip(test, rates, p))]
    fitted = model.steps[-1][1]
    names = FEATURES + (BULLPEN_FEATURES if chosen_config['bullpen'] else [])
    if include_handedness:
        from dual_agent.mlb_handedness import HAND_FEATURES
        names += HAND_FEATURES
    if include_context:
        from dual_agent.mlb_matchup_context import FORM_FEATURES, ENVIRONMENT_FEATURES
        names += FORM_FEATURES+ENVIRONMENT_FEATURES
    retained = model.steps[0][1].get_feature_names_out(names).tolist()
    extra_scores=[]
    if bullpen_coverage:
        for flag,label in [(False,'Matchup without bullpen'),(True,'Matchup with verified recent bullpen workload')]:
            config=min([row for row in development_scores if row['bullpen']==flag],key=lambda row:row['Log loss'])
            comparison=fit_run_model(train,config['alpha'],flag, include_handedness, include_context)
            _,probability=predict_matchups(comparison,test)
            extra_scores.append(score(label,y,probability))
    result = {'version': 6, 'cohort': cohort, 'training_games': len(train), 'test_games': len(test),
              'scores': [score('Historical team baseline', y, control), score('Matchup runs only', y, raw_matchup_p), score('Matchup + Elo + logistic win layer', y, hybrid_p), score('Development-selected recommendation model',y,p)]+extra_scores,
              'accuracy_change': float(differences.mean()), 'accuracy_change_95_interval': np.quantile(boot, [.025, .975]).tolist(),
              'run_MAE': float(mean_absolute_error(np.asarray([[g['home_runs'], g['away_runs']] for g in test]), rates)),
              'handedness_connected':include_handedness, 'handedness_source':handedness_source, 'handedness_coverage':handedness_coverage, 'handedness_note':'Prior completed plate appearances only. Handedness history remains delayed by 48 hours. Split rates shrink toward the player prior overall rates with 100 pseudo-observations; at least eight lineup players required. No full-season split endpoint is used.', 'context_connected':include_context, 'context_coverage':context_coverage, 'unlearned_features':[name for name in names if name not in retained], 'context_note':'Rest and form use prior warehouse records with a 48-hour buffer. Rest is measured since last known start, not verified latest start. Thirty-day rates blend toward prior season rates. Park ratio uses only prior same-season scores with 50-game shrinkage. Weather/roof/dimensions require schema-versioned frozen captures no later than archived prediction time; unsupported inputs remain unknown.', 'selected_alpha': selected, 'development_scores': development_scores,
              'win_layer_selected':bool(win_layer_selected),'bullpen_freshness_source':bullpen_source,'confidence_diagnostics':{'development':confidence_diagnostics(dp,dy),'evaluation':confidence_diagnostics(p,y)},'elo_connected':True,'elo_source':elo_source,'win_layer_bundle':win_bundle,'win_layer_development':win_development,
              'recommendation_metrics':recommended_metrics, 'recommendation_policy':{'target_accuracy':.70,'minimum_development_picks':50,'cutoff_locked_on_season':2025,'no_qualifying_rule':chosen is None,'note':'70% on recommended games is the objective, not a guaranteed probability or observed future record. If no cutoff meets 70%, show the best supported development group as research picks below target; fallback is all games if no group has 50 picks.'},
              'confidence_selection': {'threshold': chosen, 'development': thresholds, 'test_games': int(mask.sum()), 'test_accuracy': float(np.mean((p[mask] >= .5) == y[mask])) if mask.any() else None},
              'bullpen_selected':chosen_config['bullpen'], 'bullpen_coverage':bullpen_coverage, 'bullpen_coverage_by_season':pd.DataFrame(bullpen_coverage).groupby('season').prior_relief_history.agg(['sum','count']).reset_index().to_dict('records') if bullpen_coverage else [], 'coefficients': dict(zip(retained, fitted.coef_.tolist())), 'predictions': predictions,
              'missing_inputs': [* ([] if include_handedness else ['Verified historical pitcher/hitter handedness splits']), 'Confirmed bullpen health/rest availability; completion time does not establish health', 'Historical weather and roof status'],
              'limitations': ['2026 has been inspected in earlier experiments; it is not an untouched holdout.', 'Archive provider reconstruction/corrections remain possible; prior score availability uses a conservative 48-hour delay.', 'Independent Poisson scoring is an approximation; regulation ties are split equally.', 'The logistic win layer uses earlier-block run forecasts. Hyperparameters still share 2025 development data; a new future record is required.', 'Historical recommendations are evaluations, not timestamped live picks. The live engine remains unpromoted.', 'Confidence subgroup results must include their sample size. Small subgroups do not establish reliability.'],
              'bullpen_note':'Archived bullpen list is a feed observation, not confirmed availability. Relief counts use listed pitchers with prior relief appearances. Workload covers the immediate three days before capture and requires verified prior-game completion. Quality retains a 48-hour buffer over its prior 30-day window. Missing history stays unknown. Stored zero values cannot establish source completeness.', 'retained_feature_rows':games, 'created_at':pd.Timestamp.now(tz='UTC').isoformat(), 'model_bundle':{'columns':retained,'median':model.steps[0][1].statistics_[np.isfinite(model.steps[0][1].statistics_)].tolist(),'mean':model.steps[1][1].mean_.tolist(),'scale':model.steps[1][1].scale_.tolist(),'weights':fitted.coef_.tolist(),'intercept':float(fitted.intercept_)}, 'live_model_changed': False}
    (Path(cache_root)/'mlb_matchup_model_result.json').write_text(json.dumps(result, indent=2))
    return result


def save_mlb_matchup_record(result):
    """Archive model and retained pregame features separately from player model."""
    from uuid import uuid4
    from dual_agent import supabase_db as storage
    try:
        ready=storage.ensure_market_edge_storage_bucket()
        if not ready.get('success'):
            raise RuntimeError(ready.get('error','Storage bucket unavailable'))
        payload=json.dumps(storage._json_safe(result),allow_nan=False).encode('utf-8')
        path='mlb/matchup_model/'+pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid4().hex+'.json'
        bucket=storage.get_supabase_client().storage.from_(storage.MLB_STORAGE_BUCKET)
        bucket.upload(path,payload,{'content-type':'application/json','upsert':'false'})
        return {'success':True,'archive_file':path}
    except Exception as exc:
        return {'success':False,'message':str(exc)}

def generate_saved_matchup_recommendations(result, games):
    """Score fresh enriched game rows and persist every qualifying research pick.

    Caller supplies frozen pregame rows with the same feature schema as training.
    This entry point does not collect data or replace the live engine/scanner.
    """
    from dual_agent.mlb_matchup_elo import predict_win_bundle, freeze_recommendation, save_recommendation
    import hashlib
    if result.get('version',0)<6: raise ValueError('Rebuild the matchup model with verified recent bullpen workload first')
    threshold=result['confidence_selection']['threshold']
    if threshold is None: return {'recommendations':[],'errors':[],'note':'No development-qualified 70% recommendation rule'}
    model_id=hashlib.sha256(json.dumps({'runs':result['model_bundle'],'wins':result['win_layer_bundle'],'use_win_layer':result.get('win_layer_selected',True),'threshold':threshold},sort_keys=True).encode()).hexdigest()
    names=FEATURES+(BULLPEN_FEATURES if result['bullpen_selected'] else [])
    if result['handedness_connected']:
        from dual_agent.mlb_handedness import HAND_FEATURES
        names+=HAND_FEATURES
    if result['context_connected']:
        from dual_agent.mlb_matchup_context import FORM_FEATURES, ENVIRONMENT_FEATURES
        names+=FORM_FEATURES+ENVIRONMENT_FEATURES
    rows=[dict(g,home_runs=0,away_runs=0) for g in games]  # labels unused for inference
    X,_=matchup_rows(rows,result['bullpen_selected'],result['handedness_connected'],result['context_connected'])
    bundle=result['model_bundle'];indices=[names.index(name) for name in bundle['columns']]
    X=X[:,indices];X=np.where(np.isfinite(X),X,np.asarray(bundle['median']))
    z=((X-np.asarray(bundle['mean']))/np.asarray(bundle['scale']))@np.asarray(bundle['weights'])+bundle['intercept']
    rates=np.exp(np.clip(z,-20,20)).reshape(-1,2)
    raw=skellam.sf(0,rates[:,0],rates[:,1])+.5*skellam.pmf(0,rates[:,0],rates[:,1])
    probabilities=predict_win_bundle(games,rates,raw,result['win_layer_bundle']) if result.get('win_layer_selected',True) else raw
    recommendations=[];errors=[]
    for game,probability in zip(games,probabilities):
        if max(probability,1-probability)<threshold: continue
        try:
            record=freeze_recommendation(game,probability,threshold,model_id)
            saved=save_recommendation(record)
            recommendations.append(dict(record,archive_file=saved['archive_file']))
        except Exception as exc:
            errors.append({'game_id':game['game_id'],'error':str(exc)})
    return {'recommendations':recommendations,'errors':errors,'note':'Research recommendations; each successful record saved before game time'}
