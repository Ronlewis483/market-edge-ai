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


BULLPEN_FEATURES = ['opponent_listed_relief_count', 'opponent_relief_pitches_lagged_3d',
                    'opponent_relief_outs_lagged_3d', 'opponent_repeat_use_fraction_lagged_3d',
                    'opponent_relief_era_prior_30d', 'opponent_relief_whip_prior_30d']


def attach_bullpen_history(games, player_logs):
    """Frozen bullpen membership plus strictly prior warehouse appearances.

    Postgame logs have no exact result-availability timestamp. Use start+48h
    as conservative eligibility assumption, and label recent windows as lagged.
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
    groups={int(team):frame.sort_values('start_time') for team,frame in logs.groupby('team_id')}
    coverage=[]
    for game in games:
        cutoff=pd.to_datetime(game['timecode'],format='%Y%m%d_%H%M%S',utc=True)-pd.Timedelta(hours=48)
        game['bullpen_sides']={}
        for side in ['home','away']:
            identity=game['recorded_identities'][side]
            listed=set(int(pid) for pid in identity.get('bullpen_ids',[]))-{int(identity['pitcher_id'])}
            history=groups.get(int(identity['team_id']))
            values=[None]*len(BULLPEN_FEATURES)
            if history is not None and listed:
                prior=history[(history.start_time<cutoff) & (history.start_time>=cutoff-pd.Timedelta(days=30)) & (pd.to_numeric(history.player_id,errors='coerce').isin(listed)) & (pd.to_numeric(history.game_id,errors='coerce')!=game['game_id'])]
                if not prior.empty:
                    recent=prior[prior.start_time>=cutoff-pd.Timedelta(days=3)]
                    cohort=set(int(pid) for pid in prior.player_id.unique())
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


def matchup_rows(games, include_bullpen=False):
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
            targets.append(game[side + '_runs'])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def fit_run_model(games, alpha, include_bullpen=False):
    X, y = matchup_rows(games, include_bullpen)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          PoissonRegressor(alpha=alpha, max_iter=2000, tol=1e-7))
    model.fit(X, y)
    model.matchup_has_bullpen = include_bullpen
    return model


def predict_matchups(model, games):
    X, _ = matchup_rows(games, getattr(model, "matchup_has_bullpen", False))
    rates = model.predict(X).reshape(-1, 2)
    # Independent Poisson scoring; tied regulation games split equally.
    probabilities = skellam.sf(0, rates[:, 0], rates[:, 1]) + .5 * skellam.pmf(0, rates[:, 0], rates[:, 1])
    return rates, np.clip(probabilities, 1e-6, 1-1e-6)


def score(label, actual, probabilities):
    return {'Model': label, 'Games': len(actual), 'Accuracy': float(np.mean((probabilities >= .5) == actual)),
            'Brier': float(brier_score_loss(actual, probabilities)),
            'Log loss': float(log_loss(actual, probabilities, labels=[0, 1]))}


def run_mlb_matchup_model(cache_root='mlb_accuracy_results', progress=None, cohort='seasonwide', games_per_season=600, player_logs=None):
    from dual_agent.mlb_historical_accuracy_test import run_historical_accuracy_test
    from threadpoolctl import threadpool_limits
    run_historical_accuracy_test(cache_root, progress=progress, expanded=True,
                                 games_per_season=games_per_season, cohort=cohort)
    games = json.loads((Path(cache_root)/'mlb_matchup_training_rows.json').read_text())
    bullpen_coverage=attach_bullpen_history(games, player_logs) if player_logs is not None else []
    if player_logs is not None and not any(row['prior_relief_history'] for row in bullpen_coverage):
        raise ValueError('No eligible prior relief history matched the archived bullpen identities')
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
                model = fit_run_model(development_train, alpha, include_bullpen)
                _, p = predict_matchups(model, development_test)
                development_scores.append({'alpha': alpha, 'bullpen':include_bullpen, 'Log loss': float(log_loss([g['home_win'] for g in development_test], p, labels=[0, 1]))})
        chosen_config = min(development_scores, key=lambda r: r['Log loss'])
        selected = chosen_config['alpha']
        model = fit_run_model(train, selected, chosen_config['bullpen'])
        rates, p = predict_matchups(model, test)
        baseline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=42))
        baseline.fit(np.asarray([g['features'][:8] for g in train]), [g['home_win'] for g in train])
        control = baseline.predict_proba(np.asarray([g['features'][:8] for g in test]))[:, 1]
    differences = ((p >= .5) == y).astype(int) - ((control >= .5) == y).astype(int)
    daily = pd.DataFrame({'date': [g['start_time'][:10] for g in test], 'difference': differences}).groupby('date').difference.agg(['sum', 'count'])
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        draw = daily.iloc[rng.integers(0, len(daily), len(daily))]
        boot.append(float(draw['sum'].sum()/draw['count'].sum()))
    # Confidence threshold is selected on 2025, never on the 2026 outcomes.
    development_model = fit_run_model(development_train, selected, chosen_config['bullpen'])
    _, dp = predict_matchups(development_model, development_test)
    dy = np.asarray([g['home_win'] for g in development_test])
    thresholds = []
    for threshold in [.55, .60, .65]:
        mask = np.maximum(dp, 1-dp) >= threshold
        if mask.sum() >= 50:
            thresholds.append({'threshold': threshold, 'games': int(mask.sum()), 'accuracy': float(np.mean((dp[mask] >= .5) == dy[mask]))})
    chosen = max(thresholds, key=lambda r: (r['accuracy'], r['games']))['threshold'] if thresholds else None
    mask = np.maximum(p, 1-p) >= chosen if chosen is not None else np.zeros(len(p), dtype=bool)
    predictions = [{'game_id': g['game_id'], 'Start UTC': g['start_time'], 'Home': g['home_team'], 'Away': g['away_team'],
                    'Projected home runs': float(rate[0]), 'Projected away runs': float(rate[1]),
                    'Home win probability': float(probability), 'Actual home runs': g['home_runs'], 'Actual away runs': g['away_runs'],
                    'Correct': bool((probability >= .5) == g['home_win'])} for g, rate, probability in zip(test, rates, p)]
    fitted = model.steps[-1][1]
    names = FEATURES + (BULLPEN_FEATURES if chosen_config['bullpen'] else [])
    retained = model.steps[0][1].get_feature_names_out(names).tolist()
    extra_scores=[]
    if bullpen_coverage:
        for flag,label in [(False,'Matchup without bullpen'),(True,'Matchup with lagged bullpen history')]:
            config=min([row for row in development_scores if row['bullpen']==flag],key=lambda row:row['Log loss'])
            comparison=fit_run_model(train,config['alpha'],flag)
            _,probability=predict_matchups(comparison,test)
            extra_scores.append(score(label,y,probability))
    result = {'version': 2, 'cohort': cohort, 'training_games': len(train), 'test_games': len(test),
              'scores': [score('Historical team baseline', y, control), score('Development-selected matchup model', y, p)]+extra_scores,
              'accuracy_change': float(differences.mean()), 'accuracy_change_95_interval': np.quantile(boot, [.025, .975]).tolist(),
              'run_MAE': float(mean_absolute_error(np.asarray([[g['home_runs'], g['away_runs']] for g in test]), rates)),
              'selected_alpha': selected, 'development_scores': development_scores,
              'confidence_selection': {'threshold': chosen, 'development': thresholds, 'test_games': int(mask.sum()), 'test_accuracy': float(np.mean((p[mask] >= .5) == y[mask])) if mask.any() else None},
              'bullpen_selected':chosen_config['bullpen'], 'bullpen_coverage':bullpen_coverage, 'bullpen_coverage_by_season':pd.DataFrame(bullpen_coverage).groupby('season').prior_relief_history.agg(['sum','count']).reset_index().to_dict('records') if bullpen_coverage else [], 'coefficients': dict(zip(retained, fitted.coef_.tolist())), 'predictions': predictions,
              'missing_inputs': ['Verified historical pitcher/hitter handedness splits', 'Confirmed bullpen health/rest availability; immediate prior 48 hours are excluded from warehouse-based workload', 'Historical weather and roof status'],
              'limitations': ['2026 has been inspected in earlier experiments; it is not an untouched holdout.', 'Archive provider reconstruction/corrections remain possible; prior score availability uses a conservative 48-hour delay.', 'Independent Poisson scoring is an approximation; regulation ties are split equally.', 'Confidence subgroup results must include their sample size. Small subgroups do not establish reliability.'],
              'bullpen_note':'Archived bullpen list is a feed observation, not confirmed availability. Relief counts use listed pitchers with prior relief appearances. Workload covers three days ending 48 hours before capture; quality covers 30 days before that cutoff. Missing history stays unknown. Stored zero values cannot establish source completeness.', 'live_model_changed': False}
    (Path(cache_root)/'mlb_matchup_model_result.json').write_text(json.dumps(result, indent=2))
    return result
