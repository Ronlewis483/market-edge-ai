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


def matchup_rows(games):
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
            targets.append(game[side + '_runs'])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def fit_run_model(games, alpha):
    X, y = matchup_rows(games)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          PoissonRegressor(alpha=alpha, max_iter=2000, tol=1e-7))
    model.fit(X, y)
    return model


def predict_matchups(model, games):
    X, _ = matchup_rows(games)
    rates = model.predict(X).reshape(-1, 2)
    # Independent Poisson scoring; tied regulation games split equally.
    probabilities = skellam.sf(0, rates[:, 0], rates[:, 1]) + .5 * skellam.pmf(0, rates[:, 0], rates[:, 1])
    return rates, np.clip(probabilities, 1e-6, 1-1e-6)


def score(label, actual, probabilities):
    return {'Model': label, 'Games': len(actual), 'Accuracy': float(np.mean((probabilities >= .5) == actual)),
            'Brier': float(brier_score_loss(actual, probabilities)),
            'Log loss': float(log_loss(actual, probabilities, labels=[0, 1]))}


def run_mlb_matchup_model(cache_root='mlb_accuracy_results', progress=None, cohort='seasonwide', games_per_season=600):
    from dual_agent.mlb_historical_accuracy_test import run_historical_accuracy_test
    from threadpoolctl import threadpool_limits
    run_historical_accuracy_test(cache_root, progress=progress, expanded=True,
                                 games_per_season=games_per_season, cohort=cohort)
    games = json.loads((Path(cache_root)/'mlb_matchup_training_rows.json').read_text())
    development_train = [g for g in games if g['season'] == 2024]
    development_test = [g for g in games if g['season'] == 2025]
    train = [g for g in games if g['season'] < 2026]
    test = [g for g in games if g['season'] == 2026]
    if min(len(development_train), len(development_test), len(test)) < 100:
        raise ValueError('Need at least 100 usable games per season for matchup evaluation')
    y = np.asarray([g['home_win'] for g in test])
    development_scores = []
    with threadpool_limits(limits=2):
        for alpha in [.1, 1., 10.]:
            model = fit_run_model(development_train, alpha)
            _, p = predict_matchups(model, development_test)
            development_scores.append({'alpha': alpha, 'Log loss': float(log_loss([g['home_win'] for g in development_test], p, labels=[0, 1]))})
        selected = min(development_scores, key=lambda r: r['Log loss'])['alpha']
        model = fit_run_model(train, selected)
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
    development_model = fit_run_model(development_train, selected)
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
    result = {'version': 1, 'cohort': cohort, 'training_games': len(train), 'test_games': len(test),
              'scores': [score('Historical team baseline', y, control), score('Offense versus opposing starter run model', y, p)],
              'accuracy_change': float(differences.mean()), 'accuracy_change_95_interval': np.quantile(boot, [.025, .975]).tolist(),
              'run_MAE': float(mean_absolute_error(np.asarray([[g['home_runs'], g['away_runs']] for g in test]), rates)),
              'selected_alpha': selected, 'development_scores': development_scores,
              'confidence_selection': {'threshold': chosen, 'development': thresholds, 'test_games': int(mask.sum()), 'test_accuracy': float(np.mean((p[mask] >= .5) == y[mask])) if mask.any() else None},
              'coefficients': dict(zip(FEATURES, fitted.coef_.tolist())), 'predictions': predictions,
              'missing_inputs': ['Verified historical pitcher/hitter handedness splits', 'Pregame bullpen availability and three-day workload', 'Historical weather and roof status'],
              'limitations': ['2026 has been inspected in earlier experiments; it is not an untouched holdout.', 'Archive provider reconstruction/corrections remain possible; prior score availability uses a conservative 48-hour delay.', 'Independent Poisson scoring is an approximation; regulation ties are split equally.', 'Confidence subgroup results must include their sample size. Small subgroups do not establish reliability.'],
              'live_model_changed': False}
    (Path(cache_root)/'mlb_matchup_model_result.json').write_text(json.dumps(result, indent=2))
    return result
