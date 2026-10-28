"""Saved-box NBA winner features and offline training; no sports API requests.

Run: python -m dual_agent.nba_winner_box_context
Live use requires NBA_WINNER_BOX_CONTEXT_ENABLED=true and passing saved checks.
"""
import hashlib
import json
import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = 'nba/winner_box_context_v1/'
ARTIFACT = ROOT + 'latest.json.gz'
METRICS = ('effective_fg', 'turnover_rate', 'rebound_share', 'free_throw_rate',
           'opponent_effective_fg', 'opponent_turnover_rate', 'opposition_margin')
COLUMNS = [side + '_box_' + name for side in ('home', 'away', 'difference') for name in METRICS]


def key(value):
    return re.sub('[^a-z0-9]', '', str(value).lower().replace('los angeles', 'la'))


def day(value):
    return pd.Timestamp(value).date().isoformat()


def summarize(games, team, cutoff):
    """Use only previous dates, the current season segment and its last 8 games."""
    cutoff_day = day(cutoff)
    prior = [g for g in games if g['date'] < cutoff_day and str(team) in g['teams']]
    prior.sort(key=lambda g: (g['date'], g['id']))
    if not prior:
        return None
    last = prior[-1]
    age = (pd.Timestamp(cutoff_day) - pd.Timestamp(last['date'])).days
    if age > 10:
        return None
    segment = []
    newer = last
    for g in reversed(prior):
        if g['season'] != last['season'] or (pd.Timestamp(newer['date']) - pd.Timestamp(g['date'])).days > 45:
            break
        segment.append(g)
        newer = g
        if len(segment) == 8:
            break
    usable = [g for g in segment if g.get('boxes')]
    if len(usable) < 4:
        return None
    own = [g['boxes'][str(team)] for g in usable]
    opponents = [g['boxes'][next(t for t in g['teams'] if t != str(team))] for g in usable]
    def total(rows, field):
        return sum(r[field] for r in rows)
    def ratio(a, b):
        if b <= 0:
            raise ValueError('Unsupported box denominator')
        return a / b
    opp_margins = []
    for g in segment:
        opponent = next(t for t in g['teams'] if t != str(team))
        history = [p for p in games if p['date'] < cutoff_day and p['season'] == last['season'] and opponent in p['teams']]
        history.sort(key=lambda p: (p['date'], p['id']))
        if history:
            opp_margins.append(np.mean([p['scores'][opponent] - p['scores'][next(t for t in p['teams'] if t != opponent)] for p in history[-8:]]))
    if not opp_margins:
        return None
    values = [ratio(total(own, 'fgm') + .5 * total(own, 'fg3m'), total(own, 'fga')),
              ratio(total(own, 'turnover'), total(own, 'poss')),
              ratio(total(own, 'reb'), total(own, 'reb') + total(opponents, 'reb')),
              ratio(total(own, 'fta'), total(own, 'fga')),
              ratio(total(opponents, 'fgm') + .5 * total(opponents, 'fg3m'), total(opponents, 'fga')),
              ratio(total(opponents, 'turnover'), total(opponents, 'poss')),
              float(np.mean(opp_margins))]
    return dict(zip(METRICS, values))


def features(games, home, away, cutoff):
    h, a = summarize(games, home, cutoff), summarize(games, away, cutoff)
    if h is None or a is None:
        return None
    return {**{'home_box_' + k: h[k] for k in METRICS},
            **{'away_box_' + k: a[k] for k in METRICS},
            **{'difference_box_' + k: h[k] - a[k] for k in METRICS}}


def compact_games(checkpoint, samples):
    from .nba_pregame_snapshots import final, played_date, team_box
    from .nba_historical_player_collection import valid_box
    result = []
    aliases = {}
    for g in checkpoint.get('games', {}).values():
        if not final(g) or not played_date(g) or str(played_date(g)) >= datetime.now(timezone.utc).date().isoformat():
            continue
        teams = [str(g[s + '_team']['id']) for s in ('home', 'visitor')]
        scores = {teams[i]: float(g[s + '_team_score']) for i, s in enumerate(('home', 'visitor'))}
        if min(scores.values()) <= 0 or len(set(scores.values())) != 2:
            continue
        for s in ('home', 'visitor'):
            team = g[s + '_team']
            for field in ('full_name', 'abbreviation'):
                if team.get(field):
                    aliases[key(team[field])] = str(team['id'])
        rows = samples.get(str(g['id']), checkpoint.get('stats', {}).get(str(g['id'])))
        boxes = None
        if valid_box(g, rows):
            boxes = {}
            for s in ('home', 'visitor'):
                b = team_box(rows, g[s + '_team']['id'])
                boxes[str(g[s + '_team']['id'])] = {**{k: float(b[k]) for k in ('fgm', 'fg3m', 'fga', 'fta', 'turnover', 'reb')}, 'poss': float(b['estimated_possessions'])}
        result.append({'id': str(g['id']), 'date': str(played_date(g)), 'season': g['season'],
                       'teams': teams, 'scores': scores, 'boxes': boxes})
    return sorted(result, key=lambda g: (g['date'], g['id'])), aliases


def scores(y, p):
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    return {'games': len(y), 'accuracy': float(accuracy_score(y, p >= .5)),
            'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1]))}


def partition_by_game_count(frame):
    """Choose chronological whole-date boundaries using counts, never outcomes.

    Prefer 75/12.5/12.5 percent of games while retaining the existing
    500/50/50 minimums whenever whole-date boundaries permit them.
    """
    if frame.empty:
        return [frame.copy() for _ in range(3)]
    ds = frame['game_date'].map(day)
    counts = ds.value_counts().sort_index()
    dates = list(counts.index)
    if len(dates) < 3:
        return [frame.copy(), frame.iloc[:0].copy(), frame.iloc[:0].copy()]
    cumulative = counts.cumsum().to_numpy()
    total = len(frame)
    target = np.array([.75, .125, .125]) * total
    best = None
    for i in range(1, len(dates) - 1):
        train_count = int(cumulative[i - 1])
        for j in range(i + 1, len(dates)):
            sizes = (train_count, int(cumulative[j - 1]) - train_count,
                     total - int(cumulative[j - 1]))
            feasible = all(size >= minimum for size, minimum in zip(sizes, (500, 50, 50)))
            # Feasibility comes first; ties resolve deterministically by date.
            rank = (not feasible, float(np.sum((np.array(sizes) - target) ** 2)), i, j)
            if best is None or rank < best:
                best = rank
    _, _, i, j = best
    a, b = dates[i], dates[j]
    return [frame[ds < a].copy(), frame[(ds >= a) & (ds < b)].copy(), frame[ds >= b].copy()]


def train(client=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from .nba_pregame_snapshots import Client, PREFIX
    from .nba_historical_pregame_learning import list_samples, research_functions, HISTORY_ROOT
    from .nba_historical_player_collection import valid_box
    client = client or Client()
    checkpoint = client.read(PREFIX + 'checkpoint.json.gz', compressed=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema') != 1:
        raise ValueError('Saved NBA checkpoint missing')
    samples = {}
    for name in list_samples(client):
        saved = client.read(HISTORY_ROOT + 'games/' + name, compressed=True)
        if not isinstance(saved, dict) or not valid_box(saved.get('game', {}), saved.get('player_stats')):
            raise ValueError('Saved player box failed validation')
        samples[str(saved['game_id'])] = saved['player_stats']
    games, aliases = compact_games(checkpoint, samples)
    records = [{'game_id': g['id'], 'game_date': g['date'], 'season': g['season'],
                'home_team': g['teams'][0], 'away_team': g['teams'][1],
                'home_points': g['scores'][g['teams'][0]], 'away_points': g['scores'][g['teams'][1]],
                'home_win': int(g['scores'][g['teams'][0]] > g['scores'][g['teams'][1]])} for g in games]
    research = research_functions()
    base = research['build_balldontlie_pregame_features'](pd.DataFrame(records))
    by_id = {g['id']: g for g in games}
    rows = []
    for row in base.to_dict('records'):
        g = by_id[str(row['game_id'])]
        extra = features(games, *g['teams'], g['date'])
        if extra is not None:
            rows.append(dict(row, **extra))
    frame = pd.DataFrame(rows)
    dates = sorted({day(r['game_date']) for r in rows})
    report = {'usable_games': len(frame), 'training_performed': False, 'sports_api_requests': 0,
              'live_model_changed': False, 'status': 'waiting_for_box_coverage',
              'limitations': 'Retrospective data; not an untouched benchmark. No historical injuries, lineups, travel or odds included.'}
    if len(dates) < 3:
        client.write(ROOT + 'latest_report.json', report, replace=True)
        print(json.dumps(report)); return report
    groups = partition_by_game_count(frame)
    report['partition_method'] = 'chronological_game_counts_whole_dates_v2'
    report['partition_games'] = dict(zip(('train', 'validation', 'test'), map(len, groups)))
    report['partition_dates'] = {name: {'first': day(g['game_date'].min()),
                                      'last': day(g['game_date'].max())} if len(g) else None
                                 for name, g in zip(('train', 'validation', 'test'), groups)}
    reasons = []
    for name, g, minimum in zip(('train', 'validation', 'test'), groups, (500, 50, 50)):
        if len(g) < minimum:
            reasons.append(f'{name}: {len(g)}/{minimum} games')
        if g['home_win'].nunique() != 2 or g['home_win'].value_counts().min() < 10:
            reasons.append(f'{name}: needs at least 10 examples of each outcome')
    report['reasons'] = reasons
    if reasons:
        client.write(ROOT + 'latest_report.json', report, replace=True)
        print(json.dumps(report)); return report
    base_columns = research['NBA_LIVE_FEATURE_COLUMNS']
    expanded = base_columns + COLUMNS
    def fit(columns):
        model = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(C=.1, max_iter=2000, random_state=42))])
        model.fit(groups[0][columns], groups[0]['home_win'])
        return model
    incumbent, candidate = fit(base_columns), fit(expanded)
    passed = True
    for label, group in zip(('validation', 'test'), groups[1:]):
        before = scores(group['home_win'], incumbent.predict_proba(group[base_columns])[:, 1])
        after = scores(group['home_win'], candidate.predict_proba(group[expanded])[:, 1])
        report[label] = {'team_form': before, 'box_context': after}
        passed = passed and after['accuracy'] > before['accuracy'] and after['brier'] < before['brier'] and after['log_loss'] <= before['log_loss']
    scaler, estimator = candidate.named_steps['scaler'], candidate.named_steps['model']
    trained = datetime.now(timezone.utc).isoformat()
    artifact = {'schema': 1, 'trained_at': trained, 'columns': expanded, 'base_columns': base_columns,
                'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                'coef': estimator.coef_[0].tolist(), 'intercept': float(estimator.intercept_[0]),
                'training_games': len(groups[0]), 'checks_passed': bool(passed), 'aliases': aliases,
                'games': games, 'context_refreshed_at': trained, 'report': report}
    artifact['model_id'] = 'nba_box_context_' + hashlib.sha256(json.dumps(artifact, sort_keys=True, default=str).encode()).hexdigest()[:16]
    report.update(training_performed=True, status='candidate_saved', checks_passed=bool(passed), model_id=artifact['model_id'])
    client.write(ROOT + 'candidates/' + artifact['model_id'] + '.json.gz', artifact, compressed=True)
    client.write(ARTIFACT, artifact, compressed=True, replace=True)
    client.write(ROOT + 'latest_report.json', report, replace=True)
    print(json.dumps(report)); return report


def predict(artifact, base_features, home, away, game_time):
    """Read coefficients only; no training, player downloads or sports API calls."""
    if not artifact or artifact.get('schema') != 1 or not artifact.get('checks_passed'):
        return None
    at = pd.Timestamp(game_time)
    if at.tzinfo is None:
        at = at.tz_localize('UTC')
    now = pd.Timestamp.now(tz='UTC')
    refreshed = pd.Timestamp(artifact.get('context_refreshed_at', artifact['trained_at']))
    if (pd.Timestamp(artifact['trained_at']) >= at or now >= at
            or not pd.Timedelta(0) <= now - refreshed <= pd.Timedelta(hours=24)):
        return None
    aliases = artifact['aliases']
    h, a = aliases.get(key(home)), aliases.get(key(away))
    if not h or not a or h == a:
        return None
    extra = features(artifact['games'], h, a, at)
    if extra is None:
        return None
    row = dict(base_features.iloc[0])
    row.update(extra)
    x = np.array([row[k] for k in artifact['columns']], dtype=float)
    mean, scale, coef = [np.array(artifact[k], dtype=float) for k in ('mean', 'scale', 'coef')]
    if not (x.shape == mean.shape == scale.shape == coef.shape) or not all(np.isfinite(v).all() for v in (x, mean, scale, coef)) or (scale <= 0).any():
        raise ValueError('Invalid box-context artifact')
    logit = float(np.dot((x - mean) / scale, coef) + artifact['intercept'])
    p = 1 / (1 + math.exp(-max(-40., min(40., logit))))
    return {'home_win_probability': p, 'away_win_probability': 1-p,
            'predicted_side': 'HOME' if p >= .5 else 'AWAY', 'confidence': max(p, 1-p),
            'training_games': artifact['training_games'], 'calibration_games': 0,
            'raw_home_win_probability': p, 'probability_version': artifact['model_id'],
            'calibration_status': 'Saved box-context model · retrospective checks passed; live accuracy not established',
            'winner_model_id': artifact['model_id']}


def refresh(client=None):
    """Update context from the collector checkpoint without refitting any weights."""
    from .nba_pregame_snapshots import Client, PREFIX
    client = client or Client()
    artifact = client.read(ARTIFACT, compressed=True)
    if artifact is None:
        print(json.dumps({'status': 'train_box_context_first', 'sports_api_requests': 0}))
        return
    checkpoint = client.read(PREFIX + 'checkpoint.json.gz', compressed=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema') != 1:
        raise ValueError('Saved NBA checkpoint missing')
    updates, aliases = compact_games(checkpoint, {})
    merged = {g['id']: g for g in artifact['games']}
    for g in updates:
        old = merged.get(g['id'])
        if old and not g['boxes'] and old['teams'] == g['teams'] and old['scores'] == g['scores']:
            g['boxes'] = old['boxes']
        merged[g['id']] = g
    artifact['games'] = sorted(merged.values(), key=lambda g: (g['date'], g['id']))
    artifact['aliases'].update(aliases)
    artifact['context_refreshed_at'] = datetime.now(timezone.utc).isoformat()
    client.write(ARTIFACT, artifact, compressed=True, replace=True)
    print(json.dumps({'status': 'box_context_refreshed', 'model_id': artifact['model_id'],
                      'weights_changed': False, 'sports_api_requests': 0}))


if __name__ == '__main__':
    import sys
    refresh() if '--refresh' in sys.argv else train()
