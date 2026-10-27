"""Reconstruct pregame NBA examples and train a separate experimental adjustment.

Uses saved game results and player boxes only. No sports API calls, no injury or
odds backfill, and no changes to live model weights or candidate pointers.
"""
import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

if __package__:
    from .nba_pregame_snapshots import Client, PREFIX, final, played_date, form, history_before, matchup_stats, elo_ratings
    from .nba_historical_player_collection import ROOT as HISTORY_ROOT, valid_box
    from .nba_context_residual import BASE_MODEL, FEATURE_COLUMNS, RULES, partition, fit
    from .nba_winner_comparison import load_model, predict
else:
    from nba_pregame_snapshots import Client, PREFIX, final, played_date, form, history_before, matchup_stats, elo_ratings
    from nba_historical_player_collection import ROOT as HISTORY_ROOT, valid_box
    from nba_context_residual import BASE_MODEL, FEATURE_COLUMNS, RULES, partition, fit
    from nba_winner_comparison import load_model, predict

OUTPUT = 'nba/historical_pregame_learning_v1/'


def list_samples(client):
    """List only the separate historical sample objects, never secrets or odds."""
    names = []
    offset = 0
    while True:
        url = client.env['SUPABASE_URL'].rstrip('/') + '/storage/v1/object/list/' + quote(client.bucket, safe='')
        response = client.http.post(url, headers=client.headers(), json={
            'prefix': HISTORY_ROOT + 'games', 'limit': 100, 'offset': offset,
            'sortBy': {'column': 'name', 'order': 'asc'}}, timeout=(5, 20))
        if response.status_code != 200:
            raise RuntimeError('Historical sample listing failed')
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError('Invalid historical sample listing')
        names.extend(x['name'] for x in batch if str(x.get('name', '')).endswith('.json.gz'))
        if len(batch) < 100:
            return names
        offset += len(batch)


def research_functions():
    """Use the same pregame features as the saved neural winner model."""
    path = Path(__file__).with_name('nba_research.py')
    tree = ast.parse(path.read_text())
    names = {'prepare_balldontlie_games_for_research', '_summarize_balldontlie_history',
             'build_balldontlie_pregame_features'}
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                 t.id == 'NBA_LIVE_FEATURE_COLUMNS' for t in n.targets))]
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def prepare_rows(games, boxes, now=None):
    """Strictly previous calendar dates; missing player context stays unknown."""
    now = now or datetime.now(timezone.utc)
    valid = []
    for g in games:
        try:
            d = played_date(g)
            hs, aws = float(g['home_team_score']), float(g['visitor_team_score'])
            h, a = g['home_team']['id'], g['visitor_team']['id']
            if final(g) and d and d < now.date() and hs > 0 and aws > 0 and hs != aws and h != a:
                valid.append(g)
        except (KeyError, TypeError, ValueError):
            continue
    by_id = {str(g['id']): g for g in valid}
    records = [{'game_id': g['id'], 'game_date': str(played_date(g)), 'season': g['season'],
                'home_team': str(g['home_team']['id']), 'away_team': str(g['visitor_team']['id']),
                'home_points': g['home_team_score'], 'away_points': g['visitor_team_score'],
                'home_win': int(g['home_team_score'] > g['visitor_team_score'])} for g in valid]
    if not records:
        return [], {'valid_completed_games': 0, 'saved_player_boxes': 0}
    model = load_model()
    if model['meta'].get('model_id') != BASE_MODEL:
        raise ValueError('Unexpected saved neural model')
    calibration_end = pd.Timestamp(model['meta']['calibration_end']).date()
    research = research_functions()
    frame = research['build_balldontlie_pregame_features'](pd.DataFrame(records))
    predictors = {str(r['game_id']): r for _, r in frame.iterrows()}
    rows = []
    skipped = {'before_base_cutoff': 0, 'missing_base_history': 0, 'missing_prior_boxes': 0}
    for gid, g in sorted(by_id.items(), key=lambda x: (played_date(x[1]), x[0])):
        d = played_date(g)
        if d <= calibration_end:
            skipped['before_base_cutoff'] += 1
            continue
        if gid not in predictors:
            skipped['missing_base_history'] += 1
            continue
        # Using midnight UTC, rather than guessed tipoff times, excludes the
        # target day's games and all same-day player-box outcomes.
        cutoff = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
        prior = history_before(valid, cutoff)
        season_prior = [p for p in prior if p.get('season') == g.get('season')]
        h, a = g['home_team']['id'], g['visitor_team']['id']
        hm, am = matchup_stats(season_prior, boxes, h), matchup_stats(season_prior, boxes, a)
        if hm.get('games', 0) < 2 or am.get('games', 0) < 2:
            skipped['missing_prior_boxes'] += 1
            continue
        features = {name: None for name in FEATURE_COLUMNS}
        ratings, _ = elo_ratings(prior)
        for side, tid, match in (('home', h, hm), ('away', a, am)):
            team_form = form(season_prior, tid, d)
            features[side + '_elo'] = ratings.get(tid, 1500.)
            features[side + '_form_margin_8'] = team_form.get('margin_8')
            features[side + '_form_days_since_last_game'] = team_form.get('days_since_last_game')
            features[side + '_matchup_estimated_pace_48'] = match.get('estimated_pace_48')
        for name in FEATURE_COLUMNS:
            if name.endswith('__missing'):
                features[name] = int(features.get(name[:-9]) is None)
        base = predict(pd.DataFrame([predictors[gid]]), cutoff)['learning_home_probability']
        # Conservative availability proxy: prior day-end; actual historical
        # publication times are unavailable and no frozen forecast is claimed.
        observed = cutoff + timedelta(days=1)
        if observed > now:
            continue
        rows.append({'game_id': gid, 'start_time': cutoff.isoformat(), 'input_time': cutoff.isoformat(),
                     'label_observed_at': observed.isoformat(), 'base_probability': base,
                     'base_model_id': BASE_MODEL, 'features': features,
                     'label': int(g['home_team_score'] > g['visitor_team_score']),
                     'source': 'retrospective_reconstruction',
                     'home_prior_box_games': hm['games'], 'away_prior_box_games': am['games']})
    return rows, {'valid_completed_games': len(valid), 'saved_player_boxes': len(boxes), **skipped}


def run(client=None, now=None):
    client = client or Client()
    now = now or datetime.now(timezone.utc)
    checkpoint = client.read(PREFIX + 'checkpoint.json.gz', compressed=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema') != 1:
        raise ValueError('Collector checkpoint unavailable')
    boxes = {}
    for name in list_samples(client):
        sample = client.read(HISTORY_ROOT + 'games/' + name, compressed=True)
        if not isinstance(sample, dict):
            raise ValueError('Missing saved sample')
        game, stats = sample.get('game'), sample.get('player_stats')
        if not isinstance(game, dict) or str(game.get('id')) != str(sample.get('game_id')) or not valid_box(game, stats):
            raise ValueError('Invalid saved player box')
        boxes[str(game['id'])] = stats
    rows, diagnostics = prepare_rows(list(checkpoint.get('games', {}).values()), boxes, now)
    groups = partition(rows)
    reasons = []
    for name, minimum in RULES['minimum_games'].items():
        if len(groups[name]) < minimum:
            reasons.append(f'{name}: {len(groups[name])}/{minimum} games')
        if min(sum(r['label'] == y for r in groups[name]) for y in (0, 1)) < RULES['minimum_classes'][name]:
            reasons.append(name + ': insufficient examples of both outcomes')
    report = {'status': 'waiting_for_history', 'reconstructed_games': len(rows),
              'partition_games': {k: len(v) for k, v in groups.items()}, 'reasons': reasons,
              'coverage': diagnostics, 'training_performed': False, 'live_model_changed': False,
              'sports_api_requests': 0, 'source': 'retrospective_reconstruction',
              'limitations': ['Only player boxes from previous calendar days are used.',
                  'Historical injury reports, odds, and actual saved forecasts are unavailable.',
                  'Publication timestamps are assumed conservatively; they are not independently verified.',
                  'The 2026 period was previously inspected as a neural holdout; results are exploratory, not an untouched benchmark.',
                  'Experimental candidate is separate and cannot automatically replace live picks.']}
    if not reasons:
        package, metrics = fit(groups, now)
        report.update(training_performed=True, metrics=metrics,
                      status='candidate_saved' if package else 'candidate_did_not_pass_validation')
        if package:
            path = OUTPUT + 'candidates/' + package['metadata']['model_id'] + '.json.gz'
            if client.read(path, compressed=True) is None:
                client.write(path, package, compressed=True)
            report['candidate_path'] = path
    client.write(OUTPUT + 'latest_report.json', report, replace=True)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        print('Historical pregame learning stopped: ' + type(exc).__name__, flush=True)
        raise SystemExit(1)
