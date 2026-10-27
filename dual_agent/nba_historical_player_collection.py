"""Collect a small, separate NBA player-box history sample; no live-model changes.

Run from the repository root: python dual_agent/nba_historical_player_collection.py
Requires the existing BALLDONTLIE and Supabase settings. No Odds API requests.
"""
import os
from datetime import datetime, timedelta, timezone
import json

if __package__:
    from .nba_pregame_snapshots import Client, PREFIX, final, played_date, number, team_box
else:
    from nba_pregame_snapshots import Client, PREFIX, final, played_date, number, team_box

ROOT = 'nba/historical_player_sample_v1/'


def valid_box(game, rows):
    """Require both player boxes to add up to the final scores before saving."""
    if not isinstance(rows, list) or not rows:
        return False
    gid = str(game['id'])
    if any(str((r.get('game') or {}).get('id')) != gid for r in rows):
        return False
    for side in ('home', 'visitor'):
        team = game.get(side + '_team') or {}
        score = number(game.get(side + '_team_score'))
        if team.get('id') is None or score is None:
            return False
        box = team_box(rows, team['id'])
        if box is None or box['pts'] != score:
            return False
    return True


def run(client=None, now=None):
    client = client or Client()
    now = now or datetime.now(timezone.utc)
    if os.environ.get('NBA_HISTORY_STOP_WHEN_TRAINED') == 'true':
        learning = client.read('nba/historical_pregame_learning_v1/latest_report.json')
        if (isinstance(learning, dict) and learning.get('training_performed') is True
                and learning.get('status') in ('candidate_saved', 'candidate_did_not_pass_validation')):
            result = {'status': 'stopped_training_attempt_completed', 'new_saved': 0,
                      'balldontlie_requests': 0, 'odds_api_requests': 0,
                      'training_status': learning['status'], 'live_model_changed': False}
            print(json.dumps(result), flush=True)
            return result
    batch_limit = int(os.environ.get('NBA_HISTORY_SAMPLE_GAMES', '10'))
    if not 1 <= batch_limit <= 100:
        raise ValueError('NBA_HISTORY_SAMPLE_GAMES must be between 1 and 100')
    checkpoint = client.read(PREFIX + 'checkpoint.json.gz', compressed=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema') != 1:
        raise ValueError('NBA collector checkpoint missing; run the collector first')
    cutoff = now.date() - timedelta(days=1)
    games = [g for g in checkpoint.get('games', {}).values()
             if final(g) and played_date(g) and played_date(g) < cutoff]
    games.sort(key=lambda g: (played_date(g), str(g['id'])), reverse=True)
    counts = {'eligible_games_in_checkpoint': len(games), 'already_saved': 0,
              'new_saved': 0, 'reused_collector_boxes': 0, 'invalid_boxes': 0,
              'attempted_new_games': 0, 'request_budget_reached': False}
    for game in games:
        gid = str(game['id'])
        path = ROOT + 'games/' + gid + '.json.gz'
        existing = client.read(path, compressed=True)
        if existing is not None:
            if not isinstance(existing, dict) or str(existing.get('game_id')) != gid:
                raise ValueError('Existing historical sample is invalid; it was not replaced')
            counts['already_saved'] += 1
            continue
        if counts['attempted_new_games'] >= batch_limit:
            break
        counts['attempted_new_games'] += 1
        rows = checkpoint.get('stats', {}).get(gid)
        reused = rows is not None
        if rows is None:
            if client.calls >= client.max_calls:
                counts['request_budget_reached'] = True
                break
            try:
                rows = client.pages('stats', {'game_ids[]': [game['id']]})
            except RuntimeError as exc:
                if str(exc).startswith('BALLDONTLIE run budget reached'):
                    counts['request_budget_reached'] = True
                    break
                raise
        if not valid_box(game, rows):
            counts['invalid_boxes'] += 1
            continue
        payload = {'schema': 1, 'game_id': game['id'], 'game_date': str(played_date(game)),
                   'game': game, 'player_stats': rows, 'collected_at': now.isoformat(),
                   'source': 'collector_cache' if reused else 'balldontlie',
                   'use_policy': 'Completed game only; player stats are available for later games, never this game as a pregame feature.'}
        client.write(path, payload, compressed=True)
        counts['new_saved'] += 1
        counts['reused_collector_boxes'] += int(reused)
    counts.update({'status': 'sample_collected' if counts['new_saved'] else 'no_new_boxes_saved',
                   'balldontlie_requests': client.calls, 'odds_api_requests': 0,
                   'training_performed': False, 'live_model_changed': False,
                   'note': 'Separate completed-game samples only; not historical injury reports, lineups, or pregame forecasts.'})
    client.write(ROOT + 'latest_report.json', counts, replace=True)
    print(json.dumps(counts), flush=True)
    return counts


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        print('Historical player collection stopped: ' + type(exc).__name__, flush=True)
        raise SystemExit(1)
