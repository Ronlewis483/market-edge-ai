"""Opponent interactions from saved completed boxes, shared by fit and inference.

These are matchup descriptors, not calibrated win probabilities or causal claims.
No provider requests, target-day results, betting lines or future roster data.
"""
import math

import pandas as pd

VERSION = 1
METRICS = ('shooting', 'turnovers', 'rebounding', 'free_throws', 'scoring',
           'offensive_rating', 'defensive_rating')
COLUMNS = [side + '_matchup_' + metric
           for side in ('home', 'away', 'difference') for metric in METRICS]


def profile(games, team, cutoff):
    team, cutoff = str(team), pd.Timestamp(cutoff).date()
    prior = sorted((g for g in games if team in g['teams'] and
                    pd.Timestamp(g['date']).date() < cutoff),
                   key=lambda g: (g['date'], g['id']))
    if not prior or (cutoff - pd.Timestamp(prior[-1]['date']).date()).days > 10:
        return None
    segment = []
    newer = prior[-1]
    for game in reversed(prior):
        if game['season'] != prior[-1]['season'] or (
                pd.Timestamp(newer['date']) - pd.Timestamp(game['date'])).days > 45:
            break
        segment.append(game)
        newer = game
        if len(segment) == 8:
            break
    pairs = []
    for game in segment:
        boxes = game.get('boxes') or {}
        opponent = next(t for t in game['teams'] if t != team)
        own, defense = boxes.get(team), boxes.get(opponent)
        if not own or not defense:
            continue
        fields = ('fgm', 'fg3m', 'fga', 'fta', 'turnover', 'reb', 'poss')
        if any(k not in b or not isinstance(b[k], (int, float)) or
               not math.isfinite(b[k]) or b[k] < 0 for b in (own, defense) for k in fields):
            continue
        if any(b['fga'] <= 0 or b['poss'] <= 0 or b['fgm'] > b['fga'] or
               b['fg3m'] > b['fgm'] for b in (own, defense)):
            continue
        pairs.append((game, own, defense))
    if len(pairs) < 4:
        return None
    def total(index, field):
        return sum(p[index][field] for p in pairs)
    rebounds = total(1, 'reb') + total(2, 'reb')
    if rebounds <= 0:
        return None
    possessions = sum((a['poss'] + b['poss']) / 2 for _, a, b in pairs)
    return dict(
        shooting=(total(1, 'fgm') + .5 * total(1, 'fg3m')) / total(1, 'fga'),
        allowed_shooting=(total(2, 'fgm') + .5 * total(2, 'fg3m')) / total(2, 'fga'),
        turnovers=total(1, 'turnover') / total(1, 'poss'),
        forced_turnovers=total(2, 'turnover') / total(2, 'poss'),
        rebounding=total(1, 'reb') / rebounds,
        free_throws=total(1, 'fta') / total(1, 'fga'),
        allowed_free_throws=total(2, 'fta') / total(2, 'fga'),
        offensive_rating=100 * sum(g['scores'][team] for g, _, _ in pairs) / possessions,
        defensive_rating=100 * sum(g['scores'][next(t for t in g['teams'] if t != team)]
                                   for g, _, _ in pairs) / possessions,
    )


def features(games, home, away, cutoff):
    h, a = profile(games, home, cutoff), profile(games, away, cutoff)
    if h is None or a is None:
        return None
    def attack(own, opponent):
        return dict(
            shooting=own['shooting'] * opponent['allowed_shooting'],
            turnovers=own['turnovers'] * opponent['forced_turnovers'],
            rebounding=own['rebounding'] * (1 - opponent['rebounding']),
            free_throws=own['free_throws'] * opponent['allowed_free_throws'],
            scoring=own['offensive_rating'] * opponent['defensive_rating'] / 100,
            offensive_rating=own['offensive_rating'],
            defensive_rating=own['defensive_rating'],
        )
    h, a = attack(h, a), attack(a, h)
    return {**{'home_matchup_' + k: h[k] for k in METRICS},
            **{'away_matchup_' + k: a[k] for k in METRICS},
            **{'difference_matchup_' + k: h[k] - a[k] for k in METRICS}}
