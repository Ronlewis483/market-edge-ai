"""Read-only NFL spreads and MLB run lines; automatically collect NFL props.
Existing winner and prop equations are reused unchanged. Spread estimates are
new, unvalidated estimates, never substituted for winner confidence.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from scipy.stats import skellam

NFL_FEATURES = ['win_pct_diff', 'avg_point_diff_diff', 'recent_5_win_pct_diff',
                'recent_5_point_diff_diff', 'rest_diff']
SPORTS = {'NFL': 'americanfootball_nfl', 'MLB': 'baseball_mlb'}


def records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict('records')
    return list(value or [])


def game_fields(row):
    home = row.get('home_team', row.get('Home'))
    away = row.get('away_team', row.get('Away'))
    if (not home or not away) and row.get('Game'):
        teams = str(row['Game']).split(' @ ', 1)
        if len(teams) == 2:
            away, home = teams
    stamp = pd.to_datetime(row.get('commence_time', row.get('start_time', row.get('game_time'))), utc=True, errors='coerce')
    return home, away, stamp


def matching_game_rows(game, candidates):
    """Exact teams and kickoff only; no same-day guesses or doubleheader mixing."""
    home, away, stamp = game_fields(game)
    if pd.isna(stamp) or not home or not away:
        return []
    matches = []
    for row in records(candidates):
        rh, ra, rt = game_fields(row)
        if rh == home and ra == away and pd.notna(rt) and abs(rt-stamp) <= pd.Timedelta(minutes=15):
            matches.append(row)
    return matches


def fetch_spread_events(league, key):
    if not key:
        raise ValueError('Sportsbook API key unavailable.')
    response = requests.get(f'https://api.the-odds-api.com/v4/sports/{SPORTS[league]}/odds',
        params={'apiKey': key, 'regions': 'us', 'markets': 'spreads', 'oddsFormat': 'decimal'}, timeout=20)
    if response.status_code != 200:
        # Do not include credential-bearing request URLs in errors.
        raise ValueError(f'Spread feed unavailable (HTTP {response.status_code}).')
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError('Unexpected spread-feed response.')
    return payload


def paired_spread_lines(event, now):
    """Require two distinct books offering the same opposite-sided fresh line."""
    home, away, _ = game_fields(event)
    grouped = defaultdict(dict)
    for book in event.get('bookmakers', []):
        book_id = book.get('key')
        if not book_id:
            continue
        for market in book.get('markets', []):
            stamp = pd.to_datetime(market.get('last_update') or book.get('last_update'), utc=True, errors='coerce')
            if market.get('key') != 'spreads' or pd.isna(stamp) or not pd.Timedelta(0) <= now-stamp <= pd.Timedelta(minutes=30):
                continue
            outcomes = market.get('outcomes', [])
            if len(outcomes) != 2:
                continue
            by_team = {o.get('name'): o for o in outcomes}
            if set(by_team) != {home, away}:
                continue
            try:
                line = float(by_team[home]['point'])
                other = float(by_team[away]['point'])
                hp, ap = float(by_team[home]['price']), float(by_team[away]['price'])
            except (ValueError, KeyError, TypeError):
                continue
            if not all(np.isfinite(v) for v in (line, other, hp, ap)) or abs(line+other) > 1e-8 or min(hp, ap) <= 1:
                continue
            grouped[line][book_id] = (hp, ap)
    return [(line, books) for line, books in grouped.items() if len(books) >= 2]


def nfl_margin_model(history, now):
    """Fit margins separately from the winner model; use later held-out errors."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    frame = history.copy()
    needed = NFL_FEATURES + ['home_score', 'away_score', 'start_time']
    if not set(needed).issubset(frame.columns):
        raise ValueError('Completed NFL scoring history missing required margin features.')
    frame['start_time'] = pd.to_datetime(frame['start_time'], utc=True, errors='coerce')
    for column in NFL_FEATURES + ['home_score', 'away_score']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.loc[frame.start_time < now].replace([np.inf, -np.inf], np.nan).dropna(subset=needed).sort_values('start_time')
    if len(frame) < 150:
        raise ValueError('Need at least 150 completed NFL games for a spread estimate.')
    # Split entire kickoff cohorts to keep simultaneous games on the same side.
    split_time = frame.iloc[int(len(frame)*.8)]['start_time']
    earlier, later = frame.loc[frame.start_time < split_time], frame.loc[frame.start_time >= split_time]
    if len(earlier) < 100 or len(later) < 30:
        raise ValueError('Insufficient independent later-game margin sample.')
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(earlier[NFL_FEATURES], earlier.home_score-earlier.away_score)
    residuals = (later.home_score-later.away_score).to_numpy()-model.predict(later[NFL_FEATURES])
    model.fit(frame[NFL_FEATURES], frame.home_score-frame.away_score)
    return model, residuals


def nfl_cover_distribution(margin, residuals, home_line):
    adjusted = np.rint(margin+np.asarray(residuals))+home_line
    n = len(adjusted)
    if not n:
        raise ValueError('No independent margin errors available.')
    # Discrete simulated final margins retain pushes at whole-number spreads.
    whole_line = abs(home_line-round(home_line)) < 1e-8
    denominator = n + (1.5 if whole_line else 1.0)
    home = float((np.sum(adjusted > 0)+.5)/denominator)
    away = float((np.sum(adjusted < 0)+.5)/denominator)
    push = float((np.sum(adjusted == 0)+.5)/denominator) if whole_line else 0.0
    return home, away, push


def mlb_cover_distribution(home_runs, away_runs, home_line):
    if not all(np.isfinite(v) and v > 0 for v in (home_runs, away_runs)):
        raise ValueError('Projected run rates unavailable.')
    # The base model allocates ties equally. Final baseball margins have no zero:
    # conservatively allocate modeled tie mass to +1/-1 for extra innings.
    tie = float(skellam.pmf(0, home_runs, away_runs))
    threshold = -home_line
    home = float(skellam.sf(threshold, home_runs, away_runs))
    integer = abs(threshold-round(threshold)) < 1e-8
    push = float(skellam.pmf(round(threshold), home_runs, away_runs)) if integer else 0.0
    home -= tie*int(home_line > 0)
    push -= tie*int(abs(home_line) < 1e-8)
    for margin in (1, -1):
        home += .5*tie*int(margin+home_line > 0)
        push += .5*tie*int(abs(margin+home_line) < 1e-8)
    home, push = max(0., min(1., home)), max(0., min(1., push))
    return home, max(0., 1-home-push), push


def generate_spread_picks(league, predictions, events, history=None, now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    rows = records(predictions)
    margin_model = residuals = None
    if league == 'NFL':
        margin_model, residuals = nfl_margin_model(history, now)
        from dual_agent.nfl_research import build_nfl_future_matchup_features
    picks, issues = [], []
    for game in rows:
        home, away, kickoff = game_fields(game)
        if pd.isna(kickoff) or not now < kickoff <= now+pd.Timedelta(days=7):
            continue
        matches = matching_game_rows(game, events)
        if len(matches) != 1:
            issues.append(f'{away} @ {home}: no unique sportsbook spread event.')
            continue
        lines = paired_spread_lines(matches[0], now)
        if not lines:
            issues.append(f'{away} @ {home}: need two fresh books at the same spread.')
            continue
        try:
            if league == 'NFL':
                features = build_nfl_future_matchup_features(history, home, away, kickoff)
                x = features[NFL_FEATURES].apply(pd.to_numeric, errors='coerce')
                if x.empty or not np.isfinite(x.to_numpy()).all():
                    raise ValueError('Matchup margin inputs incomplete.')
                margin = float(margin_model.predict(x.iloc[[0]])[0])
            else:
                hr, ar = float(game['Projected home runs']), float(game['Projected away runs'])
                margin = hr-ar
                if game.get('Missing model inputs') or str(game.get('Input status', '')).startswith('Early forecast'):
                    raise ValueError('Pregame player inputs incomplete; spread estimate withheld.')
            # Prefer the widely offered line, rather than shopping alternative lines for inflated cover chance.
            line, books = sorted(lines, key=lambda item: (-len(item[1]), abs(item[0]), item[0]))[0]
            hc, ac, push = nfl_cover_distribution(margin, residuals, line) if league == 'NFL' else mlb_cover_distribution(hr, ar, line)
            if 1-push <= 0:
                raise ValueError('Spread outcomes cannot be estimated.')
            home_side = hc >= ac
            chance = (hc if home_side else ac)/(1-push)
            prices = [pair[0 if home_side else 1] for pair in books.values()]
            market_chances = [(1/(pair[0] if home_side else pair[1]))/(1/pair[0]+1/pair[1]) for pair in books.values()]
            picks.append({'home_team': home, 'away_team': away, 'commence_time': kickoff.isoformat(),
                'spread_team': home if home_side else away, 'spread_line': line if home_side else -line,
                'cover_chance': chance, 'push_chance': push, 'projected_home_margin': margin,
                'market_chance': float(np.mean(market_chances)), 'decimal_odds': max(prices),
                'books': len(books), 'captured_at': now.isoformat(), 'validated': False,
                'method': 'Historical margin model + held-out errors' if league == 'NFL' else 'Saved projected runs + run-difference distribution',
                'sample_size': len(residuals) if league == 'NFL' else None})
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f'{away} @ {home}: {exc}')
    return {'picks': picks, 'issues': issues}


def automatic_nfl_props(predictions, key):
    """Reuse existing NFL prop guards/ranking, scoped to predicted games."""
    from dual_agent.nfl_player_props_ui import cached_events, cached_props, cached_player_history, upcoming_week_events
    from dual_agent.nfl_player_props import MARKETS, normalize_props
    from dual_agent.nfl_player_prop_prediction_pipeline import run_nfl_player_prop_prediction_pipeline
    now = pd.Timestamp.now(tz='UTC')
    predicted = records(predictions)
    selected = [event for event in upcoming_week_events(cached_events(key), now)
                if len(matching_game_rows(event, predicted)) == 1]
    frames, issues = [], []
    def load(event):
        data, _ = cached_props(key, event['id'], tuple(MARKETS.values()))
        return normalize_props(data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(load, event): event for event in selected}
        for future in as_completed(pending):
            event = pending[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                else:
                    issues.append(f"{event.get('away_team')} @ {event.get('home_team')}: no offered prop lines.")
            except Exception as exc:
                issues.append(f"{event.get('away_team')} @ {event.get('home_team')}: prop feed unavailable ({type(exc).__name__}).")
    if not frames:
        return pd.DataFrame(), issues
    history = cached_player_history((2025, 2026))
    # The existing pipeline selects its top players internally. Run that same
    # equation per event so the slate-wide player limit cannot starve a game.
    forecasts = []
    lines = pd.concat(frames, ignore_index=True)
    for event_id, group in lines.groupby('event_id', sort=False):
        try:
            props = run_nfl_player_prop_prediction_pipeline(group, history,
                window=12, max_results=200, horizon_days=7)
            if not props.empty:
                forecasts.append(props)
        except Exception as exc:
            event = next((e for e in selected if e['id'] == event_id), {})
            issues.append(f"{event.get('away_team')} @ {event.get('home_team')}: prop model unavailable ({type(exc).__name__}).")
    return pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(), issues


def enhance_result(league, result, key, history=None):
    """Failures in add-ons never discard working winner predictions."""
    result = dict(result)
    predictions = result.get('all_predictions', result.get('predictions'))
    if not records(predictions):
        return result
    try:
        events = fetch_spread_events(league, key)
        result['spreads'] = generate_spread_picks(league, predictions, events, history)
    except Exception as exc:
        result['spreads'] = {'picks': [], 'issues': [f'Spread estimates unavailable: {exc}' if isinstance(exc, ValueError) else f'Spread estimates unavailable ({type(exc).__name__}).']}
    if league == 'NFL':
        try:
            props, issues = automatic_nfl_props(predictions, key)
            result['nfl_player_props'], result['prop_issues'] = props, issues
        except Exception as exc:
            result['nfl_player_props'] = pd.DataFrame()
            result['prop_issues'] = [f'Player props unavailable ({type(exc).__name__}).']
    return result
