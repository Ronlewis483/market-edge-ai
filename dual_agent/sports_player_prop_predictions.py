"""Pregame player-prop estimates using actual lines and prior appearances.

These are historical hit-rate estimates shrunk toward bookmaker consensus,
not calibrated win probabilities from the game-winner model.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import requests

MARKETS = {'MLB': {
    'batter_hits': ('Hits', 'hits'),
    'batter_home_runs': ('Home runs', 'batting_home_runs'),
    'batter_total_bases': ('Total bases', 'total_bases'),
    'batter_rbis': ('RBIs', 'rbi'),
    'batter_runs_scored': ('Runs scored', 'batting_runs'),
    'batter_hits_runs_rbis': ('Hits + runs + RBIs', 'hits_runs_rbis'),
    'batter_singles': ('Singles', 'singles'),
    'batter_doubles': ('Doubles', 'doubles'),
    'batter_triples': ('Triples', 'triples'),
    'batter_walks': ('Batter walks', 'batting_walks'),
    'batter_strikeouts': ('Batter strikeouts', 'batting_strikeouts'),
    'batter_stolen_bases': ('Stolen bases', 'stolen_bases'),
    'pitcher_strikeouts': ('Pitcher strikeouts', 'pitcher_strikeouts'),
    'pitcher_hits_allowed': ('Hits allowed', 'pitcher_hits'),
    'pitcher_walks': ('Pitcher walks', 'pitcher_walks'),
    'pitcher_earned_runs': ('Earned runs allowed', 'earned_runs'),
    'pitcher_outs': ('Outs recorded', 'pitching_outs')},
    'NBA': {'player_points': ('Points', 'pts'), 'player_rebounds': ('Rebounds', 'reb'), 'player_assists': ('Assists', 'ast')}}


def name_key(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value)).lower() if c.isalnum())


def estimate_prop(values, line, side, market_probability):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)][-30:]
    wins = int((values > line).sum()) if side == 'Over' else int((values < line).sum())
    pushes = int((values == line).sum())
    decisions = len(values)-pushes
    if len(values) < 10 or decisions < 10:
        return None
    # Preserve the existing short-sample shrinkage and eligibility criteria.
    chance = (wins+10*market_probability)/(decisions+10)
    return {'Estimated chance': float(chance), 'Projected stat': float(values.mean()),
            'Prior games': len(values), 'Historical wins': wins, 'Historical pushes': pushes,
            'Historical hit rate': wins/decisions, 'Estimate method': 'Prior 30 appearances + market shrinkage'}


def _get(url, key, params=None, balldontlie=False):
    response = requests.get(url, params=params, headers={'Authorization': key} if balldontlie else None, timeout=20)
    if response.status_code != 200:
        raise RuntimeError('Player-prop data request failed (HTTP '+str(response.status_code)+').')
    return response.json()


MLB_PROP_MATCH_VERSION = 4


def mlb_team_key(value):
    key = name_key(value)
    return {'ladodgers': 'losangelesdodgers', 'laangels': 'losangelesangels',
            'oaklandathletics': 'athletics', 'sacramentoathletics': 'athletics',
            'clevelandindians': 'clevelandguardians'}.get(key, key)


def match_mlb_prop_events(games, events, now):
    """One-to-one matching; allow changed times only for a unique same-day game."""
    valid = {}
    for event in events:
        stamp = pd.to_datetime(event.get('commence_time'), utc=True, errors='coerce')
        if event.get('id') and pd.notna(stamp) and stamp > now:
            valid[str(event['id'])] = (event, stamp)
    prepared = []
    for game in games:
        start = pd.to_datetime(game.get('start_time') or game.get('commence_time'), utc=True, errors='coerce')
        if pd.notna(start) and now < start <= now+pd.Timedelta(days=7):
            prepared.append((game, start, mlb_team_key(game['home_team']), mlb_team_key(game['away_team'])))
    proposed = []; diagnostics = []
    for game, start, home, away in prepared:
        details = {'Game': game['away_team']+' @ '+game['home_team'],
                   'MLB start UTC': start.isoformat(), 'Odds events returned': len(valid)}
        pair = [(event, stamp) for event, stamp in valid.values()
                if mlb_team_key(event.get('home_team')) == home and mlb_team_key(event.get('away_team')) == away]
        strict = [(e,t) for e,t in pair if abs(t-start) <= pd.Timedelta(minutes=15)]
        if len(strict) == 1:
            event, stamp = strict[0]; method = 'Teams and start within 15 minutes'
        elif len(strict) > 1:
            diagnostics.append(dict(details, Reason='Multiple sportsbook events match this start time.')); continue
        else:
            day = start.tz_convert('America/Chicago').date()
            same_day = [(e,t) for e,t in pair if t.tz_convert('America/Chicago').date() == day]
            same_games = [g for g,t,h,a in prepared if h==home and a==away and t.tz_convert('America/Chicago').date()==day]
            if len(same_day)==1 and len(same_games)==1 and abs(same_day[0][1]-start)<=pd.Timedelta(hours=3):
                event, stamp = same_day[0]; method = 'Unique same-day matchup; start times differ'
            else:
                reversed_pair = any(mlb_team_key(e.get('home_team'))==away and mlb_team_key(e.get('away_team'))==home for e,t in valid.values())
                reason = ('No event returned for these teams and home/away order.' if not pair else
                          'Start-time mismatch or multiple same-day games; automatic match withheld.')
                if not pair and reversed_pair: reason = 'Odds feed has the opposite home/away order; automatic match withheld.'
                diagnostics.append(dict(details, Reason=reason, **{'Odds starts UTC': [t.isoformat() for e,t in pair]})); continue
        proposed.append((game, event, dict(details, **{'Odds start UTC':stamp.isoformat(), 'Match':method})))
    counts = {}
    for game,event,details in proposed: counts[event['id']] = counts.get(event['id'],0)+1
    matched = {}; notes = {}
    for game,event,details in proposed:
        gid = int(game['game_id'])
        if counts[event['id']] != 1:
            diagnostics.append(dict(details, Reason='One sportsbook event would map to multiple MLB games.')); continue
        matched[gid] = event; notes[gid] = details
    return matched, notes, diagnostics


def fetch_prop_quotes(league, games, odds_key):
    sport = 'baseball_mlb' if league=='MLB' else 'basketball_nba'
    base = 'https://api.the-odds-api.com/v4/sports/'+sport
    now = pd.Timestamp.now(tz='UTC'); quotes = []; errors = []
    params = {'apiKey': odds_key}
    if league == 'MLB':
        params.update(dateFormat='iso', commenceTimeFrom=now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                      commenceTimeTo=(now+pd.Timedelta(days=7,hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ'))
    events = _get(base+'/events', odds_key, params)
    matches = {}; match_notes = {}
    if league == 'MLB':
        matches, match_notes, errors = match_mlb_prop_events(games, events, now)
    from dual_agent.mlb_historical_odds import team
    def collect(game):
        quotes = []; errors = []
        start = pd.to_datetime(game.get('start_time') or game.get('commence_time'), utc=True)
        if not now<start<=now+pd.Timedelta(days=7): return quotes, errors
        if league == 'MLB':
            event = matches.get(int(game['game_id']))
            if event is None: return quotes, errors
        else:
            matched = [event for event in events if team(event.get('home_team'))==team(game['home_team']) and team(event.get('away_team'))==team(game['away_team']) and abs(pd.to_datetime(event['commence_time'], utc=True)-start)<=pd.Timedelta(minutes=15)]
            if len(matched)!=1:
                errors.append('No unique prop event match for '+game['away_team']+' at '+game['home_team']); return quotes, errors
            event = matched[0]
        try:
            payload = _get(base+'/events/'+event['id']+'/odds', odds_key, {'apiKey': odds_key, 'regions': 'us', 'markets': ','.join(MARKETS[league]), 'oddsFormat': 'decimal'})
        except RuntimeError as exc:
            errors.append(str(exc)); return quotes, errors
        stamp = pd.Timestamp.now(tz='UTC')
        if league == 'MLB' and not payload.get('bookmakers'):
            errors.append({'Game':game['away_team']+' @ '+game['home_team'], 'Reason':'Event matched, but no sportsbook prop lines were returned.'})
        for book in payload.get('bookmakers', []):
            for market in book.get('markets', []):
                key = market.get('key')
                if key not in MARKETS[league]: continue
                updated = pd.to_datetime(market.get('last_update') or book.get('last_update'), utc=True, errors='coerce')
                if pd.isna(updated) or updated>stamp or stamp-updated>pd.Timedelta(minutes=30) or updated>=start: continue
                paired = {}
                for outcome in market.get('outcomes', []):
                    try:
                        player = str(outcome['description']); line = float(outcome['point']); price = float(outcome['price']); side = outcome['name']
                        if not np.isfinite([line,price]).all() or price<=1 or side not in ['Over','Under']: continue
                        paired.setdefault((player,line), {})[side] = price
                    except (KeyError, TypeError, ValueError): continue
                for (player,line), sides in paired.items():
                    if league != 'MLB' and set(sides) != {'Over','Under'}: continue
                    over = 1/sides['Over'] if 'Over' in sides else None
                    under = 1/sides['Under'] if 'Under' in sides else None
                    quotes.append({'player':player,'line':line,'market':key,'over_probability':over/(over+under) if over is not None and under is not None else None,
                                   'over_decimal_odds':sides.get('Over'), 'under_decimal_odds':sides.get('Under'), 'book':book.get('key'), 'updated_at':updated.isoformat(), 'event_id':event['id'],
                                   'home_team':game['home_team'],'away_team':game['away_team'],'start_time':start.isoformat(),
                                   'game_id':game.get('game_id'), 'capture_time':stamp.isoformat(),
                                   **({'Event match':match_notes[int(game['game_id'])]['Match']} if league=='MLB' else {})})
        return quotes, errors
    # Separate HTTP requests; bounded concurrency also preserves slate order.
    with ThreadPoolExecutor(max_workers=4 if league == 'MLB' else 1) as pool:
        for game_quotes, game_errors in pool.map(collect, games):
            quotes.extend(game_quotes); errors.extend(game_errors)
    return quotes, errors


def _nba_history(player, api_key):
    """Cache immutable completed box scores daily; never include ongoing games."""
    now = pd.Timestamp.now(tz='UTC')
    root = Path('nba_player_history_cache'); root.mkdir(exist_ok=True)
    path = root/(name_key(player)+'_'+now.strftime('%Y%m%d')+'.json')
    if path.exists(): return json.loads(path.read_text())
    base = 'https://api.balldontlie.io/nba/v1'
    result = _get(base+'/players', api_key, {'search':player,'per_page':100}, True)
    matches = [p for p in result.get('data', []) if name_key(p.get('first_name','')+' '+p.get('last_name',''))==name_key(player)]
    if len(matches)!=1: return []
    rows = []; cursor = None
    for _ in range(10):
        params = {'player_ids[]':matches[0]['id'],'start_date':(now-pd.Timedelta(days=365)).strftime('%Y-%m-%d'), 'end_date':(now-pd.Timedelta(days=2)).strftime('%Y-%m-%d'),'per_page':100}
        if cursor is not None: params['cursor'] = cursor
        response = _get(base+'/stats', api_key, params, True)
        for row in response.get('data', []):
            game = row.get('game', {})
            date = pd.to_datetime(game.get('date'), utc=True, errors='coerce')
            minutes = str(row.get('min') or '0').split(':')[0]
            try: played = float(minutes or 0)>0
            except ValueError: played = False
            if (str(game.get('status','')).startswith('Final') and pd.notna(date)
                    and date+pd.Timedelta(hours=48)<now and played):
                rows.append(dict(row, source_date=date.isoformat()))
        cursor = response.get('meta', {}).get('next_cursor')
        if cursor is None: break
    else:
        raise RuntimeError('NBA history pagination incomplete; player omitted.')
    rows = list({r['game']['id']:r for r in rows}.values())
    rows.sort(key=lambda r:(r['source_date'],r['game']['id']))
    path.write_text(json.dumps(rows))
    return rows


def mlb_prop_identities(snapshots):
    """Resolve names within each game's teams; posted lineups override guesses."""
    candidates = {}
    for gid, snapshot in (snapshots or {}).items():
        feed = snapshot.get('raw_feed', {})
        box = feed.get('liveData', {}).get('boxscore', {}).get('teams', {})
        people = feed.get('gameData', {}).get('players', {})
        for side in ['home', 'away']:
            team = box.get(side, {})
            lineup = {int(pid) for pid in team.get('battingOrder', [])}
            posted = len(lineup) == 9
            pitchers = team.get('pitchers', [])
            recorded = int(pitchers[0]) if pitchers else None
            probable = (snapshot.get('probable_pitchers', {}).get(side) or {})
            probable_id = int(probable['id']) if probable.get('id') else None
            conflict = recorded is not None and probable_id is not None and recorded != probable_id
            starter = None if conflict else recorded if recorded is not None else probable_id
            players = dict(team.get('players', {}))
            if starter is not None and 'ID'+str(starter) not in players:
                person = people.get('ID'+str(starter), {})
                players['ID'+str(starter)] = {'person': {'id': starter,
                    'fullName': person.get('fullName') or probable.get('fullName')}, 'position': {'code': '1'}}
            for player in players.values():
                person = player.get('person', {})
                if not person.get('id') or not person.get('fullName'):
                    continue
                pid = int(person['id'])
                position = player.get('position', {})
                # A sportsbook line is still required. Bench players are excluded
                # once a full lineup is posted; unknown lineups stay provisional.
                hitter = pid in lineup or (not posted and position.get('code') != '1'
                    and position.get('abbreviation') != 'P')
                pitching = pid == starter
                hitting_status = 'Confirmed lineup' if pid in lineup else 'Provisional — lineup unannounced'
                pitching_status = 'Recorded starter' if recorded == starter and starter is not None else 'Probable starter — subject to change'
                value = (pid, hitter, pitching, hitting_status, pitching_status)
                key = (int(gid), name_key(person['fullName']))
                candidates.setdefault(key, {})[pid] = value
    return {key: next(iter(values.values())) for key, values in candidates.items() if len(values) == 1}


def generate_player_prop_picks(league, games, odds_key, player_logs=None, snapshots=None, nba_key=None):
    if not odds_key:
        return {'picks':[], 'message':'Player props need the configured Odds API key.', 'errors':[]}
    quotes, errors = fetch_prop_quotes(league, games, odds_key)
    if not quotes:
        return {'picks':[], 'message':'No fresh eligible sportsbook player-prop lines are available for upcoming games.', 'errors':errors}
    groups = {}
    for q in quotes:
        groups.setdefault((q['event_id'],q['player'],q['market'],q['line']), []).append(q)
    history_by_name = {}; identities = {}; mlb_histories = {}
    if league=='NBA':
        if not nba_key:
            return {'picks':[], 'message':'NBA prop estimates need BALLDONTLIE_API_KEY and access to player game stats.', 'errors':errors}
        names = sorted({q['player'] for q in quotes})
        def collect(player):
            try:return player, _nba_history(player,nba_key), None
            except Exception:return player, [], 'Prior game stats unavailable for '+player
        with ThreadPoolExecutor(max_workers=3) as pool:
            for player, history, error in pool.map(collect,names):
                history_by_name[player] = history
                if error:errors.append(error)
    else:
        if player_logs is None or player_logs.empty:
            return {'picks':[], 'message':'MLB player warehouse is unavailable; no prop estimates generated.', 'errors':errors}
        identities = mlb_prop_identities(snapshots)
        wanted = {identity[0] for identity in identities.values()}
        player_ids = pd.to_numeric(player_logs.player_id, errors='coerce')
        subset = player_logs.loc[player_ids.isin(wanted)].copy()
        subset['player_id'] = player_ids.loc[subset.index]
        subset['prior_start'] = pd.to_datetime(subset.start_time, utc=True, errors='coerce')
        subset['game_id'] = pd.to_numeric(subset.game_id, errors='coerce')
        # Missing components stay unknown, rather than becoming fabricated zeros.
        def number(column):
            return pd.to_numeric(subset[column], errors='coerce') if column in subset else pd.Series(np.nan, index=subset.index)
        singles = number('hits')-number('doubles')-number('triples')-number('batting_home_runs')
        subset['singles'] = singles.where(singles >= 0)
        subset['total_bases'] = subset['singles']+2*number('doubles')+3*number('triples')+4*number('batting_home_runs')
        subset['hits_runs_rbis'] = number('hits')+number('batting_runs')+number('rbi')
        mlb_histories = {int(pid): frame for pid,frame in subset.groupby('player_id')}
    picks = []; exclusions = []
    def excluded(first, player, market, line, reason):
        exclusions.append({'Player': player, 'Market': market, 'Line': line,
            'Game': first['away_team']+' @ '+first['home_team'], 'Reason': reason})
    for (_,player,market,line), group in groups.items():
        first = group[0]; start = pd.Timestamp(first['start_time'])
        unique = {q['book']:q for q in group if q['book']}
        if not unique:
            if league == 'MLB': excluded(first, player, market, line, 'No sportsbook offers this exact line.')
            continue
        participation = None
        paired_probabilities = [q['over_probability'] for q in unique.values() if q.get('over_probability') is not None]
        # A one-sided quote cannot establish a margin-free market probability.
        # Keep the ten-observation shrinkage, using an explicit neutral prior.
        probability = float(np.median(paired_probabilities)) if paired_probabilities else .5
        stat = MARKETS[league][market][1]
        if league=='NBA':
            values = [r.get(stat) for r in history_by_name.get(player, []) if pd.Timestamp(r['source_date'])+pd.Timedelta(hours=48)<start]
        else:
            identity = identities.get((int(first['game_id']),name_key(player)))
            if not identity:
                excluded(first, player, market, line, 'No unique player identity found in this game feed.')
                continue
            pitching_market = market.startswith('pitcher_')
            if (not pitching_market and not identity[1]) or (pitching_market and not identity[2]):
                excluded(first, player, market, line, 'Player is outside the posted lineup or is not a recorded/probable starter.')
                continue
            participation = identity[4] if pitching_market else identity[3]
            logs = mlb_histories.get(identity[0])
            if logs is None:
                excluded(first, player, market, line, 'No saved prior appearances for this player.')
                continue
            cutoff = min(start,pd.Timestamp(first['capture_time']))-pd.Timedelta(hours=48)
            logs = logs[(logs.prior_start<cutoff)&(logs.game_id!=int(first['game_id']))]
            if pitching_market: logs = logs[pd.to_numeric(logs.games_started, errors='coerce')>0]
            else: logs = logs[pd.to_numeric(logs.plate_appearances, errors='coerce')>0]
            logs = logs.sort_values('prior_start').drop_duplicates('game_id', keep='last')
            if stat not in logs:
                excluded(first, player, market, line, 'Saved history is missing the statistics for this market.')
                continue
            values = pd.to_numeric(logs[stat], errors='coerce').tolist()
        values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        if league == 'MLB':
            sample = values[-30:]
            if len(sample) < 10 or sum(v != line for v in sample) < 10:
                excluded(first, player, market, line, f'Need 10 usable non-push appearances; found {len(sample)} appearances and {sum(v != line for v in sample)} non-push results.')
                continue
        qualified = False
        for side,market_p in [('Over',probability),('Under',1-probability)]:
            price_key = 'over_decimal_odds' if side == 'Over' else 'under_decimal_odds'
            available = [q for q in unique.values() if (q.get(price_key) or 0) > 1]
            if not available:
                continue
            estimate = estimate_prop(values,line,side,market_p)
            if estimate and estimate['Estimated chance'] >= .65:
                qualified = True
                if not paired_probabilities:
                    estimate['Estimate method'] = 'Prior 30 appearances + neutral shrinkage; one-sided sportsbook quote'
                best = max(available, key=lambda q:q[price_key])
                push_chance = estimate['Historical pushes']/estimate['Prior games']
                expected_return = (1-push_chance)*(estimate['Estimated chance']*best[price_key]-1)
                if not np.isfinite(expected_return):
                    continue
                picks.append({'Player':player,'Market':MARKETS[league][market][0],'Pick':side,'Line':line,
                              'Market chance':market_p,'Best sportsbook':best['book'],'Best decimal odds':best[price_key], 'Sportsbook offers':[{'book':q['book'], 'decimal_odds':q[price_key]} for q in available], 'Estimated return per unit':expected_return,'Break-even cover chance':1/best[price_key],'Estimated push chance':push_chance,'Books':len(available),'Game':first['away_team']+' @ '+first['home_team'],
                              'start_time':first['start_time'],'Captured UTC':first['capture_time'],
                              **({'Participation status': participation, 'Event match': first.get('Event match')} if league=='MLB' else {}), **estimate})
        if league == 'MLB' and not qualified:
            excluded(first, player, market, line, 'No offered side meets the 65% estimated cover chance minimum.')
    # One side of one line per player/market/game; avoid ranking duplicate books or alternatives.
    picks.sort(key=lambda r:(-r['Estimated chance'],-r['Estimated return per unit'],-r['Prior games'],r['Player']))
    seen=set(); chosen=[]
    for pick in picks:
        key=(pick['Game'],pick['start_time'],pick['Player'],pick['Market'])
        if key not in seen:
            seen.add(key);chosen.append(pick)
        # Retain eligible picks for every matchup; the global list stays top ten.
    return {'picks':chosen[:10],'game_picks':chosen,'errors':errors, **({'exclusions': exclusions, 'quoted_candidates':len(groups)} if league=='MLB' else {}), 'message':f'{len(chosen)} player-prop estimates ranked from actual lines and prior appearances.',
            'note':'Estimated chance is conditional on no push and player participation. Estimates are not calibrated; no availability or injury clearance is implied.'}
