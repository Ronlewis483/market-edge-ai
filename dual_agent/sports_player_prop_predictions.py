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

MARKETS = {'MLB': {'batter_hits': ('Hits', 'hits'), 'pitcher_strikeouts': ('Strikeouts', 'pitcher_strikeouts')},
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
    # Ten pseudo-observations keep short runs of wins from dominating the ranking.
    chance = (wins+10*market_probability)/(decisions+10)
    return {'Estimated chance': float(chance), 'Projected stat': float(values.mean()),
            'Prior games': len(values), 'Historical wins': wins, 'Historical pushes': pushes,
            'Historical hit rate': wins/decisions, 'Estimate method': 'Prior 30 appearances + market shrinkage'}


def _get(url, key, params=None, balldontlie=False):
    response = requests.get(url, params=params, headers={'Authorization': key} if balldontlie else None, timeout=20)
    if response.status_code != 200:
        raise RuntimeError('Player-prop data request failed (HTTP '+str(response.status_code)+').')
    return response.json()


def fetch_prop_quotes(league, games, odds_key):
    sport = 'baseball_mlb' if league=='MLB' else 'basketball_nba'
    base = 'https://api.the-odds-api.com/v4/sports/'+sport
    events = _get(base+'/events', odds_key, {'apiKey': odds_key})
    now = pd.Timestamp.now(tz='UTC'); quotes = []; errors = []
    from dual_agent.mlb_historical_odds import team
    for game in games:
        start = pd.to_datetime(game.get('start_time') or game.get('commence_time'), utc=True)
        if not now<start<=now+pd.Timedelta(hours=24): continue
        matched = [event for event in events if team(event.get('home_team'))==team(game['home_team']) and team(event.get('away_team'))==team(game['away_team']) and abs(pd.to_datetime(event['commence_time'], utc=True)-start)<=pd.Timedelta(minutes=15)]
        if len(matched)!=1:
            errors.append('No unique prop event match for '+game['away_team']+' at '+game['home_team']); continue
        event = matched[0]
        try:
            payload = _get(base+'/events/'+event['id']+'/odds', odds_key, {'apiKey': odds_key, 'regions': 'us', 'markets': ','.join(MARKETS[league]), 'oddsFormat': 'decimal'})
        except RuntimeError as exc:
            errors.append(str(exc)); continue
        stamp = pd.Timestamp.now(tz='UTC')
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
                    if set(sides)!= {'Over','Under'}: continue
                    over = 1/sides['Over']; under = 1/sides['Under']
                    quotes.append({'player':player,'line':line,'market':key,'over_probability':over/(over+under),
                                   'book':book.get('key'), 'updated_at':updated.isoformat(), 'event_id':event['id'],
                                   'home_team':game['home_team'],'away_team':game['away_team'],'start_time':start.isoformat(),
                                   'game_id':game.get('game_id'), 'capture_time':stamp.isoformat()})
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


def generate_player_prop_picks(league, games, odds_key, player_logs=None, snapshots=None, nba_key=None):
    if not odds_key:
        return {'picks':[], 'message':'Player props need the configured Odds API key.', 'errors':[]}
    quotes, errors = fetch_prop_quotes(league, games, odds_key)
    if not quotes:
        return {'picks':[], 'message':'No fresh two-sided sportsbook player-prop lines are available for eligible upcoming games.', 'errors':errors}
    groups = {}
    for q in quotes:
        groups.setdefault((q['event_id'],q['player'],q['market'],q['line']), []).append(q)
    history_by_name = {}; identities = {}
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
        for gid,snapshot in (snapshots or {}).items():
            box = snapshot.get('raw_feed', {}).get('liveData', {}).get('boxscore', {}).get('teams', {})
            for side in ['home','away']:
                team = box.get(side, {})
                allowed = set(team.get('battingOrder', []))|set(team.get('pitchers', [])[:1])
                for player in team.get('players', {}).values():
                    person=player.get('person', {})
                    if person.get('id') in allowed:
                        identities[(int(gid),name_key(person.get('fullName')))] = (int(person['id']), person['id'] in team.get('battingOrder', []), person['id'] in team.get('pitchers', [])[:1])
    picks = []
    for (_,player,market,line), group in groups.items():
        first = group[0]; start = pd.Timestamp(first['start_time'])
        unique = {q['book']:q for q in group if q['book']}
        if len(unique)<2: continue
        probability = float(np.median([q['over_probability'] for q in unique.values()]))
        stat = MARKETS[league][market][1]
        if league=='NBA':
            values = [r.get(stat) for r in history_by_name.get(player, []) if pd.Timestamp(r['source_date'])+pd.Timedelta(hours=48)<start]
        else:
            identity = identities.get((int(first['game_id']),name_key(player)))
            if not identity or (market=='batter_hits' and not identity[1]) or (market=='pitcher_strikeouts' and not identity[2]): continue
            logs = player_logs[pd.to_numeric(player_logs.player_id, errors='coerce')==identity[0]].copy()
            logs['prior_start'] = pd.to_datetime(logs.start_time, utc=True, errors='coerce')
            cutoff = min(start,pd.Timestamp(first['capture_time']))-pd.Timedelta(hours=48)
            logs = logs[(logs.prior_start<cutoff)&(pd.to_numeric(logs.game_id, errors='coerce')!=int(first['game_id']))]
            if market=='pitcher_strikeouts': logs = logs[pd.to_numeric(logs.games_started, errors='coerce')>0]
            else: logs = logs[pd.to_numeric(logs.plate_appearances, errors='coerce')>0]
            logs = logs.sort_values('prior_start').drop_duplicates('game_id', keep='last')
            values = pd.to_numeric(logs[stat], errors='coerce').tolist()
        values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        for side,market_p in [('Over',probability),('Under',1-probability)]:
            estimate = estimate_prop(values,line,side,market_p)
            if estimate:
                picks.append({'Player':player,'Market':MARKETS[league][market][0],'Pick':side,'Line':line,
                              'Market chance':market_p,'Books':len(unique),'Game':first['away_team']+' @ '+first['home_team'],
                              'start_time':first['start_time'],'Captured UTC':first['capture_time'], **estimate})
    # One side of one line per player/market/game; avoid ranking duplicate books or alternatives.
    picks.sort(key=lambda r:(-r['Estimated chance'],-r['Prior games'],r['Player']))
    seen=set(); chosen=[]
    for pick in picks:
        key=(pick['Game'],pick['start_time'],pick['Player'],pick['Market'])
        if key not in seen:
            seen.add(key);chosen.append(pick)
        if len(chosen)==10:break
    return {'picks':chosen,'errors':errors,'message':f'{len(chosen)} player-prop estimates ranked from actual lines and prior appearances.',
            'note':'Estimated chance is conditional on no push and player participation. Estimates are not calibrated; no availability or injury clearance is implied.'}
