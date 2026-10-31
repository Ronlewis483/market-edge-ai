"""Pregame player-prop estimates using actual lines and prior appearances.

NBA uses saved neural weights for promoted categories, with the existing
minutes model as fallback. MLB retains its existing historical calculations.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import threading
import re
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import requests
from dual_agent.odds_request_cache import get as cached_odds_get
from dual_agent.preferred_sportsbooks import allowed, standalone_value

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
    'NBA': {'player_points': ('Points', 'pts'), 'player_rebounds': ('Rebounds', 'reb'), 'player_assists': ('Assists', 'ast'),
    'player_threes': ('Three-pointers made', 'fg3m'), 'player_blocks': ('Blocks', 'blk'),
    'player_steals': ('Steals', 'stl'), 'player_turnovers': ('Turnovers', 'turnover'),
    'player_points_rebounds_assists': ('Points + rebounds + assists', ('pts', 'reb', 'ast')),
    'player_points_rebounds': ('Points + rebounds', ('pts', 'reb')),
    'player_points_assists': ('Points + assists', ('pts', 'ast')),
    'player_rebounds_assists': ('Rebounds + assists', ('reb', 'ast')),
    'player_blocks_steals': ('Blocks + steals', ('blk', 'stl'))}}


def name_key(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value)).lower() if c.isalnum())


def estimate_prop(values, line, side, market_probability, minimum=10, minimum_decisions=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)][-30:]
    wins = int((values > line).sum()) if side == 'Over' else int((values < line).sum())
    pushes = int((values == line).sum())
    decisions = len(values)-pushes
    if minimum_decisions is None:
        minimum_decisions = minimum
    if len(values) < minimum or decisions < minimum_decisions:
        return None
    # Preserve the existing short-sample shrinkage and eligibility criteria.
    chance = (wins+10*market_probability)/(decisions+10)
    return {'Estimated chance': float(chance), 'Projected stat': float(values.mean()),
            'Prior games': len(values), 'Non-push games': decisions, 'Historical wins': wins, 'Historical pushes': pushes,
            'Historical hit rate': wins/decisions, 'Estimate method': 'Prior 30 appearances + market shrinkage'}


def nba_market_value(row, stat):
    """Sum supported components only when every component is actually present."""
    fields = stat if isinstance(stat, tuple) else (stat,)
    numbers = [pd.to_numeric(row.get(field), errors='coerce') for field in fields]
    return float(sum(numbers)) if all(pd.notna(n) and np.isfinite(n) for n in numbers) else None


def _get(url, key, params=None, balldontlie=False):
    if balldontlie:
        # Share plan-aware pacing and 429 handling with NBA game history.
        from dual_agent.balldontlie_data import _balldontlie_get
        return _balldontlie_get(url, params=params, api_key=key)
    response = cached_odds_get(url, params=params, headers={'Authorization': key} if balldontlie else None, timeout=20)
    if response.status_code != 200:
        raise RuntimeError('Player-prop data request failed (HTTP '+str(response.status_code)+').')
    return response.json()


MLB_PROP_MATCH_VERSION = 6


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
        reversed_exact = [(e,t) for e,t in valid.values() if mlb_team_key(e.get('home_team')) == away and mlb_team_key(e.get('away_team')) == home and abs(t-start) <= pd.Timedelta(minutes=15)]
        if not strict and not pair and len(reversed_exact) == 1:
            pair = reversed_exact; strict = reversed_exact
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
            from dual_agent.preferred_sportsbooks import BOOKMAKERS
            payload = _get(base+'/events/'+event['id']+'/odds', odds_key, {'apiKey': odds_key, 'bookmakers': BOOKMAKERS, 'markets': ','.join(MARKETS[league]), 'oddsFormat': 'decimal'})
        except RuntimeError as exc:
            errors.append(str(exc)); return quotes, errors
        stamp = pd.Timestamp.now(tz='UTC')
        if league == 'MLB' and not payload.get('bookmakers'):
            errors.append({'Game':game['away_team']+' @ '+game['home_team'], 'Reason':'Event matched, but no sportsbook prop lines were returned.'})
        for book in payload.get('bookmakers', []):
            if not allowed(book.get('key')):
                continue
            for market in book.get('markets', []):
                key = market.get('key')
                if key not in MARKETS[league]: continue
                updated = pd.to_datetime(market.get('last_update') or book.get('last_update'), utc=True, errors='coerce')
                if pd.isna(updated) or updated>stamp or stamp-updated>pd.Timedelta(minutes=30) or updated>=start: continue
                paired = {}
                for outcome in market.get('outcomes', []):
                    try:
                        player = str(outcome['description']); price = float(outcome['price']); side = outcome['name']
                        if league == 'MLB' and key == 'batter_home_runs' and side in ('Yes', 'No'):
                            line = .5; side = 'Over' if side == 'Yes' else 'Under'
                        else:
                            line = float(outcome['point'])
                        if not np.isfinite([line,price]).all() or price<=1 or side not in ['Over','Under']: continue
                        paired.setdefault((player,line), {})[side] = price
                    except (KeyError, TypeError, ValueError): continue
                for (player,line), sides in paired.items():
                    # One actual offered side is enough for NBA and MLB.
                    # Only paired quotes contribute a margin-free market prior.
                    over = 1/sides['Over'] if 'Over' in sides else None
                    under = 1/sides['Under'] if 'Under' in sides else None
                    quotes.append({'player':player,'line':line,'market':key,'over_probability':over/(over+under) if over is not None and under is not None else None,
                                   'over_decimal_odds':sides.get('Over'), 'under_decimal_odds':sides.get('Under'), 'book':book.get('key'), 'updated_at':updated.isoformat(), 'event_id':event['id'],
                                   'home_team':game['home_team'],'away_team':game['away_team'],'start_time':start.isoformat(),
                                   'game_id':game.get('game_id'), 'capture_time':stamp.isoformat(),
                                   **({'Event match':match_notes[int(game['game_id'])]['Match']} if league=='MLB' else {})})
        if league == 'MLB':
            found = {q['market'] for q in quotes}
            for market_key, (label, _) in MARKETS['MLB'].items():
                if market_key not in found:
                    errors.append({'Game': game['away_team']+' @ '+game['home_team'],
                        'MLB start UTC': start.isoformat(), 'Market': label,
                        'Reason': 'No usable fresh '+label.lower()+' offer was returned for this matched game.'})
        return quotes, errors
    # Separate HTTP requests; bounded concurrency also preserves slate order.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for game_quotes, game_errors in pool.map(collect, games):
            quotes.extend(game_quotes); errors.extend(game_errors)
    return quotes, errors


def _nba_history(player, api_key):
    """Use the identical identity resolution and history cache as the batch loader."""
    for name, rows, issue in _nba_history_batches([player], api_key):
        if issue:
            raise RuntimeError(issue)
        return rows
    return []


def _nba_request_issue(exc):
    """Expose useful status without leaking provider response bodies or credentials."""
    import re
    match = re.search(r'(?:error|HTTP)\s+(\d{3})', str(exc), re.I)
    code = match.group(1) if match else None
    if code == '401': return 'HTTP 401: player-data API key was rejected'
    if code == '403': return 'HTTP 403: player-data access was denied; check subscription permissions'
    if code == '429' or 'rate limit' in str(exc).lower(): return 'Provider rate limit persisted after retries'
    if code: return 'Provider returned HTTP '+code
    return type(exc).__name__+': request could not complete'


def nba_identity_key(name):
    """Use the shared sport-scoped alias normalization."""
    from dual_agent.player_aliases import name_key as player_name_key
    return player_name_key(name, 'NBA')


def _nba_load_history_archive(path):
    """Disk first; restore the shared private archive only after local cache loss."""
    try:
        archive = json.loads(path.read_text())
        if archive.get('version') == 3 and isinstance(archive.get('players'), dict):
            return archive
    except (OSError, ValueError, AttributeError):
        pass
    try:
        from dual_agent.balldontlie_data import _setting
        url, key = _setting('SUPABASE_URL', ''), _setting('SUPABASE_KEY', '')
        if url and key:
            response = requests.get(str(url).rstrip('/')+'/storage/v1/object/authenticated/market-edge-data/nba/props/player_histories_v3.json',
                headers={'apikey': key, 'Authorization': 'Bearer '+key}, timeout=3)
            if response.status_code == 200:
                archive = response.json()
                if archive.get('version') == 3 and isinstance(archive.get('players'), dict):
                    path.write_text(json.dumps(archive))
                    return archive
    except Exception:
        pass  # Optional remote storage must not prevent prop generation.
    return {'version': 3, 'players': {}}


def _nba_save_remote_history_archive(archive):
    """One private archive write per completed scan, rather than per player."""
    try:
        from dual_agent.balldontlie_data import _setting
        url, key = _setting('SUPABASE_URL', ''), _setting('SUPABASE_KEY', '')
        if not url or not key:
            archive['storage_status'] = 'Local history saved; remote storage is not configured.'
            return
        response = requests.post(str(url).rstrip('/')+'/storage/v1/object/market-edge-data/nba/props/player_histories_v3.json',
            headers={'apikey': key, 'Authorization': 'Bearer '+key,
                     'Content-Type': 'application/json', 'x-upsert': 'true'},
            data=json.dumps(archive, allow_nan=False).encode(), timeout=3)
        archive['storage_status'] = ('Player history backed up to Supabase.' if response.status_code in (200,201)
            else 'Local history saved; remote backup unavailable (HTTP '+str(response.status_code)+').')
    except Exception:
        archive['storage_status'] = 'Local history saved; remote backup unavailable.'


_NBA_PROP_HISTORY_LOCK = threading.RLock()


def _nba_history_batches(names, api_key, status=None):
    with _NBA_PROP_HISTORY_LOCK:
        yield from _nba_history_batches_locked(names, api_key, status)


def _nba_history_batches_locked(names, api_key, status=None):
    """Yield completed player histories; never cache an incomplete stats batch."""
    now = pd.Timestamp.now(tz='UTC')
    root = Path('nba_player_history_cache'); root.mkdir(exist_ok=True)
    archive_path = root/'player_histories_v3.json'
    archive = _nba_load_history_archive(archive_path)
    entries = archive.setdefault('players', {})
    day = now.strftime('%Y%m%d')
    base = 'https://api.balldontlie.io/nba/v1'
    def notify(message):
        if status: status(message)
    def save(path, data):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
    missing = []
    known = {}
    for name in dict.fromkeys(names):
        key = nba_identity_key(name)
        entry = entries.get(key, {})
        if not entry:
            # Reuse histories saved under an older nickname spelling. Never
            # merge different provider IDs simply because names normalize alike.
            old_entries = [v for k,v in entries.items()
                           if nba_identity_key(k) == key and isinstance(v,dict) and v.get('player_id')]
            if old_entries and len({str(v['player_id']) for v in old_entries}) == 1:
                entry = max(old_entries, key=lambda v:str(v.get('checked_at','')))
                entries[key] = entry
        # Migrate the previous complete daily caches instead of downloading again.
        if not entry:
            for legacy in sorted(root.glob(name_key(name)+'_bdl_v2_*.json'), reverse=True):
                try:
                    rows = json.loads(legacy.read_text())
                    ids = {r['player']['id'] for r in rows}
                    if rows and len(ids) == 1:
                        entry = {'player_id': ids.pop(), 'rows': rows,
                                 'checked_at': pd.Timestamp(legacy.stat().st_mtime, unit='s', tz='UTC').isoformat()}
                        entries[key] = entry
                        break
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        known[name] = entry
        checked = pd.to_datetime(entry.get('checked_at'), utc=True, errors='coerce')
        if entry.get('player_id') and isinstance(entry.get('rows'), list) and pd.notna(checked) and pd.Timedelta(0) <= now-checked < pd.Timedelta(minutes=30):
            yield name, entry['rows'], None if entry['rows'] else 'No completed played-game statistics returned for '+name+'.'
        else:
            missing.append(name)
    if not missing:
        if not archive_path.exists():
            save(archive_path, archive)
            _nba_save_remote_history_archive(archive)
            save(archive_path, archive)
        return
    catalog = []
    if any(not known[name].get('player_id') for name in missing):
        catalog_path = root/('active_players_'+day+'.json')
        try:
            catalog = json.loads(catalog_path.read_text())
            if not isinstance(catalog, list): raise ValueError('Invalid player catalog')
        except (OSError, ValueError):
            catalog = []; cursor = None; seen = set()
            try:
                for page in range(1, 101):
                    notify(f'Resolving player identities · roster page {page}')
                    params = {'per_page': 100}
                    if cursor is not None: params['cursor'] = cursor
                    payload = _get(base+'/players/active', api_key, params, True)
                    catalog.extend(payload.get('data', []))
                    cursor = payload.get('meta', {}).get('next_cursor')
                    if cursor is None: break
                    if str(cursor) in seen: raise RuntimeError('Repeated roster cursor')
                    seen.add(str(cursor))
                else: raise RuntimeError('Player identity pagination incomplete')
                save(catalog_path, catalog)
            except Exception as exc:
                for name in missing: yield name, [], 'NBA player directory: '+_nba_request_issue(exc)
                return
    from dual_agent.player_aliases import PlayerResolver
    resolver = PlayerResolver(catalog, 'NBA', 'balldontlie')
    resolved = []
    for name in missing:
        if known[name].get('player_id'):
            resolved.append((name, known[name]['player_id']))
            continue
        matched = resolver.resolve(name)
        matches = [matched] if matched is not None else []
        if len(matches) != 1:
            # Search is first OR last name, not an assumed full-name match.
            notify('Resolving player identity · '+name)
            try:
                payload = _get(base+'/players', api_key,
                               {'search': re.sub(r'\s+(?:jr\.?|sr\.?|ii|iii|iv)$', '', name, flags=re.I).split()[-1], 'per_page': 100}, True)
                # Keep the current roster in scope: fallback search must not hide a known ambiguous identity.
                search = {str(r['id']):r for r in catalog+payload.get('data',[]) if r.get('id') is not None}
                matched = PlayerResolver(list(search.values()), 'NBA', 'balldontlie').resolve(name)
                matches = [matched] if matched is not None else []
            except Exception as exc:
                yield name, [], 'Player identity for '+name+': '+_nba_request_issue(exc)
                continue
        if len(matches) != 1:
            yield name, [], 'No unique NBA player identity for '+name+'.'
        else:
            resolved.append((name, matches[0]['id']))
    resolved.sort(key=lambda item: not bool(known[item[0]].get('checked_at')))
    batches = []
    for has_history in (True, False):
        subset = [item for item in resolved if bool(known[item[0]].get('checked_at')) == has_history]
        batches.extend(subset[i:i+8] for i in range(0, len(subset), 8))
    for offset, batch in enumerate(batches):
        ids = {pid for name, pid in batch}; rows_by_id = {pid: [] for pid in ids}
        checked_dates = [pd.to_datetime(known[name].get('checked_at'), utc=True, errors='coerce') for name, pid in batch]
        since = max(now-pd.Timedelta(days=365), min(checked_dates)-pd.Timedelta(days=3)) if all(pd.notna(d) for d in checked_dates) else now-pd.Timedelta(days=365)
        try:
            incremental = all(pd.notna(d) for d in checked_dates)
            windows = [(since, now)] if incremental else [
                (now-pd.Timedelta(days=90), now),
                (now-pd.Timedelta(days=180), now-pd.Timedelta(days=90)),
                (now-pd.Timedelta(days=365), now-pd.Timedelta(days=180))]
            active_ids = set(ids)
            for window_start, window_end in windows:
                if not active_ids: break
                cursor = None; seen = set()
                for page in range(1, 101):
                    notify(f"{'Updating new results' if incremental else 'Loading recent player history'} · batch {offset+1} of {len(batches)} · page {page}")
                    params = {'player_ids[]': sorted(active_ids), 'per_page': 100,
                              'start_date': window_start.strftime('%Y-%m-%d'),
                              'end_date': window_end.strftime('%Y-%m-%d')}
                    if cursor is not None: params['cursor'] = cursor
                    payload = _get(base+'/stats', api_key, params, True)
                    for row in payload.get('data', []):
                        pid = (row.get('player') or {}).get('id'); game = row.get('game', {})
                        date = pd.to_datetime(game.get('date'), utc=True, errors='coerce')
                        played = nba_minutes(row.get('min')) is not None
                        if pid in active_ids and str(game.get('status','')).startswith('Final') and pd.notna(date) and date<now and played:
                            rows_by_id[pid].append(dict(row, source_date=date.isoformat(), history_provider='balldontlie'))
                    cursor = payload.get('meta', {}).get('next_cursor')
                    if cursor is None: break
                    if str(cursor) in seen: raise RuntimeError('Repeated stats cursor')
                    seen.add(str(cursor))
                else: raise RuntimeError('Stats pagination incomplete')
                if not incremental:
                    # Finish each date window: provider ordering is not assumed.
                    # Only extend history for players still short of 30 usable
                    # results in at least one supported statistical category.
                    for pid in list(active_ids):
                        unique = {r['game']['id']: r for r in rows_by_id[pid]}.values()
                        if all(sum(nba_market_value(r, stat) is not None for r in unique) >= 30
                               for label, stat in MARKETS['NBA'].values()):
                            active_ids.remove(pid)
        except Exception as exc:
            # Discard all partial batch rows: incomplete samples must not drive picks.
            issue = _nba_request_issue(exc)
            for name, pid in batch:
                yield name, [], 'NBA stats for '+name+': '+issue
            if issue.startswith(('HTTP 401:', 'HTTP 403:')):
                notify('Player-history scan stopped: '+issue)
                return  # Retrying other players cannot repair denied API access.
            continue
        for name, pid in batch:
            previous = known[name].get('rows', [])
            rows = list({r['game']['id']: r for r in previous+rows_by_id[pid] if pd.Timestamp(r['source_date']) > now-pd.Timedelta(days=365)}.values())
            rows.sort(key=lambda r:(r['source_date'], r['game']['id']))
            entries[nba_identity_key(name)] = {'player_id': pid, 'checked_at': now.isoformat(), 'rows': rows}
            # Checkpoint each completed player locally; partial batches are never saved.
            save(archive_path, archive)
            yield name, rows, None if rows else 'No completed played-game statistics returned for '+name+'.'

    _nba_save_remote_history_archive(archive)
    save(archive_path, archive)


def mlb_prop_identities(snapshots, quotes=None):
    """Resolve names within each game's teams; posted lineups override guesses."""
    candidates = {}
    roster_records = {}
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
                roster_records.setdefault(int(gid), {})[pid] = dict(people.get('ID'+str(pid), {}), **person)
    from dual_agent.player_aliases import PlayerResolver
    resolvers = {gid:PlayerResolver(list(players.values()), 'MLB', 'mlb_statsapi') for gid,players in roster_records.items()}
    for quote in quotes or []:
        gid = quote.get('game_id')
        if gid is None: continue
        gid = int(gid)
        resolver = resolvers.get(gid)
        matched = resolver.resolve(quote['player']) if resolver is not None else None
        if matched is not None:
            pid = int(matched['id'])
            original = candidates.get((gid,name_key(matched['fullName'])), {})
            if pid in original:
                candidates.setdefault((gid,name_key(quote['player'])), {})[pid] = original[pid]
    return {key: next(iter(values.values())) for key, values in candidates.items() if len(values) == 1}


def nba_sportsbook_prop_picks(quotes, errors):
    """Market probabilities only; never invent a player-history model or sample."""
    groups = {}
    for quote in quotes:
        groups.setdefault((quote['event_id'], quote['player'], quote['market'], quote['line']), []).append(quote)
    picks = []; listings = []
    for (_, player, market, line), group in groups.items():
        first = group[0]
        # Pick'em indicative feed prices do not establish standalone win chances.
        priced = {q['book']: q for q in group if standalone_value(q['book'])}
        paired = [q['over_probability'] for q in priced.values() if q.get('over_probability') is not None]
        for side, field in [('Over','over_decimal_odds'), ('Under','under_decimal_odds')]:
            offered = [q for q in priced.values() if (q.get(field) or 0)>1]
            unpriced = [q for q in group if not standalone_value(q['book']) and (q.get(field) or 0)>1]
            for quote in unpriced:
                listings.append({'Player':player, 'Market':MARKETS['NBA'][market][0], 'Pick':side,
                    'Line':line, 'Best sportsbook':quote['book'],
                    'Game':first['away_team']+' @ '+first['home_team'], 'start_time':first['start_time']})
            if not offered: continue
            if paired:
                over = float(np.median(paired))
                chance = over if side=='Over' else 1-over
                method = 'Median of same-line, same-book paired implied probabilities with margin normalized out'
            else:
                chance = float(np.median([1/q[field] for q in offered]))
                method = 'Single-side implied probability from quoted prices; bookmaker margin remains'
            best = max(offered, key=lambda q:q[field])
            picks.append({'Player':player, 'Market':MARKETS['NBA'][market][0], 'Pick':side,
                'Line':line, 'Estimated chance':chance, 'Probability source':'sportsbook_implied',
                'Estimate method':method, 'Prior games':0, 'Projected stat':None,
                'Best sportsbook':best['book'], 'Best decimal odds':best[field],
                'Sportsbook offers':[{'book':q['book'],'decimal_odds':q[field]} for q in offered],
                'Books':len(offered), 'Game':first['away_team']+' @ '+first['home_team'],
                'start_time':first['start_time'], 'Captured UTC':first['capture_time']})
    picks.sort(key=lambda row:(-row['Estimated chance'],row['Player'],row['Market'],row['Line']))
    return {'picks':picks[:10], 'game_picks':picks, 'errors':list(errors),
        'probability_source':'sportsbook_implied', 'unpriced_listings':listings,
        'message':f'{len(picks)} actual sportsbook sides ranked by market-implied probability.',
        'note':'Sportsbook-implied probabilities are not independent model predictions. No player history or injury verification was used.'}


def nba_minutes(value):
    """Provider minutes may be MM:SS or decimal; missing is never zero."""
    try:
        if ':' in str(value):
            minute, second = str(value).split(':')
            if not 0 <= float(second) < 60: return None
            total = float(minute)+float(second)/60
        else:
            total = float(value)
        return total if np.isfinite(total) and 0 < total <= 70 else None
    except (ValueError, TypeError):
        return None


def load_nba_matchup_context(api_key):
    """One cached league injury snapshot; defense uses existing game cache only."""
    now = pd.Timestamp.now(tz='UTC'); root = Path('nba_player_history_cache'); root.mkdir(exist_ok=True)
    path = root/'injury_context_v1.json'; injury = None
    try:
        saved = json.loads(path.read_text())
        stamp = pd.to_datetime(saved.get('checked_at'),utc=True,errors='coerce')
        if pd.notna(stamp) and pd.Timedelta(0)<=now-stamp<pd.Timedelta(minutes=15): injury=saved
    except (OSError,ValueError): pass
    if injury is None:
        try:
            if not api_key: raise RuntimeError('Player-data API key unavailable')
            items=[]; cursor=None; seen=set()
            for page in range(20):
                params={'per_page':100}
                if cursor is not None: params['cursor']=cursor
                payload=_get('https://api.balldontlie.io/v1/player_injuries',api_key,params,True)
                batch=payload.get('data')
                if not isinstance(batch,list): raise ValueError('Invalid injury response')
                items.extend(batch); cursor=(payload.get('meta') or {}).get('next_cursor')
                if cursor is None: break
                if str(cursor) in seen: raise ValueError('Repeated injury cursor')
                seen.add(str(cursor))
            else: raise ValueError('Injury pagination incomplete')
            injury={'checked_at':now.isoformat(),'complete':True,'records':items}
        except Exception as exc:
            injury={'checked_at':now.isoformat(),'complete':False,'records':[],
                    'issue':'NBA injury feed unavailable: '+_nba_request_issue(exc)}
        temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(injury)); temp.replace(path)
    from dual_agent.balldontlie_data import _setting
    game_root=Path(_setting('NBA_HISTORY_CACHE_DIR','.market_edge_cache/nba_history'))
    games={}; teams={}; player_teams={}
    for cache in game_root.glob('season_*_v1.json'):
        try:
            saved=json.loads(cache.read_text())
            if not saved.get('complete'): continue
            for game in saved.get('games',[]):
                if game.get('id') is not None: games[game['id']]=game
                for side in ('home_team','visitor_team'):
                    team=game.get(side) or {}
                    if team.get('id') is not None: teams[team['id']]=team
        except (OSError,ValueError): continue
    # Cached active directory provides team membership without new roster calls.
    for cache in sorted(root.glob('active_players_*.json'),reverse=True)[:1]:
        try:
            for player in json.loads(cache.read_text()):
                team=player.get('team') or {}
                if team.get('id') is not None:
                    teams[team['id']]=team; player_teams[player['id']]=team['id']
        except (OSError,ValueError,KeyError): pass
    injuries={}; blocked=[]
    for item in injury['records']:
        player=item.get('player') or {}; pid=player.get('id')
        if pid is not None:
            injuries.setdefault(pid,[]).append(item)
            if player.get('team_id') is not None: player_teams[pid]=player['team_id']
        if (name_key(item.get('status')) in {'out','doubtful','suspended','inactive','injuredreserve'}
                or re.search(r'(?:will be limited to (?:about )?\d+(?:-\d+)? minutes|on a minutes restriction|under a minutes restriction|minutes will be restricted)',str(item.get('description','')),re.I)):
            blocked.append(nba_identity_key(player.get('first_name','')+' '+player.get('last_name','')))
    return {'checked_at':injury['checked_at'],'injury_complete':injury['complete'],
            'injuries':injuries,'blocked_player_names':blocked,'games':list(games.values()),
            'teams':teams,'player_teams':player_teams,'issue':injury.get('issue')}


def nba_availability(rows, context):
    context=context or {}; ids={r.get('player',{}).get('id') for r in rows if r.get('player',{}).get('id') is not None}
    if len(ids)!=1: return False,'Player availability identity unverified',False
    pid=next(iter(ids)); reports=context.get('injuries',{}).get(pid,[])
    statuses=[str(r.get('status','')) for r in reports]
    blocked=any(name_key(s) in {'out','doubtful','suspended','inactive','injuredreserve'} for s in statuses)
    restriction=any(re.search(r'(?:will be limited to (?:about )?\d+(?:-\d+)? minutes|on a minutes restriction|under a minutes restriction|minutes will be restricted)',str(r.get('description','')),re.I) for r in reports)
    if reports:
        status='Injury feed: '+', '.join(dict.fromkeys(statuses))+' · checked '+str(context.get('checked_at'))
    elif context.get('injury_complete'):
        status='No injury listing in checked feed; participation is not confirmed'
    else:
        status='Injury status unverified; forecast is conditional on playing'
    if restriction: status+=' · minutes restriction reported; future cap unverified'
    return blocked,status,restriction


def nba_points_defense(rows, matchup, context, cutoff):
    """Conservative points-only allowance adjustment, not defense by position."""
    context=context or {}; teams=context.get('teams',{})
    def team_key(value):
        k=name_key(value)
        return {'laclippers':'losangelesclippers'}.get(k,k)
    ids={r.get('player',{}).get('id') for r in rows if r.get('player',{}).get('id') is not None}
    pid=next(iter(ids)) if len(ids)==1 else None
    own=context.get('player_teams',{}).get(pid)
    if own is None and rows: own=(rows[-1].get('team') or {}).get('id')
    own_team=teams.get(own) or next((r.get('team') for r in reversed(rows) if (r.get('team') or {}).get('id')==own),{})
    own_name=team_key(own_team.get('full_name'))
    home,away=team_key(matchup.get('home_team')),team_key(matchup.get('away_team'))
    opponent_name=away if own_name==home else home if own_name==away else None
    opponent_ids=[tid for tid,t in teams.items() if team_key(t.get('full_name'))==opponent_name]
    if len(opponent_ids)!=1: return 1.,'Opponent team identity unverified; no defensive adjustment'
    opponent=opponent_ids[0]; all_values=[]; all_weights=[]; against=[]
    for game in context.get('games',[]):
        if not str(game.get('status','')).startswith('Final') or game.get('season_type')=='preseason': continue
        date=pd.to_datetime(game.get('date'),utc=True,errors='coerce')
        # Date-only feeds cannot establish an intraday pregame cutoff.
        if pd.isna(date) or date.date()>=cutoff.date() or date<cutoff-pd.Timedelta(days=365): continue
        try: hs,vs=float(game['home_team_score']),float(game['visitor_team_score'])
        except (KeyError,TypeError,ValueError): continue
        if not np.isfinite([hs,vs]).all() or min(hs,vs)<=0: continue
        weight=2**(-max(0.,(cutoff-date).total_seconds()/86400)/90.)
        all_values.extend([hs,vs]);all_weights.extend([weight,weight])
        h=(game.get('home_team') or {}).get('id');v=(game.get('visitor_team') or {}).get('id')
        if h==opponent: against.append((date,vs,weight))
        elif v==opponent: against.append((date,hs,weight))
    against=sorted(against,key=lambda r:r[0])[-20:]
    if len(against)<5 or len(all_values)<60:
        return 1.,'Insufficient verified opponent scoring history; no defensive adjustment'
    w=np.array([g[2] for g in against]);effective=w.sum()**2/(w*w).sum()
    baseline=np.average(all_values,weights=all_weights)
    allowance=np.average([g[1] for g in against],weights=w)
    ratio=1+(allowance/baseline-1)*effective/(effective+20)
    factor=float(np.clip(ratio,.92,1.08))
    return factor,f'Points component only: {factor:.3f}× allowance factor · {len(against)} prior opponent games; not pace or position adjusted'


def estimate_nba_matchup(rows, stat, line, side, matchup, context=None):
    """Recency-weighted per-minute counts with a mixture of recent minutes.

    Negative-binomial dispersion includes historical excess count variance and
    rate-estimation uncertainty. Fixed weighting choices are not calibrated.
    """
    from scipy.stats import nbinom
    if side not in ('Over','Under') or not np.isfinite(float(line)): return None
    cutoff=min(pd.Timestamp(matchup['start_time']),pd.Timestamp(matchup['capture_time']))
    prior=[r for r in rows if pd.Timestamp(r['source_date'])<cutoff]
    prior.sort(key=lambda r:(r['source_date'],str((r.get('game') or {}).get('id'))))
    player_ids={r.get('player',{}).get('id') for r in prior if r.get('player',{}).get('id') is not None}
    if len(player_ids)!=1: return None
    blocked,status,restriction=nba_availability(prior,context)
    if blocked or restriction: return None
    sample=[]
    for row in prior:
        minute=nba_minutes(row.get('min')); value=nba_market_value(row,stat)
        if minute is not None and value is not None and value>=0: sample.append((row,minute,value))
    sample=sample[-30:]
    if len(sample)<10: return None
    actual=np.array([r[2] for r in sample]); minutes=np.array([r[1] for r in sample])
    baseline=estimate_nba_history(actual,line,side)
    if baseline is None: return None
    n=len(sample); weight=2**(-np.arange(n-1,-1,-1)/10.)
    effective=weight.sum()**2/(weight*weight).sum(); weight*=effective/weight.sum()
    total=float(weight@actual); exposure=float(weight@minutes)
    rate=(total+.5)/(exposure+1.)
    observed_mean=rate*minutes
    excess=max(0.,float(np.average((actual-observed_mean)**2-observed_mean,weights=weight)))
    denominator=float(np.average(observed_mean**2,weights=weight))
    dispersion=(excess/denominator if denominator>0 else 0.)+1/(total+.5)
    role=[nba_minutes(r.get('min')) for r in prior]
    role=np.array([m for m in role if m is not None][-10:])
    if len(role)<5: return None
    role=np.clip(role,0,48)  # Regulation minutes; no overtime prediction.
    role_weight=2**(-np.arange(len(role)-1,-1,-1)/3.); role_weight/=role_weight.sum()
    factor,opponent_note=nba_points_defense(prior,matchup,context,cutoff)
    components=stat if isinstance(stat,tuple) else (stat,)
    if 'pts' in components and factor!=1:
        points=np.array([nba_market_value(r[0],'pts') for r in sample],dtype=float)
        if np.isfinite(points).all():
            point_rate=float(weight@points)/max(exposure,1.)
            rate=max(0.,rate+(factor-1)*point_rate)
    elif 'pts' not in components:
        opponent_note='No verified category-specific defensive adjustment for this market'
    means=np.maximum(rate*role,1e-12); shape=1/max(dispersion,1e-9)
    probability=shape/(shape+means)
    above=1-nbinom.cdf(np.floor(line),shape,probability)
    below=nbinom.cdf(np.ceil(line)-1,shape,probability)
    push=nbinom.pmf(line,shape,probability) if float(line).is_integer() and line>=0 else np.zeros(len(role))
    over=float(role_weight@above);under=float(role_weight@below);push=float(role_weight@push)
    if over+under<=0: return None
    chance=(over if side=='Over' else under)/(over+under)
    baseline.update({'Estimated chance':float(chance),'Over chance':over/(over+under),'Under chance':under/(over+under),
        'Projected stat':float(role_weight@means),'Expected minutes':float(role_weight@role),
        'Minutes sample':len(role),'Effective historical games':round(float(effective),2),
        'Estimated push chance':push,'Availability status':status,
        'Opponent context':opponent_note,'NBA prop model version':2,
        'Estimate method':'Recency-weighted per-minute negative-binomial model + recent-minutes mixture; uncalibrated; sportsbook prices excluded',
        'Model calculated UTC':pd.Timestamp.now(tz='UTC').isoformat()})
    # The trained model reuses these same completed-game rows. Never collect
    # another history, train a network, or request another sportsbook here.
    try:
        from dual_agent.nba_prop_neural import STATS, estimate as neural_estimate, load_model
        model_meta = load_model()['meta']
        if cutoff.date() <= pd.Timestamp(model_meta['promotion_end']).date():
            baseline['Neural fallback reason'] = 'Game predates the saved model promotion cutoff'
            return baseline
        complete = []
        for row in prior:
            date = pd.Timestamp(row['source_date'])
            game = row.get('game') or {}
            if date.date() >= cutoff.date() or date < cutoff-pd.Timedelta(days=365):
                continue
            if not str(game.get('status', '')).startswith('Final'):
                continue
            if str(game.get('season_type', '')).lower() in {'preseason', 'allstar', 'all-star'}:
                continue
            minute = nba_minutes(row.get('min'))
            numbers = [nba_market_value(row, field) for field in STATS]
            if minute is not None and all(v is not None and v >= 0 for v in numbers):
                complete.append((row, minute, numbers))
        # Exact game identity prevents duplicate provider rows inflating support.
        unique = {}
        for item in complete:
            gid = (item[0].get('game') or {}).get('id')
            if gid is not None:
                unique[str(gid)] = item
        complete = sorted(unique.values(), key=lambda item:pd.Timestamp(item[0]['source_date']))[-30:]
        if len(complete) >= 10:
            vector = np.array([item[2] for item in complete], dtype=float)
            mins = np.array([item[1] for item in complete], dtype=float)
            last_date = pd.Timestamp(complete[-1][0]['source_date'])
            result = neural_estimate(vector, mins, stat, line, side, factor,
                                     (cutoff.date()-last_date.date()).days)
            # Historical support shown on the card must refer to the same sample
            # supplied to the neural network, not an older or larger sample.
            history = estimate_nba_history([nba_market_value(item[0],stat) for item in complete],line,side)
            if result is not None and history is not None:
                baseline.update(history)
                baseline.update(result)
                baseline['Minutes sample'] = 8
                baseline['Recent form window'] = 8
                baseline['Opponent context'] = opponent_note
                baseline['Availability status'] = status
                return baseline
        baseline['Neural fallback reason'] = 'Category not promoted or complete neural history unavailable'
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        baseline['Neural fallback reason'] = 'Saved NBA neural model unavailable: '+type(exc).__name__
    return baseline


def estimate_nba_history(values, line, side):
    """Independent, smoothed prior-game frequency; no sportsbook probability input."""
    estimate = estimate_prop(values,line,side,.5,minimum=10,minimum_decisions=10)
    if estimate is None: return None
    estimate['Estimated chance'] = (estimate['Historical wins']+1)/(estimate['Non-push games']+2)
    estimate['Estimate method'] = 'Last 30 completed played-game results + Beta(1,1) smoothing; sportsbook prices excluded'
    estimate['Probability source'] = 'player_history'
    estimate['Statistics source'] = 'BALLDONTLIE'
    estimate['Availability status'] = 'Participation and expected minutes not verified'
    return estimate


def load_nba_prop_tracking(restore=False):
    path = Path('nba_player_history_cache')/'prop_tracking_v1.json'
    try:
        payload = json.loads(path.read_text())
        if payload.get('version') == 1 and isinstance(payload.get('records'), dict): return payload
    except (OSError, ValueError, AttributeError):
        pass
    if restore:
        try:
            from dual_agent.balldontlie_data import _setting
            url, key = _setting('SUPABASE_URL', ''), _setting('SUPABASE_KEY', '')
            if url and key:
                response = requests.get(str(url).rstrip('/')+'/storage/v1/object/authenticated/market-edge-data/nba/props/tracking_v1.json',
                    headers={'apikey': key, 'Authorization': 'Bearer '+key}, timeout=3)
                payload = response.json() if response.status_code == 200 else {}
                if payload.get('version') == 1 and isinstance(payload.get('records'), dict):
                    path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(payload)); return payload
        except Exception:
            pass
    empty = {'version': 1, 'records': {}}
    if restore:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(empty))
    return empty


def update_nba_prop_tracking(props):
    """Freeze one stronger offered side per exact line; settle verified finals only."""
    import hashlib
    with _NBA_PROP_HISTORY_LOCK:
        ledger = load_nba_prop_tracking(restore=True)
        records = ledger['records']; now = pd.Timestamp.now(tz='UTC')
        candidates = sorted(props.get('game_picks', props.get('picks', [])),
                            key=lambda r: -float(r.get('Estimated chance', 0)))
        for row in candidates:
            start = pd.to_datetime(row.get('start_time'), utc=True, errors='coerce')
            capture = pd.to_datetime(row.get('Captured UTC'), utc=True, errors='coerce')
            if pd.isna(start) or pd.isna(capture) or not capture <= now < start: continue
            if now-capture > pd.Timedelta(minutes=30) or not row.get('Best sportsbook'): continue
            identity = '|'.join([str(row.get('Game')), start.isoformat(), nba_identity_key(row.get('Player')), str(row.get('Market')), str(float(row['Line']))])
            version = int(row.get('NBA prop model version',1))
            legacy_id = hashlib.sha256(identity.encode()).hexdigest()
            record_id = hashlib.sha256((identity+'|model='+str(version)).encode()).hexdigest()
            if version==1 and legacy_id in records: continue
            if record_id not in records:
                records[record_id] = dict(row, **{'Tracking status': 'Pending',
                    'Recorded UTC': now.isoformat(), 'Actual stat': None})
        try:
            archive = json.loads((Path('nba_player_history_cache')/'player_histories_v3.json').read_text())
        except (OSError, ValueError):
            archive = {'players': {}}
        entries = archive.get('players', {})
        # Resolve both teams from provider IDs/names already present in player logs.
        teams = {}
        def team_name(value):
            key = name_key(value)
            return {'laclippers': 'losangelesclippers'}.get(key, key)
        for entry in entries.values():
            for row in entry.get('rows', []):
                team = row.get('team') or {}
                if team.get('id') is not None and team.get('full_name'):
                    teams[team['id']] = team_name(team['full_name'])
        lookup = {label: stat for label, stat in MARKETS['NBA'].values()}
        for record in records.values():
            if record.get('Tracking status') != 'Pending': continue
            start = pd.to_datetime(record.get('start_time'), utc=True, errors='coerce')
            if pd.isna(start) or start >= now: continue
            pair = str(record.get('Game', '')).split(' @ ', 1)
            if len(pair) != 2: continue
            away, home = map(team_name, pair)
            entry = entries.get(nba_identity_key(record['Player']), {})
            matches = {}
            for row in entry.get('rows', []):
                game = row.get('game') or {}
                date = pd.to_datetime(game.get('date'), utc=True, errors='coerce')
                if (not str(game.get('status','')).startswith('Final') or pd.isna(date)
                        or date.date() != start.tz_convert('America/Chicago').date()): continue
                if teams.get(game.get('home_team_id')) != home or teams.get(game.get('visitor_team_id')) != away: continue
                matches[game.get('id')] = row
            if len(matches) != 1 or record['Market'] not in lookup: continue
            actual = nba_market_value(next(iter(matches.values())), lookup[record['Market']])
            if actual is None: continue
            line = float(record['Line']); side = record['Pick']
            if side not in ('Over','Under'): continue
            status = 'Push' if actual == line else ('Covered' if (actual > line) == (side == 'Over') else 'Missed')
            record.update({'Tracking status': status, 'Actual stat': actual, 'Settled UTC': now.isoformat()})
        root = Path('nba_player_history_cache'); root.mkdir(exist_ok=True)
        path = root/'prop_tracking_v1.json'; temp = path.with_suffix('.tmp')
        temp.write_text(pd.Series(ledger).to_json(date_format='iso')); temp.replace(path)
        return ledger


def save_nba_prop_results(result, remote=False):
    """Checkpoint display results without changing their original quote times."""
    try:
        root = Path('nba_player_history_cache'); root.mkdir(exist_ok=True)
        frame = result.get('all_predictions', result.get('predictions'))
        rows = json.loads(frame.to_json(orient='records', date_format='iso')) if isinstance(frame, pd.DataFrame) else list(frame or [])
        if not rows: return
        payload = {'version': 1, 'saved_at': pd.Timestamp.now(tz='UTC').isoformat(),
                   'predictions': rows, 'player_props': result.get('player_props', {})}
        # pandas converts missing numeric values to JSON null.
        data = pd.Series(payload).to_json(date_format='iso').encode()
        path = root/'latest_prop_results_v1.json'
        temporary = path.with_suffix('.tmp'); temporary.write_bytes(data); temporary.replace(path)
        if remote:
            from dual_agent.balldontlie_data import _setting
            url, key = _setting('SUPABASE_URL', ''), _setting('SUPABASE_KEY', '')
            if url and key:
                requests.post(str(url).rstrip('/')+'/storage/v1/object/market-edge-data/nba/props/latest_results_v1.json',
                    headers={'apikey': key, 'Authorization': 'Bearer '+key,
                             'Content-Type': 'application/json', 'x-upsert': 'true'}, data=data, timeout=3)
                ledger_path = root/'prop_tracking_v1.json'
                if ledger_path.exists():
                    requests.post(str(url).rstrip('/')+'/storage/v1/object/market-edge-data/nba/props/tracking_v1.json',
                        headers={'apikey': key, 'Authorization': 'Bearer '+key,
                                 'Content-Type': 'application/json', 'x-upsert': 'true'},
                        data=ledger_path.read_bytes(), timeout=3)
    except Exception:
        pass  # Display/storage failure must not discard calculated predictions.


def load_nba_prop_results():
    """Restore last results for immediate display, not as verified current offers."""
    path = Path('nba_player_history_cache')/'latest_prop_results_v1.json'
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        try:
            from dual_agent.balldontlie_data import _setting
            url, key = _setting('SUPABASE_URL', ''), _setting('SUPABASE_KEY', '')
            if not url or not key: return None
            response = requests.get(str(url).rstrip('/')+'/storage/v1/object/authenticated/market-edge-data/nba/props/latest_results_v1.json',
                headers={'apikey': key, 'Authorization': 'Bearer '+key}, timeout=3)
            if response.status_code != 200: return None
            payload = response.json()
        except Exception:
            return None
    if not isinstance(payload, dict) or payload.get('version') != 1: return None
    frame = pd.DataFrame(payload.get('predictions', []))
    if frame.empty or not {'home_team','away_team','commence_time','confidence'}.issubset(frame): return None
    saved = pd.to_datetime(payload.get('saved_at'), utc=True, errors='coerce')
    if pd.isna(saved): return None
    times = pd.to_datetime(frame['commence_time'], utc=True, errors='coerce')
    now = pd.Timestamp.now(tz='UTC')
    # Existing status polling retires confirmed finals; bound old restore data.
    frame = frame.loc[(times >= now-pd.Timedelta(days=1)) & (times <= now+pd.Timedelta(days=7))].copy()
    if frame.empty: return None
    return {'predictions': frame.head(10), 'all_predictions': frame,
            'player_props': payload.get('player_props', {}), 'saved_props_at': saved.isoformat(),
            'nba_props_pending': False, 'restored_saved_results': True}


def generate_player_prop_picks(league, games, odds_key, player_logs=None, snapshots=None, nba_key=None, progress=None, nba_history_enabled=True):
    scan_started = time.perf_counter()
    calculation_seconds = 0.0
    timings = {}
    if not odds_key and league != 'NBA':
        return {'picks':[], 'message':'Player props need the configured Odds API key.', 'errors':[]}
    if league == 'NBA' and progress:
        progress({'picks': [], 'game_picks': [], 'errors': [], 'loading_progress': {'stage': 'Finding sportsbook lines', 'completed': 0, 'total': 0}})
    prepared_nba_context = None
    prepared_star_rosters = None
    if league == 'NBA' and nba_history_enabled:
        from concurrent.futures import ThreadPoolExecutor
        from dual_agent.nba_star_spotlight import collect_star_rosters
        def timed_input(function, *args):
            started = time.perf_counter()
            return function(*args), time.perf_counter()-started
        # Same calls and freshness guards, with independent sources overlapped.
        with ThreadPoolExecutor(max_workers=3) as input_pool:
            context_job = input_pool.submit(timed_input, load_nba_matchup_context, nba_key)
            roster_job = input_pool.submit(timed_input, collect_star_rosters, games)
            quote_job = input_pool.submit(timed_input, fetch_prop_quotes, league, games, odds_key) if odds_key else None
            try:
                if quote_job is not None:
                    (quotes, errors), seconds = quote_job.result()
                    timings['Sportsbook collection'] = seconds
                else:
                    quotes, errors = [], ['Sportsbook API key unavailable.']
            except Exception as exc:
                quotes, errors = [], [f'NBA sportsbook lines unavailable: {type(exc).__name__}: {exc}.']
            prepared_nba_context, timings['Injury and opponent context'] = context_job.result()
            prepared_star_rosters, timings['Team rosters'] = roster_job.result()
    else:
        try:
            quotes, errors = fetch_prop_quotes(league, games, odds_key) if odds_key else ([], ['Sportsbook API key unavailable.'])
        except Exception as exc:
            if league != 'NBA':
                raise
            quotes, errors = [], [f'NBA sportsbook lines unavailable: {type(exc).__name__}: {exc}.']
        timings['Sportsbook collection'] = time.perf_counter()-scan_started
    if not quotes and league != 'NBA':
        return {'picks':[], 'message':'No fresh eligible sportsbook player-prop lines are available for upcoming games.', 'errors':errors}
    if league == 'NBA' and not nba_history_enabled:
        return nba_sportsbook_prop_picks(quotes, errors)
    groups = {}
    for q in quotes:
        groups.setdefault((q['event_id'],q['player'],q['market'],q['line']), []).append(q)
    history_by_name = {}; identities = {}; mlb_histories = {}; nba_context = {}
    if league=='NBA':
        from dual_agent.nba_star_spotlight import collect_star_rosters, build_star_spotlights, key as star_key
        context_started = time.perf_counter()
        nba_context = prepared_nba_context
        # Source timings were measured in their concurrent tasks.
        if nba_context.get('issue'): errors.append(nba_context['issue'])
        if progress:
            progress({'picks': [], 'game_picks': [], 'errors': list(errors), 'loading_progress': {'stage': 'Finding team star rosters', 'completed': 0, 'total': 0}})
        roster_started = time.perf_counter()
        star_rosters = prepared_star_rosters
        if not nba_key:
            errors.append('NBA projections need BALLDONTLIE_API_KEY and access to player game stats.')
        # Fetch each unique player once, reusing the existing daily history cache.
        # Stars remain visible even without any available sportsbook market.
        names_by_key = {star_key(q['player']): q['player'] for q in quotes}
        pending_ledger = load_nba_prop_tracking(restore=True)
        for record in pending_ledger['records'].values():
            kickoff = pd.to_datetime(record.get('start_time'), utc=True, errors='coerce')
            if (record.get('Tracking status') == 'Pending' and pd.notna(kickoff)
                    and kickoff < pd.Timestamp.now(tz='UTC')-pd.Timedelta(hours=2)):
                names_by_key.setdefault(star_key(record['Player']), record['Player'])
        for roster in star_rosters.values():
            for player in roster['players']:
                names_by_key.setdefault(star_key(player['player']), player['player'])
        # Earliest games first; no games or candidates are removed from the scan.
        priority = {}
        for quote in quotes:
            k = star_key(quote['player']); stamp = pd.Timestamp(quote['start_time'])
            priority[k] = min(priority.get(k, stamp), stamp)
        for game in games:
            for side in ('home', 'away'):
                for roster in star_rosters.values():
                    if roster.get('team') == game[side+'_team']:
                        for player in roster['players']:
                            k = star_key(player['player']); stamp = pd.Timestamp(game['commence_time'])
                            priority[k] = min(priority.get(k, stamp), stamp)
        names = sorted(names_by_key.values(), key=lambda name:(priority.get(star_key(name), pd.Timestamp.max.tz_localize('UTC')), name)) if nba_key else []
        completed = 0
        last_scored = -1
        last_publish = 0.0
        def publish(stage):
            nonlocal last_scored, last_publish, calculation_seconds
            if not progress: return
            # Network-page updates should not rescore the whole slate.
            if completed == last_scored or (completed < len(names) and completed % 8 and time.monotonic()-last_publish < 3):
                return
            last_scored = completed
            last_publish = time.monotonic()
            by_key = {star_key(name): rows for name, rows in history_by_name.items()}
            ready = {q['player']: by_key[star_key(q['player'])] for q in quotes if star_key(q['player']) in by_key}
            ready.update(history_by_name)
            ready_groups = {key: group for key, group in groups.items() if star_key(key[1]) in by_key}
            scoring_started = time.perf_counter()
            partial = _rank_prop_candidates('NBA', games, ready_groups,
                ready, list(errors), star_rosters, {}, {}, nba_context=nba_context)
            calculation_seconds += time.perf_counter()-scoring_started
            partial.update(probability_source='player_history', statistics_source='BALLDONTLIE')
            partial['loading_progress'] = {'completed': completed, 'total': len(names), 'stage': stage,
                'sportsbook_lines': len(quotes), 'players_with_history': sum(bool(rows) for rows in history_by_name.values()),
                'calculated_props': len(partial.get('game_picks', []))}
            progress(partial)
        history_started = time.perf_counter()
        history_scoring_before = calculation_seconds
        publish('Checking saved player histories')
        for player, history, error in _nba_history_batches(names, nba_key, status=publish):
            history_by_name[player] = history
            completed += 1
            if error: errors.append(error)
            publish('Processed '+player)
        last_scored = -1
        publish('Player history checks finished; completing prop calculations')
        timings['Player history and storage'] = max(0.0, time.perf_counter()-history_started-(calculation_seconds-history_scoring_before))
        # Provider spellings may differ only in accents or punctuation.
        by_key = {star_key(name): rows for name, rows in history_by_name.items()}
        for q in quotes:
            history_by_name[q['player']] = by_key.get(star_key(q['player']), [])
    else:
        if player_logs is None or player_logs.empty:
            return {'picks':[], 'message':'MLB player warehouse is unavailable; no prop estimates generated.', 'errors':errors}
        identities = mlb_prop_identities(snapshots, quotes=quotes)
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
    scoring_started = time.perf_counter()
    if league == 'MLB' and progress:
        # Publish completed games independently; reuse quote/history inputs.
        scored = []; excluded_rows = []; pitcher_rows = []; result = {}
        for game in sorted(games, key=lambda g:g.get('start_time',g.get('commence_time',''))):
            game_groups = {key:group for key,group in groups.items() if group[0].get('game_id')==game.get('game_id')}
            partial = _rank_prop_candidates(league,[game],game_groups,history_by_name,errors,{},identities,mlb_histories)
            scored.extend(partial.get('game_picks', []))
            excluded_rows.extend(partial.get('exclusions', [])); pitcher_rows.extend(partial.get('pitcher_forecasts', []))
            ordered = sorted(scored,key=lambda r:(-r['Estimated chance'],-r['Prior games'],r['Player']))
            result = dict(partial,game_picks=ordered,picks=ordered[:10],exclusions=list(excluded_rows),
                          pitcher_forecasts=list(pitcher_rows),quoted_candidates=len(groups),
                          message=f'{len(ordered)} player-prop estimates ranked from actual lines and prior appearances.')
            progress(result)
        if not result:
            result = _rank_prop_candidates(league,games,groups,history_by_name,errors,{},identities,mlb_histories)
    else:
        result = _rank_prop_candidates(league, games, groups, history_by_name, errors,
                                     star_rosters if league == 'NBA' else {}, identities, mlb_histories, nba_context=nba_context)
    calculation_seconds += time.perf_counter()-scoring_started
    if league == 'NBA':
        timings['Prediction calculations'] = calculation_seconds
        timings['Total'] = time.perf_counter()-scan_started
        result['loading_timings_seconds'] = {label: round(seconds, 2) for label, seconds in timings.items()}
        result['scan_complete'] = True
        result.update(probability_source='player_history', statistics_source='BALLDONTLIE',
            note='Promoted NBA categories use trained neural weights with chronological residual calibration and an eight-game recent-form/minutes window. Other categories retain the existing minutes model. Probabilities are conditional on participation and no push; actual sportsbook accuracy is not established.')
        try:
            stored = json.loads((Path('nba_player_history_cache')/'player_histories_v3.json').read_text())
            result['history_storage_status'] = stored.get('storage_status', 'Player history saved locally.')
        except (OSError, ValueError):
            result['history_storage_status'] = 'History storage status unavailable.'
    return result


def _rank_prop_candidates(league, games, groups, history_by_name, errors,
                          star_rosters, identities, mlb_histories, nba_context=None):
    if league == 'NBA':
        from dual_agent.nba_star_spotlight import build_star_spotlights
    picks = []; exclusions = []; pitcher_forecasts = []
    def excluded(first, player, market, line, reason):
        exclusions.append({'Player': player, 'Market': MARKETS[league].get(market, (market, None))[0], 'Line': line,
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
            nba_rows = history_by_name.get(player, [])
            unavailable, availability_status, restricted = nba_availability(nba_rows,nba_context)
            if unavailable or restricted:
                excluded(first,player,market,line,availability_status)
                continue
            values = [nba_market_value(r, stat) for r in history_by_name.get(player, []) if pd.Timestamp(r['source_date'])+(pd.Timedelta(0) if r.get('history_provider') in {'api_sports','balldontlie'} else pd.Timedelta(hours=48))<min(start,pd.Timestamp(first['capture_time']))]
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
            if not pitching_market and (len(sample) < 10 or sum(v != line for v in sample) < 8):
                excluded(first, player, market, line, f'Need 10 prior appearances and 8 non-push results; found {len(sample)} appearances and {sum(v != line for v in sample)} non-push results.')
                continue
        qualified = False
        for side,market_p in [('Over',probability),('Under',1-probability)]:
            price_key = 'over_decimal_odds' if side == 'Over' else 'under_decimal_odds'
            available = [q for q in unique.values() if (q.get(price_key) or 0) > 1]
            if not available:
                continue
            estimate = estimate_prop(values,line,side,market_p,
                minimum=1 if league == 'MLB' and pitching_market else 10,
                minimum_decisions=8 if league == 'MLB' and not pitching_market else None)
            if league == 'NBA':
                estimate = estimate_nba_matchup(nba_rows,stat,line,side,first,nba_context)
            if estimate and league == 'MLB' and pitching_market:
                best_forecast = max(available, key=lambda q:q[price_key])
                pitcher_forecasts.append({'game_id': int(first['game_id']), 'player_id': identity[0],
                    'Player': player, 'Market': MARKETS[league][market][0], 'Pick': side, 'Line': line,
                    'Best sportsbook': best_forecast['book'], 'Best decimal odds': best_forecast[price_key],
                    'Estimated chance': estimate['Estimated chance'], 'Projected stat': estimate['Projected stat'],
                    'Prior games': estimate['Prior games'],
                    'Non-push games': estimate['Prior games']-estimate['Historical pushes'],
                    'Participation status': participation,
                    'start_time': first['start_time'], 'Captured UTC': first['capture_time'],
                    'Estimate method': estimate['Estimate method'] if paired_probabilities else 'Prior appearances + neutral shrinkage; one-sided quote'})
            required_decisions = 8 if league == 'MLB' and not pitching_market else 10
            enough_history = estimate and estimate['Prior games'] >= 10 and estimate['Prior games']-estimate['Historical pushes'] >= required_decisions
            if enough_history:

                if league != 'NBA' and not paired_probabilities:
                    estimate['Estimate method'] = 'Prior 30 appearances + neutral shrinkage; one-sided sportsbook quote'
                best = max(available, key=lambda q:q[price_key])
                if league == 'NBA':
                    opposite = 'Under' if side == 'Over' else 'Over'
                    # Both directions come from the same fitted predictive distribution.
                    # Stored Over/Under chances already exclude the same push mass.
                    estimate['Opposite side offered'] = any((q.get('under_decimal_odds' if side == 'Over' else 'over_decimal_odds') or 0) > 1 for q in unique.values())
                value_eligible = False
                push_chance = estimate.get('Estimated push chance',estimate['Historical pushes']/estimate['Prior games'])
                expected_return = (1-push_chance)*(estimate['Estimated chance']*best[price_key]-1)
                qualified = True
                picks.append({'Player':player,'Market':MARKETS[league][market][0],'Pick':side,'Line':line,
                              'Market chance':market_p,'Best sportsbook':best['book'],'Best decimal odds':best[price_key], 'Sportsbook offers':[{'book':q['book'], 'decimal_odds':q[price_key], 'updated_at':q['updated_at']} for q in available], 'Estimated return per unit':expected_return,'Value eligible':value_eligible,'Break-even cover chance':1/best[price_key],'Estimated push chance':push_chance,'Books':len(available),'Game':first['away_team']+' @ '+first['home_team'],
                              'start_time':first['start_time'],'Captured UTC':first['capture_time'],
                              **({'Participation status': participation, 'Event match': first.get('Event match'), 'game_id':int(first['game_id']), 'player_id':int(identity[0])} if league=='MLB' else {}), **estimate})
        if league in ('MLB', 'NBA') and not qualified:
            excluded(first, player, market, line, 'No offered side has sufficient usable history for an estimate.')
    # One side of one line per player/market/game; avoid ranking duplicate books or alternatives.
    picks.sort(key=lambda r:(-r['Estimated chance'],-r['Prior games'],r['Player']))
    seen=set(); chosen=[]
    for pick in picks:
        key=(pick['Game'],pick['start_time'],pick['Player'],pick['Market']) + (pick['Pick'], pick['Line'])
        if key not in seen:
            seen.add(key);chosen.append(pick)
        # Retain eligible picks for every matchup; the global list stays top ten.
    return {'picks':chosen[:10],'game_picks':chosen,'errors':errors,
            **({'star_spotlights': build_star_spotlights(games, star_rosters, history_by_name, chosen), 'unavailable_players':(nba_context or {}).get('blocked_player_names',[]), 'nba_prop_model_version':3} if league == 'NBA' else {}),
            **({'exclusions': exclusions, 'quoted_candidates':len(groups), 'pitcher_forecasts':pitcher_forecasts} if league in ('MLB','NBA') else {}), 'message':f'{len(chosen)} player-prop estimates ranked from actual lines and prior appearances.',
            'note':'Estimated chance is conditional on no push and player participation. Estimates are not calibrated; no availability or injury clearance is implied.'}
