"""NBA star cards selected from reported current-season roster salaries.

Salary selects the spotlight only; it never affects any prediction equation.
No Odds API calls are made here. Missing salaries/history remain unknown.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import unicodedata

import pandas as pd
import requests
import streamlit as st

BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'


def key(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value)).lower() if c.isalnum())


@st.cache_data(ttl=86400, show_spinner=False)
def team_index():
    response = requests.get(BASE+'/teams', timeout=8)
    response.raise_for_status()
    teams = response.json()['sports'][0]['leagues'][0]['teams']
    result = {}
    for entry in teams:
        team = entry['team']
        for label in ('displayName', 'shortDisplayName', 'abbreviation'):
            if team.get(label):
                result[key(team[label])] = str(team['id'])
    # Some moneyline feeds spell out Los Angeles for the Clippers.
    if key('LA Clippers') in result:
        result[key('Los Angeles Clippers')] = result[key('LA Clippers')]
    return result


def season_year(start):
    stamp = pd.Timestamp(start)
    return stamp.year + (stamp.month >= 7)


def current_salary(athlete, season):
    contracts = [athlete.get('contract') or {}] + (athlete.get('contracts') or [])
    for contract in contracts:
        if (contract.get('season') or {}).get('year') != season:
            continue
        try:
            value = float(contract['salary'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def select_stars(payload, season):
    """Keep injured stars visible, without implying that they will play."""
    if (payload.get('season') or {}).get('year') != season:
        raise ValueError('Roster season does not match this game.')
    players = []
    for athlete in payload.get('athletes', []):
        salary = current_salary(athlete, season)
        name = athlete.get('displayName') or athlete.get('fullName')
        if salary is None or not name or not athlete.get('id'):
            continue
        injuries = [str(r.get('status')) for r in athlete.get('injuries', []) if r.get('status')]
        status = ' / '.join(dict.fromkeys(injuries)) or 'Game availability unconfirmed'
        blocked = any(key(s) in {'out', 'injuredreserve', 'suspension', 'suspended'} for s in injuries)
        roster_status = key((athlete.get('status') or {}).get('type', ''))
        if roster_status in {'inactive', 'suspended'}:
            blocked = True
            if not injuries:
                status = str((athlete.get('status') or {}).get('name', 'Inactive'))
        players.append({'player': name, 'player_id': str(athlete['id']),
            'salary': salary, 'season': season, 'status': status,
            'unavailable': blocked, 'headshot': (athlete.get('headshot') or {}).get('href'),
            'selection_basis': 'Top two reported current-season salaries'})
    players.sort(key=lambda p: (-p['salary'], p['player']))
    # Do not silently substitute salary figures from a previous season.
    return players[:2]


@st.cache_data(ttl=1800, show_spinner=False)
def roster_stars(team, season):
    team_id = team_index().get(key(team))
    if team_id is None:
        raise ValueError('No exact roster identity found for '+team)
    url = BASE+'/teams/'+team_id+'/roster'
    response = requests.get(url, params={'season': season}, timeout=8)
    response.raise_for_status()
    payload = response.json()
    actual = str((payload.get('team') or {}).get('id', ''))
    if actual != team_id:
        raise ValueError('Roster team identity does not match '+team)
    players = select_stars(payload, season)
    return {'team': team, 'players': players,
        'captured_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'source': 'ESPN roster / reported season salaries', 'source_url': url,
        'issue': None if len(players) == 2 else 'Fewer than two players have reported salaries for this season.'}


def collect_star_rosters(games):
    wanted = sorted({(g[side+'_team'], season_year(g['commence_time']))
        for g in games for side in ('away', 'home')})
    # Warm the shared team identity cache before concurrent roster requests.
    try:
        team_index()
    except Exception as exc:
        return {(team, year): {'team': team, 'players': [],
            'issue': 'Star roster unavailable: '+str(exc)} for team, year in wanted}
    def fetch(item):
        team, year = item
        try:
            return item, roster_stars(team, year)
        except Exception as exc:
            return item, {'team': team, 'players': [], 'issue': 'Star roster unavailable: '+str(exc)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(fetch, wanted))


def build_star_spotlights(games, rosters, histories, picks):
    from dual_agent.sports_player_prop_predictions import MARKETS, nba_market_value
    history_index = {key(name): rows for name, rows in histories.items()}
    result = []
    for game in games:
        start = pd.Timestamp(game['commence_time'])
        teams = []
        for side in ('away', 'home'):
            team = game[side+'_team']
            roster = rosters.get((team, season_year(start)), {'team': team, 'players': []})
            players = []
            for original in roster['players']:
                player = dict(original)
                history = history_index.get(key(player['player']), [])
                history = [row for row in history
                    if pd.Timestamp(row['source_date'])+pd.Timedelta(hours=48) < start]
                projections = []
                for _, (label, stat) in MARKETS['NBA'].items():
                    values = [nba_market_value(row, stat) for row in history]
                    values = [v for v in values if v is not None and math.isfinite(v)][-30:]
                    if values:
                        projections.append({'market': label, 'projection': sum(values)/len(values),
                            'prior_games': len(values)})
                matching = [p for p in picks if key(p['Player']) == key(player['player'])
                    and p.get('Game') == game['away_team']+' @ '+game['home_team']
                    and pd.Timestamp(p['start_time']) == start]
                matching.sort(key=lambda p: (-p['Estimated chance'], -p['Prior games']))
                # One most-likely actual side per category in this spotlight.
                seen = set(); best = []
                for pick in matching:
                    if pick['Market'] not in seen:
                        seen.add(pick['Market']); best.append(pick)
                player.update(projections=projections, props=[] if player['unavailable'] else best,
                    projection_note='Prior-appearance averages, conditional on playing; not adjusted for minutes or opponent.')
                players.append(player)
            teams.append(dict(roster, players=players))
        result.append({'home_team': game['home_team'], 'away_team': game['away_team'],
            'commence_time': start.isoformat(), 'teams': teams})
    return result
