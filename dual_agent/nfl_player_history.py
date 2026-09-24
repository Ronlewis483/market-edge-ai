"""Historical NFL player game logs for the independent player-props module."""
from datetime import datetime, timezone
import pandas as pd
import nflreadpy as nfl

STAT_COLUMNS = {
    'Passing yards': ('passing_yards',),
    'Rushing yards': ('rushing_yards',),
    'Receiving yards': ('receiving_yards',),
    'Receptions': ('receptions',),
    'Passing touchdowns': ('passing_tds', 'passing_touchdowns'),
    'Passing completions': ('completions', 'passing_completions'),
    'Interceptions thrown': ('interceptions', 'passing_interceptions'),
}


def _column(frame, *candidates):
    return next((name for name in candidates if name in frame.columns), None)


def load_player_history(seasons):
    """Return completed regular-season game stats in validate_history() format.

    Missing or ambiguous schedule matches are excluded, not assigned guessed dates.
    Weekly player stats are joined to the actual game kickoff by season/week/team.
    """
    seasons = sorted({int(s) for s in seasons})
    if not seasons or len(seasons) > 4:
        raise ValueError('Choose between one and four NFL seasons.')
    stats = nfl.load_player_stats(seasons, summary_level='week').to_pandas()
    schedule = nfl.load_schedules(seasons).to_pandas()
    if stats.empty or schedule.empty:
        raise ValueError('No NFL player stats or schedules returned for selected seasons.')

    name_col = _column(stats, 'player_name', 'player_display_name')
    team_col = _column(stats, 'recent_team', 'team')
    if not name_col or not team_col or not {'season', 'week'}.issubset(stats.columns):
        raise ValueError('Player stats schema changed: player name, team, season or week missing.')
    kickoff_col = _column(schedule, 'gameday', 'game_date')
    if not kickoff_col or not {'season', 'week', 'home_team', 'away_team'}.issubset(schedule.columns):
        raise ValueError('NFL schedule schema changed: game date, teams, season or week missing.')

    schedule = schedule.copy()
    if 'game_type' in schedule:
        schedule = schedule[schedule['game_type'].eq('REG')]
    if {'home_score', 'away_score'}.issubset(schedule.columns):
        schedule = schedule.dropna(subset=['home_score', 'away_score'])
    else:
        raise ValueError('Schedule has no final-score columns; cannot verify completed games.')

    # The schedule's game day is a calendar date, not a precise kickoff timestamp.
    # Use the following UTC day as a conservative post-game availability timestamp.
    schedule['game_time'] = (pd.to_datetime(schedule[kickoff_col], utc=True, errors='coerce')
                             + pd.Timedelta(days=1))
    schedule = schedule[schedule['game_time'] < pd.Timestamp.now(tz='UTC')]
    home = schedule[['season', 'week', 'home_team', 'game_time']].rename(columns={'home_team': 'team'})
    away = schedule[['season', 'week', 'away_team', 'game_time']].rename(columns={'away_team': 'team'})
    games = pd.concat([home, away], ignore_index=True).dropna(subset=['game_time'])
    # Never assign a game if season/week/team is not unique.
    games = games.drop_duplicates(['season', 'week', 'team'], keep=False)

    stats = stats.copy().rename(columns={name_col: 'player', team_col: 'team'})
    stats = stats.merge(games, on=['season', 'week', 'team'], how='inner', validate='many_to_one')
    if stats.empty:
        raise ValueError('No completed player games matched the schedule.')

    records = []
    for label, aliases in STAT_COLUMNS.items():
        column = _column(stats, *aliases)
        if column:
            records.append(pd.DataFrame({
                'player': stats['player'], 'market': label,
                'game_time': stats['game_time'], 'value': pd.to_numeric(stats[column], errors='coerce')
            }))
    rush = _column(stats, 'rushing_yards')
    rec = _column(stats, 'receiving_yards')
    if rush and rec:
        records.append(pd.DataFrame({
            'player': stats['player'], 'market': 'Rushing + receiving yards',
            'game_time': stats['game_time'],
            'value': pd.to_numeric(stats[rush], errors='coerce').fillna(0)
                     + pd.to_numeric(stats[rec], errors='coerce').fillna(0)
        }))
    rush_td = _column(stats, 'rushing_tds', 'rushing_touchdowns')
    rec_td = _column(stats, 'receiving_tds', 'receiving_touchdowns')
    if rush_td and rec_td:
        records.append(pd.DataFrame({
            'player': stats['player'], 'market': 'Anytime touchdown',
            'game_time': stats['game_time'],
            'value': pd.to_numeric(stats[rush_td], errors='coerce').fillna(0)
                     + pd.to_numeric(stats[rec_td], errors='coerce').fillna(0)
        }))
    if not records:
        raise ValueError('No supported player statistic columns were returned.')
    result = pd.concat(records, ignore_index=True).dropna(subset=['player', 'game_time', 'value'])
    result = result[result['player'].astype(str).str.strip().ne('')]
    return result.drop_duplicates(['player', 'market', 'game_time']).sort_values('game_time').reset_index(drop=True)
