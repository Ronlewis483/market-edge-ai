"""NCAAF-only saved-history regression and exact listed pregame props.

No NFL weights, invented lines or fabricated zero appearances. Collection and
training are explicit bounded maintenance actions, not part of winner generation.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import threading

ROOT = Path('ncaaf_history_cache/player_props')
_LOCK = threading.RLock()
MARKETS = {
 'player_pass_yds': ('Passing yards','passing','YDS'),
 'player_pass_tds': ('Passing touchdowns','passing','TD'),
 'player_pass_interceptions': ('Passing interceptions','passing','INT'),
 'player_pass_attempts': ('Passing attempts','passing','C/ATT'),
 'player_pass_completions': ('Passing completions','passing','C/ATT'),
 'player_rush_yds': ('Rushing yards','rushing','YDS'),
 'player_rush_attempts': ('Rushing attempts','rushing','CAR'),
 'player_reception_yds': ('Receiving yards','receiving','YDS'),
 'player_receptions': ('Receptions','receiving','REC'),
}


def _read(name, default):
    try: return json.loads((ROOT/name).read_text())
    except (OSError, ValueError): return default


def _write(name, data):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/name; tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False)); tmp.replace(path)


def _weights_path():
    local=ROOT/'weights.json'
    return local if local.exists() else Path(__file__).with_name('ncaaf_player_prop_regression_weights.json')


def _weights():
    try:return json.loads(_weights_path().read_text())
    except (OSError,ValueError):return {}


def _number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError): return None


def parse_boxes(boxes, games, now=None):
    """Only explicit category stats, linked to a completed CFBD game/team."""
    from dual_agent.ncaaf_prediction import stamp, name_key
    now = now or datetime.now(timezone.utc)
    records = {}
    for box in boxes:
        game = games.get(str(box.get('id')))
        if not game or game.get('completed') is not True: continue
        start = stamp(game.get('startDate'))
        if not start or start >= now-timedelta(days=2): continue
        team_ids = {name_key(game.get(side+'Team')):str(game.get(side+'Id')) for side in ('home','away')}
        for team in box.get('teams', []):
            tid = team_ids.get(name_key(team.get('team')))
            if not tid or tid == 'None': continue
            for cat in team.get('categories', []):
                for stat in cat.get('types', []):
                    for market, (_, category, field) in MARKETS.items():
                        if str(cat.get('name','')).lower()!=category or str(stat.get('name','')).upper()!=field: continue
                        for athlete in stat.get('athletes', []):
                            if not athlete.get('id') or not athlete.get('name'): continue
                            value = athlete.get('stat')
                            if field == 'C/ATT':
                                pair = str(value).split('/')
                                if len(pair)!=2: continue
                                completed, attempted = map(_number, pair)
                                if completed is None or attempted is None or not 0<=completed<=attempted: continue
                                value = attempted if market.endswith('attempts') else completed
                            else: value = _number(value)
                            if value is None or not float(value).is_integer(): continue
                            if 'yds' not in market and value<0: continue
                            row = {'player_id':str(athlete['id']), 'player':athlete['name'],
                                'team_id':tid,'team':team['team'], 'game_id':str(game['id']),
                                'date':start.isoformat(),'market':market,'value':float(value)}
                            key = (row['player_id'],row['game_id'],market)
                            # Conflicts are retained as invalid, never last-write-wins.
                            if key in records and records[key] and records[key]['value']!=row['value']: records[key]=None
                            elif key not in records: records[key]=row
    return [r for r in records.values() if r is not None]
