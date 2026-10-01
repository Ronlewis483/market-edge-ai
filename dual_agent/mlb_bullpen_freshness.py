"""Verified prior-game completion times for immediate bullpen workload."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import urllib.request
import pandas as pd


def completion_time(payload):
    plays=payload.get('allPlays',[])
    if not plays or not all(p.get('about',{}).get('isComplete') for p in plays):
        raise ValueError('Incomplete play history')
    ended=[pd.to_datetime(p['about']['endTime'],utc=True) for p in plays if p.get('about',{}).get('endTime')]
    if len(ended)!=len(plays): raise ValueError('Missing completed-play timestamps')
    return max(ended).isoformat()


def collect_bullpen_completion_times(games,schedules,cache_root,progress=None):
    """Fetch recent final games only; fail rather than use incomplete workload."""
    schedule={int(g['gamePk']):g for g in schedules}
    by_team={}
    for gid,g in schedule.items():
        start=pd.to_datetime(g['gameDate'],utc=True)
        for side in ['home','away']:
            by_team.setdefault((int(g['season']),int(g['teams'][side]['team']['id'])),[]).append((gid,g,start))
    requested={}
    for target in games:
        cap=pd.to_datetime(target['timecode'],format='%Y%m%d_%H%M%S',utc=True)
        teams={int(target['recorded_identities'][s]['team_id']) for s in ['home','away']}
        recent=[row for team in teams for row in by_team.get((int(target['season']),team),[])]
        for gid,g,start in recent:
            if (gid!=target['game_id'] and cap-pd.Timedelta(days=3)<=start<cap and
                g.get('gameType')=='R' and not g.get('resumeDate') and
                g.get('status',{}).get('abstractGameState')=='Final'):
                requested[gid]=g
    root=Path(cache_root)/'bullpen_completion_cache';root.mkdir(parents=True,exist_ok=True)
    def fetch(gid):
        path=root/f'{gid}_v1.json'
        if path.exists():
            value=json.loads(path.read_text());pd.to_datetime(value['ended'],utc=True)
            return value['ended']
        with urllib.request.urlopen(f'https://statsapi.mlb.com/api/v1/game/{gid}/playByPlay',timeout=25) as response:
            ended=completion_time(json.load(response))
        path.write_text(json.dumps({'game_id':gid,'ended':ended,'source':'MLB completed play history'}))
        return ended
    records={};errors=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending={pool.submit(fetch,gid):gid for gid in requested}
        for index,future in enumerate(as_completed(pending),1):
            gid=pending[future]
            try:records[gid]=future.result()
            except Exception as exc:errors.append({'game_id':gid,'error':str(exc)})
            if progress and (index%10==0 or index==len(pending)):progress(index,len(pending))
    if errors:raise ValueError(f'Recent bullpen completion history incomplete: {len(errors)} failures; cached successes retained. First: {errors[0]}')
    return records,{'requested_games':len(requested),'verified_games':len(records),'window_days':3,'source':'MLB completed play history','suspended_resumed_games':'excluded'}
