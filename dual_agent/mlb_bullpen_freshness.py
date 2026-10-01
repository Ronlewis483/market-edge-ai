"""Verified prior-game completion times for immediate bullpen workload."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import tempfile
import time
import urllib.request
import pandas as pd


def completion_time(payload,return_details=False):
    plays=payload.get('allPlays',[])
    if not plays:raise ValueError('Empty play history')
    incomplete=[p for p in plays if not p.get('about',{}).get('isComplete')]
    if any(p.get('result',{}).get('eventType')!='game_advisory' for p in incomplete):
        raise ValueError('Incomplete baseball play history')
    ended=[pd.to_datetime(p['about']['endTime'],utc=True) for p in plays if p.get('about',{}).get('endTime')]
    if len(ended)!=len(plays) or any(pd.isna(t) for t in ended):raise ValueError('Missing completed-play timestamps')
    if not any(p.get('about',{}).get('isComplete') for p in plays):raise ValueError('No completed baseball plays')
    # A rain-delay advisory is not proof of when the game was declared final.
    # The caller requests only final, non-resumed games. Use a conservative
    # availability buffer rather than inventing an exact completion time.
    kind='advisory_48h_buffer' if incomplete else 'completed_plays'
    stamp=max(ended)+(pd.Timedelta(hours=48) if incomplete else pd.Timedelta(0))
    return (stamp.isoformat(),kind) if return_details else stamp.isoformat()


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
            try:
                value=json.loads(path.read_text());stamp=pd.to_datetime(value['ended'],utc=True)
                if pd.isna(stamp):raise ValueError('Invalid cached timestamp')
                return value['ended'],value.get('eligibility_kind','completed_plays')
            except (ValueError,KeyError,TypeError,OSError):pass
        error=None
        for attempt in range(3):
            temporary=None
            try:
                with urllib.request.urlopen(f'https://statsapi.mlb.com/api/v1/game/{gid}/playByPlay',timeout=25) as response:
                    ended,kind=completion_time(json.load(response),return_details=True)
                value={'game_id':gid,'ended':ended,'eligibility_kind':kind,'source':'MLB completed play history'}
                with tempfile.NamedTemporaryFile(mode='w',dir=root,prefix=f'{gid}_',suffix='.tmp',delete=False) as handle:
                    temporary=Path(handle.name);json.dump(value,handle);handle.flush();os.fsync(handle.fileno())
                os.replace(temporary,path)
                return ended,kind
            except Exception as exc:
                error=exc
                if attempt<2:time.sleep(.5*(attempt+1))
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
        raise ValueError(f'Game {gid} failed after 3 attempts: {error}') from error
    records={};errors=[];conservative=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending={pool.submit(fetch,gid):gid for gid in requested}
        for index,future in enumerate(as_completed(pending),1):
            gid=pending[future]
            try:
                ended,kind=future.result();records[gid]=ended
                if kind=='advisory_48h_buffer':conservative.append(gid)
            except Exception as exc:errors.append({'game_id':gid,'error':str(exc)})
            if progress and (index%10==0 or index==len(pending)):progress(index,len(pending))
    if errors:raise ValueError(f'Recent bullpen completion history incomplete: {len(errors)} failures; cached successes retained. First: {errors[0]}')
    return records,{'requested_games':len(requested),'verified_games':len(records)-len(conservative),'conservative_game_ids':sorted(conservative),'advisory_note':'Final rain-advisory games use last recorded event plus 48 hours; before that point, affected recent workload remains unknown.','window_days':3,'source':'MLB completed play history','suspended_resumed_games':'excluded'}
