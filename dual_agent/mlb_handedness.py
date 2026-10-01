"""Prior-appearance handedness reconstruction. Never use season-end split totals."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import json
import os
import tempfile
import time
import urllib.request
import numpy as np

HIT_BASES={'single':1,'double':2,'triple':3,'home_run':4}
NON_AB={'walk','intent_walk','intentional_walk','hit_by_pitch','sac_fly','sac_bunt','sac_fly_double_play','sac_bunt_double_play','catcher_interf'}
OUT_AB={'field_out','strikeout','strikeout_double_play','force_out','grounded_into_double_play','double_play','triple_play','fielders_choice','fielders_choice_out','field_error','other_out'}
HAND_FEATURES=['lineup_obp_vs_starter_hand','lineup_slg_vs_starter_hand','starter_allowed_obp_vs_lineup_sides','hand_split_lineup_coverage']


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def parse_appearances(payload, game):
    rows=[]
    for play in payload.get('allPlays',[]):
        about=play.get('about',{});event=play.get('result',{}).get('eventType')
        if not about.get('isComplete') or event not in HIT_BASES.keys() | NON_AB | OUT_AB:
            continue
        match=play.get('matchup',{});hand=match.get('pitchHand',{}).get('code');bat=match.get('batSide',{}).get('code')
        if hand not in {'L','R'} or bat not in {'L','R'} or not about.get('endTime'):continue
        rows.append({'game_id':game['gamePk'],'season':int(game['season']),'start_time':game['gameDate'],
                     'ended':about['endTime'],'batter':match['batter']['id'],'pitcher':match['pitcher']['id'],
                     'hand':hand,'bat':bat,'pa':1,'ab':int(event not in NON_AB),'hit':int(event in HIT_BASES),
                     'tb':HIT_BASES.get(event,0),'bb':int(event in {'walk','intent_walk','intentional_walk'}),
                     'hbp':int(event=='hit_by_pitch'),'sf':int(event in {'sac_fly','sac_fly_double_play'})})
    return rows


def valid_appearance_cache(rows, game):
    required={'game_id','season','start_time','ended','batter','pitcher','hand','bat','pa','ab','hit','tb','bb','hbp','sf'}
    if not isinstance(rows,list) or not rows:return False
    for row in rows:
        if not isinstance(row,dict) or not required.issubset(row):return False
        if row['game_id']!=game['gamePk'] or row['season']!=int(game['season']):return False
        if row['hand'] not in {'L','R'} or row['bat'] not in {'L','R'}:return False
        try:
            timestamp(row['start_time']);timestamp(row['ended'])
        except (TypeError,ValueError,AttributeError):return False
    return True


def fetch_prior_appearances(game, cache, attempts=3):
    """Recover invalid caches; retry downloads; atomically retain valid rows."""
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    path=cache/f"{game['gamePk']}_appearances_v1.json"
    if path.exists():
        try:
            rows=json.loads(path.read_text())
            if valid_appearance_cache(rows,game):return rows
        except (OSError,ValueError,TypeError):
            pass  # Redownload this entry only; other cache files remain intact.
    error=None
    for attempt in range(attempts):
        temp_path=None
        try:
            request=urllib.request.Request(
                f"https://statsapi.mlb.com/api/v1/game/{game['gamePk']}/playByPlay",
                headers={'Accept':'application/json','User-Agent':'MarketEdgeAI-history/1.0'})
            with urllib.request.urlopen(request,timeout=25) as response:
                data=json.load(response)
            rows=parse_appearances(data,game)
            if not valid_appearance_cache(rows,game):
                raise ValueError('No valid supported completed plate appearances')
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=cache,
                                             prefix=f"{game['gamePk']}_",suffix='.tmp',delete=False) as handle:
                temp_path=Path(handle.name)
                json.dump(rows,handle,allow_nan=False)
                handle.flush();os.fsync(handle.fileno())
            os.replace(temp_path,path)
            return rows
        except Exception as exc:
            error=exc
            if attempt+1<attempts:time.sleep(.5*(attempt+1))
        finally:
            if temp_path is not None:temp_path.unlink(missing_ok=True)
    raise ValueError(f"Game {game['gamePk']} failed after {attempts} attempts: {error}") from error


def collect_prior_appearances(games, cache_root, progress=None):
    root=Path(cache_root);cache=root/'handedness_play_cache';cache.mkdir(exist_ok=True)
    maximum={}
    for g in games:
        cap=datetime.strptime(g['timecode'],'%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)-timedelta(hours=48)
        maximum[g['season']]=max(maximum.get(g['season'],cap),cap)
    requested=[]
    for year,limit in maximum.items():
        schedule=json.loads((root/'historical_pilot_cache'/f'schedule_{year}.json').read_text())
        requested.extend(g for day in schedule['dates'] for g in day['games'] if g.get('gameType')=='R' and g['status'].get('abstractGameState')=='Final' and not g.get('resumeDate') and timestamp(g['gameDate'])<limit)
    def fetch(g):
        return fetch_prior_appearances(g,cache)
    records=[];errors=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending={pool.submit(fetch,g):g for g in requested}
        for i,future in enumerate(as_completed(pending),1):
            try:records.extend(future.result())
            except Exception as exc:errors.append({'game_id':pending[future]['gamePk'],'error':str(exc)})
            if progress and (i%10==0 or i==len(pending)):progress(i,len(pending))
    if errors:
        # Missing prior games can bias player splits; stop instead of treating them as zero.
        raise ValueError(f"Handedness history incomplete: {len(errors)} prior games failed. Cached successes retained; rerun to resume. First failure: {errors[0]}")
    return records,{'prior_games':len(requested),'plate_appearances':len(records),'failed_games':len(errors)}


def stats(rows):
    return {key:sum(r[key] for r in rows) for key in ['pa','ab','hit','tb','bb','hbp','sf']}


def obp(values):
    denominator=values['ab']+values['bb']+values['hbp']+values['sf']
    return (values['hit']+values['bb']+values['hbp'])/denominator if denominator else None


def shrunk_rates(split,overall):
    prior=obp(overall)
    if prior is None or not overall['ab']:return None,None
    denominator=split['ab']+split['bb']+split['hbp']+split['sf']
    # Fixed sample-support shrinkage toward this player's prior overall rates.
    return ((split['hit']+split['bb']+split['hbp']+100*prior)/(denominator+100),
            (split['tb']+100*overall['tb']/overall['ab'])/(split['ab']+100))


def attach_handedness(games,records,progress=None):
    batters=defaultdict(list);pitchers=defaultdict(list)
    for row in records:
        batters[(row['season'],row['batter'])].append(row)
        pitchers[(row['season'],row['pitcher'])].append(row)
    coverage=[]
    for index,game in enumerate(games,1):
        cutoff=datetime.strptime(game['timecode'],'%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)-timedelta(hours=48)
        def eligible(rows):
            return [r for r in rows if r['game_id']!=game['game_id'] and timestamp(r['start_time'])<cutoff and timestamp(r['ended'])<cutoff]
        game['hand_sides']={}
        for side,other in [('home','away'),('away','home')]:
            opponent=game['recorded_identities'][other]['pitcher_id']
            prior_pitcher=eligible(pitchers[(game['season'],opponent)])
            hands={r['hand'] for r in prior_pitcher}
            # Ambidextrous or inconsistent pitcher metadata stays unknown.
            hand=next(iter(hands)) if len(hands)==1 else None
            obps=[];slgs=[];allowed=[]
            for pid in game['recorded_identities'][side]['lineup_ids']:
                prior=eligible(batters[(game['season'],pid)])
                matching=[r for r in prior if r['hand']==hand]
                if not hand or not matching:continue
                rate,power=shrunk_rates(stats(matching),stats(prior))
                if rate is None:continue
                obps.append(rate);slgs.append(power)
                sides={r['bat'] for r in matching}
                if len(sides)==1:
                    bat=next(iter(sides));pitch_split=[r for r in prior_pitcher if r['bat']==bat]
                    pitcher_rate,_=shrunk_rates(stats(pitch_split),stats(prior_pitcher))
                    if pitcher_rate is not None:allowed.append(pitcher_rate)
            count=len(obps)
            game['hand_sides'][side]=[float(np.mean(obps)) if count>=8 else None,
                                      float(np.mean(slgs)) if count>=8 else None,
                                      float(np.mean(allowed)) if len(allowed)>=8 else None,count/9]
            coverage.append({'game_id':game['game_id'],'season':game['season'],'side':side,'starter_hand':hand,'lineup_players_with_splits':count,'pitcher_split_players':len(allowed)})
        if progress and (index%10==0 or index==len(games)):progress(index,len(games))
    return coverage
