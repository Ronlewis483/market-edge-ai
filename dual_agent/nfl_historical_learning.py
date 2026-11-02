"""Offline NFL historical reconstruction; never mix it into live forecast counts.

Uses public nflverse schedules/results, a conservative result-availability delay,
and the current five-feature/logistic base equation. Saves a separate connected
candidate. Validated bootstrap activation is opt-in; incumbents stay protected.
"""
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import argparse
import hashlib
import inspect
import json

import numpy as np
import pandas as pd

from dual_agent.nfl_connected_engine import FEATURES
from dual_agent.sports_connected_winner import ROOT, Store, clean_team, digest

SOURCE='https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
PREFIX=ROOT+'nfl/historical_v1/'
COLUMNS=['game_id','season','game_type','gameday','gametime','home_team','away_team','home_score','away_score']
METHOD='three_season_prior_results_logistic_v1'
LIMITATIONS=[
 'Retrospective reconstruction, not archived live forecasts or an untouched benchmark.',
 'Only prior completed scores, team form and rest are used. No historical injuries, QB availability, pregame weather or odds are reconstructed.',
 'Results are assumed available 24 hours after kickoff; publication timestamps and historical data revisions are not independently verified.',
 'Home/away assignment and kickoff timestamps use the current schedule archive. Corrections may differ from what was published before the game.',
 'The base model is refitted once per forecast date using earlier available results; game features use a three-season window.',
 'Historical candidate eligibility does not establish prospective accuracy or betting profitability.'
]


def base_identity():
    from dual_agent.nfl_research import predict_nfl_matchup
    return 'nfl_live:'+hashlib.sha256(inspect.getsource(predict_nfl_matchup).encode()).hexdigest()[:20]


def load_source(store,now):
    cached=store.read(PREFIX+'schedule_snapshot.json.gz')
    if cached and cached.get('source')==SOURCE:
        at=pd.Timestamp(cached['fetched_at'])
        if pd.Timedelta(0)<=now-at<pd.Timedelta(hours=24):return pd.DataFrame(cached['rows']),cached['sha256'],False
    import requests
    response=requests.get(SOURCE,timeout=(5,45))
    if response.status_code!=200:raise RuntimeError('NFL public schedule download failed (HTTP '+str(response.status_code)+')')
    frame=pd.read_csv(StringIO(response.text),dtype={'game_id':str})
    if not set(COLUMNS)<=set(frame):raise ValueError('NFL schedule schema changed; required columns missing')
    frame=frame[COLUMNS].copy()
    sha=hashlib.sha256(response.content).hexdigest()
    rows=json.loads(frame.to_json(orient='records'))
    store.write(PREFIX+'schedule_snapshot.json.gz',{'source':SOURCE,'sha256':sha,'fetched_at':now.isoformat(),'rows':rows},replace=True)
    return frame,sha,True


def completed_games(frame,start_season,now):
    missing=set(COLUMNS)-set(frame)
    if missing:raise ValueError('Missing NFL schedule columns: '+str(sorted(missing)))
    frame=frame[COLUMNS].copy()
    frame['season']=pd.to_numeric(frame.season,errors='coerce')
    frame=frame[(frame.season>=start_season-2)&frame.game_type.isin(['REG','WC','DIV','CON','SB','POST'])].copy()
    local=pd.to_datetime(frame.gameday.astype(str)+' '+frame.gametime.astype(str),errors='coerce')
    frame['start_time']=local.dt.tz_localize('America/New_York',ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    for col in ('home_score','away_score'):frame[col]=pd.to_numeric(frame[col],errors='coerce')
    frame=frame.dropna(subset=['game_id','start_time','home_team','away_team','home_score','away_score','season'])
    frame=frame[(frame.start_time+pd.Timedelta(hours=24)<now)&(frame.home_score>=0)&(frame.away_score>=0)]
    frame=frame[np.isfinite(frame.home_score)&np.isfinite(frame.away_score)]
    frame['home_team']=frame.home_team.map(lambda x:clean_team(x,'NFL').upper())
    frame['away_team']=frame.away_team.map(lambda x:clean_team(x,'NFL').upper())
    codes=set('ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS'.split())
    frame=frame[frame.home_team.isin(codes)&frame.away_team.isin(codes)&(frame.season<=now.year)]
    frame=frame[(frame.home_team!=frame.away_team)&frame.home_score.mod(1).eq(0)&frame.away_score.mod(1).eq(0)]
    # Conflicting duplicate labels/timestamps never choose a convenient last row.
    conflicting=frame.groupby('game_id')[['home_score','away_score','start_time','home_team','away_team']].transform('nunique').gt(1).any(axis=1)
    return frame[~conflicting].drop_duplicates('game_id').sort_values(['start_time','game_id']).reset_index(drop=True)


def features_for_window(games,season):
    """Rebuild the same cumulative/recent-five features within three seasons."""
    frame=games[(games.season>=season-2)&(games.season<=season)]
    history=defaultdict(list);pending=[];result=[]
    def team_features(team,at):
        rows=history[team]
        if not rows:return [.5,0.,.5,0.,7.]
        recent=rows[-5:]
        return [np.mean([r['win'] for r in rows]),np.mean([r['margin'] for r in rows]),
                np.mean([r['win'] for r in recent]),np.mean([r['margin'] for r in recent]),
                (at-rows[-1]['start']).total_seconds()/86400]
    for row in frame.to_dict('records'):
        at=row['start_time'];known_before=at.floor('D')-pd.Timedelta(hours=1);still=[]
        for prior in pending:
            if prior['start_time']+pd.Timedelta(hours=24)<known_before:
                h,a=prior['home_score'],prior['away_score']
                # Match the existing feature builder's win coding, including ties.
                history[prior['home_team']].append(dict(start=prior['start_time'],win=.5 if h==a else int(h>a),margin=h-a))
                history[prior['away_team']].append(dict(start=prior['start_time'],win=.5 if a==h else int(a>h),margin=a-h))
            else:still.append(prior)
        pending=still
        home=team_features(row['home_team'],at);away=team_features(row['away_team'],at)
        values=[float(h-a) for h,a in zip(home,away)]
        result.append(dict(row,x=values,label_time=at+pd.Timedelta(hours=24),
                           outcome=None if row['home_score']==row['away_score'] else int(row['home_score']>row['away_score'])))
        pending.append(row)
    return result


def reconstruct(games,start_season,base_id,min_base=200):
    from sklearn.linear_model import LogisticRegression
    output=[];fits=0
    for season in sorted(set(int(v) for v in games.season if v>=start_season)):
        rows=features_for_window(games,season)
        targets=[r for r in rows if int(r['season'])==season and r['outcome'] is not None]
        for day in sorted({r['start_time'].floor('D') for r in targets}):
            batch=[r for r in targets if r['start_time'].floor('D')==day]
            captured=min(r['start_time'] for r in batch)-pd.Timedelta(hours=1)
            prior=[r for r in rows if r['label_time']<day-pd.Timedelta(hours=1) and r['outcome'] is not None]
            if len(prior)<min_base or len({r['outcome'] for r in prior})<2:continue
            model=LogisticRegression(max_iter=2000).fit(np.array([r['x'] for r in prior]),[r['outcome'] for r in prior]);fits+=1
            probabilities=model.predict_proba([r['x'] for r in batch])[:,1]
            for row,p in zip(batch,probabilities):
                output.append(dict(sport='NFL',base_id=base_id,event_id=str(row['game_id']),
                    start_time=row['start_time'].isoformat(),captured_at=captured.isoformat(),
                    label_observed_at=row['label_time'].isoformat(),home_team=row['home_team'],away_team=row['away_team'],
                    original_probability=float(p),context=dict(zip(FEATURES,row['x'])),outcome=int(row['outcome']),
                    source='historical_reconstruction',historical_training_games=len(prior)))
    return output,fits


class CandidateStore:
    """Reuse the current NFL trainer, confined to the historical namespace."""
    def __init__(self,store):self.store=store;self.active_candidate=None
    def read(self,path):
        if path==ROOT+'nfl/active.json.gz':return None
        return self.store.read(self.redirect(path))
    def redirect(self,path):
        prefix=ROOT+'nfl/'
        if not path.startswith(prefix):raise ValueError('Historical trainer tried another sport namespace')
        return PREFIX+path[len(prefix):]
    def write(self,path,value,replace=False):
        if path==ROOT+'nfl/active.json.gz':self.active_candidate=value;return
        value=dict(value,source='historical_reconstruction')
        self.store.write(self.redirect(path),value,replace=replace)


def activate_bootstrap(store,base_id):
    """Publish only a passing candidate into an empty compatible deployment."""
    candidate=store.read(PREFIX+'eligible_candidate.json.gz')
    active=store.read(ROOT+'nfl/active.json.gz')
    if active:return {'activation_status':'incumbent_retained_use_later_result_comparison'}
    if not candidate or candidate.get('base_id')!=base_id or not candidate.get('checks_passed'):
        return {'activation_status':'waiting_for_passing_historical_candidate'}
    store.write(ROOT+'nfl/candidates/'+candidate['model_id']+'.json.gz',candidate,replace=True)
    store.write(ROOT+'nfl/active.json.gz',candidate,replace=True)
    return {'activated':True,'live_model_changed':True,'activation_status':'validated_historical_bootstrap_activated'}


def run(start_season=2018,store=None,now=None,input_csv=None,activate=False):
    if not 1999<=start_season<=datetime.now(timezone.utc).year:raise ValueError('Invalid NFL history start season')
    now=pd.Timestamp(now or datetime.now(timezone.utc));now=now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    store=store or Store();report={'sport':'NFL','source':'historical_reconstruction','method':METHOD,
        'start_season':start_season,'checked_at':now.isoformat(),'activated':False,'live_model_changed':False,
        'paid_sports_api_requests':0,'limitations':LIMITATIONS}
    try:
        if input_csv:
            frame=pd.read_csv(input_csv,dtype={'game_id':str});sha=hashlib.sha256(Path(input_csv).read_bytes()).hexdigest();downloaded=False
        else:frame,sha,downloaded=load_source(store,now)
        games=completed_games(frame,start_season,now);base_id=base_identity()
        fingerprint=digest([METHOD,base_id,start_season,json.loads(games.to_json(orient='records',date_format='iso'))])
        prior=store.read(PREFIX+'training_report.json.gz')
        if prior and prior.get('input_fingerprint')==fingerprint and prior.get('trained'):
            report.update(prior,status='unchanged_historical_data_skipped',checked_at=now.isoformat(),activated=False,live_model_changed=False)
            if activate:
                report.update(activate_bootstrap(store,base_id))
            return report
        rows,fits=reconstruct(games,start_season,base_id)
        report.update(base_id=base_id,raw_source_sha256=sha,input_fingerprint=fingerprint,
            public_source_downloaded=downloaded,completed_games=len(games),reconstructed_games=len(rows),base_fits=fits)
        store.write(PREFIX+'reconstructed_rows.json.gz',{'schema':1,'source':'historical_reconstruction','rows':rows,'limitations':LIMITATIONS},replace=True)
        from dual_agent.nfl_connected_engine import train_connected
        confined=CandidateStore(store)
        training=train_connected('NFL',base_id,rows,confined)
        report.update(training);report.update(source='historical_reconstruction',activated=False,live_model_changed=False,
            historical_checks_passed=bool(training.get('checks_passed')),eligible_for_live_evaluation=bool(confined.active_candidate))
        if training.get('trained'):
            report['status']='historical_candidate_saved'
            if confined.active_candidate:
                candidate=dict(confined.active_candidate,source='historical_reconstruction',limitations=LIMITATIONS)
                store.write(PREFIX+'eligible_candidate.json.gz',candidate,replace=True)
        if activate:report.update(activate_bootstrap(store,base_id))
        store.write(PREFIX+'training_report.json.gz',report,replace=True)
        return report
    except Exception as exc:
        report.update(status='historical_training_failed',error_type=type(exc).__name__)
        raise
    finally:
        Path('nfl_historical_learning_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-season',type=int,default=2018)
    parser.add_argument('--input-csv',default=None)
    parser.add_argument('--activate-bootstrap',action='store_true')
    args=parser.parse_args();run(args.start_season,input_csv=args.input_csv,activate=args.activate_bootstrap)
