"""Prepare saved NBA pregame inputs for later training; never train or place bets.

Run after the snapshot collector: python dual_agent/nba_training_dataset.py
Uses Supabase only. No Odds API or BALLDONTLIE calls. Missing inputs stay null.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from statistics import mean, median
# GitHub runs this file directly, so Python places dual_agent/ on sys.path.
# Package imports use a relative sibling; direct scripts use the local sibling.
# Do not catch dependency ImportErrors and mistake them for launch-path issues.
if __package__:
    from .nba_pregame_snapshots import Client, PREFIX, stamp, number, final, start_time, played_date, safe_issue
else:
    from nba_pregame_snapshots import Client, PREFIX, stamp, number, final, start_time, played_date, safe_issue

DATA_PREFIX='nba/training_v1/'
SCHEMA=1
FORM_FIELDS=('recent_games','support_games','win_rate_8','margin_8','points_for_8',
             'points_against_8','win_rate_30','margin_sd_30','days_since_last_game',
             'back_to_back','games_last_4_days','home_games_8')
MATCH_FIELDS=('games','estimated_pace_48','estimated_offensive_rating',
              'estimated_defensive_rating','effective_fg','rebounds','turnovers')
ROTATION_FIELDS=('players_with_minutes','top5_minutes_if_playing','top8_minutes_if_playing',
                 'mean_prior_appearances','reported_unavailable_count',
                 'restricted_minutes_count','reported_unavailable_prior_minutes')
SIDE_FIELDS=tuple('form_'+x for x in FORM_FIELDS)+('elo','elo_prior_games',)+tuple('matchup_'+x for x in MATCH_FIELDS)+tuple('rotation_'+x for x in ROTATION_FIELDS)
MARKET_FIELDS=('snapshot_lead_hours','market_home_probability','market_home_probability_spread','market_books')
BASE_FEATURES=tuple(side+'_'+f for side in ('home','away') for f in SIDE_FIELDS)+tuple('difference_'+f for f in SIDE_FIELDS)+MARKET_FIELDS
FEATURE_COLUMNS=BASE_FEATURES+tuple(f+'__missing' for f in BASE_FEATURES)


def source_history_valid(ids,game,checkpoint,captured,tid):
    """Validate frozen source references without recalculating inputs from later data."""
    if not ids or len(set(map(str,ids)))!=len(ids): return False
    dates=[]
    for gid in ids:
        g=checkpoint['games'].get(str(gid))
        if not g or str(gid)==str(game['id']): return False
        if tid not in ((g.get('home_team') or {}).get('id'),(g.get('visitor_team') or {}).get('id')): return False
        date=played_date(g)
        if date is None or date>=captured.date() or g.get('season')!=game.get('season'): return False
        dates.append(date)
    dates.sort()
    return all((b-a).days<=45 for a,b in zip(dates,dates[1:]))


def extract_features(snapshot,game,checkpoint):
    """An explicit feature allowlist keeps labels, final stats and forecasts out of X."""
    captured=stamp(snapshot.get('captured_at'));start=stamp(snapshot.get('scheduled_start'))
    if captured is None or start is None or captured>=start: raise ValueError('Snapshot is not verified pregame')
    if str(snapshot.get('game_id'))!=str(game['id']): raise ValueError('Snapshot game identity mismatch')
    if start_time(game)!=start: raise ValueError('Snapshot tipoff differs from current saved schedule; no guessed reschedule join')
    source_times=snapshot.get('source_times') or {}
    for name,value in source_times.items():
        checked=stamp(value.get('checked_at')) if isinstance(value,dict) else stamp(value)
        if checked and checked>captured: raise ValueError('Source timestamp is newer than snapshot')
    features={k:None for k in BASE_FEATURES};gaps=[]
    features['snapshot_lead_hours']=(start-captured).total_seconds()/3600
    reliability=snapshot.get('confidence_reliability') or {}
    for side,provider in (('home','home'),('away','visitor')):
        context=(snapshot.get('teams') or {}).get(provider) or {}
        if (context.get('team') or {}).get('id')!=game[provider+'_team']['id']:
            raise ValueError('Snapshot home/away team identity mismatch')
        form=context.get('form_and_schedule') or {}
        valid_form=source_history_valid(form.get('source_game_ids'),game,checkpoint,captured,game[provider+'_team']['id'])
        if valid_form:
            for key in FORM_FIELDS: features[side+'_form_'+key]=number(form.get(key))
        else: gaps.append(side+': form source window missing, crosses season/gap, or is not prior history')
        if not valid_form or (number(form.get('recent_games')) or 0)<8:
            gaps.append(side+': fewer than eight verified recent form games')
        strength=context.get('opponent_adjusted_strength') or {}
        features[side+'_elo']=number(strength.get('elo'))
        features[side+'_elo_prior_games']=number(strength.get('prior_games'))
        matchup=context.get('matchup_performance') or {}
        valid_matchup=source_history_valid(matchup.get('source_game_ids'),game,checkpoint,captured,game[provider+'_team']['id'])
        if valid_matchup:
            for key in MATCH_FIELDS: features[side+'_matchup_'+key]=number(matchup.get(key))
        else: gaps.append(side+': matchup source history not verified')
        if not valid_matchup or (number(matchup.get('games')) or 0)<8:
            gaps.append(side+': fewer than eight complete matchup boxes')
        rotation=context.get('availability_and_rotation') or {}
        players=rotation.get('players') or []
        roster_time=stamp(source_times.get('active_roster'))
        roster_fresh=(reliability.get('active_roster_complete') is True and roster_time is not None
                      and timedelta(0)<=captured-roster_time<=timedelta(hours=6))
        rotation_valid=bool(rotation.get('window_policy')) and roster_fresh
        if rotation_valid:
            usable=[p for p in players if number(p.get('recent_minutes_if_playing')) is not None
                    and (number(p.get('prior_appearances')) or 0)>0]
            usable.sort(key=lambda p:-float(p['recent_minutes_if_playing']))
            features[side+'_rotation_players_with_minutes']=len(usable)
            for n in (5,8):
                if len(usable)>=n:
                    features[side+f'_rotation_top{n}_minutes_if_playing']=sum(float(p['recent_minutes_if_playing']) for p in usable[:n])
            if usable: features[side+'_rotation_mean_prior_appearances']=mean(float(p['prior_appearances']) for p in usable)
        else: gaps.append(side+': roster/rotation stale or legacy window policy unverified')
        injury_time=stamp(source_times.get('injuries'))
        injuries_fresh=(roster_fresh and reliability.get('injury_feed_complete') is True and injury_time is not None
                        and timedelta(0)<=captured-injury_time<=timedelta(hours=6))
        if injuries_fresh:
            unavailable=[p for p in players if p.get('unavailable_reported') is True]
            features[side+'_rotation_reported_unavailable_count']=len(unavailable)
            features[side+'_rotation_restricted_minutes_count']=sum(p.get('minutes_restriction_reported') is True for p in players)
            # Unknown absent-player minutes are never treated as zero impact.
            if rotation_valid and all(number(p.get('recent_minutes_if_playing')) is not None for p in unavailable):
                features[side+'_rotation_reported_unavailable_prior_minutes']=sum(float(p['recent_minutes_if_playing']) for p in unavailable)
        else: gaps.append(side+': current availability feed not verified at capture')
        gaps.extend([side+': confirmed pregame starters unavailable',side+': replacement impact not trained'])
    for field in SIDE_FIELDS:
        h,a=features['home_'+field],features['away_'+field]
        if h is not None and a is not None: features['difference_'+field]=h-a
    prices={}
    for offer in (snapshot.get('sportsbook_context') or {}).get('offers',[]):
        updated=stamp(offer.get('updated_at'));hp=number(offer.get('home_decimal'));ap=number(offer.get('away_decimal'))
        if (updated is None or not timedelta(0)<=captured-updated<=timedelta(minutes=30)
            or hp is None or ap is None or min(hp,ap)<=1): continue
        book=str(offer.get('bookmaker') or '')
        if not book: continue
        prob=(1/hp)/(1/hp+1/ap)
        # Retain only the latest valid quote per book.
        if book not in prices or prices[book][0]<updated: prices[book]=(updated,prob)
    if prices:
        vals=[p for _,p in prices.values()]
        features.update(market_home_probability=median(vals),market_home_probability_spread=max(vals)-min(vals),market_books=len(vals))
    else: gaps.append('No fresh two-sided sportsbook quote in this snapshot')
    for name in BASE_FEATURES: features[name+'__missing']=int(features[name] is None)
    return features,sorted(set(gaps))


def split_rows(rows):
    """Date-grouped ordered splits; purge labels not observed before next partition."""
    labeled=[r for r in rows if r['label'] is not None]
    dates=sorted({r['scheduled_start'][:10] for r in labeled})
    if len(dates)<4:
        return {'status':'waiting_for_at_least_four_distinct_game_dates','dates':len(dates),
                'note':'No random split and no same-game duplication across partitions.'}
    n=len(dates);a=max(1,min(n-3,int(.7*n)));b=max(a+1,min(n-2,int(.8*n)));c=max(b+1,min(n-1,int(.9*n)))
    cut=[dates[a],dates[b],dates[c]]
    for row in labeled:
        date=row['scheduled_start'][:10]
        row['partition']=('train' if date<cut[0] else 'validation' if date<cut[1] else 'calibration' if date<cut[2] else 'test')
    order=('train','validation','calibration','test')
    for i,name in enumerate(order[:-1]):
        later=[stamp(r['captured_at']) for r in labeled if r['partition'] in order and order.index(r['partition'])>i]
        boundary=min(later) if later else None
        for row in labeled:
            if row['partition']==name and (boundary is None or stamp(row['label_observed_at'])>=boundary):
                row['partition']='purged';row['partition_issue']='Final result was not observed before the next partition inputs'
        # Purged rows are outside every partition; do not evaluate their order next pass.
    return {'status':'prepared_not_trained','cutoff_dates':cut,'counts':dict(Counter(r['partition'] for r in labeled)),
            'note':'70/10/10/10 date groups where possible; tiny partitions are not proof of trainability. Fit imputation/scaling on train only.'}


def build(client=None,now=None):
    client=client or Client();now=now or datetime.now(timezone.utc)
    checkpoint=client.read(PREFIX+'checkpoint.json.gz',compressed=True)
    if not checkpoint or checkpoint.get('schema')!=1: raise ValueError('Missing or incompatible collector checkpoint; collect first')
    cache=client.read(DATA_PREFIX+'feature_cache.json.gz',compressed=True)
    if cache is None: cache={'schema':SCHEMA,'snapshots':{}}
    if cache.get('schema')!=SCHEMA: raise ValueError('Dataset feature cache schema mismatch; existing data not reset')
    cache.setdefault('outcomes',{})
    rows=[];excluded=[];downloads=0
    for gid,capture in sorted(checkpoint.get('captures',{}).items()):
        game=checkpoint['games'].get(str(gid))
        if not game or start_time(game) is None:
            excluded.append({'game_id':gid,'reason':'No verified scheduled start'});continue
        start=start_time(game);deadline=start-timedelta(minutes=60);candidates=[]
        for path in sorted(set(capture.get('paths',[]))):
            if not path.startswith(PREFIX+'snapshots/'+str(gid)+'/'): raise ValueError('Unexpected snapshot reference')
            cached=cache['snapshots'].get(path)
            if cached is None:
                snapshot=client.read(path);downloads+=1
                if not isinstance(snapshot,dict): raise ValueError('Referenced immutable snapshot missing')
                # Store immutable raw inputs once. Revalidate references against checkpoint each run.
                cached={'snapshot':snapshot};cache['snapshots'][path]=cached
            snapshot=cached['snapshot'];captured=stamp(snapshot.get('captured_at'))
            if captured is not None and captured<=deadline and stamp(snapshot.get('scheduled_start'))==start:
                candidates.append((captured,path,snapshot))
        if not candidates:
            excluded.append({'game_id':gid,'reason':'No snapshot saved at least 60 minutes before this scheduled start'});continue
        captured,path,snapshot=max(candidates,key=lambda x:(x[0],x[1]))
        try: features,gaps=extract_features(snapshot,game,checkpoint)
        except ValueError as exc:
            excluded.append({'game_id':gid,'reason':str(exc)});continue
        # Labels live outside the feature allowlist and never enter the immutable snapshot.
        label=None;observed=None;outcome_path=None
        if final(game):
            digest=checkpoint.get('settled',{}).get(str(gid))
            if digest:
                outcome_path=PREFIX+'outcomes/'+str(gid)+'/'+digest+'.json'
                outcome=cache['outcomes'].get(outcome_path)
                if outcome is None:
                    outcome=client.read(outcome_path)
                    if not isinstance(outcome,dict): raise ValueError('Final-result sidecar missing')
                    cache['outcomes'][outcome_path]=outcome
                observed=stamp(outcome.get('observed_at'));hs=number(outcome.get('home_score'));aws=number(outcome.get('away_score'))
                if (str(outcome.get('game_id'))==str(gid) and observed and start<observed<=now
                    and hs is not None and aws is not None and min(hs,aws)>0 and hs!=aws
                    and hs==number(game.get('home_team_score')) and aws==number(game.get('visitor_team_score'))):
                    label=int(hs>aws)
                else: raise ValueError('Final-result sidecar identity, timestamp or scores inconsistent')
            else: gaps.append('Final outcome not yet persisted by collector')
        rows.append({'game_id':str(gid),'season':game.get('season'),'scheduled_start':start.isoformat(),
            'captured_at':captured.isoformat(),'home_team':game['home_team']['full_name'],
            'away_team':game['visitor_team']['full_name'],'snapshot_path':path,'outcome_path':outcome_path,
            'features':features,'label':label,'label_observed_at':observed.isoformat() if label is not None else None,
            'partition':'unassigned' if label is not None else 'pending','coverage_gaps':sorted(set(gaps)),
            'full_package_ready':False})
    rows.sort(key=lambda r:(r['scheduled_start'],r['game_id']))
    split=split_rows(rows)
    missing={name:sum(r['features'][name] is None for r in rows) for name in BASE_FEATURES}
    report={'schema':SCHEMA,'prepared_games':len(rows),'labeled_games':sum(r['label'] is not None for r in rows),
        'pending_games':sum(r['label'] is None for r in rows),'excluded_games':len(excluded),
        'missing_feature_counts':missing,'split':split,'training_performed':False,
        'live_model_changed':False,'odds_api_requests':0,'balldontlie_requests':0,
        'limitations':['Confirmed starters and trained replacement impact are not supplied.',
            'Unlisted injury status does not mean healthy. Reported unavailable minutes are descriptive, not projected lost impact.',
            'Legacy cross-season form windows are masked; no historical injuries are invented.',
            'Model probabilities and actual game/player statistics are excluded from predictor columns.',
            'This is a winner dataset, not a player-prop training dataset.']}
    dataset={'schema':SCHEMA,'feature_columns':list(FEATURE_COLUMNS),'label_column':'label',
        'policy':{'snapshot_selection':'latest saved snapshot at least 60 minutes before tipoff; one row per game',
                  'missing_values':'null plus explicit missing indicators; fit imputation on train only',
                  'partitioning':'date groups in chronological order; purge labels observed after next partition inputs'},
        'rows':rows,'excluded':excluded,'report':report}
    # Content-addressed versions make each prepared dataset reproducible; pointer moves last.
    raw=json.dumps(dataset,sort_keys=True,allow_nan=False,separators=(',',':'))
    digest=hashlib.sha256(raw.encode()).hexdigest();path=DATA_PREFIX+'datasets/'+digest+'.json.gz'
    if cache.get('last_dataset_hash')!=digest:
        existing=client.read(path,compressed=True)
        if existing is None: client.write(path,dataset,compressed=True)
        elif existing!=dataset: raise ValueError('Dataset hash conflict; existing version preserved')
    client.write(DATA_PREFIX+'latest.json',{'schema':SCHEMA,'dataset_path':path,'dataset_hash':digest,
        'updated_at':now.isoformat(),'prepared_games':len(rows),'labeled_games':report['labeled_games'],
        'pending_games':report['pending_games'],'split_status':split['status']},replace=True)
    cache['last_dataset_hash']=digest
    client.write(DATA_PREFIX+'feature_cache.json.gz',cache,replace=True,compressed=True)
    print(json.dumps({'NBA training dataset':{k:report[k] for k in ('prepared_games','labeled_games','pending_games','excluded_games','training_performed')},
                      'new_snapshot_downloads':downloads,'split_status':split['status'],'dataset_path':path}),flush=True)
    return dataset


if __name__=='__main__':
    try: build()
    except Exception as exc:
        print('NBA dataset preparation stopped: '+safe_issue(exc),flush=True)
        raise SystemExit(1)
