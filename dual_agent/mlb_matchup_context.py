"""Prior-only rest, recent form, and dated environment inputs for run model."""
import numpy as np
import pandas as pd

FORM_FEATURES=['opponent_starter_rest_days','opponent_starter_last_start_pitches',
 'opponent_starter_last_two_start_pitches','opponent_starter_recent_k9_blended',
 'opponent_starter_recent_bb9_blended','lineup_recent_avg_blended',
 'lineup_recent_slg_blended','lineup_recent_walk_rate_blended','lineup_recent_k_rate_blended']
ENVIRONMENT_FEATURES=['prior_park_scoring_ratio','forecast_temperature_f','forecast_wind_mph',
 'forecast_humidity_pct','forecast_precipitation_probability_pct','wind_out_to_center_mph',
 'roof_closed','stadium_center_field_ft','stadium_elevation_ft']


def attach_matchup_context(games,player_logs,schedule_games,context_dataset=None):
    needed={'player_id','game_id','start_time','games_started','pitches','pitching_outs',
     'pitcher_strikeouts','pitcher_walks','at_bats','hits','doubles','triples','batting_home_runs',
     'plate_appearances','batting_walks','batting_strikeouts'}
    if player_logs is None or not needed.issubset(player_logs.columns):
        raise ValueError('Warehouse missing rest/form columns: '+', '.join(sorted(needed-set(player_logs.columns if player_logs is not None else []))))
    logs=player_logs.copy()
    logs['start_time']=pd.to_datetime(logs.start_time,utc=True,errors='coerce')
    logs=logs.dropna(subset=['start_time','player_id']).drop_duplicates(['game_id','player_id'])
    for key in needed-{'start_time'}:logs[key]=pd.to_numeric(logs[key],errors='coerce')
    by_player={int(pid):frame.sort_values('start_time') for pid,frame in logs.groupby('player_id')}
    finished=[]
    venue_by_game={}
    for game in schedule_games:
        venue_by_game[int(game['gamePk'])]=game.get('venue',{}).get('id')
        h=game['teams']['home'].get('score');a=game['teams']['away'].get('score')
        if game.get('resumeDate') or h is None or a is None or h==a or game['status'].get('abstractGameState')!='Final':continue
        finished.append({'game_id':game['gamePk'],'season':int(game['season']),'start':pd.Timestamp(game['gameDate']), 'venue':game.get('venue',{}).get('id'),'runs':h+a})
    environment_by_game={}
    for record in (context_dataset or {}).get('rows',[]):
        try:
            capture=pd.Timestamp(record['capture_finished_at'])
            start=pd.Timestamp(record['start_time'])
            if capture.tzinfo is None or start.tzinfo is None or capture>=start:continue
            values=record.get('context_features',{})
            if record.get('context_version')!=1 or not isinstance(values,dict):continue
            environment_by_game.setdefault(int(record['game_id']),[]).append((capture,values))
        except (KeyError,TypeError,ValueError):continue
    def total(frame,key):
        return float(frame[key].sum()) if not frame.empty and frame[key].notna().all() else None
    def prior(pid,cutoff,gid,year):
        frame=by_player.get(int(pid))
        if frame is None:return logs.iloc[:0]
        return frame[(frame.start_time<cutoff)&(frame.game_id!=gid)&(frame.start_time.dt.year==year)]
    def blend(recent,alltime,numerator,denominator,pseudo=50):
        n,d=total(alltime,numerator),total(alltime,denominator)
        rn,rd=total(recent,numerator),total(recent,denominator)
        return (rn+pseudo*n/d)/(rd+pseudo) if d and n is not None and rn is not None and rd is not None else None
    coverage=[]
    for game in games:
        capture=pd.to_datetime(game['timecode'],format='%Y%m%d_%H%M%S',utc=True)
        cutoff=capture-pd.Timedelta(hours=48)
        game['form_sides']={}
        for side,other in [('home','away'),('away','home')]:
            identity=game['recorded_identities'][other]
            history=prior(identity['pitcher_id'],cutoff,game['game_id'],game['season'])
            starts=history[history.games_started>0]
            rest=last=two=None
            if not starts.empty:
                latest=starts.iloc[-1]
                rest=float((capture-latest.start_time).total_seconds()/86400)
                last=float(latest.pitches) if pd.notna(latest.pitches) else None
                two=total(starts.tail(2),'pitches')
            recent=history[history.start_time>=cutoff-pd.Timedelta(days=30)]
            k=blend(recent,history,'pitcher_strikeouts','pitching_outs',90)
            bb=blend(recent,history,'pitcher_walks','pitching_outs',90)
            metrics=[[],[],[],[]]
            for pid in game['recorded_identities'][side]['lineup_ids']:
                batting=prior(pid,cutoff,game['game_id'],game['season']).copy()
                batting=batting[batting.plate_appearances>0]
                batting['total_bases']=batting.hits+batting.doubles+2*batting.triples+3*batting.batting_home_runs
                window=batting[batting.start_time>=cutoff-pd.Timedelta(days=30)]
                for values,num,den in zip(metrics,['hits','total_bases','batting_walks','batting_strikeouts'],['at_bats','at_bats','plate_appearances','plate_appearances']):
                    value=blend(window,batting,num,den)
                    if value is not None:values.append(value)
            game['form_sides'][side]=[rest,last,two,k*27 if k is not None else None,bb*27 if bb is not None else None]+[float(np.mean(values)) if len(values)>=8 else None for values in metrics]
        known=[row for row in finished if row['season']==game['season'] and row['start']<cutoff and row['game_id']!=game['game_id']]
        venue=venue_by_game.get(game['game_id'])
        local=[row for row in known if venue is not None and row['venue']==venue]
        mean=np.mean([row['runs'] for row in known]) if known else None
        ratio=(sum(row['runs'] for row in local)+50*mean)/((len(local)+50)*mean) if local and mean and mean>0 else None
        candidates=[(time,values) for time,values in environment_by_game.get(game['game_id'],[]) if time<=capture]
        values=max(candidates,key=lambda pair:pair[0])[1] if candidates else {}
        environment=[ratio]
        for name in ENVIRONMENT_FEATURES[1:]:
            try:
                number=float(values.get(name))
                environment.append(number if np.isfinite(number) else None)
            except (ValueError,TypeError):environment.append(None)
        game['environment']=environment
        coverage.append({'game_id':game['game_id'],'season':game['season'],'park_prior_games':len(local),'dated_environment_capture':bool(candidates), 'home_form_values':sum(v is not None for v in game['form_sides']['home']),'away_form_values':sum(v is not None for v in game['form_sides']['away'])})
    return coverage
