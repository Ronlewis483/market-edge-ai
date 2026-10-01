"""Import-safe historical MLB paired accuracy test. No live model promotion."""

def run_historical_accuracy_test(cache_root="mlb_accuracy_results", progress=None, expanded=True, games_per_season=600, cohort="seasonwide"):
    if cohort not in {"seasonwide", "june_pilot"} or games_per_season < 100:
        raise ValueError("Use seasonwide or june_pilot with at least 100 games per season")
    import urllib.request,json,time,math,concurrent.futures
    from datetime import datetime,timezone,timedelta
    from pathlib import Path
    ROOT=Path(cache_root);ROOT.mkdir(exist_ok=True);CACHE=ROOT/'historical_pilot_cache';CACHE.mkdir(exist_ok=True)
    def get(url,cache=None):
        if cache and cache.exists():return json.loads(cache.read_text())
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=25) as response:data=json.load(response)
                if cache:cache.write_text(json.dumps(data))
                return data
            except Exception:
                if attempt==2:raise
                time.sleep(.5)
    def ts(value):return datetime.fromisoformat(value.replace('Z','+00:00'))
    def num(value):
        try:
            v=float(value);return v if math.isfinite(v) else None
        except (TypeError,ValueError):return None
    allgames=[]
    for year in [2024,2025,2026]:
        data=get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={year}-03-01&endDate={year}-09-30&gameType=R',CACHE/f'schedule_{year}.json')
        for day in data.get('dates',[]):
            for g in day['games']:
                if g['status'].get('abstractGameState')=='Final' and g.get('gameType')=='R' and not g.get('resumeDate'):
                    allgames.append(g)
    # Rebuild baseline prior-day team histories. Prior-date final availability is an explicit assumption.
    teamhistory={};baseline={}
    for day in sorted({g['gameDate'][:10] for g in allgames}):
        current=sorted([g for g in allgames if g['gameDate'][:10]==day],key=lambda g:g['gamePk'])
        for g in current:
            both={}
            for side in ['home','away']:
                history=teamhistory.get((g['season'],g['teams'][side]['team']['id']),[])
                def mean(key,n=None):
                    h=history[-n:] if n else history
                    return sum(x[key] for x in h)/len(h) if h else (0.5 if key=='win' else 0.)
                both[side]=[len(history),mean('win'),mean('for'),mean('against'),mean('diff'),mean('win',5),mean('win',10),mean('diff',5)]
            baseline[g['gamePk']]=[h-a for h,a in zip(both['home'],both['away'])]
        for g in current:
            for side,other in [('home','away'),('away','home')]:
                runs=g['teams'][side].get('score');against=g['teams'][other].get('score')
                if runs is not None and against is not None and runs!=against:
                    teamhistory.setdefault((g['season'],g['teams'][side]['team']['id']),[]).append({'win':int(runs>against),'for':runs,'against':against,'diff':runs-against})
    sample=[]
    for year in [2024,2025,2026]:
        games=sorted([g for g in allgames if int(g['season'])==year and g['gameDate'][:10]>=f'{year}-05-01'],key=lambda g:(g['gameDate'],g['gamePk']))
        if expanded and cohort == "seasonwide":
            # Evenly spaced chronological positions cover May through September.
            count=min(games_per_season,len(games))
            games=[games[round(i*(len(games)-1)/(count-1))] for i in range(count)] if count>1 else games[:count]
        else:
            games=[g for g in games if g['gameDate'][:10]>=f'{year}-06-01'][:150]
        sample.extend(games)
    def worker(g):
        gid=g['gamePk'];base=f'https://statsapi.mlb.com/api/v1.1/game/{gid}/feed/live';start=ts(g['gameDate']);cutoff=start-timedelta(minutes=5)
        try:
            timestamps=get(base+'/timestamps',CACHE/f'{gid}_timestamps.json')
            codes=[c for c in timestamps if datetime.strptime(c,'%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)<cutoff]
            if not codes:return {'game_id':gid,'error':'No pregame timecode'}
            code=max(codes);feed=get(base+'?timecode='+code,CACHE/f'{gid}_{code}.json')
            if feed.get('metaData',{}).get('timeStamp')!=code:return {'game_id':gid,'error':'Timecode mismatch'}
            live=feed.get('liveData',{});plays=live.get('plays',{}).get('allPlays',[])
            if any(p.get('about',{}).get('isComplete') or any(e.get('isPitch') or e.get('details',{}).get('isScoringPlay') for e in p.get('playEvents',[])) for p in plays):
                return {'game_id':gid,'error':'Gameplay occurred in requested snapshot'}
            box=live.get('boxscore',{}).get('teams',{});sidevals={};ids={}
            for side in ['home','away']:
                team=box.get(side,{});players=team.get('players',{});order=team.get('battingOrder',[]);pitchers=team.get('pitchers',[])
                if len(order)!=9 or not pitchers:return {'game_id':gid,'error':'Pregame lineup or starting pitcher missing'}
                # Use only identities recorded in archived liveData, never hydrated gameData probablePitchers.
                pid=pitchers[0];pitcher=players.get('ID'+str(pid),{});stats=pitcher.get('seasonStats',{}).get('pitching',{})
                era,whip=num(stats.get('era')),num(stats.get('whip'))
                if era is None or whip is None:return {'game_id':gid,'error':'Prior pitcher rates missing'}
                ops=[num(players.get('ID'+str(p),{}).get('seasonStats',{}).get('batting',{}).get('ops')) for p in order]
                ops=[v for v in ops if v is not None]
                if len(ops)<8:return {'game_id':gid,'error':'Prior lineup rates missing'}
                sidevals[side]=[era,whip,sum(ops)/len(ops)]
                if expanded:
                    innings=str(stats.get('inningsPitched', ''))
                    try:
                        whole, _, partial=innings.partition('.')
                        outs=int(whole)*3+int(partial or 0)
                        if int(partial or 0)>2 or outs<=0: raise ValueError()
                    except ValueError:
                        outs=0
                    def rate(key):
                        value=num(stats.get(key))
                        return value*27/outs if value is not None and outs else None
                    def lineup_rate(key):
                        values=[num(players.get('ID'+str(player),{}).get('seasonStats',{}).get('batting',{}).get(key)) for player in order]
                        values=[value for value in values if value is not None]
                        return sum(values)/len(values) if len(values)>=8 else None
                    sidevals[side] += [rate('strikeOuts'), rate('baseOnBalls'), lineup_rate('obp'), lineup_rate('slg')]
                ids[side]={'pitcher_id':pid,'lineup_ids':order}
            h=g['teams']['home'].get('score');a=g['teams']['away'].get('score')
            if h is None or a is None or h==a:return {'game_id':gid,'error':'Final label invalid'}
            return {'game_id':gid,'season':int(g['season']),'start_time':g['gameDate'],'timecode':code,'home_win':int(h>a),
                'features':baseline[gid]+[h-a if h is not None and a is not None else None for h,a in zip(sidevals['home'],sidevals['away'])], 'recorded_identities':ids,
                'source_url':base+'?timecode='+code, 'player_sides':sidevals, 'home_runs':h, 'away_runs':a, 'home_team':g['teams']['home']['team']['name'], 'away_team':g['teams']['away']['team']['name']}
        except Exception as exc:return {'game_id':gid,'error':type(exc).__name__}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        pending=[pool.submit(worker,g) for g in sample]
        for future in concurrent.futures.as_completed(pending):
            results.append(future.result())
            if len(results)%25==0:
                (ROOT/'historical_pilot_rows.json').write_text(json.dumps(results))
                progress(len(results), len(sample)) if progress else None
    (ROOT/'historical_pilot_rows.json').write_text(json.dumps(results))
    pass
    
    import json
    from pathlib import Path
    import numpy as np,pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
    root=ROOT;rows=json.loads((root/'historical_pilot_rows.json').read_text());usable=[r for r in rows if 'features' in r];
    # Rebuild baseline using a 48-hour delay before admitting prior final scores.
    # This avoids assuming a game that started yesterday had finished at midnight.
    history={}
    schedules_by_year={}
    for year in [2024,2025,2026]:
        schedule=json.loads((root/'historical_pilot_cache'/f'schedule_{year}.json').read_text())
        schedules_by_year[year]=schedule
        for day in schedule['dates']:
            for g in day['games']:
                if g['status'].get('abstractGameState')!='Final' or g.get('gameType')!='R' or g.get('resumeDate'):continue
                for side,other in [('home','away'),('away','home')]:
                    runs=g['teams'][side].get('score');against=g['teams'][other].get('score')
                    if runs is None or against is None or runs==against:continue
                    history.setdefault((int(g['season']),g['teams'][side]['team']['id']),[]).append({'date':g['gameDate'],'for':runs,'against':against,'diff':runs-against,'win':int(runs>against),'game_id':g['gamePk']})
    for h in history.values():h.sort(key=lambda x:(x['date'],x['game_id']))
    for r in usable:
        schedule=schedules_by_year[r['season']]
        g=next(g for d in schedule['dates'] for g in d['games'] if g['gamePk']==r['game_id'])
        cutoff=pd.to_datetime(r['timecode'],format='%Y%m%d_%H%M%S',utc=True)-pd.Timedelta(hours=48)
        sides={}
        for side in ['home','away']:
            h=[x for x in history.get((r['season'],g['teams'][side]['team']['id']),[]) if pd.Timestamp(x['date'])<cutoff]
            def mean(key,n=None):
                a=h[-n:] if n else h
                return sum(x[key] for x in a)/len(a) if a else (.5 if key=='win' else 0.)
            sides[side]=[len(h),mean('win'),mean('for'),mean('against'),mean('diff'),mean('win',5),mean('win',10),mean('diff',5)]
        r['features'][:8]=[h-a for h,a in zip(sides['home'],sides['away'])]
        r['team_sides']=sides
    
    df=pd.DataFrame(usable).sort_values(['start_time','game_id']).drop_duplicates('game_id');train=df[df.season<2026];test=df[df.season==2026]
    if len(train)<200 or len(test)<50:raise ValueError(f'Insufficient pilot coverage: train={len(train)}, test={len(test)}')
    def fit(frame,width,strength):
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=strength,max_iter=3000,random_state=42))
        model.fit(np.asarray(frame.features.tolist(), dtype=float)[:,:width],frame.home_win)
        return model
    development_scores=[]
    selected_width,selected_strength=11,1.
    if expanded:
        development_train=df[df.season==2024];development_test=df[df.season==2025]
        if len(development_train)<100 or len(development_test)<100:
            raise ValueError('Need at least 100 usable games in each development season')
        for width in [11,15]:
            for strength in [.1,1.]:
                candidate=fit(development_train,width,strength)
                probability=candidate.predict_proba(np.asarray(development_test.features.tolist(),dtype=float)[:,:width])[:,1]
                development_scores.append({'Feature count':width,'C':strength,'Development games':len(development_test),'Log loss':float(log_loss(development_test.home_win,probability,labels=[0,1]))})
        chosen=min(development_scores,key=lambda row:row['Log loss'])
        selected_width,selected_strength=chosen['Feature count'],chosen['C']
    X=np.asarray(train.features.tolist(),dtype=float);T=np.asarray(test.features.tolist(),dtype=float);scores=[];pred={}
    candidate_name='Development-selected player candidate' if expanded else 'Team baseline plus archived pitchers and lineups'
    for name,width,strength in [('Historical team baseline',8,1.),(candidate_name,selected_width,selected_strength)]:
        model=fit(train,width,strength)
        p=model.predict_proba(T[:,:width])[:,1];correct=(p>=.5).astype(int)==test.home_win.to_numpy();pred[name]={'probability':p.tolist(),'correct':correct.astype(int).tolist()}
        scores.append({'Model':name,'Training games':len(train),'Test games':len(test),'Accuracy':float(correct.mean()),'AUC':float(roc_auc_score(test.home_win,p)),'Brier':float(brier_score_loss(test.home_win,p)),'Log loss':float(log_loss(test.home_win,p,labels=[0,1]))})
    left,right=list(pred);difference=np.asarray(pred[right]['correct'])-np.asarray(pred[left]['correct']);daily=pd.DataFrame({'day':test.start_time.str[:10].to_numpy(),'diff':difference}).groupby('day')['diff'].agg(['sum','count']);rng=np.random.default_rng(42);boot=[]
    for i in range(2000):
        s=daily.iloc[rng.integers(0,len(daily),len(daily))];boot.append(float(s['sum'].sum()/s['count'].sum()))
    report={'requested_games':len(rows),'usable_games':len(df),'counts_by_season':{str(k):int(v) for k,v in df.groupby('season').size().items()},'scores':scores,'accuracy_change':float(difference.mean()),'accuracy_change_95_interval':np.quantile(boot,[.025,.975]).tolist(),'test_dates':sorted(daily.index.tolist()),'excluded_reasons':pd.Series([r['error'] for r in rows if 'error'in r]).value_counts().to_dict(),'scope':'Pilot using a conservative 48-hour team-history buffer and archived liveData pitcher and lineup season rates added to prior-date team features. Weather, stadium updates, injuries, and workload are not validated in this pilot.','limitations':['Chronological seasonal split with a short June evaluation window; not full-season validation.','Team-history features admit final scores only from games starting over 48 hours before the archived capture. Known resumeDate games are excluded; exact historical result-availability timestamps remain unverified.','Hydrated gameData fields were excluded after a known contamination was observed.','Archived liveData timestamps and no-gameplay state were checked; statistical corrections or later reconstruction by the provider cannot be ruled out.','Models are research experiments. Live model is unchanged.']}
    (root/'mlb_matchup_training_rows.json').write_text(json.dumps(df.to_dict('records')))
    report['cohort']=cohort if expanded else 'june_pilot'
    report['protocol_version']=2 if expanded else 1
    report['development_scores']=development_scores
    report['selected_candidate']={'feature_count':selected_width,'C':selected_strength,'selection_season':2025 if expanded else None}
    report['scope']='Archived pregame player-rate comparison with a 48-hour team-history buffer. Weather, stadium updates, injuries, and workload are not validated.'
    if expanded:
        report['limitations'][0]=('Evenly spaced May–September samples.' if cohort == 'seasonwide' else 'Cached June pilot for code verification only.') + '  Candidate selection uses 2024 training and 2025 validation, then refits on both seasons for 2026 evaluation. Some 2026 games were inspected in earlier experiments; this is not an untouched holdout.'
    feature_names=['games_played_diff','win_pct_diff','avg_runs_for_diff','avg_runs_against_diff','avg_run_diff_diff','recent_5_win_pct_diff','recent_10_win_pct_diff','recent_5_run_diff_diff','starter_era_diff','starter_whip_diff','lineup_ops_diff','starter_k9_diff','starter_bb9_diff','lineup_obp_diff','lineup_slg_diff']
    report['feature_coverage']=[{'Feature':feature_names[i],'Games with values':int(np.isfinite(np.asarray(df.features.tolist(),dtype=float)[:,i]).sum())} for i in range(len(df.iloc[0].features))]
    (root/'mlb_accuracy_pilot_report.json').write_text(json.dumps(report,indent=2));out=[]
    for j,r in enumerate(test.to_dict('records')):
        out.append({'game_id':r['game_id'],'start_time':r['start_time'],'pregame_timecode':r['timecode'],'actual_home_win':r['home_win'],'baseline_probability':pred[left]['probability'][j],'player_probability':pred[right]['probability'][j]})
    pd.DataFrame(out).to_csv(root/'mlb_accuracy_pilot_predictions.csv',index=False)
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(run_historical_accuracy_test(), indent=2))
