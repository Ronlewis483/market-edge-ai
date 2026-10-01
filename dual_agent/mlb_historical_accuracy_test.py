"""Import-safe historical MLB paired accuracy test. No live model promotion."""

def run_historical_accuracy_test(cache_root="mlb_accuracy_results", progress=None):
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
        games=sorted([g for g in allgames if int(g['season'])==year and g['gameDate'][:10]>=f'{year}-06-01'],key=lambda g:(g['gameDate'],g['gamePk']))[:150]
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
                sidevals[side]=[era,whip,sum(ops)/len(ops)];ids[side]={'pitcher_id':pid,'lineup_ids':order}
            h=g['teams']['home'].get('score');a=g['teams']['away'].get('score')
            if h is None or a is None or h==a:return {'game_id':gid,'error':'Final label invalid'}
            return {'game_id':gid,'season':int(g['season']),'start_time':g['gameDate'],'timecode':code,'home_win':int(h>a),
                'features':baseline[gid]+[h-a for h,a in zip(sidevals['home'],sidevals['away'])], 'recorded_identities':ids,
                'source_url':base+'?timecode='+code}
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
    
    df=pd.DataFrame(usable).sort_values(['start_time','game_id']).drop_duplicates('game_id');train=df[df.season<2026];test=df[df.season==2026]
    if len(train)<200 or len(test)<50:raise ValueError(f'Insufficient pilot coverage: train={len(train)}, test={len(test)}')
    X=np.asarray(train.features.tolist());T=np.asarray(test.features.tolist());scores=[];pred={}
    for name,width in [('Historical team baseline',8),('Team baseline plus archived pitchers and lineups',11)]:
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1.,max_iter=3000,random_state=42));model.fit(X[:,:width],train.home_win)
        p=model.predict_proba(T[:,:width])[:,1];correct=(p>=.5).astype(int)==test.home_win.to_numpy();pred[name]={'probability':p.tolist(),'correct':correct.astype(int).tolist()}
        scores.append({'Model':name,'Training games':len(train),'Test games':len(test),'Accuracy':float(correct.mean()),'AUC':float(roc_auc_score(test.home_win,p)),'Brier':float(brier_score_loss(test.home_win,p)),'Log loss':float(log_loss(test.home_win,p,labels=[0,1]))})
    left,right=list(pred);difference=np.asarray(pred[right]['correct'])-np.asarray(pred[left]['correct']);daily=pd.DataFrame({'day':test.start_time.str[:10].to_numpy(),'diff':difference}).groupby('day')['diff'].agg(['sum','count']);rng=np.random.default_rng(42);boot=[]
    for i in range(2000):
        s=daily.iloc[rng.integers(0,len(daily),len(daily))];boot.append(float(s['sum'].sum()/s['count'].sum()))
    report={'requested_games':len(rows),'usable_games':len(df),'counts_by_season':{str(k):int(v) for k,v in df.groupby('season').size().items()},'scores':scores,'accuracy_change':float(difference.mean()),'accuracy_change_95_interval':np.quantile(boot,[.025,.975]).tolist(),'test_dates':sorted(daily.index.tolist()),'excluded_reasons':pd.Series([r['error'] for r in rows if 'error'in r]).value_counts().to_dict(),'scope':'Pilot using a conservative 48-hour team-history buffer and archived liveData pitcher and lineup season rates added to prior-date team features. Weather, stadium updates, injuries, and workload are not validated in this pilot.','limitations':['Chronological seasonal split with a short June evaluation window; not full-season validation.','Team-history features admit final scores only from games starting over 48 hours before the archived capture. Known resumeDate games are excluded; exact historical result-availability timestamps remain unverified.','Hydrated gameData fields were excluded after a known contamination was observed.','Archived liveData timestamps and no-gameplay state were checked; statistical corrections or later reconstruction by the provider cannot be ruled out.','Models are research experiments. Live model is unchanged.']}
    (root/'mlb_accuracy_pilot_report.json').write_text(json.dumps(report,indent=2));out=[]
    for j,r in enumerate(test.to_dict('records')):
        out.append({'game_id':r['game_id'],'start_time':r['start_time'],'pregame_timecode':r['timecode'],'actual_home_win':r['home_win'],'baseline_probability':pred[left]['probability'][j],'player_probability':pred[right]['probability'][j]})
    pd.DataFrame(out).to_csv(root/'mlb_accuracy_pilot_predictions.csv',index=False)
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(run_historical_accuracy_test(), indent=2))
