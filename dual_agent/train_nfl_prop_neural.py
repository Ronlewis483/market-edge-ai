"""Offline reproducible NFL neural training. Never imported by Streamlit.

Usage: python train_nfl_prop_neural.py --data-dir PATH --output-dir dual_agent
Requires numpy, pandas, scipy and scikit-learn. Downloads public nflverse PBP
only when missing. Validation and probability checks use chronological games.
"""
import argparse
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import hashlib
import json
import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from nfl_prop_neural import MARKETS,FEATURES,features,network

SNAPSHOTS=(0,600,1200,1800,2400,3000,3300)
COLS=['game_id','play_id','game_date','week','season_type','qtr','game_seconds_remaining',
      'passer_player_id','passer_player_name','receiver_player_id','receiver_player_name',
      'rusher_player_id','rusher_player_name','passing_yards','receiving_yards','rushing_yards',
      'pass_attempt','complete_pass','rush_attempt','sack','play_type','qb_kneel','qb_spike']


def download(root,year):
    p=root/f'play_by_play_{year}.csv.gz'
    if not p.exists():
        url=f'https://github.com/nflverse/nflverse-data/releases/download/pbp/{p.name}'
        with urllib.request.urlopen(url,timeout=45) as response,p.open('wb') as out:
            while chunk:=response.read(1024*1024):out.write(chunk)
    return p


def events_from_pbp(root,year):
    path=download(root,year)
    df=pd.read_csv(path,usecols=lambda c:c in COLS,low_memory=False)
    df=df.loc[df.season_type.isin(['REG','POST'])].copy()
    df=df.sort_values(['game_date','game_id','play_id'])
    df['elapsed']=(3600-pd.to_numeric(df.game_seconds_remaining,errors='coerce')).clip(lower=0)
    # Next observed play start is a conservative bound on completion time.
    df['end_elapsed']=df.groupby('game_id').elapsed.shift(-1)
    df['end_elapsed']=df.end_elapsed.fillna(3600)
    df=df.loc[df.play_type.ne('no_play')&df.elapsed.notna()].copy()
    pieces=[]
    def role(idcol,mask,fields):
        selected=df.loc[mask&df[idcol].notna()].copy()
        for market,column,transform in fields:
            part=selected[['game_id','game_date','week','end_elapsed',idcol]].rename(columns={idcol:'player_id'})
            values=transform(selected) if transform else pd.to_numeric(selected[column],errors='coerce').fillna(0)
            part=part.assign(market=market,value=np.asarray(values,dtype=float),activity=1)
            pieces.append(part)
    passes=pd.to_numeric(df.pass_attempt,errors='coerce').eq(1)&~pd.to_numeric(df.sack,errors='coerce').eq(1)
    rushes=pd.to_numeric(df.rush_attempt,errors='coerce').eq(1)
    role('passer_player_id',passes,[('Passing yards','passing_yards',None),
         ('Passing completions','complete_pass',None),('Passing attempts',None,lambda d:np.ones(len(d)))])
    role('receiver_player_id',passes,[('Receiving yards','receiving_yards',None),('Receptions','complete_pass',None)])
    role('rusher_player_id',rushes,[('Rushing yards','rushing_yards',None),('Rushing attempts',None,lambda d:np.ones(len(d)))])
    plays=pd.concat(pieces,ignore_index=True)
    plays=plays.groupby(['game_id','game_date','week','player_id','market','end_elapsed'],as_index=False).agg(value=('value','sum'),activity=('activity','sum'))
    plays['season']=year
    return plays


def build_dataset(root):
    cached=root/'neural_snapshots_v2.npz'
    if cached.exists():
        with np.load(cached,allow_pickle=False) as d:return {k:d[k] for k in d.files}
    history=defaultdict(lambda:deque(maxlen=12));records=[]
    for year in range(2020,2026):
        plays=events_from_pbp(root,year)
        for (date,game),game_rows in plays.groupby(['game_date','game_id'],sort=True):
            pending=[]
            groups={key:rows for key,rows in game_rows.groupby(['player_id','market'],sort=False)}
            # A played game with zero targets/carries is still a zero-stat
            # historical appearance. Do not learn only from productive games.
            for pid,market in ((pid,market) for pid in game_rows.player_id.unique() for market in MARKETS):
                rows=groups.get((pid,market),game_rows.iloc[:0])
                rows=rows.sort_values('end_elapsed');total=float(rows.value.sum())
                key=(str(pid),market);prior=list(history[key])
                # Only earlier dates, never a future game or another snapshot.
                vals=np.array([v for day,v in prior if day<str(date)],dtype=float)[-12:]
                if len(vals)>=8 and np.any(vals!=0):
                    for elapsed in SNAPSHOTS:
                        known=rows.loc[rows.end_elapsed<=elapsed]
                        if elapsed and known.empty:continue
                        current=float(known.value.sum()) if elapsed else 0.
                        x,baseline,scale=features(vals,current,market,elapsed)
                        records.append((x,(total-current-baseline)/scale,total,current,baseline,scale,
                                        MARKETS.index(market),elapsed,year,int(game_rows.week.iloc[0]),str(date),str(game)))
                pending.append((key,total))
            for key,total in pending:history[key].append((str(date),total))
        print('built season',year,'cumulative snapshots',len(records),flush=True)
    names=['x','target','actual','current','baseline','scale','market','elapsed','season','week','date','game']
    result={name:np.asarray([row[i] for row in records]) for i,name in enumerate(names)}
    np.savez_compressed(cached,**result)
    return result


def residual_prob(pred,current,scale,residuals,line,count):
    # Continuous evaluation at half-integer lines; avoid artificial pushes.
    points=current+pred+scale*residuals
    if count:points=np.maximum(points,current)
    return (np.sum(points>line)+1)/(len(points)+2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',default='nfl_neural_training_data')
    parser.add_argument('--output-dir',default='dual_agent');args=parser.parse_args()
    root=Path(args.data_dir);root.mkdir(parents=True,exist_ok=True);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda y:download(root,y),range(2020,2026)))
    data=build_dataset(root);x=data['x'].astype(float);y=data['target'].astype(float)
    train=data['season']<=2023
    validation=(data['season']==2024)&(data['week']<=9)
    calibration=(data['season']==2024)&(data['week']>9)
    promotion=(data['season']==2025)&(data['week']<=9)
    test=(data['season']==2025)&(data['week']>9)
    scaler=StandardScaler().fit(x[train]);xt=scaler.transform(x[train]);xv=scaler.transform(x[validation])
    model=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='relu',solver='adam',
        alpha=.02,batch_size=1024,learning_rate_init=.0006,random_state=42,shuffle=True,max_iter=1)
    best=None;best_loss=float('inf');waiting=0;rng=np.random.default_rng(42)
    for epoch in range(60):
        order=rng.permutation(len(xt));model.partial_fit(xt[order],y[train][order])
        loss=float(np.mean((model.predict(xv)-y[validation])**2))
        print('epoch',epoch+1,'validation standardized MSE',round(loss,5),flush=True)
        if loss<best_loss-1e-4:best=copy.deepcopy(model);best_loss=loss;waiting=0
        else:waiting+=1
        if waiting>=8:break
    model=best;prediction=model.predict(scaler.transform(x))
    arrays={'x_mean':scaler.mean_,'x_scale':scaler.scale_}
    for i,(w,b) in enumerate(zip(model.coefs_,model.intercepts_)):arrays['w'+str(i)]=w;arrays['b'+str(i)]=b
    # Residual distributions are never fitted on training or test labels.
    report={'scope':'NFL props; historical PBP reconstruction; no historical sportsbook lines',
      'architecture':[len(FEATURES),64,32,16,1],'snapshots_seconds':list(SNAPSHOTS),
      'splits':{k:{'snapshots':int(mask.sum()),'games':int(len(set(data['game'][mask]))),
                   'start':str(min(data['date'][mask])),'end':str(max(data['date'][mask]))}
                for k,mask in [('train',train),('validation',validation),('calibration',calibration),('promotion',promotion),('test',test)]},'groups':{}}
    promoted={}
    for mi,market in enumerate(MARKETS):
        for phase in ('pregame','live'):
            phase_mask=(data['elapsed']==0) if phase=='pregame' else (data['elapsed']>0)
            cal=calibration&(data['market']==mi)&phase_mask
            hold=test&(data['market']==mi)&phase_mask
            residual=np.sort(y[cal]-prediction[cal]);base_residual=np.sort(y[cal])
            if len(residual)>2000:residual=np.quantile(residual,np.linspace(0,1,2000))
            if len(base_residual)>2000:base_residual=np.quantile(base_residual,np.linspace(0,1,2000))
            arrays['residual_'+str(mi)+'_'+phase]=residual
            count='yards' not in market.lower()
            def evaluate(mask):
                positions=np.where(mask)[0];actual=data['actual'][mask]
                learned=data['current'][mask]+data['baseline'][mask]+data['scale'][mask]*prediction[mask]
                base=data['current'][mask]+data['baseline'][mask]
                if count:learned=np.maximum(learned,data['current'][mask])
                mae=float(np.mean(abs(actual-learned)));base_mae=float(np.mean(abs(actual-base)))
                bs=[];base_bs=[];wins=[]
                # Thresholds depend only on prior/current inputs, never the
                # outcome; these are NOT actual historical sportsbook lines.
                for j in positions:
                    for offset in (-.5,0.,.5):
                        line=math_floor(data['current'][j]+data['baseline'][j]+offset*data['scale'][j])+.5
                        if count and line<data['current'][j]:continue
                        p=residual_prob(data['baseline'][j]+data['scale'][j]*prediction[j],data['current'][j],data['scale'][j],residual,line,count)
                        q=residual_prob(data['baseline'][j],data['current'][j],data['scale'][j],base_residual,line,count)
                        truth=float(data['actual'][j]>line);bs.append((p-truth)**2);base_bs.append((q-truth)**2);wins.append((p>=.5)==bool(truth))
                return {'snapshots':len(positions),'neural_mae':mae,'baseline_mae':base_mae,
                  'constructed_threshold_brier':float(np.mean(bs)),'baseline_threshold_brier':float(np.mean(base_bs)),
                  'constructed_threshold_accuracy':float(np.mean(wins))}
            gate=evaluate(promotion&(data['market']==mi)&phase_mask)
            untouched=evaluate(hold)
            group=market+'|'+phase
            passed=(len(residual)>=100 and gate['snapshots']>=200 and gate['neural_mae']<=gate['baseline_mae'] and gate['constructed_threshold_brier']<=gate['baseline_threshold_brier'])
            promoted[group]=bool(passed)
            report['groups'][group]={'promotion_gate':gate,'untouched_test':untouched,'promoted':bool(passed)}
            print(group,report['groups'][group],flush=True)
    metadata={'schema':1,'model_id':'nfl_neural_v1','features':FEATURES,'markets':MARKETS,
       'training_end':report['splits']['train']['end'],'calibration_end':report['splits']['calibration']['end'],
       'promoted':promoted,'architecture':report['architecture'],'test_end':report['splits']['test']['end']}
    arrays['metadata']=np.asarray(json.dumps(metadata));np.savez_compressed(out/'nfl_prop_neural_weights.npz',**arrays)
    # Exact portable inference parity; no sklearn serialization in production.
    assert np.max(abs(network(x[:100],arrays)-model.predict(scaler.transform(x[:100]))))<1e-8
    report['limitations']=['Reconstructed PBP totals can differ from official box scores on rare lateral/stat-correction plays.','Threshold accuracy uses constructed half-point thresholds, not sportsbook lines or actual bets.','Game snapshots are correlated within games; sample counts are not independent game counts.','No future accuracy or profitability is guaranteed; drift and sportsbook settlement rules remain.','Live inputs omit possession, score effects, injuries, substitutions and an explicit overtime process.']
    report['weights_sha256']=hashlib.sha256((out/'nfl_prop_neural_weights.npz').read_bytes()).hexdigest()
    (out/'nfl_prop_neural_report.json').write_text(json.dumps(report,indent=2))
    print('Saved weights and evaluation report',flush=True)


def math_floor(value):return int(np.floor(value))


if __name__=='__main__':main()
