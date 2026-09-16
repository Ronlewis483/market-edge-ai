from datetime import datetime,timedelta,timezone
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from .config import secret

H=5
FEATURE_GROUPS={
"momentum":["ret_1","ret_5","ret_20","ret_60","ma_gap_20","ma_gap_50","rsi14"],
"volatility":["vol_5","vol_20","vol_ratio","atr14_pct","range_pct","drawdown_20"],
"volume":["volume_z20"],
"market":["spy_ret_1","spy_ret_5","spy_ret_20","spy_vol_20","qqq_ret_5","qqq_ret_20"],
"relative":["rel_spy_5","rel_spy_20"]
}
ALL=sum(FEATURE_GROUPS.values(),[])

DEFAULT_UNIVERSE="""AAPL,MSFT,NVDA,AMZN,META,GOOGL,GOOG,TSLA,AVGO,AMD,ORCL,CRM,ADBE,NFLX,INTC,QCOM,TXN,MU,AMAT,CSCO,IBM,
JPM,BAC,WFC,C,GS,MS,V,MA,AXP,BLK,SCHW,
LLY,JNJ,UNH,ABBV,MRK,PFE,TMO,ABT,AMGN,GILD,
WMT,COST,HD,LOW,TGT,NKE,MCD,SBUX,KO,PEP,PG,
XOM,CVX,COP,SLB,EOG,
CAT,DE,GE,BA,HON,UPS,FDX,RTX,LMT,
NEE,DUK,SO,
LIN,APD,FCX,NEM,
DIS,CMCSA,T,VZ,
SPY,QQQ,IWM,DIA,XLF,XLK,XLE,XLV,XLI,XLY,XLP,XLU""".replace("\\n","")

def client():
    k,s=secret("ALPACA_API_KEY"),secret("ALPACA_API_SECRET")
    if not k or not s: raise RuntimeError("Missing Alpaca credentials.")
    return StockHistoricalDataClient(k,s)

def bars(symbols,years=8):
    syms=sorted(set(symbols+["SPY","QQQ"]))
    end=datetime.now(timezone.utc); start=end-timedelta(days=int(years*365.25))
    q=StockBarsRequest(symbol_or_symbols=syms,timeframe=TimeFrame.Day,start=start,end=end,feed=DataFeed.IEX)
    d=client().get_stock_bars(q).df.reset_index()
    d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
    return d.sort_values(["symbol","timestamp"])

def rsi(s,n=14):
    d=s.diff(); u=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=u.ewm(alpha=1/n,adjust=False).mean()/dn.ewm(alpha=1/n,adjust=False).mean().replace(0,np.nan)
    return 100-100/(1+rs)

def base_features(g):
    g=g.copy().sort_values("timestamp"); c=g.close
    for n in [1,5,20,60]: g[f"ret_{n}"]=c.pct_change(n)
    g["vol_5"]=g.ret_1.rolling(5).std(); g["vol_20"]=g.ret_1.rolling(20).std()
    g["vol_ratio"]=g.vol_5/g.vol_20.replace(0,np.nan)
    g["volume_z20"]=(g.volume-g.volume.rolling(20).mean())/g.volume.rolling(20).std().replace(0,np.nan)
    prev=c.shift()
    tr=pd.concat([g.high-g.low,(g.high-prev).abs(),(g.low-prev).abs()],axis=1).max(axis=1)
    g["atr14_pct"]=tr.rolling(14).mean()/c; g["range_pct"]=(g.high-g.low)/c
    for n in [20,50]: g[f"ma_gap_{n}"]=c/c.rolling(n).mean()-1
    g["rsi14"]=rsi(c); g["drawdown_20"]=c/c.rolling(20).max()-1
    g["future_ret_5"]=c.shift(-H)/c-1
    return g

def make_dataset(raw):
    x=pd.concat([base_features(g) for _,g in raw.groupby("symbol",sort=False)],ignore_index=True)
    spy=x[x.symbol=="SPY"][["timestamp","ret_1","ret_5","ret_20","vol_20","future_ret_5"]].rename(columns={
      "ret_1":"spy_ret_1","ret_5":"spy_ret_5","ret_20":"spy_ret_20","vol_20":"spy_vol_20","future_ret_5":"spy_future_ret_5"})
    q=x[x.symbol=="QQQ"][["timestamp","ret_5","ret_20"]].rename(columns={"ret_5":"qqq_ret_5","ret_20":"qqq_ret_20"})
    x=x.merge(spy,on="timestamp",how="left").merge(q,on="timestamp",how="left")
    x["rel_spy_5"]=x.ret_5-x.spy_ret_5; x["rel_spy_20"]=x.ret_20-x.spy_ret_20
    x["target_up"]=(x.future_ret_5>0).astype(float)
    x["target_beat_spy"]=(x.future_ret_5>x.spy_future_ret_5).astype(float)
    x["target_plus1"]=(x.future_ret_5>.01).astype(float)
    invalid=x.future_ret_5.isna()
    x.loc[invalid,["target_up","target_beat_spy","target_plus1"]]=np.nan
    return x

def model():
    return HistGradientBoostingClassifier(learning_rate=.04,max_iter=220,max_leaf_nodes=15,min_samples_leaf=60,l2_regularization=3,random_state=42)

def walk_forward(d,features,target,min_train_days=750,test_days=126):
    d=d.dropna(subset=features+[target]).sort_values("timestamp")
    dates=np.array(sorted(d.timestamp.dt.normalize().unique()))
    preds=[]
    start=min_train_days
    while start < len(dates)-20:
        end=min(start+test_days,len(dates))
        train_dates=dates[:start]; test_dates=dates[start:end]
        tr=d[d.timestamp.dt.normalize().isin(train_dates)]
        te=d[d.timestamp.dt.normalize().isin(test_dates)]
        if len(tr)<1000 or len(te)<50: break
        m=model(); m.fit(tr[features],tr[target].astype(int))
        p=m.predict_proba(te[features])[:,1]
        tmp=te[["timestamp","symbol",target]].copy(); tmp["p"]=p; preds.append(tmp)
        start=end
    if not preds: raise RuntimeError("Not enough data for walk-forward validation.")
    o=pd.concat(preds,ignore_index=True); y=o[target].astype(int); p=o.p.values
    base=float(d[d.timestamp < o.timestamp.min()][target].mean())
    bp=np.full(len(y),base)
    return {
      "rows":len(o),"folds":int(np.ceil(len(np.unique(o.timestamp.dt.normalize()))/test_days)),
      "base_rate":base,"auc":float(roc_auc_score(y,p)) if y.nunique()>1 else None,
      "model_brier":float(brier_score_loss(y,p)),"baseline_brier":float(brier_score_loss(y,bp)),
      "model_log_loss":float(log_loss(y,p)),"baseline_log_loss":float(log_loss(y,bp)),
      "high_conf_accuracy":float((y[p>=.60]==1).mean()) if (p>=.60).sum() else None,
      "high_conf_n":int((p>=.60).sum())
    }

def run_research(universe):
    raw=bars(universe); d=make_dataset(raw)
    configs={
      "momentum":FEATURE_GROUPS["momentum"],
      "momentum+market":FEATURE_GROUPS["momentum"]+FEATURE_GROUPS["market"],
      "momentum+relative+market":FEATURE_GROUPS["momentum"]+FEATURE_GROUPS["relative"]+FEATURE_GROUPS["market"],
      "all_features":ALL
    }
    targets={"Up in 5d":"target_up","Beat SPY in 5d":"target_beat_spy","> +1% in 5d":"target_plus1"}
    rows=[]
    for tname,target in targets.items():
        for cname,features in configs.items():
            try:
                r=walk_forward(d,features,target); r.update({"Target":tname,"Features":cname})
                r["beats_brier"]=r["model_brier"]<r["baseline_brier"]
                r["beats_logloss"]=r["model_log_loss"]<r["baseline_log_loss"]
                rows.append(r)
            except Exception as e:
                rows.append({"Target":tname,"Features":cname,"error":str(e)})
    return pd.DataFrame(rows)

def latest_scan(train_universe,scan_symbols,target="target_beat_spy",features=None):
    features=features or (FEATURE_GROUPS["momentum"]+FEATURE_GROUPS["relative"]+FEATURE_GROUPS["market"])
    raw=bars(sorted(set(train_universe+scan_symbols))); d=make_dataset(raw)
    train=d[d.symbol.isin(train_universe)].dropna(subset=features+[target]).sort_values("timestamp")
    m=model(); m.fit(train[features],train[target].astype(int))
    rows=[]
    for sym,g in d[d.symbol.isin(scan_symbols)].groupby("symbol"):
        x=g.dropna(subset=features).sort_values("timestamp").tail(1)
        if x.empty: continue
        p=float(m.predict_proba(x[features])[:,1][0]); r=x.iloc[0]
        rows.append({"Symbol":sym,"Close":float(r.close),"P":p,"Target":target})
    return pd.DataFrame(rows).sort_values("P",ascending=False)
