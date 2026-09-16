from datetime import datetime,timedelta,timezone
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from .config import secret,MODEL_DIR
from .db import log_prediction
MODEL_PATH=MODEL_DIR/"stock_model.joblib"
FEATURES=["ret_1","ret_2","ret_5","ret_10","ret_20","vol_5","vol_20","volume_z20","range_pct","ma_gap_5","ma_gap_20"]
def client():
    k,s=secret("ALPACA_API_KEY"),secret("ALPACA_API_SECRET")
    if not k or not s: raise RuntimeError("Add your Alpaca keys in Streamlit Secrets.")
    return StockHistoricalDataClient(k,s)
def bars(symbols,years=6):
    end=datetime.now(timezone.utc); start=end-timedelta(days=int(years*365.25))
    req=StockBarsRequest(symbol_or_symbols=symbols,timeframe=TimeFrame.Day,start=start,end=end,feed=DataFeed.IEX)
    d=client().get_stock_bars(req).df.reset_index(); d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
    return d.sort_values(["symbol","timestamp"])
def feat(g,h):
    g=g.copy().sort_values("timestamp"); c=g.close
    for n in [1,2,5,10,20]: g[f"ret_{n}"]=c.pct_change(n)
    g["vol_5"]=g.ret_1.rolling(5).std(); g["vol_20"]=g.ret_1.rolling(20).std()
    vm=g.volume.rolling(20).mean(); vs=g.volume.rolling(20).std()
    g["volume_z20"]=(g.volume-vm)/vs.replace(0,np.nan); g["range_pct"]=(g.high-g.low)/c
    g["ma_gap_5"]=c/c.rolling(5).mean()-1; g["ma_gap_20"]=c/c.rolling(20).mean()-1
    future=c.shift(-h); g["target"]=np.where(future.notna(),(future>c).astype(int),np.nan); return g
def dataset(d,h): return pd.concat([feat(g,h) for _,g in d.groupby("symbol",sort=False)],ignore_index=True)
def train(symbols,h=5):
    d=dataset(bars(symbols),h).dropna(subset=FEATURES+["target"]).sort_values("timestamp")
    if len(d)<300: raise RuntimeError("Not enough historical observations.")
    cut=int(len(d)*.8); tr,te=d.iloc[:cut],d.iloc[cut:]
    m=HistGradientBoostingClassifier(learning_rate=.05,max_iter=250,max_leaf_nodes=15,l2_regularization=1,random_state=42)
    m.fit(tr[FEATURES],tr.target.astype(int)); p=m.predict_proba(te[FEATURES])[:,1]; y=te.target.astype(int)
    metrics={"Train rows":len(tr),"Test rows":len(te),"Brier score":float(brier_score_loss(y,p)),"Log loss":float(log_loss(y,p)),"AUC":float(roc_auc_score(y,p)) if y.nunique()>1 else None}
    joblib.dump({"model":m,"horizon":h,"metrics":metrics},MODEL_PATH); return metrics
def scan(symbols):
    if not MODEL_PATH.exists(): raise RuntimeError("Train the stock model first.")
    b=joblib.load(MODEL_PATH); h=b["horizon"]; d=dataset(bars(symbols,1),h); out=[]
    for sym,g in d.groupby("symbol"):
        x=g.dropna(subset=FEATURES).sort_values("timestamp").tail(1)
        if x.empty: continue
        p=float(b["model"].predict_proba(x[FEATURES])[:,1][0]); r=x.iloc[0]
        log_prediction("stock",sym,p,f"{h} trading days",metadata={"close":float(r.close)})
        out.append({"Symbol":sym,"Close":float(r.close),"Probability higher":p,"Signal":"Bullish" if p>=.5 else "Bearish","Strength":abs(p-.5)*2})
    return pd.DataFrame(out).sort_values("Strength",ascending=False),b["metrics"]
