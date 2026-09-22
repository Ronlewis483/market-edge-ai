from datetime import datetime,timedelta,timezone
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from .config import secret,MODEL_DIR
from .db import log_prediction

HORIZONS=[1,5,20]
F=["ret_1","ret_2","ret_5","ret_10","ret_20","ret_60","vol_5","vol_20","vol_ratio",
"volume_z20","range_pct","atr14_pct","ma_gap_5","ma_gap_20","ma_gap_50","rsi14","drawdown_20",
"spy_ret_1","spy_ret_5","spy_ret_20","spy_vol_20","qqq_ret_5","qqq_ret_20","rel_spy_5","rel_spy_20"]

def path(h): return MODEL_DIR/f"stock_model_h{h}.joblib"

def client():
    k = secret("ALPACA_API_KEY")
    s = secret("ALPACA_SECRET_KEY")

    if not k or not s:
        raise RuntimeError("Missing Alpaca credentials.")

    return StockHistoricalDataClient(k, s)
def bars(symbols,years=7):
    syms=sorted(set(symbols+["SPY","QQQ"])); end=datetime.now(timezone.utc); start=end-timedelta(days=int(years*365.25))
    q=StockBarsRequest(symbol_or_symbols=syms,timeframe=TimeFrame.Day,start=start,end=end,feed=DataFeed.IEX)
    d=client().get_stock_bars(q).df.reset_index(); d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
    return d.sort_values(["symbol","timestamp"])
def rsi(s,n=14):
    d=s.diff(); u=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=u.ewm(alpha=1/n,adjust=False).mean()/dn.ewm(alpha=1/n,adjust=False).mean().replace(0,np.nan)
    return 100-100/(1+rs)
def features(g,h):
    g=g.copy().sort_values("timestamp"); c=g.close
    for n in [1,2,5,10,20,60]: g[f"ret_{n}"]=c.pct_change(n)
    g["vol_5"]=g.ret_1.rolling(5).std(); g["vol_20"]=g.ret_1.rolling(20).std(); g["vol_ratio"]=g.vol_5/g.vol_20.replace(0,np.nan)
    g["volume_z20"]=(g.volume-g.volume.rolling(20).mean())/g.volume.rolling(20).std().replace(0,np.nan)
    prev=c.shift(); tr=pd.concat([g.high-g.low,(g.high-prev).abs(),(g.low-prev).abs()],axis=1).max(axis=1)
    g["atr14_pct"]=tr.rolling(14).mean()/c; g["range_pct"]=(g.high-g.low)/c
    for n in [5,20,50]: g[f"ma_gap_{n}"]=c/c.rolling(n).mean()-1
    g["rsi14"]=rsi(c); g["drawdown_20"]=c/c.rolling(20).max()-1
    fut=c.shift(-h); g["target"]=np.where(fut.notna(),(fut>c).astype(int),np.nan)
    return g
def dataset(d,h):
    x=pd.concat([features(g,h) for _,g in d.groupby("symbol",sort=False)],ignore_index=True)
    spy=x[x.symbol=="SPY"][["timestamp","ret_1","ret_5","ret_20","vol_20"]].rename(columns={"ret_1":"spy_ret_1","ret_5":"spy_ret_5","ret_20":"spy_ret_20","vol_20":"spy_vol_20"})
    q=x[x.symbol=="QQQ"][["timestamp","ret_5","ret_20"]].rename(columns={"ret_5":"qqq_ret_5","ret_20":"qqq_ret_20"})
    x=x.merge(spy,on="timestamp",how="left").merge(q,on="timestamp",how="left")
    x["rel_spy_5"]=x.ret_5-x.spy_ret_5; x["rel_spy_20"]=x.ret_20-x.spy_ret_20
    return x

def split_dates(d, h=1):
    dates = np.array(
        sorted(d.timestamp.dt.normalize().unique())
    )

    if len(dates) < 100:
        raise ValueError(
            "Insufficient historical dates for validation."
        )

    a_idx = int(len(dates) * 0.70)
    b_idx = int(len(dates) * 0.85)

    a = dates[a_idx]
    b = dates[b_idx]

    # Exclude observations whose future outcomes
    # overlap the following evaluation period.
    train_end = dates[max(0, a_idx - h)]
    cal_end = dates[max(a_idx, b_idx - h)]

    normalized = d.timestamp.dt.normalize()

    train = d[normalized < train_end].copy()

    calibration = d[
        (normalized >= a) &
        (normalized < cal_end)
    ].copy()

    test = d[normalized >= b].copy()

    if (
        train.empty or
        calibration.empty or
        test.empty
    ):
        raise ValueError(
            "Insufficient data after applying "
            "the prediction-horizon separation."
        )

    return train, calibration, test

def train(symbols, h):

    # Download historical market data.
    d = dataset(bars(symbols), h)

    d = (
        d[d.symbol.isin(symbols)]
        .dropna(subset=F + ["target"])
        .sort_values("timestamp")
    )

    # Separate historical training, calibration,
    # and testing periods.
    tr, cal, te = split_dates(d, h)

    # Train the existing prediction model.
    m = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=350,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=2,
        random_state=42,
    )

    m.fit(tr[F], tr.target.astype(int))

    # Calibrate probability estimates.
    pc = m.predict_proba(cal[F])[:, 1]

    iso = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.02,
        y_max=0.98,
    )

    iso.fit(pc, cal.target.astype(int))

    # Evaluate on unseen historical test data.
    raw = m.predict_proba(te[F])[:, 1]

    p = iso.predict(raw)

    y = te.target.astype(int)

    base = float(tr.target.mean())

    bp = np.full(len(y), base)

    # Convert predicted probabilities into
    # directional predictions.
    predicted_direction = (p >= 0.50).astype(int)

    actual_direction = y.to_numpy()

    # Directional accuracy.
    directional_accuracy = float(
        np.mean(
            predicted_direction == actual_direction
        )
    )

    # Precision of bullish predictions.
    bullish_predictions = predicted_direction == 1

    if bullish_predictions.sum() > 0:

        bullish_precision = float(
            np.mean(
                actual_direction[bullish_predictions] == 1
            )
        )

    else:
        bullish_precision = None

    # Precision of bearish predictions.
    bearish_predictions = predicted_direction == 0

    if bearish_predictions.sum() > 0:

        bearish_precision = float(
            np.mean(
                actual_direction[bearish_predictions] == 0
            )
        )

    else:
        bearish_precision = None

    # Historical majority-class baseline.
    test_positive_rate = float(y.mean())

    majority_baseline_accuracy = max(
        test_positive_rate,
        1.0 - test_positive_rate,
    )

    # Probability validation.
    baseline_brier = float(
        brier_score_loss(y, bp)
    )

    model_brier = float(
        brier_score_loss(y, p)
    )

    baseline_log_loss = float(
        log_loss(y, bp, labels=[0, 1])
    )

    model_log_loss = float(
        log_loss(y, p, labels=[0, 1])
    )

    auc = (
        float(roc_auc_score(y, p))
        if y.nunique() > 1
        else None
    )


    # ==========================================
    # MARKET EDGE AI - PREDICTION DIAGNOSTICS
    # ==========================================

    # Analyze raw and calibrated probabilities.
    raw_probability_stats = {
        "min": float(np.min(raw)),
        "max": float(np.max(raw)),
        "mean": float(np.mean(raw)),
        "std": float(np.std(raw)),
    }

    calibrated_probability_stats = {
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "mean": float(np.mean(p)),
        "std": float(np.std(p)),
    }

    # Count bullish and bearish predictions.
    bullish_count = int(np.sum(predicted_direction == 1))
    bearish_count = int(np.sum(predicted_direction == 0))

    # Evaluate alternative probability thresholds.
    threshold_results = {}

    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:

        threshold_predictions = (
            p >= threshold
        ).astype(int)

        threshold_accuracy = float(
            np.mean(
                threshold_predictions == actual_direction
            )
        )

        threshold_results[str(threshold)] = {
            "accuracy": threshold_accuracy,
            "bullish_predictions": int(
                np.sum(threshold_predictions == 1)
            ),
            "bearish_predictions": int(
                np.sum(threshold_predictions == 0)
            ),
        }

    # Analyze predictions farther from 50%.
    confidence_results = {}

    for confidence in [0.55, 0.60, 0.65, 0.70]:

        confident_bullish = p >= confidence
        confident_bearish = p <= (1.0 - confidence)

        selected = confident_bullish | confident_bearish

        selected_count = int(np.sum(selected))

        if selected_count > 0:

            selected_predictions = (
                p[selected] >= 0.50
            ).astype(int)

            selected_actual = actual_direction[selected]

            selected_accuracy = float(
                np.mean(
                    selected_predictions == selected_actual
                )
            )

        else:
            selected_accuracy = None

        confidence_results[str(confidence)] = {
            "observations": selected_count,
            "coverage": float(
                selected_count / len(p)
            ),
            "directional_accuracy": selected_accuracy,
        }

    # Preserve diagnostic results.
    prediction_diagnostics = {
        "raw_probability_stats": raw_probability_stats,
        "calibrated_probability_stats":
            calibrated_probability_stats,
        "bullish_predictions": bullish_count,
        "bearish_predictions": bearish_count,
        "threshold_results": threshold_results,
        "confidence_results": confidence_results,
    }
    
    # Save validation results.
    met = {
        "horizon": h,

        "train_rows": len(tr),
        "calibration_rows": len(cal),
        "test_rows": len(te),

        "base_rate": base,

        "baseline_brier": baseline_brier,
        "model_brier": model_brier,

        "baseline_log_loss": baseline_log_loss,
        "model_log_loss": model_log_loss,

        "auc": auc,

        "directional_accuracy": directional_accuracy,

        "bullish_precision": bullish_precision,

        "bearish_precision": bearish_precision,

        "majority_baseline_accuracy":
            majority_baseline_accuracy,

                "test_positive_rate": test_positive_rate,

        "prediction_diagnostics": prediction_diagnostics,
    }

    # Require improvement over probability baselines
    # and the historical majority-class benchmark.
    met["passes_benchmarks"] = bool(
        model_brier < baseline_brier
        and model_log_loss < baseline_log_loss
        and auc is not None
        and auc > 0.50
        and directional_accuracy
        > majority_baseline_accuracy
    )

    # Preserve compatibility with the existing
    # Research Lab and stock scanner.
    joblib.dump(
        {
            "model": m,
            "cal": iso,
            "base": base,
            "metrics": met,
        },
        path(h),
    )

    return met
def train_all(symbols): return {str(h):train(symbols,h) for h in HORIZONS}
def scan(symbols):
    rawdata=bars(symbols,1); rows=[]
    for h in HORIZONS:
        if not path(h).exists(): continue
        b=joblib.load(path(h)); d=dataset(rawdata,h)
        for sym,g in d[d.symbol.isin(symbols)].groupby("symbol"):
            x=g.dropna(subset=F).sort_values("timestamp").tail(1)
            if x.empty: continue
            raw=float(b["model"].predict_proba(x[F])[:,1][0]); p=float(b["cal"].predict([raw])[0]); base=float(b["base"]); edge=p-base; r=x.iloc[0]
            rows.append({"Symbol":sym,"Horizon":f"{h}d","Close":float(r.close),"Raw P":raw,"Calibrated P":p,"Historical base":base,"Model vs base":edge,"Direction":"Bullish" if edge>=0 else "Bearish"})
            log_prediction("stock_v3",sym,p,f"{h} trading days",base,edge,{"raw":raw,"close":float(r.close)})
    if not rows: raise RuntimeError("Train all horizons first.")
        # V3 deployment refresh
    return pd.DataFrame(rows)
def metrics_all():
    return {str(h):joblib.load(path(h))["metrics"] for h in HORIZONS if path(h).exists()}
