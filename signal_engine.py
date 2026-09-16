
import re
import pandas as pd
from .validated_model import VALIDATED_STOCK_MODEL

GATES = {"min_probability": 0.60, "min_edge_vs_base": 0.055}

def clean_symbols(symbols):
    if symbols is None: return []
    if isinstance(symbols, str):
        symbols = re.split(r"[\s,]+", symbols)
    out=[]; seen=set()
    for x in symbols:
        x=re.sub(r"\s+","",str(x)).strip().upper()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def stock_decision(df):
    m=VALIDATED_STOCK_MODEL
    if not m.get("validated") or df is None or df.empty:
        return {"status":"NO QUALIFYING TRADE","reason":"No usable validated signal."}
    d=df.copy()
    # V4 research/latest_scan convention uses P, Symbol, Close.
    if "P" not in d.columns:
        raise ValueError("Scanner output is missing probability column P.")
    d["P"]=pd.to_numeric(d["P"],errors="coerce")
    d["edge"]=d["P"]-float(m["base_rate"])
    ranked=d.sort_values(["P","edge"],ascending=False).copy()
    q=ranked[(ranked["P"]>=GATES["min_probability"]) &
             (ranked["edge"]>=GATES["min_edge_vs_base"])]
    if q.empty:
        top=ranked.iloc[0].to_dict() if len(ranked) else {}
        return {"status":"NO QUALIFYING TRADE",
                "reason":"No stock clears the validated probability/edge gates.",
                "top":top,"ranked":ranked}
    r=q.iloc[0]
    return {"status":"MAKE THIS TRADE","symbol":r["Symbol"],
            "direction":"LONG / RELATIVE OUTPERFORMANCE",
            "probability":float(r["P"]),"edge":float(r["edge"]),
            "close":float(r["Close"]),"horizon":"5 trading days",
            "target":"Outperform SPY over the next 5 trading days",
            "ranked":ranked}

def sports_decision():
    return {"status":"NO QUALIFYING SPORTS PICK",
            "reason":"NBA model has not yet passed historical validation."}
