
import pandas as pd

# Promotion gates are intentionally conservative.
# A signal is shown only when the selected research model has demonstrated
# out-of-sample benchmark improvement and the live probability clears the gate.
STOCK_GATES = {
    "min_auc": 0.515,
    "require_brier_win": True,
    "require_logloss_win": True,
    "min_probability": 0.60,
    "min_edge_vs_base": 0.055,
}

def research_passes(row):
    if row is None:
        return False
    auc = float(row.get("auc", 0) or 0)
    brier_ok = bool(row.get("beats_brier", False))
    log_ok = bool(row.get("beats_logloss", False))
    return (
        auc >= STOCK_GATES["min_auc"]
        and (brier_ok if STOCK_GATES["require_brier_win"] else True)
        and (log_ok if STOCK_GATES["require_logloss_win"] else True)
    )

def select_research_model(research_df):
    """Choose only among benchmark-passing 5-day relative-performance models."""
    if research_df is None or len(research_df) == 0:
        return None
    d = research_df.copy()
    d = d[
        (d["Target"] == "Beat SPY in 5d")
        & (d["beats_brier"] == True)
        & (d["beats_logloss"] == True)
        & (d["auc"] >= STOCK_GATES["min_auc"])
    ]
    if d.empty:
        return None
    # Prefer probability quality first, discrimination second.
    d["brier_gain"] = d["baseline_brier"] - d["model_brier"]
    d["logloss_gain"] = d["baseline_log_loss"] - d["model_log_loss"]
    return d.sort_values(["brier_gain","logloss_gain","auc"], ascending=False).iloc[0].to_dict()

def stock_decision(scan_df, research_row):
    if not research_passes(research_row) or scan_df is None or scan_df.empty:
        return {"status":"NO QUALIFYING TRADE","reason":"No validated model/signal clears the research gates."}

    d = scan_df.copy()
    base = float(research_row["base_rate"])
    d["edge"] = d["P"].astype(float) - base
    d = d[
        (d["P"].astype(float) >= STOCK_GATES["min_probability"])
        & (d["edge"] >= STOCK_GATES["min_edge_vs_base"])
    ].sort_values(["edge","P"], ascending=False)

    if d.empty:
        return {"status":"NO QUALIFYING TRADE","reason":"Validated model is active, but no stock clears today's probability and edge thresholds."}

    r = d.iloc[0]
    return {
        "status":"MAKE THIS TRADE",
        "symbol":r["Symbol"],
        "direction":"LONG / RELATIVE OUTPERFORMANCE",
        "probability":float(r["P"]),
        "edge":float(r["edge"]),
        "close":float(r["Close"]),
        "horizon":"5 trading days",
        "target":"Outperform SPY over the next 5 trading days",
        "research_auc":float(research_row["auc"]),
        "feature_set":research_row["Features"],
    }

def sports_decision():
    # Sports remains locked until its historical walk-forward model exists.
    return {
        "status":"NO QUALIFYING SPORTS PICK",
        "reason":"Sports signal engine is locked until the NBA historical model passes its validation gates."
    }
