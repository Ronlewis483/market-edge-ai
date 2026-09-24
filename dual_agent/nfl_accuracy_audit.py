
"""
MARKET EDGE AI
NFL Prediction Accuracy Audit

Evaluates completed, out-of-sample NFL predictions.

Required input columns:
    probability: Predicted probability of a home win.
    actual: Actual result (1 = home win, 0 = away win).

Optional:
    start_time: Historical game kickoff.
    baseline_probability: Pre-game baseline home-win probability.

IMPORTANT:
Only use predictions generated before their games began.
Never use training-set predictions as validation results.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# -------------------------------------------------
# DATA VALIDATION
# -------------------------------------------------

def prepare_audit_data(predictions):

    if predictions is None:
        raise ValueError(
            "Historical predictions are unavailable."
        )

    df = pd.DataFrame(predictions).copy()

    required = ["probability", "actual"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing audit columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "No historical predictions available."
        )

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="coerce",
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["probability", "actual"]
    ).copy()

    df = df[
        df["probability"].between(0, 1)
        & df["actual"].isin([0, 1])
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid completed predictions found."
        )

    df["actual"] = df["actual"].astype(int)

    # A probability of exactly 50% is treated
    # as a home-team prediction.

    df["predicted_home"] = (
        df["probability"] >= 0.5
    ).astype(int)

    df["confidence"] = np.maximum(
        df["probability"],
        1 - df["probability"],
    )

    df["correct"] = (
        df["predicted_home"] == df["actual"]
    ).astype(int)

    if "start_time" in df.columns:

        df["start_time"] = pd.to_datetime(
            df["start_time"],
            utc=True,
            errors="coerce",
        )

    return df.reset_index(drop=True)


# -------------------------------------------------
# OVERALL MODEL PERFORMANCE
# -------------------------------------------------

def calculate_overall_accuracy(df):

    actual = df["actual"]

    probabilities = df["probability"]

    predicted = df["predicted_home"]

    accuracy = accuracy_score(
        actual,
        predicted,
    )

    brier = brier_score_loss(
        actual,
        probabilities,
    )

    loss = log_loss(
        actual,
        probabilities,
        labels=[0, 1],
    )

    if actual.nunique() == 2:

        auc = roc_auc_score(
            actual,
            probabilities,
        )

    else:

        auc = np.nan

    return {
        "total_predictions": int(len(df)),
        "correct_predictions": int(
            df["correct"].sum()
        ),
        "incorrect_predictions": int(
            len(df) - df["correct"].sum()
        ),
        "accuracy": float(accuracy),
        "brier_score": float(brier),
        "log_loss": float(loss),
        "roc_auc": float(auc),
    }


# -------------------------------------------------
# CONFIDENCE THRESHOLD ANALYSIS
# -------------------------------------------------

def analyze_confidence_thresholds(df):

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    results = []

    total = len(df)

    for threshold in thresholds:

        selected = df[
            df["confidence"] >= threshold
        ]

        count = len(selected)

        if count == 0:

            results.append({
                "threshold": threshold,
                "predictions": 0,
                "correct": 0,
                "accuracy": np.nan,
                "coverage": 0.0,
                "average_confidence": np.nan,
            })

            continue

        results.append({
            "threshold": threshold,
            "predictions": int(count),
            "correct": int(
                selected["correct"].sum()
            ),
            "accuracy": float(
                selected["correct"].mean()
            ),
            "coverage": float(
                count / total
            ),
            "average_confidence": float(
                selected["confidence"].mean()
            ),
        })

    return pd.DataFrame(results)


# -------------------------------------------------
# PROBABILITY CALIBRATION
# -------------------------------------------------

def analyze_calibration(df):

    bins = np.linspace(
        0,
        1,
        11,
    )

    data = df.copy()

    data["probability_band"] = pd.cut(
        data["probability"],
        bins=bins,
        include_lowest=True,
    )

    calibration = (
        data.groupby(
            "probability_band",
            observed=False,
        )
        .agg(
            predictions=("actual", "size"),
            average_probability=(
                "probability",
                "mean",
            ),
            actual_win_rate=(
                "actual",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration["calibration_error"] = (
        calibration["average_probability"]
        - calibration["actual_win_rate"]
    ).abs()

    calibration["probability_band"] = (
        calibration["probability_band"].astype(str)
    )

    return calibration


# -------------------------------------------------
# HISTORICAL PERFORMANCE OVER TIME
# -------------------------------------------------

def analyze_performance_over_time(df):

    if "start_time" not in df.columns:
        return pd.DataFrame()

    data = df.dropna(
        subset=["start_time"]
    ).copy()

    if data.empty:
        return pd.DataFrame()

    data["month"] = (
        data["start_time"]
        .dt.strftime("%Y-%m")
    )

    results = (
        data.groupby("month")
        .agg(
            predictions=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            average_confidence=(
                "confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    return results


# -------------------------------------------------
# BASELINE COMPARISON
# -------------------------------------------------

def analyze_baseline(df):

    if "baseline_probability" not in df.columns:
        return None

    data = df.copy()

    data["baseline_probability"] = (
        pd.to_numeric(
            data["baseline_probability"],
            errors="coerce",
        )
    )

    data = data.dropna(
        subset=["baseline_probability"]
    )

    data = data[
        data["baseline_probability"].between(0, 1)
    ]

    if data.empty:
        return None

    baseline_predictions = (
        data["baseline_probability"] >= 0.5
    ).astype(int)

    return {
        "sample_size": int(len(data)),
        "model_accuracy": float(
            accuracy_score(
                data["actual"],
                data["predicted_home"],
            )
        ),
        "baseline_accuracy": float(
            accuracy_score(
                data["actual"],
                baseline_predictions,
            )
        ),
        "model_brier": float(
            brier_score_loss(
                data["actual"],
                data["probability"],
            )
        ),
        "baseline_brier": float(
            brier_score_loss(
                data["actual"],
                data["baseline_probability"],
            )
        ),
    }


# -------------------------------------------------
# COMPLETE NFL ACCURACY AUDIT
# -------------------------------------------------

def run_nfl_accuracy_audit(
    historical_predictions,
    min_samples=30,
):

    df = prepare_audit_data(
        historical_predictions
    )

    overall = calculate_overall_accuracy(df)

    thresholds = analyze_confidence_thresholds(df)

    calibration = analyze_calibration(df)

    performance = analyze_performance_over_time(df)

    baseline = analyze_baseline(df)

    sufficient_sample = (
        len(df) >= min_samples
    )

    return {
        "success": True,
        "overall": overall,
        "confidence_thresholds": thresholds,
        "calibration": calibration,
        "performance_over_time": performance,
        "baseline_comparison": baseline,
        "sufficient_sample": sufficient_sample,
        "minimum_samples": min_samples,
        "audited_predictions": df,
    }
