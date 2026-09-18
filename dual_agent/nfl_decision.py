import pandas as pd


# Minimum sample size before we trust a confidence band
MIN_BAND_SAMPLE = 20

# Decision thresholds
BET_MIN_CONFIDENCE = 0.65
LEAN_MIN_CONFIDENCE = 0.60

# Historical accuracy required for each decision
BET_MIN_HISTORICAL_ACCURACY = 0.70
LEAN_MIN_HISTORICAL_ACCURACY = 0.65


def build_nfl_confidence_profile(predictions):
    """
    Build historical confidence-band performance from
    walk-forward predictions.

    This gives the decision engine evidence about how
    reliable different confidence levels have actually been.
    """

    if predictions is None or predictions.empty:
        return pd.DataFrame()

    df = predictions.copy()

    if "home_win_probability" not in df.columns:
        raise ValueError(
            "Predictions must contain 'home_win_probability'."
        )

    if "correct" not in df.columns:
        raise ValueError(
            "Predictions must contain 'correct'."
        )

    # Confidence in whichever team the model selected
    df["confidence"] = df["home_win_probability"].where(
        df["home_win_probability"] >= 0.50,
        1.0 - df["home_win_probability"],
    )

    bins = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.01,
    ]

    labels = [
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80-85%",
        "85-90%",
        "90-95%",
        "95-100%",
    ]

    df["confidence_band"] = pd.cut(
        df["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    profile = (
        df.groupby(
            "confidence_band",
            observed=False,
        )
        .agg(
            predictions=("correct", "size"),
            correct=("correct", "sum"),
            average_confidence=("confidence", "mean"),
            actual_accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    profile["sample_reliable"] = (
        profile["predictions"] >= MIN_BAND_SAMPLE
    )

    profile["calibration_gap"] = (
        profile["actual_accuracy"]
        - profile["average_confidence"]
    )

    return profile


def get_nfl_decision(
    home_team,
    away_team,
    home_win_probability,
    confidence_profile,
):
    """
    Convert an NFL model probability into a controlled
    BET / LEAN / PASS decision.

    This function does NOT consider sportsbook odds yet.
    That comes in the next layer.
    """

    probability = float(home_win_probability)

    if probability >= 0.50:
        predicted_team = home_team
        confidence = probability
    else:
        predicted_team = away_team
        confidence = 1.0 - probability

    # Find historical confidence band
    matching_band = None

    for _, row in confidence_profile.iterrows():

        band = str(row["confidence_band"])

        try:
            low = float(
                band.split("-")[0]
            ) / 100.0

            high = float(
                band.split("-")[1].replace("%", "")
            ) / 100.0

        except Exception:
            continue

        if low <= confidence < high:
            matching_band = row
            break

    # If we have no trustworthy historical band,
    # automatically stay conservative.
    if matching_band is None:
        return {
            "decision": "PASS",
            "predicted_team": predicted_team,
            "confidence": confidence,
            "historical_accuracy": None,
            "historical_sample": 0,
            "reason": "No historical confidence-band evidence.",
        }

    historical_accuracy = matching_band["actual_accuracy"]
    historical_sample = int(matching_band["predictions"])
    sample_reliable = bool(matching_band["sample_reliable"])

    if not sample_reliable:
        decision = "PASS"
        reason = (
            "Confidence band does not yet have enough "
            "historical predictions."
        )

    elif (
        confidence >= BET_MIN_CONFIDENCE
        and historical_accuracy >= BET_MIN_HISTORICAL_ACCURACY
    ):
        decision = "BET"
        reason = (
            "Model confidence and historical band "
            "performance both passed BET thresholds."
        )

    elif (
        confidence >= LEAN_MIN_CONFIDENCE
        and historical_accuracy >= LEAN_MIN_HISTORICAL_ACCURACY
    ):
        decision = "LEAN"
        reason = (
            "Prediction has useful evidence but does not "
            "meet the full BET threshold."
        )

    else:
        decision = "PASS"
        reason = (
            "Prediction did not satisfy the required "
            "confidence and historical-performance gates."
        )

    return {
        "decision": decision,
        "predicted_team": predicted_team,
        "confidence": confidence,
        "historical_accuracy": historical_accuracy,
        "historical_sample": historical_sample,
        "reason": reason,
    }
