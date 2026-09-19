import math


def american_odds_to_probability(odds):
    """
    Convert American moneyline odds to implied probability.

    Examples:
        -150 -> 0.6000
        +130 -> 0.4348
    """
    odds = float(odds)

    if odds == 0:
        raise ValueError("American odds cannot be 0.")

    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)

    return 100.0 / (odds + 100.0)


def calculate_no_vig_probabilities(home_odds, away_odds):
    """
    Remove sportsbook vig from a two-way moneyline market.
    """
    home_implied = american_odds_to_probability(home_odds)
    away_implied = american_odds_to_probability(away_odds)

    total_implied = home_implied + away_implied

    if total_implied <= 0:
        raise ValueError("Invalid implied probability total.")

    return {
        "home_implied_probability": home_implied,
        "away_implied_probability": away_implied,
        "home_no_vig_probability": home_implied / total_implied,
        "away_no_vig_probability": away_implied / total_implied,
        "market_hold": total_implied - 1.0,
    }


def calculate_nfl_market_edge(
    home_model_probability,
    home_odds,
    away_odds,
):
    """
    Compare NFL model probabilities against the no-vig market.

    Positive edge means the model assigns a greater probability
    to that outcome than the market does.
    """
    home_model_probability = float(home_model_probability)

    if not 0.0 <= home_model_probability <= 1.0:
        raise ValueError(
            "home_model_probability must be between 0 and 1."
        )

    away_model_probability = 1.0 - home_model_probability

    market = calculate_no_vig_probabilities(
        home_odds,
        away_odds,
    )

    home_edge = (
        home_model_probability
        - market["home_no_vig_probability"]
    )

    away_edge = (
        away_model_probability
        - market["away_no_vig_probability"]
    )

    if home_edge >= away_edge:
        best_side = "HOME"
        best_edge = home_edge
        best_model_probability = home_model_probability
        best_market_probability = market[
            "home_no_vig_probability"
        ]
        best_odds = float(home_odds)
    else:
        best_side = "AWAY"
        best_edge = away_edge
        best_model_probability = away_model_probability
        best_market_probability = market[
            "away_no_vig_probability"
        ]
        best_odds = float(away_odds)

    return {
        **market,
        "home_model_probability": home_model_probability,
        "away_model_probability": away_model_probability,
        "home_edge": home_edge,
        "away_edge": away_edge,
        "best_side": best_side,
        "best_edge": best_edge,
        "best_model_probability": best_model_probability,
        "best_market_probability": best_market_probability,
        "best_odds": best_odds,
    }


def expected_value_per_dollar(model_probability, american_odds):
    """
    Expected profit/loss for each $1 risked.

    Positive value = positive model EV.
    Negative value = negative model EV.
    """
    probability = float(model_probability)
    odds = float(american_odds)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "model_probability must be between 0 and 1."
        )

    if odds == 0:
        raise ValueError("American odds cannot be 0.")

    if odds > 0:
        profit_if_win = odds / 100.0
    else:
        profit_if_win = 100.0 / abs(odds)

    probability_loss = 1.0 - probability

    return (
        probability * profit_if_win
        - probability_loss
    )



def classify_nfl_market_edge(
    model_probability,
    market_probability,
    american_odds,
    historical_accuracy=None,
    historical_sample=None,
):
    """
    Classify an NFL moneyline opportunity using model
    probability, market edge, expected value, and
    historical validation.

    BET requirements:
        Model probability >= 65%
        Model edge >= 3 percentage points
        Positive expected value
        Historical accuracy >= 70%
        Historical sample >= 25 games

    LEAN requirements:
        Model probability >= 60%
        Model edge >= 1 percentage point
        Positive expected value

    Otherwise PASS.
    """

    # -----------------------------------------
    # 1. Convert and validate inputs
    # -----------------------------------------

    model_probability = float(model_probability)
    market_probability = float(market_probability)
    american_odds = float(american_odds)

    if not 0.0 <= model_probability <= 1.0:
        raise ValueError(
            "Model probability must be between 0 and 1."
        )

    if not 0.0 <= market_probability <= 1.0:
        raise ValueError(
            "Market probability must be between 0 and 1."
        )

    if not (
        american_odds <= -100
        or american_odds >= 100
    ):
        raise ValueError(
            "Invalid American moneyline odds."
        )

    # -----------------------------------------
    # 2. Calculate model edge
    # -----------------------------------------

    edge = model_probability - market_probability

    # -----------------------------------------
    # 3. Calculate expected value
    # -----------------------------------------

    ev = expected_value_per_dollar(
        model_probability,
        american_odds,
    )

    # -----------------------------------------
    # 4. Validate historical performance
    # -----------------------------------------

    history_ok = False

    if (
        historical_accuracy is not None
        and historical_sample is not None
    ):
        historical_accuracy = float(
            historical_accuracy
        )

        historical_sample = int(
            historical_sample
        )

        history_ok = (
            historical_accuracy >= 0.70
            and historical_sample >= 25
        )

    # -----------------------------------------
    # 5. Classify opportunity
    # -----------------------------------------

    decision = "PASS"

    reason = "No classification reason provided."

    if (
        model_probability >= 0.65
        and edge >= 0.03
        and ev > 0
        and history_ok
    ):

        decision = "BET"

        reason = (
            "Model confidence, market edge, "
            "expected value, and historical "
            "validation passed BET thresholds."
        )

    elif (
        model_probability >= 0.60
        and edge >= 0.01
        and ev > 0
    ):

        decision = "LEAN"

        reason = (
            "Positive model edge and expected "
            "value exist, but the opportunity "
            "did not pass every BET threshold."
        )

    else:

        decision = "PASS"

        reasons = []

        if model_probability < 0.60:
            reasons.append(
                "Model win probability is below "
                "the 60% minimum for LEAN."
            )

        if edge < 0.01:
            reasons.append(
                "Model edge is below the "
                "1 percentage point minimum."
            )

        if ev <= 0:
            reasons.append(
                "Expected value is not positive."
            )

        if reasons:
            reason = " ".join(reasons)

    # -----------------------------------------
    # 6. Return classification results
    # -----------------------------------------

    return {
        "decision": decision,
        "reason": reason,
        "model_probability": model_probability,
        "market_probability": market_probability,
        "edge": edge,
        "expected_value": ev,
        "ev": ev,
        "american_odds": american_odds,
        "historical_accuracy": historical_accuracy,
        "historical_sample": historical_sample,
        "history_ok": history_ok,
    }
