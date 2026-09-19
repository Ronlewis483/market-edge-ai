import pandas as pd

from dual_agent.nfl_market import (
    calculate_nfl_market_edge,
    classify_nfl_market_edge,
)


def build_live_nfl_opportunities(
    model_predictions,
    best_lines,
):
    """
    Join NFL model predictions to the best available sportsbook
    moneylines and calculate the market edge for every matchup.

    Expected model prediction columns:
        home_team
        away_team
        home_win_probability

    Expected best-line columns:
        home_team
        away_team
        commence_time
        best_home_moneyline
        best_home_sportsbook
        best_away_moneyline
        best_away_sportsbook
    """

    if model_predictions is None or len(model_predictions) == 0:
        return pd.DataFrame()

    if best_lines is None or len(best_lines) == 0:
        return pd.DataFrame()

    predictions = pd.DataFrame(model_predictions).copy()
    lines = pd.DataFrame(best_lines).copy()

    required_prediction_columns = {
        "home_team",
        "away_team",
        "home_win_probability",
    }

    missing = required_prediction_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            "Model predictions are missing required columns: "
            + ", ".join(sorted(missing))
        )

    merged = lines.merge(
        predictions[
            [
                "home_team",
                "away_team",
                "home_win_probability",
            ]
        ],
        on=["home_team", "away_team"],
        how="inner",
    )

    opportunities = []

    for _, game in merged.iterrows():

        market_result = calculate_nfl_market_edge(
            home_model_probability=float(
                game["home_win_probability"]
            ),
            home_odds=
                game["best_home_moneyline"]
            ),
            away_odds=
                game["best_away_moneyline"]
            ),
        )

        classification = classify_nfl_market_edge(
            market_result
        )

        opportunities.append(
            {
                "commence_time": game["commence_time"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],

                "home_win_probability":
                    game["home_win_probability"],

                "best_home_moneyline":
                    game["best_home_moneyline"],

                "best_home_sportsbook":
                    game["best_home_sportsbook"],

                "best_away_moneyline":
                    game["best_away_moneyline"],

                "best_away_sportsbook":
                    game["best_away_sportsbook"],

                "best_side":
                    market_result["best_side"],

                "model_probability":
                    market_result["model_probability"],

                "market_no_vig_probability":
                    market_result["market_no_vig_probability"],

                "model_edge":
                    market_result["model_edge"],

                "expected_value":
                    market_result["expected_value"],

                "decision":
                    classification["decision"],

                "reason":
                    classification["reason"],
            }
        )

    result = pd.DataFrame(opportunities)

    if result.empty:
        return result

    decision_order = {
        "BET": 0,
        "LEAN": 1,
        "PASS": 2,
    }

    result["_decision_order"] = (
        result["decision"]
        .map(decision_order)
        .fillna(3)
    )

    result = result.sort_values(
        by=[
            "_decision_order",
            "model_edge",
            "expected_value",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    result = result.drop(
        columns=["_decision_order"]
    )

    return result.reset_index(drop=True)
