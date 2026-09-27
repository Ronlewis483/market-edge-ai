
import streamlit as st
from supabase import create_client
from datetime import datetime, timezone

@st.cache_resource
def get_supabase_client():
    """Connect Market Edge AI to Supabase."""

    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]

    return create_client(url, key)


def test_connection():
    """Verify that the betting database is accessible."""

    try:
        client = get_supabase_client()

        response = (
            client.table("bets")
            .select("id")
            .limit(1)
            .execute()
        )

        return True, "Supabase connection successful!"

    except Exception as e:
        return False, str(e)


def save_bet(bet_data):
    """Save a new bet to Supabase."""
    try:
        client = get_supabase_client()

        response = (
            client.table("bets")
            .insert(bet_data)
            .execute()
        )

        return True, response.data

    except Exception as e:
        return False, str(e)


def get_all_bets():
    """Retrieve saved bets, newest first."""
    try:
        client = get_supabase_client()

        response = (
            client.table("bets")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return response.data

    except Exception as e:
        st.error(f"Unable to load betting history: {e}")
        return []


def update_bet_result(bet_id, status, profit_loss):
    """Record the final result of a bet."""
    try:
        client = get_supabase_client()

        response = (
            client.table("bets")
            .update({
                "status": status,
                "profit_loss": profit_loss,
            })
            .eq("id", bet_id)
            .execute()
        )

        return True, response.data

    except Exception as e:
        return False, str(e)


def save_prediction_snapshot(
    league,
    prediction,
    opportunity=None,
    model_version=None,
):
    """
    Persist the exact state of a prediction when it was generated.

    This creates an audit trail so predictions can later be
    searched, graded, calibrated, and compared with actual results.
    """

    try:
        client = get_supabase_client()

        if client is None:
            return {
                "success": False,
                "error": "Supabase client is not available.",
            }

        opportunity = opportunity or {}

        record = {
            "league": str(league).upper(),

            "game_id": prediction.get("game_id"),
            "game_time": prediction.get("commence_time"),

            "away_team": prediction.get("away_team"),
            "home_team": prediction.get("home_team"),

            "predicted_team": prediction.get("predicted_team"),

            "home_win_probability": prediction.get(
                "home_win_probability"
            ),

            "away_win_probability": prediction.get(
                "away_win_probability"
            ),

            "confidence": prediction.get("confidence"),

            # Market information
            "sportsbook": opportunity.get("sportsbook"),
            "moneyline": opportunity.get("moneyline"),

            "market_probability": opportunity.get(
                "market_probability"
            ),

            "model_edge": opportunity.get("model_edge"),

            "expected_value": opportunity.get(
                "expected_value"
            ),

            # Decision layer
            "recommendation": opportunity.get(
                "recommendation"
            ),

            "reliability": opportunity.get(
                "reliability"
            ),

            # Model tracking
            "model_version": model_version,

            # Result fields intentionally empty at prediction time
            "result": None,
            "actual_winner": None,

            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        response = (
            client
            .table("prediction_history")
            .insert(record)
            .execute()
        )

        return {
            "success": True,
            "data": response.data,
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }


def get_prediction_history(
    league=None,
    team=None,
    limit=500,
):
    """
    Retrieve historical prediction snapshots.

    Optional filters:
        league -> NFL / NBA / MLB
        team   -> searches home and away teams
    """

    try:
        client = get_supabase_client()

        if client is None:
            return []

        query = (
            client
            .table("prediction_history")
            .select("*")
            .order(
                "generated_at",
                desc=True,
            )
            .limit(limit)
        )

        if league:
            query = query.eq(
                "league",
                str(league).upper(),
            )

        response = query.execute()

        rows = response.data or []

        if team:

            search_term = str(team).lower().strip()

            rows = [
                row
                for row in rows
                if (
                    search_term
                    in str(
                        row.get(
                            "home_team",
                            "",
                        )
                    ).lower()
                    or
                    search_term
                    in str(
                        row.get(
                            "away_team",
                            "",
                        )
                    ).lower()
                )
            ]

        return rows

    except Exception as exc:
        print(
            "Prediction history lookup failed:",
            exc,
        )
        return []
