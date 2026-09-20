
import streamlit as st
from supabase import create_client


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
