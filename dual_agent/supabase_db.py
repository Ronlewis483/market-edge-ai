
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
