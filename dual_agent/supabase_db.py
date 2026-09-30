
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

# ==========================================
# MLB PITCHER HISTORY PERSISTENCE
# ==========================================

import io
import pandas as pd


MLB_STORAGE_BUCKET = "market-edge-data"

MLB_PITCHER_HISTORY_FILE = (
    "mlb/mlb_pitcher_history.csv"
)


def ensure_market_edge_storage_bucket():
    """
    Make sure the permanent Market Edge storage
    bucket exists.

    This is handled entirely through Python.
    """

    try:
        client = get_supabase_client()

        buckets = client.storage.list_buckets()

        bucket_names = []

        for bucket in buckets:

            if isinstance(bucket, dict):
                bucket_name = bucket.get("name")

            else:
                bucket_name = getattr(
                    bucket,
                    "name",
                    None,
                )

            if bucket_name:
                bucket_names.append(bucket_name)

        if MLB_STORAGE_BUCKET not in bucket_names:

            client.storage.create_bucket(
                MLB_STORAGE_BUCKET,
                options={
                    "public": False,
                },
            )

        return {
            "success": True,
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }


def save_mlb_pitcher_history(
    pitcher_logs,
):
    """
    Permanently save MLB historical pitcher logs
    to Supabase Storage.

    Existing records are deduplicated before saving.

    This replaces the previous saved file so the
    stored dataset always represents the newest
    complete checkpoint.
    """

    try:

        if (
            pitcher_logs is None
            or pitcher_logs.empty
        ):
            return {
                "success": False,
                "error": (
                    "Pitcher history is empty. "
                    "Nothing was saved."
                ),
            }

        client = get_supabase_client()

        bucket_result = (
            ensure_market_edge_storage_bucket()
        )

        if not bucket_result.get(
            "success",
            False,
        ):
            return bucket_result

        clean_logs = pitcher_logs.copy()

        # ----------------------------------
        # REMOVE DUPLICATE PITCHER/GAME ROWS
        # ----------------------------------

        dedupe_columns = [
            column
            for column in [
                "game_id",
                "pitcher_id",
            ]
            if column in clean_logs.columns
        ]

        if dedupe_columns:

            clean_logs = (
                clean_logs
                .drop_duplicates(
                    subset=dedupe_columns,
                    keep="last",
                )
                .reset_index(drop=True)
            )

        # ----------------------------------
        # CONVERT DATAFRAME TO CSV BYTES
        # ----------------------------------

        csv_bytes = clean_logs.to_csv(
            index=False
        ).encode("utf-8")

        # ----------------------------------
        # UPSERT PERMANENT FILE
        # ----------------------------------

        response = (
            client.storage
            .from_(MLB_STORAGE_BUCKET)
            .upload(
                path=MLB_PITCHER_HISTORY_FILE,
                file=csv_bytes,
                file_options={
                    "content-type": "text/csv",
                    "upsert": "true",
                },
            )
        )

        return {
            "success": True,
            "rows_saved": len(clean_logs),
            "path": MLB_PITCHER_HISTORY_FILE,
            "response": response,
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }


def load_mlb_pitcher_history():
    """
    Load the permanently saved MLB pitcher-history
    dataset from Supabase Storage.

    Returns an empty DataFrame if no permanent
    dataset exists yet.
    """

    try:

        client = get_supabase_client()

        bucket_result = (
            ensure_market_edge_storage_bucket()
        )

        if not bucket_result.get(
            "success",
            False,
        ):
            return pd.DataFrame()

        file_bytes = (
            client.storage
            .from_(MLB_STORAGE_BUCKET)
            .download(
                MLB_PITCHER_HISTORY_FILE
            )
        )

        if not file_bytes:
            return pd.DataFrame()

        pitcher_logs = pd.read_csv(
            io.BytesIO(file_bytes)
        )

        # ----------------------------------
        # RESTORE IMPORTANT DATA TYPES
        # ----------------------------------

        if "start_time" in pitcher_logs.columns:

            pitcher_logs["start_time"] = (
                pd.to_datetime(
                    pitcher_logs["start_time"],
                    utc=True,
                    errors="coerce",
                )
            )

        for column in [
            "game_id",
            "pitcher_id",
        ]:

            if column in pitcher_logs.columns:

                pitcher_logs[column] = (
                    pd.to_numeric(
                        pitcher_logs[column],
                        errors="coerce",
                    )
                )

        # ----------------------------------
        # REMOVE DUPLICATES AS SAFETY
        # ----------------------------------

        dedupe_columns = [
            column
            for column in [
                "game_id",
                "pitcher_id",
            ]
            if column in pitcher_logs.columns
        ]

        if dedupe_columns:

            pitcher_logs = (
                pitcher_logs
                .drop_duplicates(
                    subset=dedupe_columns,
                    keep="last",
                )
                .reset_index(drop=True)
            )

        return pitcher_logs

    except Exception as exc:

        print(
            "MLB pitcher-history load failed:",
            exc,
        )

        return pd.DataFrame()
