
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
import numpy as np


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

# ==========================================
# MLB DAILY PREGAME INTELLIGENCE STORAGE
# ==========================================
# Current-day game context is stored separately from the
# permanent historical pitcher cache. A latest snapshot is
# convenient for live use, while timestamped snapshots let
# Market Edge preserve exactly what was known before a game.

import json
from datetime import datetime, timezone

MLB_PREGAME_STORAGE_PREFIX = "mlb/pregame"
MLB_PREGAME_LATEST_FILE = f"{MLB_PREGAME_STORAGE_PREFIX}/latest.json"


MLB_PREGAME_STORAGE_VERSION = 2


def _json_safe(value):
    """Convert captures including DataFrames and nested numpy values to strict JSON."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.datetime64):
        return _json_safe(pd.Timestamp(value))
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_mlb_pregame_intelligence(snapshot, snapshot_date=None):
    """Archive first, then update latest. Report partial success explicitly."""
    from uuid import uuid4
    result = {"success": False, "archive_saved": False, "latest_saved": False}
    if not isinstance(snapshot, dict):
        return dict(result, message="No valid MLB pregame snapshot supplied.")
    try:
        now = datetime.now(timezone.utc)
        date_value = snapshot_date or snapshot.get("snapshot_date") or snapshot.get("game_date") or snapshot.get("date") or now.date()
        snapshot_date = pd.Timestamp(date_value).date().isoformat()
        archive_file = (f"{MLB_PREGAME_STORAGE_PREFIX}/{snapshot_date}/"
                        f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}.json")
        result.update(snapshot_date=snapshot_date, archive_file=archive_file,
                      latest_file=MLB_PREGAME_LATEST_FILE, saved_at=now.isoformat())
        payload = {"snapshot_date": snapshot_date, "saved_at": now.isoformat(),
                   "archive_file": archive_file, "data": _json_safe(snapshot)}
        file_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        bucket_result = ensure_market_edge_storage_bucket()
        if not bucket_result.get("success"):
            raise RuntimeError(bucket_result.get("error") or "Storage bucket unavailable.")
        client = get_supabase_client()
        bucket = client.storage.from_(MLB_STORAGE_BUCKET)
        bucket.upload(archive_file, file_bytes, {"content-type": "application/json", "upsert": "false"})
        result["archive_saved"] = True
        bucket.upload(MLB_PREGAME_LATEST_FILE, file_bytes, {"content-type": "application/json", "upsert": "true"})
        result.update(success=True, latest_saved=True)
    except Exception as exc:
        result["message"] = str(exc)
    return result


def load_mlb_pregame_intelligence(snapshot_date=None, archive_file=None):
    """
    Load MLB pregame intelligence.

    Default: load mlb/pregame/latest.json.
    Pass archive_file to load a specific frozen historical snapshot.
    """
    client = get_supabase_client()

    object_path = archive_file or MLB_PREGAME_LATEST_FILE

    try:
        file_bytes = (
            client.storage
            .from_(MLB_STORAGE_BUCKET)
            .download(object_path)
        )

        if not file_bytes:
            return None

        payload = json.loads(file_bytes.decode("utf-8"))

        if snapshot_date is not None:
            wanted = str(snapshot_date)[:10]
            actual = str(payload.get("snapshot_date", ""))[:10]
            if actual != wanted:
                return None

        return payload

    except Exception as exc:
        print("MLB pregame-intelligence load failed:", exc)
        return None

# ==========================================
# MLB HISTORICAL PLAYER CACHE PERSISTENCE
# CHUNKED STORAGE VERSION
# ==========================================

MLB_PLAYER_HISTORY_FILE = "mlb/mlb_player_history.csv"

MLB_PLAYER_HISTORY_CHUNK_PREFIX = (
    "mlb/player_history_chunks"
)

MLB_PLAYER_HISTORY_MANIFEST_FILE = (
    f"{MLB_PLAYER_HISTORY_CHUNK_PREFIX}/manifest.json"
)


def _normalize_mlb_player_history(player_logs):
    """
    Normalize, deduplicate, and sort MLB player-game history.
    """

    if player_logs is None or player_logs.empty:
        return pd.DataFrame()

    clean_logs = player_logs.copy()

    # ----------------------------------
    # RESTORE / NORMALIZE START TIME
    # ----------------------------------

    if "start_time" in clean_logs.columns:
        clean_logs["start_time"] = pd.to_datetime(
            clean_logs["start_time"],
            utc=True,
            errors="coerce",
        )

    # ----------------------------------
    # NUMERIC COLUMNS
    # ----------------------------------

    numeric_columns = [
        "game_id",
        "season_id",
        "team_id",
        "opponent_team_id",
        "player_id",
        "batting_order",
        "batting_games",
        "plate_appearances",
        "at_bats",
        "hits",
        "doubles",
        "triples",
        "batting_home_runs",
        "batting_runs",
        "rbi",
        "batting_walks",
        "batting_strikeouts",
        "stolen_bases",
        "pitching_games",
        "games_started",
        "pitching_outs",
        "batters_faced",
        "pitcher_strikeouts",
        "pitcher_walks",
        "pitcher_hits",
        "pitcher_home_runs",
        "earned_runs",
        "pitches",
        "strikes",
    ]

    for column in numeric_columns:
        if column in clean_logs.columns:
            clean_logs[column] = pd.to_numeric(
                clean_logs[column],
                errors="coerce",
            )

    # ----------------------------------
    # BOOLEAN BULLPEN FLAG
    # ----------------------------------

    if "is_bullpen" in clean_logs.columns:
        clean_logs["is_bullpen"] = (
            clean_logs["is_bullpen"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin([
                "true",
                "1",
                "yes",
            ])
        )

    # ----------------------------------
    # DEDUPLICATE GAME / PLAYER
    # ----------------------------------

    dedupe_columns = [
        column
        for column in [
            "game_id",
            "player_id",
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
    # STABLE ORDER
    # ----------------------------------

    sort_columns = [
        column
        for column in [
            "start_time",
            "game_id",
            "side",
            "player_id",
        ]
        if column in clean_logs.columns
    ]

    if sort_columns:
        clean_logs = (
            clean_logs
            .sort_values(sort_columns)
            .reset_index(drop=True)
        )

    return clean_logs


def _load_mlb_player_history_manifest():
    """
    Load the chunk manifest.

    If no manifest exists yet, return an empty manifest.
    """

    try:
        client = get_supabase_client()

        file_bytes = (
            client.storage
            .from_(MLB_STORAGE_BUCKET)
            .download(
                MLB_PLAYER_HISTORY_MANIFEST_FILE
            )
        )

        if not file_bytes:
            raise FileNotFoundError

        manifest = json.loads(
            file_bytes.decode("utf-8")
        )

        if not isinstance(manifest, dict):
            raise ValueError(
                "Invalid MLB player-history manifest."
            )

        manifest.setdefault("version", 1)
        manifest.setdefault("chunks", [])
        manifest.setdefault("saved_game_ids", [])

        return manifest

    except Exception:
        return {
            "version": 1,
            "chunks": [],
            "saved_game_ids": [],
            "updated_at": None,
        }


def _save_mlb_player_history_manifest(manifest):
    """
    Save the small chunk manifest.
    """

    try:
        client = get_supabase_client()

        payload = json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        response = (
            client.storage
            .from_(MLB_STORAGE_BUCKET)
            .upload(
                path=MLB_PLAYER_HISTORY_MANIFEST_FILE,
                file=payload,
                file_options={
                    "content-type": "application/json",
                    "upsert": "true",
                },
            )
        )

        return {
            "success": True,
            "response": response,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


def _load_mlb_player_history_base():
    """
    Load the original large MLB player-history CSV.

    IMPORTANT:
    This preserves the historical data already collected
    before chunked storage was introduced.
    """

    try:
        client = get_supabase_client()

        file_bytes = (
            client.storage
            .from_(MLB_STORAGE_BUCKET)
            .download(
                MLB_PLAYER_HISTORY_FILE
            )
        )

        if not file_bytes:
            return pd.DataFrame()

        player_logs = pd.read_csv(
            io.BytesIO(file_bytes)
        )

        return _normalize_mlb_player_history(
            player_logs
        )

    except Exception as exc:
        print(
            "MLB base player-history load failed:",
            exc,
        )

        return pd.DataFrame()


def save_mlb_player_history(player_logs):
    """
    Save only player-game rows that are not already
    permanently stored.

    The original mlb/mlb_player_history.csv remains the
    permanent base dataset.

    New data is stored in small append-only CSV chunks.
    """

    try:

        if (
            player_logs is None
            or player_logs.empty
        ):
            return {
                "success": False,
                "error": (
                    "MLB player history is empty. "
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

        clean_logs = _normalize_mlb_player_history(
            player_logs
        )

        # ==================================
        # LOAD MANIFEST
        # ==================================

        manifest = (
            _load_mlb_player_history_manifest()
        )

        saved_game_ids = set()

        for game_id in manifest.get(
            "saved_game_ids",
            [],
        ):
            try:
                saved_game_ids.add(int(game_id))
            except (TypeError, ValueError):
                pass

        # ==================================
        # INCLUDE ORIGINAL BASE GAME IDS
        # ==================================

        base_logs = (
            _load_mlb_player_history_base()
        )

        if (
            base_logs is not None
            and not base_logs.empty
            and "game_id" in base_logs.columns
        ):
            base_game_ids = (
                pd.to_numeric(
                    base_logs["game_id"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            saved_game_ids.update(
                base_game_ids
            )

        # ==================================
        # FIND ONLY UNSAVED GAMES
        # ==================================

        if "game_id" not in clean_logs.columns:
            return {
                "success": False,
                "error": (
                    "MLB player history does not "
                    "contain game_id."
                ),
            }

        clean_game_ids = pd.to_numeric(
            clean_logs["game_id"],
            errors="coerce",
        )

        new_mask = ~clean_game_ids.isin(
            saved_game_ids
        )

        new_logs = (
            clean_logs.loc[new_mask]
            .copy()
            .reset_index(drop=True)
        )

        # Nothing new means the checkpoint
        # is already safely stored.
        if new_logs.empty:
            return {
                "success": True,
                "rows_saved": 0,
                "games_saved": 0,
                "players_saved": 0,
                "path": None,
                "message": (
                    "No new MLB player-history "
                    "games needed to be saved."
                ),
            }

        # ==================================
        # BUILD SMALL CHUNK
        # ==================================

        chunk_game_ids = (
            pd.to_numeric(
                new_logs["game_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        now = datetime.now(timezone.utc)

        timestamp = now.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        chunk_path = (
            f"{MLB_PLAYER_HISTORY_CHUNK_PREFIX}/"
            f"player_history_{timestamp}.csv"
        )

        csv_bytes = new_logs.to_csv(
            index=False
        ).encode("utf-8")

        # ==================================
        # UPLOAD ONLY THIS NEW CHUNK
        # ==================================

        response = (
            client.storage
            .from_(MLB_STORAGE_BUCKET)
            .upload(
                path=chunk_path,
                file=csv_bytes,
                file_options={
                    "content-type": "text/csv",
                    "upsert": "false",
                },
            )
        )

        # ==================================
        # UPDATE MANIFEST
        # ==================================

        chunk_record = {
            "path": chunk_path,
            "rows": int(len(new_logs)),
            "games": int(
                len(chunk_game_ids)
            ),
            "created_at": now.isoformat(),
        }

        manifest.setdefault(
            "chunks",
            [],
        ).append(
            chunk_record
        )

        saved_game_ids.update(
            chunk_game_ids
        )

        manifest["saved_game_ids"] = sorted(
            saved_game_ids
        )

        manifest["updated_at"] = (
            now.isoformat()
        )

        manifest_result = (
            _save_mlb_player_history_manifest(
                manifest
            )
        )

        if not manifest_result.get(
            "success",
            False,
        ):
            return {
                "success": False,
                "error": (
                    "MLB player-history chunk was "
                    "uploaded, but the manifest "
                    "could not be updated: "
                    f"{manifest_result.get('error')}"
                ),
                "chunk_path": chunk_path,
            }

        return {
            "success": True,
            "rows_saved": int(
                len(new_logs)
            ),
            "games_saved": int(
                len(chunk_game_ids)
            ),
            "players_saved": (
                int(
                    new_logs[
                        "player_id"
                    ].nunique()
                )
                if "player_id"
                in new_logs.columns
                else None
            ),
            "path": chunk_path,
            "response": response,
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }


def load_mlb_player_history():
    """
    Load the complete MLB player-history warehouse.

    Combines:
        1. Original base CSV.
        2. Every append-only player-history chunk.

    Returns the same DataFrame format expected by
    the rest of Market Edge.
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

        history_frames = []

        # ==================================
        # ORIGINAL BASE HISTORY
        # ==================================

        base_logs = (
            _load_mlb_player_history_base()
        )

        if (
            base_logs is not None
            and not base_logs.empty
        ):
            history_frames.append(
                base_logs
            )

        # ==================================
        # CHUNK HISTORY
        # ==================================

        manifest = (
            _load_mlb_player_history_manifest()
        )

        for chunk in manifest.get(
            "chunks",
            [],
        ):

            chunk_path = chunk.get(
                "path"
            )

            if not chunk_path:
                continue

            try:

                file_bytes = (
                    client.storage
                    .from_(MLB_STORAGE_BUCKET)
                    .download(
                        chunk_path
                    )
                )

                if not file_bytes:
                    continue

                chunk_logs = pd.read_csv(
                    io.BytesIO(file_bytes)
                )

                if not chunk_logs.empty:
                    history_frames.append(
                        chunk_logs
                    )

            except Exception as exc:

                print(
                    "MLB player-history "
                    "chunk load failed:",
                    chunk_path,
                    exc,
                )

        if not history_frames:
            return pd.DataFrame()

        # ==================================
        # COMBINE BASE + CHUNKS
        # ==================================

        player_logs = pd.concat(
            history_frames,
            ignore_index=True,
            sort=False,
        )

        return _normalize_mlb_player_history(
            player_logs
        )

    except Exception as exc:

        print(
            "MLB player-history load failed:",
            exc,
        )

        return pd.DataFrame()


def get_mlb_player_history_status():
    """
    Return status for the complete MLB player-history
    warehouse, including base + chunks.
    """

    player_logs = (
        load_mlb_player_history()
    )

    manifest = (
        _load_mlb_player_history_manifest()
    )

    if (
        player_logs is None
        or player_logs.empty
    ):
        return {
            "exists": False,
            "rows": 0,
            "games": 0,
            "players": 0,
            "first_game": None,
            "last_game": None,
            "chunks": 0,
            "path": MLB_PLAYER_HISTORY_FILE,
        }

    first_game = None
    last_game = None

    if "start_time" in player_logs.columns:

        valid_times = pd.to_datetime(
            player_logs["start_time"],
            utc=True,
            errors="coerce",
        ).dropna()

        if not valid_times.empty:
            first_game = str(
                valid_times.min()
            )

            last_game = str(
                valid_times.max()
            )

    return {
        "exists": True,
        "rows": int(
            len(player_logs)
        ),
        "games": (
            int(
                player_logs[
                    "game_id"
                ].nunique()
            )
            if "game_id"
            in player_logs.columns
            else 0
        ),
        "players": (
            int(
                player_logs[
                    "player_id"
                ].nunique()
            )
            if "player_id"
            in player_logs.columns
            else 0
        ),
        "first_game": first_game,
        "last_game": last_game,
        "chunks": len(
            manifest.get(
                "chunks",
                [],
            )
        ),
        "path": MLB_PLAYER_HISTORY_FILE,
    }
