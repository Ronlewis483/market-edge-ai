import pandas as pd
import numpy as np
from dual_agent.nba_data import test_nba_connections
from dual_agent.nba_history import test_nba_history

from dual_agent.balldontlie_data import (
    test_balldontlie_connection,
    test_full_historical_season,
    test_multiple_historical_seasons,
    audit_historical_games,
    get_multiple_historical_seasons,
)
from dual_agent.nba_research import (
    run_nba_research,
    run_multi_season_nba_research,
    nba_calibration_summary,
    prepare_balldontlie_games_for_research,
    build_balldontlie_pregame_features,
    audit_balldontlie_pregame_features,
    run_balldontlie_walkforward_model,
    build_balldontlie_future_matchup_features,
    predict_balldontlie_matchup,
    calculate_no_vig_model_edge,
    get_live_nba_moneylines,
    normalize_nba_team_name,
)

from dual_agent.nfl_data import (
    test_sportradar_connection,
    get_nfl_seasons,
    get_nfl_season_games,
    get_multiple_nfl_seasons,
    audit_nfl_games,
)

from dual_agent.nfl_research import (
    build_nfl_pregame_features,
    run_nfl_walkforward_model,
    run_nfl_walkforward_model_v2,
    compare_nfl_v1_v2,
    build_nfl_future_matchup_features,
    predict_nfl_matchup,
)

from dual_agent.nfl_decision import (
    build_nfl_confidence_profile,
    get_nfl_decision,
)

from dual_agent.nfl_market import (
    calculate_nfl_market_edge,
    classify_nfl_market_edge,
)

from dual_agent.nfl_odds import (
    get_live_nfl_moneylines,
    get_best_nfl_moneylines,
)

from dual_agent.nfl_live_engine import (
    build_live_nfl_opportunities,
)


from dual_agent.nfl_accuracy_audit import (
    run_nfl_accuracy_audit,
)

from dual_agent.nfl_prediction_pipeline import (
    run_nfl_prediction_pipeline,
)

from dual_agent.nba_prediction_pipeline import (
    run_nba_prediction_pipeline,
)


from dual_agent.nfl_player_props_ui import render_nfl_player_props

from dual_agent.nfl_player_props_v2 import (
    compare_v1_v2_walkforward,
    compare_v1_v2a_v2b_walkforward,
)

import streamlit as st

from dual_agent.supabase_db import (
    test_connection,
    save_bet,
    get_all_bets,
    update_bet_result,
    save_prediction_snapshot,
    load_mlb_pitcher_history,
    save_mlb_pitcher_history,
    load_mlb_player_history,
    save_mlb_player_history,
    get_mlb_player_history_status,
)

from dual_agent.bet_settlement import (
    auto_settle_bets,
)

from dual_agent.nfl_player_history import load_player_history

from dual_agent.mlb_research import (
    fetch_mlb_games,
    build_mlb_pregame_features,
    summarize_mlb_dataset,
    train_mlb_prediction_model,
    run_mlb_walkforward_v1,
    summarize_mlb_walkforward_v1,
    get_missing_mlb_pitcher_log_games,
    collect_mlb_pitcher_logs_batch,
    build_mlb_v2a_features,
    run_mlb_walkforward_v2a,
    summarize_mlb_walkforward_v2a,
    get_missing_mlb_player_log_games,
    collect_mlb_player_logs_batch,
)
st.set_page_config(page_title="Market Edge AI V5", page_icon="📊", layout="wide")

# ==========================================
# MARKET EDGE AI V5 - ALPACA CONNECTION TEST
# ==========================================

import requests

with st.sidebar.expander("Alpaca Connection Test"):

    if st.button("Test Alpaca API", key="test_alpaca_connection"):

        try:
            api_key = st.secrets["ALPACA_API_KEY"]
            secret_key = st.secrets["ALPACA_SECRET_KEY"]

            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            }

            response = requests.get(
                "https://data.alpaca.markets/v2/stocks/AAPL/bars",
                headers=headers,
                params={
                    "timeframe": "1Day",
                    "limit": 1,
                    "feed": "iex",
                },
                timeout=15,
            )

            if response.status_code == 200:

                data = response.json()
                bars = data.get("bars", [])

                st.success("Alpaca API connected successfully!")

                if bars:
                    st.write("Latest available AAPL bar:")
                    st.json(bars[0])
                else:
                    st.warning(
                        "Connection successful, but no price bars were returned."
                    )

            else:
                st.error(
                    f"Alpaca API returned HTTP {response.status_code}."
                )
                st.write(response.text[:500])

        except KeyError as error:
            st.error(
                f"Missing Alpaca credential in Streamlit Secrets: {error}"
            )

        except requests.RequestException as error:
            st.error(f"Alpaca connection error: {error}")

        except Exception as error:
            st.error(f"Connection test failed: {error}")
from dual_agent.research import DEFAULT_UNIVERSE, latest_scan, run_research

from dual_agent.stock import (
    train_all as train_stock_models,
    metrics_all as stock_model_metrics,
)
from dual_agent.signal_engine import clean_symbols, stock_decision, sports_decision, GATES
from dual_agent.validated_model import VALIDATED_STOCK_MODEL as M
from trading_center_ui import render_trading_center


# ==========================================
# MARKET EDGE AI — DASHBOARD V2 THEME
# ==========================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background: #0B1220;
        color: #F8FAFC;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111C30;
        border-right: 1px solid #243247;
    }

    /* Headings */
    h1, h2, h3 {
        color: #F8FAFC !important;
        letter-spacing: -0.5px;
    }

    h1 {
        font-weight: 800 !important;
    }

    /* Dashboard metric cards */
    div[data-testid="stMetric"] {
        background: #172338;
        border: 1px solid #263850;
        border-radius: 16px;
        padding: 20px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        background: #172338;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover {
        border-color: #38BDF8;
        color: #38BDF8;
    }

    .stButton > button[kind="primary"] {
        background: #0284C7;
        color: white;
        border: none;
    }

    /* Input fields */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #172338;
        border-radius: 10px;
    }

    /* Expandable sections */
    div[data-testid="stExpander"] {
        background: #111C30;
        border: 1px solid #263850;
        border-radius: 12px;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #263850;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Horizontal dividers */
    hr {
        border-color: #263850;
    }

    
/* ==========================================
   NAVIGATION & TOOLBAR VISIBILITY FIX
   ========================================== */

/* Sidebar navigation labels */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #E2E8F0 !important;
}

/* Sidebar radio navigation */
section[data-testid="stSidebar"]
div[role="radiogroup"] label {
    color: #F8FAFC !important;
}

/* Streamlit top toolbar */
header[data-testid="stHeader"] {
    background: #111C30 !important;
    color: #F8FAFC !important;
}

/* Toolbar buttons and icons */
header[data-testid="stHeader"] button {
    color: #F8FAFC !important;
}

header[data-testid="stHeader"] button svg {
    color: #F8FAFC !important;
    fill: none;
    stroke: currentColor;
}

/* Toolbar icon visibility */
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] svg {
    color: #F8FAFC !important;
}

/* Main text and input labels */
.stApp label,
.stApp .stMarkdown p {
    color: #E2E8F0;
}

/* Selected sidebar navigation item */
section[data-testid="stSidebar"]
div[role="radiogroup"]
label:has(input:checked) {
    background: #1E3A5F;
    border-radius: 8px;
    padding: 8px;
}

/* Selected navigation text */
section[data-testid="stSidebar"]
label:has(input:checked) p {
    color: #38BDF8 !important;
    font-weight: 700;
}

    </style>
    """,
    unsafe_allow_html=True,
)

def render_league_prediction_results(
    league_name,
    league_icon,
    pipeline_result,
):
    """
    Shared prediction-results UI for NFL, NBA, and MLB.

    Expected pipeline_result:
    {
        "predictions": pandas.DataFrame,
        "opportunities": pandas.DataFrame,
        ...
    }
    """

    if pipeline_result is None:
        return

    predictions = pipeline_result.get("predictions")
    opportunities = pipeline_result.get("opportunities")

    if predictions is None or predictions.empty:
        st.info(
            f"No upcoming {league_name} predictions are currently available."
        )
        return

    prediction_count = len(predictions)

    # ============================================
    # UPCOMING GAME PREDICTIONS
    # ============================================

    with st.expander(
        f"{league_icon} Upcoming {league_name} Game Predictions "
        f"({prediction_count})",
        expanded=False,
    ):
        st.caption(
            "Model-generated win probabilities for upcoming games."
        )

        # ========================================
        # CONFIDENCE SUMMARY
        # ========================================

        high_confidence_count = int(
            (predictions["confidence"] >= 0.70).sum()
        )

        moderate_count = int(
            (
                (predictions["confidence"] >= 0.58)
                & (predictions["confidence"] < 0.70)
            ).sum()
        )

        close_count = int(
            (predictions["confidence"] < 0.58).sum()
        )

        (
            summary_col1,
            summary_col2,
            summary_col3,
            summary_col4,
        ) = st.columns(4)

        summary_col1.metric(
            "Games",
            prediction_count,
        )

        summary_col2.metric(
            "High Confidence",
            high_confidence_count,
        )

        summary_col3.metric(
            "Moderate",
            moderate_count,
        )

        summary_col4.metric(
            "Close Matchups",
            close_count,
        )

        st.markdown("")

        # ========================================
        # SORT PREDICTIONS
        # ========================================

        sorted_predictions = predictions.sort_values(
            by="confidence",
            ascending=False,
        ).reset_index(drop=True)

        confidence_groups = [
            (
                "🔥 HIGH CONFIDENCE PICKS",
                "The model's strongest win-probability predictions.",
                sorted_predictions[
                    sorted_predictions["confidence"] >= 0.70
                ],
            ),
            (
                "⚡ MODERATE CONFIDENCE",
                "The model has a meaningful preference, "
                "but with less separation.",
                sorted_predictions[
                    (
                        sorted_predictions["confidence"] >= 0.58
                    )
                    & (
                        sorted_predictions["confidence"] < 0.70
                    )
                ],
            ),
            (
                "⚖️ CLOSE MATCHUPS",
                "Games where the model sees relatively "
                "little separation.",
                sorted_predictions[
                    sorted_predictions["confidence"] < 0.58
                ],
            ),
        ]

        # ========================================
        # GAME CARDS
        # ========================================

        for (
            group_title,
            group_description,
            group_predictions,
        ) in confidence_groups:

            if group_predictions.empty:
                continue

            st.markdown("---")
            st.markdown(f"### {group_title}")
            st.caption(group_description)

            group_records = group_predictions.to_dict("records")

            for index in range(
                0,
                len(group_records),
                2,
            ):
                card_columns = st.columns(2)

                games_in_row = group_records[
                    index:index + 2
                ]

                for column, game in zip(
                    card_columns,
                    games_in_row,
                ):
                    with column:

                        home_team = game.get(
                            "home_team",
                            "Home",
                        )

                        away_team = game.get(
                            "away_team",
                            "Away",
                        )

                        predicted_team = game.get(
                            "predicted_team",
                            "Unknown",
                        )

                        confidence = float(
                            game.get(
                                "confidence",
                                0.0,
                            )
                        )

                        home_probability = float(
                            game.get(
                                "home_win_probability",
                                0.0,
                            )
                        )

                        away_probability = float(
                            game.get(
                                "away_win_probability",
                                0.0,
                            )
                        )

                        # --------------------------
                        # GAME TIME
                        # --------------------------

                        commence_time = game.get(
                            "commence_time"
                        )

                        game_time = pd.to_datetime(
                            commence_time,
                            utc=True,
                            errors="coerce",
                        )

                        if pd.notna(game_time):

                            central_time = (
                                game_time.tz_convert(
                                    "America/Chicago"
                                )
                            )

                            game_time_text = (
                                central_time.strftime(
                                    "%a • %I:%M %p CT"
                                )
                                .replace(
                                    " 0",
                                    " ",
                                )
                                .upper()
                            )

                        else:
                            game_time_text = "TIME TBD"

                        # --------------------------
                        # CONFIDENCE LABEL
                        # --------------------------

                        if confidence >= 0.70:

                            confidence_label = (
                                "HIGH CONFIDENCE"
                            )
                            confidence_icon = "🔥"

                        elif confidence >= 0.58:

                            confidence_label = (
                                "MODERATE"
                            )
                            confidence_icon = "⚡"

                        else:

                            confidence_label = (
                                "CLOSE MATCHUP"
                            )
                            confidence_icon = "⚖️"

                        # --------------------------
                        # CARD
                        # --------------------------

                        with st.container(
                            border=True
                        ):

                            st.caption(
                                game_time_text
                            )

                            st.markdown(
                                f"#### {away_team} "
                                f"@ {home_team}"
                            )

                            st.markdown(
                                "##### 🏆 MODEL PICK"
                            )

                            st.markdown(
                                f"## {predicted_team}"
                            )

                            st.progress(
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        confidence,
                                    ),
                                )
                            )

                            (
                                probability_col1,
                                probability_col2,
                            ) = st.columns(2)

                            probability_col1.metric(
                                away_team,
                                f"{away_probability:.1%}",
                            )

                            probability_col2.metric(
                                home_team,
                                f"{home_probability:.1%}",
                            )

                            st.markdown(
                                f"**{confidence_icon} "
                                f"{confidence_label}** "
                                f"• {confidence:.1%}"
                            )

                            with st.expander(
                                "View model details"
                            ):

                                st.write(
                                    "**Predicted winner:** "
                                    f"{predicted_team}"
                                )

                                st.write(
                                    "**Model confidence:** "
                                    f"{confidence:.1%}"
                                )

                                st.write(
                                    "**Home win probability:** "
                                    f"{home_probability:.1%}"
                                )

                                st.write(
                                    "**Away win probability:** "
                                    f"{away_probability:.1%}"
                                )

                                training_games = game.get(
                                    "training_games"
                                )

                                if training_games is not None:

                                    st.write(
                                        "**Historical training games:** "
                                        f"{training_games}"
                                    )

        st.markdown("---")

        st.caption(
            "Predictions are ordered from highest to lowest "
            "model confidence within each section. Confidence "
            "represents estimated win probability and does not "
            "by itself indicate betting value."
        )

    # ============================================
    # MARKET OPPORTUNITIES
    # ============================================

    if (
        opportunities is not None
        and not opportunities.empty
    ):

        with st.expander(
            "💰 Market Opportunities",
            expanded=False,
        ):

            st.caption(
                "Model-vs-market analysis using available "
                "sportsbook moneylines."
            )

            st.dataframe(
                opportunities,
                use_container_width=True,
                hide_index=True,
            )

st.title("📊 Market Edge AI — V5")

# Supabase database connection test
with st.expander("Database Connection Status"):
    if st.button("Test Supabase Connection"):
        success, message = test_connection()

        
        if success:
            st.success(message)
        else:
            st.error("Database connection failed.")
            st.error(f"Error details: {message}")
            
st.caption("Persistent validated model • lightweight daily inference • research/paper mode")


# ==========================================
# MARKET EDGE AI — NAVIGATION V2
# ==========================================

with st.sidebar:

    st.markdown("## ⚡ MARKET EDGE AI")
    st.caption("Sports Intelligence & Trading")

    st.divider()

    
    navigation_options = [
        "🏠 Home",
        "🏀 Sports Center",
        "📈 Trading Center",
        "🎟️ My Bets",
        "📊 Performance",
        "🧪 Research Lab",
    ]

    if "main_navigation" not in st.session_state:
        st.session_state["main_navigation"] = "🏠 Home"

    page = st.radio(
        "NAVIGATION",
        navigation_options,
        key="main_navigation",
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("MARKET EDGE AI V5")
    st.success("System Online")

default=clean_symbols(DEFAULT_UNIVERSE)


if page == "🏠 Home":

    st.title("🏈 Sports Betting Center")

    st.write(
        "Find opportunities, review predictions, "
        "and track your betting performance."
    )

# ============================================
# HOME — PREDICTION CONTROL CENTER
# ============================================

if page == "🏠 Home":

    # ============================================
    # NFL ONE-CLICK PREDICTION CENTER
    # ============================================
    
    st.subheader("🏈 NFL Prediction Center")
    
    st.caption(
        "Generate upcoming NFL game predictions, shop available "
        "moneylines, and analyze model-vs-market opportunities."
    )
    
    if st.button(
        "⚡ Generate NFL Predictions",
        key="generate_nfl_predictions_one_click",
        type="primary",
        use_container_width=True,
    ):
        try:
    
            feature_games = st.session_state.get(
                "nfl_feature_games"
            )

            # ============================================
            # AUTO-PREPARE NFL HISTORICAL FEATURES
            # ============================================

            required_nfl_feature_columns = {
                "start_time",
                "home_score",
                "away_score",
                "home_team",
                "away_team",
            }

            feature_schema_is_stale = (
                feature_games is not None
                and hasattr(feature_games, "columns")
                and not required_nfl_feature_columns.issubset(
                    set(feature_games.columns)
                )
            )

            if (
                feature_games is None
                or (
                    hasattr(feature_games, "empty")
                    and feature_games.empty
                )
                or feature_schema_is_stale
            ):
                with st.spinner(
                    "Preparing NFL historical data automatically..."
                ):
                    season_ids = [
                        "sr:season:115087",
                        "sr:season:127985",
                    ]

                    multi_nfl = get_multiple_nfl_seasons(
                        season_ids
                    )

                    historical_games = multi_nfl["games"]

                    if (
                        historical_games is None
                        or historical_games.empty
                    ):
                        raise ValueError(
                            "NFL historical download returned no games."
                        )

                    st.session_state[
                        "multi_nfl_games"
                    ] = historical_games

                    feature_games = build_nfl_pregame_features(
                        historical_games
                    )

                    if feature_games.empty:
                        raise ValueError(
                            "NFL historical feature generation "
                            "returned no data."
                        )

                    st.session_state[
                        "nfl_feature_games"
                    ] = feature_games

                    nfl_validation = run_nfl_walkforward_model(
                        feature_games
                    )

                    st.session_state[
                        "nfl_walkforward_result"
                    ] = nfl_validation

                    historical_predictions = (
                        nfl_validation.get("predictions")
                    )

                    if historical_predictions is not None:
                        st.session_state[
                            "nfl_historical_predictions"
                        ] = historical_predictions

                    st.session_state[
                        "nfl_historical_accuracy"
                    ] = nfl_validation.get(
                        "accuracy"
                    )

                    st.session_state[
                        "nfl_historical_sample"
                    ] = nfl_validation.get(
                        "prediction_count"
                    )
    
            with st.spinner(
                "Generating NFL predictions and analyzing markets..."
            ):
    
                nfl_pipeline_result = (
                    run_nfl_prediction_pipeline(
                        feature_games=feature_games,
                        historical_accuracy=st.session_state.get(
                            "nfl_historical_accuracy"
                        ),
                        historical_sample=st.session_state.get(
                            "nfl_historical_sample"
                        ),
                    )
                )
    
                st.session_state[
                    "nfl_prediction_pipeline_result"
                ] = nfl_pipeline_result
    
                st.session_state[
                    "live_nfl_moneylines"
                ] = nfl_pipeline_result[
                    "live_odds"
                ]
    
                st.session_state[
                    "best_nfl_moneylines"
                ] = nfl_pipeline_result[
                    "best_lines"
                ]
    
                st.session_state[
                    "live_nfl_predictions"
                ] = nfl_pipeline_result[
                    "predictions"
                ]
    
                st.session_state[
                    "live_nfl_opportunities"
                ] = nfl_pipeline_result[
                    "opportunities"
                ]

        
    
        except Exception as error:
    
            st.error(
                f"NFL prediction pipeline failed: {error}"
            )
    
            st.exception(error)
    
    # ============================================
    # NFL PREDICTION RESULTS
    # ============================================
    
    nfl_pipeline_result = st.session_state.get(
        "nfl_prediction_pipeline_result",
        None,
    )
    
    if nfl_pipeline_result is not None:
    
        predictions = nfl_pipeline_result.get(
            "predictions"
        )
    
        opportunities = nfl_pipeline_result.get(
            "opportunities"
        )
    
        if (
            predictions is not None
            and not predictions.empty
        ):
    
            prediction_count = len(predictions)

        # ========================================
        # COLLAPSIBLE GAME PREDICTIONS
        # ========================================

        with st.expander(
            f"🏈 Upcoming NFL Game Predictions "
            f"({prediction_count})",
            expanded=False,
        ):

            st.caption(
                "Model-generated win probabilities "
                "for upcoming games."
            )

            # ------------------------------------
            # PREDICTION SUMMARY
            # ------------------------------------

            high_confidence_count = int(
                (
                    predictions["confidence"]
                    >= 0.70
                ).sum()
            )

            moderate_count = int(
                (
                    (
                        predictions["confidence"]
                        >= 0.58
                    )
                    & (
                        predictions["confidence"]
                        < 0.70
                    )
                ).sum()
            )

            close_count = int(
                (
                    predictions["confidence"]
                    < 0.58
                ).sum()
            )

            (
                summary_col1,
                summary_col2,
                summary_col3,
                summary_col4,
            ) = st.columns(4)

            summary_col1.metric(
                "Games",
                prediction_count,
            )

            summary_col2.metric(
                "High Confidence",
                high_confidence_count,
            )

            summary_col3.metric(
                "Moderate",
                moderate_count,
            )

            summary_col4.metric(
                "Close Matchups",
                close_count,
            )

            st.markdown("")

                        # ------------------------------------
            # GAME CARDS
            # MOST FAVORABLE → LEAST FAVORABLE
            # ------------------------------------

            sorted_predictions = predictions.sort_values(
                by="confidence",
                ascending=False,
            ).reset_index(drop=True)

            confidence_groups = [
                (
                    "🔥 HIGH CONFIDENCE PICKS",
                    "The model's strongest win-probability predictions.",
                    sorted_predictions[
                        sorted_predictions["confidence"] >= 0.70
                    ],
                ),
                (
                    "⚡ MODERATE CONFIDENCE",
                    "The model has a meaningful preference, but with less separation.",
                    sorted_predictions[
                        (
                            sorted_predictions["confidence"] >= 0.58
                        )
                        & (
                            sorted_predictions["confidence"] < 0.70
                        )
                    ],
                ),
                (
                    "⚖️ CLOSE MATCHUPS",
                    "Games where the model sees relatively little separation.",
                    sorted_predictions[
                        sorted_predictions["confidence"] < 0.58
                    ],
                ),
            ]

            for (
                group_title,
                group_description,
                group_predictions,
            ) in confidence_groups:

                if group_predictions.empty:
                    continue

                st.markdown("---")

                st.markdown(
                    f"### {group_title}"
                )

                st.caption(
                    group_description
                )

                group_records = (
                    group_predictions.to_dict(
                        "records"
                    )
                )

                for index in range(
                    0,
                    len(group_records),
                    2,
                ):

                    card_columns = st.columns(2)

                    games_in_row = group_records[
                        index:index + 2
                    ]

                    for column, game in zip(
                        card_columns,
                        games_in_row,
                    ):

                        with column:

                            home_team = game[
                                "home_team"
                            ]

                            away_team = game[
                                "away_team"
                            ]

                            predicted_team = game[
                                "predicted_team"
                            ]

                            confidence = float(
                                game["confidence"]
                            )

                            home_probability = float(
                                game[
                                    "home_win_probability"
                                ]
                            )

                            away_probability = float(
                                game[
                                    "away_win_probability"
                                ]
                            )

                            # --------------------------
                            # FORMAT GAME TIME
                            # --------------------------

                            game_time = pd.to_datetime(
                                game["commence_time"],
                                utc=True,
                                errors="coerce",
                            )

                            if pd.notna(game_time):

                                central_time = (
                                    game_time.tz_convert(
                                        "America/Chicago"
                                    )
                                )

                                game_time_text = (
                                    central_time.strftime(
                                        "%a • %I:%M %p CT"
                                    )
                                    .replace(
                                        " 0",
                                        " ",
                                    )
                                    .upper()
                                )

                            else:

                                game_time_text = (
                                    "TIME TBD"
                                )

                            # --------------------------
                            # CONFIDENCE CLASSIFICATION
                            # --------------------------

                            if confidence >= 0.70:

                                confidence_label = (
                                    "HIGH CONFIDENCE"
                                )

                                confidence_icon = "🔥"

                            elif confidence >= 0.58:

                                confidence_label = (
                                    "MODERATE"
                                )

                                confidence_icon = "⚡"

                            else:

                                confidence_label = (
                                    "CLOSE MATCHUP"
                                )

                                confidence_icon = "⚖️"

                            # --------------------------
                            # RENDER GAME CARD
                            # --------------------------

                            with st.container(
                                border=True
                            ):

                                st.caption(
                                    game_time_text
                                )

                                st.markdown(
                                    f"#### {away_team} "
                                    f"@ {home_team}"
                                )

                                st.markdown(
                                    "##### 🏆 MODEL PICK"
                                )

                                st.markdown(
                                    f"## {predicted_team}"
                                )

                                st.progress(
                                    max(
                                        0.0,
                                        min(
                                            1.0,
                                            confidence,
                                        ),
                                    )
                                )

                                (
                                    probability_col1,
                                    probability_col2,
                                ) = st.columns(2)

                                probability_col1.metric(
                                    away_team,
                                    f"{away_probability:.1%}",
                                )

                                probability_col2.metric(
                                    home_team,
                                    f"{home_probability:.1%}",
                                )

                                st.markdown(
                                    f"**{confidence_icon} "
                                    f"{confidence_label}** "
                                    f"• {confidence:.1%}"
                                )

                                with st.expander(
                                    "View model details"
                                ):

                                    training_games = (
                                        game.get(
                                            "training_games"
                                        )
                                    )

                                    st.write(
                                        "**Predicted winner:** "
                                        f"{predicted_team}"
                                    )

                                    st.write(
                                        "**Model confidence:** "
                                        f"{confidence:.1%}"
                                    )

                                    st.write(
                                        "**Home win probability:** "
                                        f"{home_probability:.1%}"
                                    )

                                    st.write(
                                        "**Away win probability:** "
                                        f"{away_probability:.1%}"
                                    )

                                    if (
                                        training_games
                                        is not None
                                    ):

                                        st.write(
                                            "**Historical training "
                                            "games:** "
                                            f"{training_games}"
                                        )

            st.markdown("---")

            st.caption(
                "Predictions are ordered from highest to lowest "
                "model confidence within each section. Confidence "
                "represents estimated win probability and does not "
                "by itself indicate betting value."
            )

            # ------------------------------------
            # MARKET OPPORTUNITIES
            # ------------------------------------

            if (
                opportunities is not None
                and not opportunities.empty
            ):

                with st.expander(
                    "💰 Market Opportunities",
                    expanded=False,
                ):

                    st.caption(
                        "Model-vs-market analysis "
                        "using available sportsbook "
                        "moneylines."
                    )

                    st.dataframe(
                        opportunities,
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================
# NFL PLAYER PROP PREDICTIONS
# ============================================

if page == "🏠 Home":
    with st.expander(
        "🎯 NFL Player Prop Predictions",
        expanded=False,
    ):
        render_nfl_player_props()


# ============================================
# SPORTS CENTER — LEAGUE PREDICTION HUB
# ============================================

if page == "🏀 Sports Center":

    st.title("🏀 Sports Center")

    st.caption(
        "AI-powered game predictions, player props, "
        "and model-vs-market opportunities."
    )

    selected_league = st.radio(
        "League",
        [
            "🏈 NFL",
            "🏀 NBA",
            "⚾ MLB",
        ],
        horizontal=True,
        key="sports_center_league",
        label_visibility="collapsed",
    )

    st.divider()

    # ========================================
    # NFL
    # ========================================

    if selected_league == "🏈 NFL":

        st.subheader(
            "🏈 NFL Prediction Center"
        )

        st.caption(
            "Generate upcoming NFL game predictions, "
            "shop available moneylines, and analyze "
            "model-vs-market opportunities."
        )

        nfl_result = st.session_state.get(
            "nfl_prediction_pipeline_result"
        )

        render_league_prediction_results(
            league_name="NFL",
            league_icon="🏈",
            pipeline_result=nfl_result,
        )

        with st.expander(
            "🎯 NFL Player Prop Predictions",
            expanded=False,
        ):
            render_nfl_player_props()

    # ========================================
    # NBA
    # ========================================

    elif selected_league == "🏀 NBA":

        st.subheader(
            "🏀 NBA Prediction Center"
        )

        st.caption(
            "Generate upcoming NBA game predictions "
            "and analyze model-vs-market opportunities."
        )

        if st.button(
            "⚡ Generate NBA Predictions",
            key="sports_center_generate_nba",
            type="primary",
            use_container_width=True,
        ):

            try:

                status_box = st.empty()

                def update_nba_status(
                    stage,
                    elapsed,
                ):
                    status_box.info(
                        f"🏀 {stage} — "
                        f"{elapsed:.1f} seconds"
                    )

                nba_result = (
                    run_nba_prediction_pipeline(
                        progress_callback=update_nba_status,
                    )
                )

                st.session_state[
                    "nba_prediction_pipeline_result"
                ] = nba_result

                status_box.success(
                    "✅ NBA predictions generated successfully."
                )

            except Exception as error:

                st.error(
                    "NBA prediction pipeline failed: "
                    f"{error}"
                )

                st.exception(error)

        nba_result = st.session_state.get(
            "nba_prediction_pipeline_result"
        )

        render_league_prediction_results(
            league_name="NBA",
            league_icon="🏀",
            pipeline_result=nba_result,
        )

    # ========================================
    # MLB
    # ========================================

    elif selected_league == "⚾ MLB":

        st.subheader(
            "⚾ MLB Prediction Center"
        )

        st.caption("Saved pregame data → timing and coverage checks → learned complete-context candidate → predictions. This candidate is experimental until its accuracy is established.")
        prediction_date = st.date_input("MLB game date", value=pd.Timestamp.now(tz="America/Chicago").date(), key="mlb_connected_date")
        if st.button("Generate MLB predictions from collected data", key="mlb_connected_generate", type="primary"):
            try:
                import dual_agent.mlb_research as research
                import dual_agent.supabase_db as storage
                with st.spinner("Checking saved outcomes and generating predictions..."):
                    dataset = storage.load_mlb_capture_learning_dataset()
                    archived = storage.load_mlb_pregame_intelligence()
                    batch = research.build_mlb_capture_learning_rows(archived["data"]) if archived and isinstance(archived.get("data"), dict) else {"rows": [], "errors": []}
                    keys = {row["capture_key"] for row in batch["rows"]}
                    refreshed = research.refresh_mlb_capture_learning_results([row for row in dataset["rows"] if row["capture_key"] not in keys])
                    batch["rows"] = refreshed["rows"] + batch["rows"]
                    merged = storage.update_mlb_capture_learning_dataset(batch)
                    if not merged.get("success"):
                        raise RuntimeError(merged.get("message", "Unable to save result checks"))
                    result = research.run_mlb_saved_player_model(merged["data"])
                    saved = storage.save_mlb_saved_player_model(result)
                    if not saved.get("success"):
                        raise RuntimeError(saved.get("message", "Unable to save model record"))
                    st.session_state["mlb_connected_result"] = result
                    st.session_state["mlb_connected_dates"] = {int(row["game_id"]): pd.Timestamp(row["start_time"]).tz_convert("America/Chicago").date() for row in merged["data"]["rows"]}
            except Exception as exc:
                st.error("MLB prediction pipeline stopped: " + str(exc))
        connected = st.session_state.get("mlb_connected_result")
        if connected:
            st.info(connected["message"])
            picks = [r for r in connected.get("upcoming_predictions", []) if r["Model"] == "Complete pregame candidate" and st.session_state.get("mlb_connected_dates", {}).get(r["game_id"]) == prediction_date]
            now_utc = pd.Timestamp.now(tz="UTC")
            picks = [r for r in picks if now_utc - pd.Timestamp(r["Captured UTC"]) <= pd.Timedelta(minutes=15)]
            if picks:
                for pick in picks:
                    probability = pick["Home win probability"]
                    pick["Predicted winner"] = pick["Home"] if probability >= .5 else pick["Away"]
                    pick["Winner probability"] = max(probability, 1-probability)
                st.dataframe(pd.DataFrame(picks), use_container_width=True, hide_index=True)
                st.caption("Experimental candidate predictions. Weather and other context affect the equation only when training observations exist.")
            else:
                st.warning("No usable candidate predictions for this date. The engine requires 200 completed captures to fit, plus eligible upcoming captures collected within 15 minutes. The scanner remains deferred.")
            if connected.get("scores"):
                st.dataframe(pd.DataFrame(connected["scores"]), use_container_width=True, hide_index=True)
            with st.expander("Data coverage and model controllers"):
                st.json({"completed_games": connected["completed_games"], "minimum_training_games": connected["minimum_training_games"], "excluded": connected["excluded"], "models": connected["models"], "coverage": connected.get("feature_coverage", [])})


# ============================================
# NBA ONE-CLICK PREDICTION CENTER
# ============================================

if page == "🏠 Home":

    st.divider()
    
    
    st.caption(
        "Generate upcoming NBA game predictions and "
        "analyze model-vs-market opportunities."
    )
    
    if st.button(
        "⚡ Generate NBA Predictions",
        key="generate_nba_predictions_one_click",
        type="primary",
        use_container_width=True,
    ):
    
        try:
            status_box = st.empty()
    
            def update_nba_status(
                stage,
                elapsed,
            ):
                status_box.info(
                    f"🏀 {stage} — "
                    f"{elapsed:.1f} seconds"
                )
    
            nba_pipeline_result = (
                run_nba_prediction_pipeline(
                    progress_callback=update_nba_status,
                )
            )
    
            st.session_state[
                "nba_prediction_pipeline_result"
            ] = nba_pipeline_result
    
            status_box.success(
                "✅ NBA predictions generated successfully."
            )
    
        except Exception as error:
            st.error(
                f"NBA prediction pipeline failed: {error}"
            )
    
            st.exception(error)

# ============================================
# NBA PREDICTION RESULTS
# ============================================

nba_pipeline_result = st.session_state.get(
    "nba_prediction_pipeline_result",
    None,
)

if nba_pipeline_result is not None:

    nba_predictions = nba_pipeline_result.get(
        "predictions"
    )

    nba_opportunities = nba_pipeline_result.get(
        "opportunities"
    )

    if (
        nba_predictions is not None
        and not nba_predictions.empty
    ):

        nba_prediction_count = len(
            nba_predictions
        )

        with st.expander(
            f"🏀 Upcoming NBA Game Predictions "
            f"({nba_prediction_count})",
            expanded=False,
        ):

            st.caption(
                "Model-generated win probabilities "
                "for upcoming NBA games."
            )

            # ====================================
            # SUMMARY
            # ====================================

            nba_high_count = int(
                (
                    nba_predictions["confidence"]
                    >= 0.70
                ).sum()
            )

            nba_moderate_count = int(
                (
                    (
                        nba_predictions["confidence"]
                        >= 0.58
                    )
                    & (
                        nba_predictions["confidence"]
                        < 0.70
                    )
                ).sum()
            )

            nba_close_count = int(
                (
                    nba_predictions["confidence"]
                    < 0.58
                ).sum()
            )

            (
                nba_summary_1,
                nba_summary_2,
                nba_summary_3,
                nba_summary_4,
            ) = st.columns(4)

            nba_summary_1.metric(
                "Games",
                nba_prediction_count,
            )

            nba_summary_2.metric(
                "High Confidence",
                nba_high_count,
            )

            nba_summary_3.metric(
                "Moderate",
                nba_moderate_count,
            )

            nba_summary_4.metric(
                "Close Matchups",
                nba_close_count,
            )

            st.markdown("")

            # ====================================
            # MOST FAVORABLE → LEAST FAVORABLE
            # ====================================

            sorted_nba_predictions = (
                nba_predictions
                .sort_values(
                    by="confidence",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            nba_confidence_groups = [
                (
                    "🔥 HIGH CONFIDENCE PICKS",
                    "The model's strongest NBA "
                    "win-probability predictions.",
                    sorted_nba_predictions[
                        sorted_nba_predictions[
                            "confidence"
                        ] >= 0.70
                    ],
                ),
                (
                    "⚡ MODERATE CONFIDENCE",
                    "The model has a meaningful "
                    "preference, but with less separation.",
                    sorted_nba_predictions[
                        (
                            sorted_nba_predictions[
                                "confidence"
                            ] >= 0.58
                        )
                        & (
                            sorted_nba_predictions[
                                "confidence"
                            ] < 0.70
                        )
                    ],
                ),
                (
                    "⚖️ CLOSE MATCHUPS",
                    "NBA games where the model sees "
                    "relatively little separation.",
                    sorted_nba_predictions[
                        sorted_nba_predictions[
                            "confidence"
                        ] < 0.58
                    ],
                ),
            ]

            for (
                group_title,
                group_description,
                group_predictions,
            ) in nba_confidence_groups:

                if group_predictions.empty:
                    continue

                st.markdown("---")

                st.markdown(
                    f"### {group_title}"
                )

                st.caption(
                    group_description
                )

                group_records = (
                    group_predictions.to_dict(
                        "records"
                    )
                )

                for index in range(
                    0,
                    len(group_records),
                    2,
                ):

                    card_columns = st.columns(2)

                    games_in_row = (
                        group_records[
                            index:index + 2
                        ]
                    )

                    for column, game in zip(
                        card_columns,
                        games_in_row,
                    ):

                        with column:

                            home_team = game[
                                "home_team"
                            ]

                            away_team = game[
                                "away_team"
                            ]

                            predicted_team = game[
                                "predicted_team"
                            ]

                            confidence = float(
                                game["confidence"]
                            )

                            home_probability = float(
                                game[
                                    "home_win_probability"
                                ]
                            )

                            away_probability = float(
                                game[
                                    "away_win_probability"
                                ]
                            )

                            # ========================
                            # GAME TIME
                            # ========================

                            game_time = pd.to_datetime(
                                game["commence_time"],
                                utc=True,
                                errors="coerce",
                            )

                            if pd.notna(game_time):

                                central_time = (
                                    game_time.tz_convert(
                                        "America/Chicago"
                                    )
                                )

                                game_time_text = (
                                    central_time.strftime(
                                        "%a • %I:%M %p CT"
                                    )
                                    .replace(
                                        " 0",
                                        " ",
                                    )
                                    .upper()
                                )

                            else:

                                game_time_text = (
                                    "TIME TBD"
                                )

                            # ========================
                            # CONFIDENCE
                            # ========================

                            if confidence >= 0.70:

                                confidence_label = (
                                    "HIGH CONFIDENCE"
                                )

                                confidence_icon = "🔥"

                            elif confidence >= 0.58:

                                confidence_label = (
                                    "MODERATE"
                                )

                                confidence_icon = "⚡"

                            else:

                                confidence_label = (
                                    "CLOSE MATCHUP"
                                )

                                confidence_icon = "⚖️"

                            # ========================
                            # NBA CARD
                            # ========================

                            with st.container(
                                border=True
                            ):

                                st.caption(
                                    game_time_text
                                )

                                st.markdown(
                                    f"#### {away_team} "
                                    f"@ {home_team}"
                                )

                                st.markdown(
                                    "##### 🏆 MODEL PICK"
                                )

                                st.markdown(
                                    f"## {predicted_team}"
                                )

                                st.progress(
                                    max(
                                        0.0,
                                        min(
                                            1.0,
                                            confidence,
                                        ),
                                    )
                                )

                                (
                                    probability_col1,
                                    probability_col2,
                                ) = st.columns(2)

                                probability_col1.metric(
                                    away_team,
                                    f"{away_probability:.1%}",
                                )

                                probability_col2.metric(
                                    home_team,
                                    f"{home_probability:.1%}",
                                )

                                st.markdown(
                                    f"**{confidence_icon} "
                                    f"{confidence_label}** "
                                    f"• {confidence:.1%}"
                                )

                                with st.expander(
                                    "View model details"
                                ):

                                    training_games = (
                                        game.get(
                                            "training_games"
                                        )
                                    )

                                    st.write(
                                        "**Predicted winner:** "
                                        f"{predicted_team}"
                                    )

                                    st.write(
                                        "**Model confidence:** "
                                        f"{confidence:.1%}"
                                    )

                                    st.write(
                                        "**Home win probability:** "
                                        f"{home_probability:.1%}"
                                    )

                                    st.write(
                                        "**Away win probability:** "
                                        f"{away_probability:.1%}"
                                    )

                                    if (
                                        training_games
                                        is not None
                                    ):

                                        st.write(
                                            "**Historical training "
                                            "games:** "
                                            f"{training_games}"
                                        )

            st.markdown("---")

            st.caption(
                "Predictions are ordered from highest "
                "to lowest model confidence within each "
                "section. Confidence represents estimated "
                "win probability and does not by itself "
                "indicate betting value."
            )

            # ====================================
            # NBA MARKET OPPORTUNITIES
            # ====================================

            if (
                nba_opportunities is not None
                and not nba_opportunities.empty
            ):

                with st.expander(
                    "💰 NBA Market Opportunities",
                    expanded=False,
                ):

                    st.caption(
                        "Model-vs-market analysis sorted "
                        "by expected ROI and model edge."
                    )

                    st.dataframe(
                        nba_opportunities,
                        use_container_width=True,
                        hide_index=True,
                    )


# ==========================================
# SPORTS CENTER — DASHBOARD NAVIGATION
# ==========================================

    
    
    sports_tab = st.radio(
        "Sports Center Navigation",
        [
            "🎯 Today's Picks",
            "🎟️ My Bets",
            "📊 Model Performance",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="sports_center_tabs",
    )

    
    # Keep Sports Center tabs local; sidebar pages use the main navigation radio.
    show_today = sports_tab == "🎯 Today's Picks"
    show_bets = sports_tab == "🎟️ My Bets"
    show_performance = sports_tab == "📊 Model Performance"

    st.divider()

    # Dashboard summary

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Today's Opportunities",
            "—"
        )

    with col2:
        st.metric(
            "My Win Rate",
            "—"
        )

    with col3:
        st.metric(
            "Total Profit / Loss",
            "—"
        )

    with col4:
        st.metric(
            "Active Bets",
            "—"
        )


    if show_today:
        st.divider()
        st.subheader("Today's Picks")

    

        st.divider()

        # Sports selection

        st.subheader("Choose Your Sport")

        sport = st.selectbox(
            "Which sport would you like to analyze?",
            [ 
                "Game Winner",
                "Player Points",
                "Player Rebounds",
                "Player Assists",
                "Player Passing Yards",
                "Player Rushing Yards",
                "Player Receiving Yards",
                "Player Receptions",
                "Player Passing Touchdowns",
                "Player Anytime Touchdown",
                "Player Rushing + Receiving Yards",
                "Player Passing Completions",
                "Player Interceptions Thrown",
                "Game Total Points",
                "Point Spread",
                "Other",
            ],
        )

        st.divider()

        # Betting market selection

        st.subheader("Choose Your Betting Market")

        market = st.selectbox(
            "What type of bet are you interested in?",
            [
                "Game Winner",
                "Player Points",
                "Player Rebounds",
                "Player Assists",
                "Player Passing Yards",
                "Player Rushing Yards",
                "Player Receiving Yards",
                "Game Total Points",
                "Point Spread",
                "Other",
            ],
        )

        st.divider()

        
        st.divider()

        # ==========================================
        # TODAY'S BETTING OPPORTUNITIES
        # ==========================================

        st.subheader("Today's Betting Opportunities")

        st.info(
            f"Selected sport: {sport}\n\n"
            f"Selected betting market: {market}"
        )

        # ------------------------------------------
        # CONNECT NFL PLAYER PROP FORECASTS
        # ------------------------------------------

        dashboard_market_map = {
            "Player Passing Yards": "Passing yards",
            "Player Rushing Yards": "Rushing yards",
            "Player Receiving Yards": "Receiving yards",
            "Player Receptions": "Receptions",
            "Player Passing Touchdowns": "Passing touchdowns",
            "Player Anytime Touchdown": "Anytime touchdown",
            "Player Rushing + Receiving Yards": "Rushing + receiving yards",
            "Player Passing Completions": "Passing completions",
            "Player Interceptions Thrown": "Interceptions thrown",
        }

        if sport == "NFL Football":

            selected_prop_market = dashboard_market_map.get(
                market
            )

            if selected_prop_market is None:

                st.info(
                    "Select a supported NFL player prop "
                    "market to view available forecasts."
                )

            else:

                # Retrieve forecasts generated by
                # NFL Player Props Research.

                saved_forecasts = st.session_state.get(
                    "nfl_dashboard_forecasts",
                    {}
                )

                matching_forecasts = []

                for forecast in saved_forecasts.values():

                    if (
                        forecast.get("market")
                        == selected_prop_market
                    ):

                        matching_forecasts.append(
                            forecast
                        )

                # ----------------------------------
                # DISPLAY AVAILABLE FORECASTS
                # ----------------------------------

                if not matching_forecasts:

                    st.info(
                        "No forecasts have been generated "
                        "for this NFL player prop market. "
                        "Open NFL Player Props Research, "
                        "load the upcoming game and "
                        "historical statistics, then "
                        "generate a player forecast."
                    )

                else:

                    st.success(
                        f"{len(matching_forecasts)} "
                        "player prop forecast(s) available."
                    )

                    # Show most recently generated first.

                    matching_forecasts = sorted(
                        matching_forecasts,
                        key=lambda item: item.get(
                            "generated_at", ""
                        ),
                        reverse=True
                    )

                    for forecast in matching_forecasts:

                        player_name = forecast["player"]
                        game_name = forecast["game"]
                        prop_side = forecast["side"]
                        prop_line = forecast["line"]

                        projected = forecast[
                            "projected_value"
                        ]

                        chance = forecast[
                            "estimated_chance"
                        ]

                        difference = forecast[
                            "projection_difference"
                        ]

                        sample_size = forecast[
                            "sample_size"
                        ]

                        historical_hits = forecast[
                            "historical_hits"
                        ]

                        eligible_games = forecast[
                            "eligible_games"
                        ]

                        # --------------------------
                        # PLAYER FORECAST CARD
                        # --------------------------

                        with st.container(border=True):

                            st.markdown(
                                f"### 🏈 {player_name}"
                            )

                            st.caption(game_name)

                            st.write(
                                f"**Market:** "
                                f"{selected_prop_market}"
                            )

                            st.write(
                                f"**Prop:** {prop_side} "
                                f"{prop_line:g}"
                            )

                            # ----------------------
                            # FORECAST METRICS
                            # ----------------------

                            c1, c2 = st.columns(2)

                            c1.metric(
                                "Projected Player Performance",
                                f"{projected:.1f}"
                            )

                            c2.metric(
                                "Sportsbook Line",
                                f"{prop_line:g}"
                            )

                            c3, c4 = st.columns(2)

                            c3.metric(
                                "Chances Player Will Make the Prop",
                                f"{chance:.1%}"
                            )

                            c4.metric(
                                "Projection vs. Line",
                                f"{difference:+.1f}"
                            )

                            st.progress(
                                max(
                                    0.0,
                                    min(1.0, chance)
                                )
                            )

                            # ----------------------
                            # HISTORICAL SUPPORT
                            # ----------------------

                            st.caption(
                                f"Historical sample: "
                                f"{sample_size} games | "
                                f"Historical hits: "
                                f"{historical_hits}/"
                                f"{eligible_games}"
                            )

                            st.caption(
                                "The displayed chance is "
                                "a preliminary estimate "
                                "based on historical hit "
                                "frequency. It is not a "
                                "calibrated probability "
                                "or a guaranteed outcome."
                            )

                            st.caption(
                                "Forecast generated: "
                                f"{forecast['generated_at']}"
                            )

        else:

            st.info(
                "NFL player prop forecasts are "
                "currently available through "
                "this dashboard connection. "
                "Other sports will be connected "
                "separately."
            )

        # Prediction area

        st.subheader("Today's Betting Opportunities")

        st.info(
            f"Selected sport: {sport}\n\n"
            f"Selected betting market: {market}"
        )

        st.write(
            "Your available predictions and betting "
            "opportunities will appear here."
        )

        st.caption(
            "Predictions are estimates, not guaranteed outcomes. "
            "Historical performance and current odds should be "
            "reviewed before placing a wager."
        )

        st.divider()

        # Strategy section

        st.subheader("Strategy Performance")

        st.write(
            "Track how different betting strategies "
            "perform over time."
        )

        st.info(
            "Strategy performance will appear here "
            "after your betting history is connected."
        )

    if show_bets:
        st.subheader("My Bets")
        st.info("Open My Bets in the sidebar to record wagers and update results.")
        if st.button("Open My Bets", key="sports_open_bets"):
            st.session_state["main_navigation"] = "🎟️ My Bets"
            st.rerun()

    if show_performance:
        st.subheader("Model Performance")
        st.info("Open Performance in the sidebar for recorded betting results. Model validation remains in Research Lab.")
        if st.button("Open Performance", key="sports_open_performance"):
            st.session_state["main_navigation"] = "📊 Performance"
            st.rerun()

if page=="🏠 Home":
    st.success(f"VALIDATED MODEL LOADED — {M['target']} • {M['features']} • AUC {M['auc']:.3f}")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("📈 Best Stock Signal")
        txt=st.text_input("Symbols to scan", "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AVGO,AMD,JPM,LLY,XOM")
        syms=clean_symbols(txt)
        if st.button("Scan Today's Market",type="primary",use_container_width=True):
            try:
                with st.spinner("Loading recent market data and scoring stocks — no retraining..."):
                    # Keep existing V4 model/data implementation, but do NOT run walk-forward research.
                    df=latest_scan(default,syms)
                    st.session_state["daily"]=stock_decision(df)
            except Exception as e:
                st.error(f"Daily scan failed: {e}")

        d=st.session_state.get("daily")
        if d:
            if d["status"]=="MAKE THIS TRADE":
                st.success("MAKE THIS TRADE")
                st.markdown(f"## {d['symbol']}")
                a,b=st.columns(2)
                a.metric("Model probability",f"{d['probability']:.1%}")
                b.metric("Edge vs base rate",f"+{d['edge']:.1%}")
                st.write(f"**Direction:** {d['direction']}")
                st.write(f"**Horizon:** {d['horizon']}")
                st.write(f"**Target:** {d['target']}")
            else:
                st.warning("NO QUALIFYING TRADE")
                st.caption(d["reason"])
                top=d.get("top",{})
                if top:
                    st.write(f"Top candidate: **{top.get('Symbol','—')}** • probability {float(top.get('P',0)):.1%} • required {GATES['min_probability']:.0%}")

            ranked=d.get("ranked")
            if ranked is not None and len(ranked):
                show=ranked.head(10).copy()
                cols=[x for x in ["Symbol","Close","P","edge"] if x in show.columns]
                show=show[cols].rename(columns={"P":"Model probability","edge":"Edge vs base"})
                st.markdown("#### Today's top candidates")
                st.dataframe(show,use_container_width=True,hide_index=True)

    with c2:
        st.subheader("🏀 Best Sports Signal")
        sd=sports_decision()
        st.warning(sd["status"])
        st.caption(sd["reason"])
        st.code("PICK THIS TEAM / PICK THIS PLAYER PROP\nor\nNO QUALIFYING SPORTS PICK")


    # ============================================
    # NBA PREDICTION CENTER
    # ============================================

    if page == "🏠 Home":

        st.divider()
        st.subheader("🏀 NBA Prediction Center")

        st.caption(
            "NBA game predictions, sportsbook comparison, "
            "and potential wager payouts."
        )

        NBA_TEAMS = [
        "Atlanta Hawks",
        "Boston Celtics",
        "Brooklyn Nets",
        "Charlotte Hornets",
        "Chicago Bulls",
        "Cleveland Cavaliers",
        "Dallas Mavericks",
        "Denver Nuggets",
        "Detroit Pistons",
        "Golden State Warriors",
        "Houston Rockets",
        "Indiana Pacers",
        "Los Angeles Clippers",
        "Los Angeles Lakers",
        "Memphis Grizzlies",
        "Miami Heat",
        "Milwaukee Bucks",
        "Minnesota Timberwolves",
        "New Orleans Pelicans",
        "New York Knicks",
        "Oklahoma City Thunder",
        "Orlando Magic",
        "Philadelphia 76ers",
        "Phoenix Suns",
        "Portland Trail Blazers",
        "Sacramento Kings",
        "San Antonio Spurs",
        "Toronto Raptors",
        "Utah Jazz",
        "Washington Wizards",
    ]

    nba_home = st.selectbox(
        "Home Team",
        NBA_TEAMS,
        index=None,
        placeholder="Select the home team",
        key="nba_home_team",
    )

    nba_away = st.selectbox(
        "Away Team",
        NBA_TEAMS,
        index=None,
        placeholder="Select the away team",
        key="nba_away_team",
    )

    if nba_home and nba_away:
        if nba_home == nba_away:
            st.error(
                "The home and away teams must be different."
            )
        else:
            st.info(
                f"Selected Matchup: {nba_away} at {nba_home}"
            )

    nba_wager = st.number_input(
        "Wager Amount ($)",
        min_value=1.0,
        value=100.0,
        step=10.0,
        key="nba_wager"
    )

    nba_odds = st.number_input(
        "American Odds",
        value=-110,
        step=5,
        key="nba_odds"
    )

    
    
    # ==========================================
    # NBA GAME DATE
    # ==========================================

    nba_game_date = st.date_input(
        "NBA Game Date",
        value="today",
        key="nba_game_date",
    )

    # ==========================================
    # HISTORICAL NBA DATA
    # ==========================================

    @st.cache_data(ttl=86400, show_spinner=False)
    def load_nba_prediction_history(seasons):

        games, summary, requests = (
            get_multiple_historical_seasons(
                list(seasons)
            )
        )

        return games

    # ==========================================
    # GENERATE NBA PREDICTION
    # ==========================================

    if st.button(
        "Generate NBA Prediction",
        key="generate_nba_prediction",
    ):

        if not nba_home or not nba_away:

            st.warning(
                "Please select both NBA teams."
            )

        elif nba_home == nba_away:

            st.warning(
                "Home and away teams must be different."
            )

        elif abs(nba_odds) < 100:

            st.warning(
                "Enter valid American odds, "
                "such as -110 or +150."
            )

        else:

            try:

                with st.spinner(
                    "Loading NBA history and "
                    "generating your prediction..."
                ):

                    # ----------------------------------
                    # 1. Convert NBA team names
                    # ----------------------------------

                    home_code = (
                        normalize_nba_team_name(
                            nba_home
                        )
                    )

                    away_code = (
                        normalize_nba_team_name(
                            nba_away
                        )
                    )

                    if not home_code or not away_code:
                        raise ValueError(
                            "Unable to identify one "
                            "of the selected NBA teams."
                        )

                    # ----------------------------------
                    # 2. Load historical NBA games
                    # ----------------------------------

                    game_year = nba_game_date.year

                    # NBA seasons cross calendar years.
                    # September-December belongs to
                    # the season starting that year.

                    season_year = (
                        game_year
                        if nba_game_date.month >= 9
                        else game_year - 1
                    )

                    seasons = tuple(
                        range(
                            season_year - 3,
                            season_year + 1,
                        )
                    )

                    historical_games = (
                        load_nba_prediction_history(
                            seasons
                        )
                    )

                    # ----------------------------------
                    # 3. Exclude games on or after
                    #    the selected prediction date
                    # ----------------------------------

                    historical_games = (
                        historical_games[
                            pd.to_datetime(
                                historical_games[
                                    "game_date"
                                ]
                            )
                            < pd.Timestamp(
                                nba_game_date
                            )
                        ].copy()
                    )

                    if historical_games.empty:
                        raise ValueError(
                            "No historical NBA games "
                            "were available before "
                            "the selected date."
                        )

                    # ----------------------------------
                    # 4. Prepare historical data
                    # ----------------------------------

                    prepared_games = (
                        prepare_balldontlie_games_for_research(
                            historical_games
                        )
                    )

                    feature_games = (
                        build_balldontlie_pregame_features(
                            prepared_games
                        )
                    )

                    # ----------------------------------
                    # 5. Build future matchup features
                    # ----------------------------------

                    matchup_features = (
                        build_balldontlie_future_matchup_features(
                            prepared_games,
                            home_code,
                            away_code,
                            nba_game_date,
                        )
                    )

                    # ----------------------------------
                    # 6. Run the actual NBA model
                    # ----------------------------------

                    prediction = (
                        predict_balldontlie_matchup(
                            feature_games,
                            matchup_features,
                        )
                    )

                # ==================================
                # DISPLAY NBA PREDICTION RESULTS
                # ==================================

                st.success(
                    "NBA prediction generated successfully!"
                )

                st.subheader(
                    f"{nba_away} at {nba_home}"
                )

                st.caption(
                    f"Game date: {nba_game_date}"
                )

                home_probability = prediction[
                    "home_win_probability"
                ]

                away_probability = prediction[
                    "away_win_probability"
                ]

                predicted_side = prediction[
                    "predicted_side"
                ]

                predicted_team = (
                    nba_home
                    if predicted_side == "HOME"
                    else nba_away
                )

                st.markdown(
                    "### Model Prediction"
                )

                st.write(
                    f"**Predicted winner: "
                    f"{predicted_team}**"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        nba_home,
                        f"{home_probability:.1%}",
                    )

                with col2:

                    st.metric(
                        nba_away,
                        f"{away_probability:.1%}",
                    )

                st.metric(
                    "Model Confidence",
                    f"{prediction['confidence']:.1%}",
                )

                st.caption(
                    "Model confidence is an estimated "
                    "probability, not a verified "
                    "historical win rate."
                )

                st.write(
                    "Historical training games:",
                    prediction["training_games"],
                )

                # ==================================
                # SPORTSBOOK COMPARISON
                # ==================================

                st.divider()

                st.subheader(
                    "Sportsbook Comparison"
                )

                betting_side = st.selectbox(
                    "Which team do these odds apply to?",
                    [nba_home, nba_away],
                    key="nba_betting_side",
                )

                selected_probability = (
                    home_probability
                    if betting_side == nba_home
                    else away_probability
                )

                if nba_odds > 0:

                    implied_probability = (
                        100 / (nba_odds + 100)
                    )

                    potential_profit = (
                        nba_wager * nba_odds / 100
                    )

                else:

                    implied_probability = (
                        abs(nba_odds)
                        / (abs(nba_odds) + 100)
                    )

                    potential_profit = (
                        nba_wager
                        * 100 / abs(nba_odds)
                    )

                estimated_edge = (
                    selected_probability
                    - implied_probability
                )

                total_payout = (
                    nba_wager + potential_profit
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Model Win Probability",
                        f"{selected_probability:.1%}",
                    )

                with col2:

                    st.metric(
                        "Sportsbook Implied Probability",
                        f"{implied_probability:.1%}",
                    )

                st.metric(
                    "Model vs. Sportsbook Difference",
                    f"{estimated_edge:+.1%}",
                )

                st.caption(
                    "This difference compares the "
                    "model estimate with the implied "
                    "probability of the entered odds. "
                    "It is not a validated betting edge."
                )

                # ==================================
                # POTENTIAL PAYOUT
                # ==================================

                st.divider()

                st.subheader(
                    "Potential Wager Payout"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Wager",
                        f"${nba_wager:,.2f}",
                    )

                with col2:

                    st.metric(
                        "Potential Profit",
                        f"${potential_profit:,.2f}",
                    )

                with col3:

                    st.metric(
                        "Total Payout",
                        f"${total_payout:,.2f}",
                    )

                st.caption(
                    "Total payout includes your "
                    "original wager. This assumes "
                    "the selected bet wins."
                )

                st.warning(
                    "Model validation status: "
                    "Not verified for live betting. "
                    "Review out-of-sample accuracy "
                    "and calibration before relying "
                    "on these probabilities."
                )

            except Exception as e:

                st.error(
                    "NBA prediction could not be generated."
                )

                st.error(
                    f"Error details: {e}"
                )

    
    # ==========================================
    # NBA LIVE MODEL HISTORICAL VALIDATION
    # ==========================================

    st.divider()
    st.subheader("🏀 NBA Live Model Validation")

    st.caption(
        "Test the NBA prediction model against "
        "completed historical games."
    )

    
    nba_test_count = st.selectbox(
        "Number of historical games to test",
        options=[25, 50, 100, 250, 500, 1000],
        index=4,
        key="nba_live_validation_count",
    )

    if st.button(
        "Run NBA Live Model Validation",
        key="run_nba_live_validation",
    ):

        try:
            from dual_agent import nba_research

            if not hasattr(
                nba_research,
                "validate_live_nba_model",
            ):
                st.error(
                    "The live-model validation function "
                    "is missing from nba_research.py. "
                    "Confirm it is committed to main."
                )

            else:
                with st.spinner(
                    "Loading NBA history and "
                    "running historical validation..."
                ):

                    current_year = pd.Timestamp.now().year

                    seasons = tuple(
                        range(
                            current_year - 4,
                            current_year + 1,
                        )
                    )

                    historical_games = (
                        load_nba_prediction_history(
                            seasons
                        )
                    )

                    prepared_games = (
                        prepare_balldontlie_games_for_research(
                            historical_games
                        )
                    )

                    feature_games = (
                        build_balldontlie_pregame_features(
                            prepared_games
                        )
                    )

                    results = (
                        nba_research.validate_live_nba_model(
                            historical_games=prepared_games,
                            feature_games=feature_games,
                            minimum_training_games=500,
                            test_games=nba_test_count,
                        )
                    )

                    st.session_state[
                        "nba_live_validation_results"
                    ] = results

                st.success(
                    "NBA historical validation completed!"
                )

        except Exception as validation_error:

            st.error(
                "NBA historical validation failed."
            )

            st.exception(validation_error)

    # ==========================================
    # DISPLAY HISTORICAL VALIDATION RESULTS
    # ==========================================

    if (
        "nba_live_validation_results"
        in st.session_state
    ):

        results = st.session_state[
            "nba_live_validation_results"
        ]

        st.subheader("Historical Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Games Tested",
                results["games_tested"],
            )

        with col2:
            st.metric(
                "Prediction Accuracy",
                f'{results["accuracy"]:.1%}',
            )

        with col3:
            st.metric(
                "Brier Score",
                f'{results["brier_score"]:.4f}',
            )

        st.metric(
            "Log Loss",
            f'{results["log_loss"]:.4f}',
        )

        st.metric(
            "Games Skipped",
            results["games_skipped"],
        )

        st.subheader("Accuracy by Confidence Range")

        st.dataframe(
            results["confidence"],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "View Individual Historical Predictions"
        ):

        
            # ==========================================
            # NBA PICK CONFIDENCE ANALYSIS
            # ==========================================
    
            st.subheader("🏀 NBA Pick Confidence Analysis")
    
            try:
                from dual_agent.nba_research import (
                    analyze_nba_pick_confidence,
                )
    
                historical_predictions = results["predictions"]
    
                confidence_results = analyze_nba_pick_confidence(
                    historical_predictions
                )
    
                st.dataframe(
                    confidence_results,
                    use_container_width=True,
                    hide_index=True,
                )
    
            except Exception as confidence_error:
                st.warning(
                    "NBA confidence analysis could not run."
                )
                st.exception(confidence_error)

            
            st.dataframe(
                results["predictions"],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "View Skipped Games"
        ):
            st.dataframe(
                results["skipped_games"],
                use_container_width=True,
                hide_index=True,
            )

    # ==========================================
    # END NBA LIVE MODEL VALIDATION
    # ==========================================


elif page == "🎟️ My Bets":

    st.title("My Bets")
    st.write(
        "Record your wagers, track wins and losses, "
        "and monitor your betting performance."
    )

    st.divider()

        # ==========================================
    # AUTOMATIC BET SETTLEMENT
    # ==========================================

    st.subheader("Automatic Results")

    st.caption(
        "Check completed NFL, NBA, MLB, and College Football "
        "games and automatically settle eligible pending bets."
    )

    if st.button(
        "🔄 Check Results",
        key="check_bet_results",
        use_container_width=True,
    ):

        with st.spinner(
            "Checking completed games..."
        ):

            settlement_bets = get_all_bets()

            # Support either the direct-list return style
            # or the (success, result) return style.
            if (
                isinstance(settlement_bets, tuple)
                and len(settlement_bets) == 2
            ):
                bets_success, settlement_bets = settlement_bets

                if not bets_success:
                    settlement_bets = []

            if settlement_bets is None:
                settlement_bets = []

            settlement_report = auto_settle_bets(
                settlement_bets
            )

        settled_count = settlement_report.get(
            "settled",
            0,
        )

        checked_count = settlement_report.get(
            "checked",
            0,
        )

        skipped_count = settlement_report.get(
            "skipped",
            0,
        )

        if settled_count > 0:

            st.success(
                f"Settled {settled_count} bet"
                f"{'' if settled_count == 1 else 's'}."
            )

            for update in settlement_report.get(
                "updated",
                [],
            ):

                result = update.get(
                    "result",
                    ""
                )

                description = update.get(
                    "description",
                    "Bet"
                )

                profit_loss = float(
                    update.get(
                        "profit_loss",
                        0,
                    )
                )

                if result == "Won":
                    icon = "✅"

                elif result == "Lost":
                    icon = "❌"

                else:
                    icon = "➖"

                st.write(
                    f"{icon} **{description}** — "
                    f"{result} "
                    f"(${profit_loss:+,.2f})"
                )

        else:

            st.info(
                "No eligible bets were ready "
                "to settle."
            )

        st.caption(
            f"Checked: {checked_count} • "
            f"Skipped/unsupported: {skipped_count}"
        )

        # --------------------------------------
        # API USAGE
        # --------------------------------------

        quota = settlement_report.get(
            "quota",
            {},
        )

        if quota:

            with st.expander(
                "API usage"
            ):

                for sport, usage in quota.items():

                    remaining = usage.get(
                        "remaining",
                        "—",
                    )

                    used = usage.get(
                        "used",
                        "—",
                    )

                    last = usage.get(
                        "last",
                        "—",
                    )

                    st.write(
                        f"**{sport}** — "
                        f"Used: {used} | "
                        f"Remaining: {remaining} | "
                        f"Last request: {last}"
                    )

        # --------------------------------------
        # ERRORS
        # --------------------------------------

        errors = settlement_report.get(
            "errors",
            [],
        )

        if errors:

            with st.expander(
                "Settlement warnings"
            ):

                for error in errors:
                    st.warning(error)


    st.subheader("Record a New Bet")

    # ==========================================
    # BET BUILDER STATE
    # ==========================================

    if "parlay_leg_count" not in st.session_state:
        st.session_state["parlay_leg_count"] = 2

    bet_mode = st.segmented_control(
        "Bet Type",
        options=[
            "Single",
            "Parlay",
        ],
        default="Single",
        key="bet_builder_mode",
    )

    # ==========================================
    # SINGLE BET
    # ==========================================

    if bet_mode == "Single":

        with st.form("single_bet_form"):

            sport = st.selectbox(
                "Sport",
                [
                    "NFL",
                    "NBA",
                    "College Football",
                    "MLB",
                    "Other",
                ],
                key="single_sport",
            )

            betting_market = st.selectbox(
                "Betting Market",
                [
                    "Game Winner",
                    "Spread",
                    "Game Total",
                    "Player Points",
                    "Player Rebounds",
                    "Player Assists",
                    "Passing Yards",
                    "Rushing Yards",
                    "Receiving Yards",
                    "Receptions",
                    "Anytime Touchdown",
                    "Three-Pointers",
                    "Other",
                ],
                key="single_market",
            )

            col1, col2 = st.columns(2)

            with col1:
                player_name = st.text_input(
                    "Player Name (optional)",
                    key="single_player",
                )

            with col2:
                team_name = st.text_input(
                    "Team Name (optional)",
                    key="single_team",
                )

            bet_description = st.text_input(
                "Describe Your Bet",
                placeholder="Example: Ravens moneyline",
                key="single_description",
            )

            col1, col2 = st.columns(2)

            with col1:
                bet_type = st.selectbox(
                    "Bet Direction",
                    [
                        "Over",
                        "Under",
                        "Moneyline",
                        "Spread",
                        "Yes",
                        "No",
                        "Other",
                    ],
                    key="single_direction",
                )

            with col2:
                betting_line = st.number_input(
                    "Betting Line",
                    value=0.0,
                    step=0.5,
                    key="single_line",
                )

            col1, col2 = st.columns(2)

            with col1:
                odds = st.number_input(
                    "American Odds",
                    value=-110,
                    step=1,
                    key="single_odds",
                )

            with col2:
                wager = st.number_input(
                    "Amount Wagered ($)",
                    min_value=0.01,
                    value=10.00,
                    step=1.00,
                    key="single_wager",
                )

            sportsbook = st.text_input(
                "Sportsbook (optional)",
                key="single_sportsbook",
            )

            notes = st.text_area(
                "Notes (optional)",
                key="single_notes",
            )

            submitted = st.form_submit_button(
                "Save Single Bet",
                type="primary",
                use_container_width=True,
            )

        if submitted:

            if not bet_description.strip():

                st.error(
                    "Please enter a description of your bet."
                )

            elif abs(odds) < 100:

                st.error(
                    "Enter valid American odds, "
                    "such as -110 or +150."
                )

            else:

                bet_data = {
                    "sport": sport,
                    "betting_market": betting_market,
                    "player_name": player_name,
                    "team_name": team_name,
                    "bet_description": bet_description,
                    "bet_type": bet_type,
                    "betting_line": betting_line,
                    "odds": odds,
                    "wager": wager,
                    "status": "Pending",
                    "profit_loss": 0,
                    "sportsbook": sportsbook,
                    "strategy": betting_market,
                    "notes": notes,
                    "source": "Manual",
                }

                success, result = save_bet(
                    bet_data
                )

                if success:

                    st.success(
                        "Your bet was saved successfully!"
                    )

                else:

                    st.error(
                        "Unable to save your bet."
                    )

    # ==========================================
    # PARLAY BET BUILDER
    # ==========================================

    else:

        st.markdown("### 🎟️ Parlay Builder")

        st.caption(
            "Add each selection as a separate leg. "
            "The entire parlay will be saved as one ticket."
        )

        # --------------------------------------
        # ADD / REMOVE LEG CONTROLS
        # --------------------------------------

        control_col1, control_col2, control_col3 = st.columns(
            [1, 1, 3]
        )

        with control_col1:

            if st.button(
                "＋ Add Leg",
                key="add_parlay_leg",
                use_container_width=True,
            ):

                st.session_state["parlay_leg_count"] += 1
                st.rerun()

        with control_col2:

            if (
                st.session_state["parlay_leg_count"] > 2
                and st.button(
                    "− Remove Leg",
                    key="remove_parlay_leg",
                    use_container_width=True,
                )
            ):

                st.session_state["parlay_leg_count"] -= 1
                st.rerun()

        st.divider()

        # --------------------------------------
        # PARLAY LEGS
        # --------------------------------------

        parlay_legs = []

        for leg_index in range(
            st.session_state["parlay_leg_count"]
        ):

            leg_number = leg_index + 1

            with st.container(border=True):

                st.markdown(f"#### LEG {leg_number}")

                # ------------------------------
                # SPORT + MARKET
                # ------------------------------

                col1, col2 = st.columns([1, 2])

                with col1:
                    leg_sport = st.selectbox(
                        "Sport",
                        [
                            "NFL",
                            "NBA",
                            "College Football",
                            "MLB",
                            "Other",
                        ],
                        key=f"parlay_sport_{leg_index}",
                    )

                with col2:
                    leg_market = st.selectbox(
                        "Market",
                        [
                            "Game Winner",
                            "Spread",
                            "Game Total",
                            "Player Points",
                            "Player Rebounds",
                            "Player Assists",
                            "Passing Yards",
                            "Rushing Yards",
                            "Receiving Yards",
                            "Receptions",
                            "Anytime Touchdown",
                            "Three-Pointers",
                            "Other",
                        ],
                        key=f"parlay_market_{leg_index}",
                    )

                # ------------------------------
                # GAME MARKETS
                # ------------------------------

                game_markets = [
                    "Game Winner",
                    "Spread",
                    "Game Total",
                ]

                if leg_market in game_markets:

                    if leg_market == "Game Total":

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            leg_team = st.text_input(
                                "Matchup",
                                placeholder="Example: Seahawks @ Cardinals",
                                key=f"parlay_team_{leg_index}",
                            )

                        with col2:
                            leg_direction = st.selectbox(
                                "Pick",
                                ["Over", "Under"],
                                key=f"parlay_direction_{leg_index}",
                            )

                        leg_line = st.number_input(
                            "Total",
                            value=0.0,
                            step=0.5,
                            key=f"parlay_line_{leg_index}",
                        )

                        leg_player = ""

                        leg_selection = (
                            f"{leg_team} "
                            f"{leg_direction} {leg_line:g}"
                        ).strip()

                    elif leg_market == "Spread":

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            leg_team = st.text_input(
                                "Team",
                                placeholder="Example: 49ers",
                                key=f"parlay_team_{leg_index}",
                            )

                        with col2:
                            leg_line = st.number_input(
                                "Spread",
                                value=0.0,
                                step=0.5,
                                key=f"parlay_line_{leg_index}",
                            )

                        leg_player = ""
                        leg_direction = "Spread"

                        if leg_team.strip():
                            leg_selection = (
                                f"{leg_team} {leg_line:+g}"
                            )
                        else:
                            leg_selection = ""

                    else:

                        leg_team = st.text_input(
                            "Team",
                            placeholder="Example: Seahawks",
                            key=f"parlay_team_{leg_index}",
                        )

                        leg_player = ""
                        leg_direction = "Moneyline"
                        leg_line = 0.0

                        if leg_team.strip():
                            leg_selection = (
                                f"{leg_team} ML"
                            )
                        else:
                            leg_selection = ""

                # ------------------------------
                # PLAYER PROP MARKETS
                # ------------------------------

                else:

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        leg_player = st.text_input(
                            "Player",
                            placeholder="Example: Jauan Jennings",
                            key=f"parlay_player_{leg_index}",
                        )

                    with col2:
                        leg_direction = st.selectbox(
                            "Pick",
                            [
                                "Over",
                                "Under",
                                "Yes",
                                "No",
                            ],
                            key=f"parlay_direction_{leg_index}",
                        )

                    leg_team = ""

                    if leg_market == "Anytime Touchdown":

                        leg_line = 0.0

                        if leg_player.strip():
                            leg_selection = (
                                f"{leg_player} "
                                f"Anytime TD — "
                                f"{leg_direction.upper()}"
                            )
                        else:
                            leg_selection = ""

                    else:

                        leg_line = st.number_input(
                            "Line",
                            value=0.0,
                            step=0.5,
                            key=f"parlay_line_{leg_index}",
                        )

                        if leg_player.strip():
                            leg_selection = (
                                f"{leg_player} "
                                f"{leg_direction.upper()} "
                                f"{leg_line:g} "
                                f"{leg_market}"
                            )
                        else:
                            leg_selection = ""

                # ------------------------------
                # TICKET PREVIEW
                # ------------------------------

                if leg_selection:

                    st.caption(
                        f"✓ {leg_selection}"
                    )

                parlay_legs.append(
                    {
                        "sport": leg_sport,
                        "market": leg_market,
                        "player": leg_player,
                        "team": leg_team,
                        "selection": leg_selection,
                        "direction": leg_direction,
                        "line": leg_line,
                    }
                )

        # ======================================
        # TICKET DETAILS
        # ======================================

        st.markdown("### Ticket Details")

        col1, col2 = st.columns(2)

        with col1:

            parlay_odds = st.number_input(
                "Combined American Odds",
                value=200,
                step=1,
                key="parlay_combined_odds",
            )

        with col2:

            parlay_wager = st.number_input(
                "Amount Wagered ($)",
                min_value=0.01,
                value=10.00,
                step=1.00,
                key="parlay_wager",
            )

        parlay_sportsbook = st.text_input(
            "Sportsbook (optional)",
            key="parlay_sportsbook",
        )

        parlay_notes = st.text_area(
            "Notes (optional)",
            key="parlay_notes",
        )

        # --------------------------------------
        # POTENTIAL PAYOUT
        # --------------------------------------

        if parlay_odds > 0:

            potential_profit = (
                parlay_wager
                * parlay_odds
                / 100
            )

        elif parlay_odds <= -100:

            potential_profit = (
                parlay_wager
                * 100
                / abs(parlay_odds)
            )

        else:

            potential_profit = 0

        potential_payout = (
            parlay_wager
            + potential_profit
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Legs",
            len(parlay_legs),
        )

        metric_col2.metric(
            "Wager",
            f"${parlay_wager:,.2f}",
        )

        metric_col3.metric(
            "Potential Payout",
            f"${potential_payout:,.2f}",
        )

        # ======================================
        # SAVE PARLAY
        # ======================================

        if st.button(
            "Save Parlay",
            key="save_parlay_ticket",
            type="primary",
            use_container_width=True,
        ):

            valid_legs = [
                leg
                for leg in parlay_legs
                if leg["selection"].strip()
            ]

            if len(valid_legs) < 2:

                st.error(
                    "A parlay must contain at least "
                    "two completed legs."
                )

            elif abs(parlay_odds) < 100:

                st.error(
                    "Enter valid combined American odds, "
                    "such as +250 or -110."
                )

            else:

                # ----------------------------------
                # Build readable ticket description.
                #
                # We intentionally use the existing
                # bet record structure so no database
                # changes are required.
                # ----------------------------------

                leg_descriptions = []

                for index, leg in enumerate(
                    valid_legs,
                    start=1,
                ):

                    leg_descriptions.append(
                        f"{index}. {leg['selection'].strip()}"
                    )

                parlay_description = " | ".join(
                    leg_descriptions
                )

                sports = []

                for leg in valid_legs:

                    if leg["sport"] not in sports:
                        sports.append(
                            leg["sport"]
                        )

                if len(sports) == 1:
                    parlay_sport = sports[0]
                else:
                    parlay_sport = "Multiple"

                parlay_bet_data = {
                    "sport": parlay_sport,
                    "betting_market": "Parlay",
                    "player_name": "",
                    "team_name": "",
                    "bet_description": parlay_description,
                    "bet_type": "Parlay",
                    "betting_line": 0,
                    "odds": parlay_odds,
                    "wager": parlay_wager,
                    "status": "Pending",
                    "profit_loss": 0,
                    "sportsbook": parlay_sportsbook,
                    "strategy": "Parlay",
                    "notes": parlay_notes,
                    "source": "Manual",
                }

                success, result = save_bet(
                    parlay_bet_data
                )

                if success:

                    st.success(
                        f"Your {len(valid_legs)}-leg "
                        "parlay was saved successfully!"
                    )

                else:

                    st.error(
                        "Unable to save your parlay."
                    )
    st.divider()

    # -----------------------------------
    # Betting history
    # -----------------------------------

    

    # ---------------------------------
    # Betting History and Full Payout
    # ---------------------------------

    st.subheader("Your Betting History")

    bets = get_all_bets()

    if not bets:
        st.info("You haven't recorded any bets yet.")

    else:
        history = []

        for bet in bets:

            wager = float(bet.get("wager") or 0)
            profit = float(bet.get("profit_loss") or 0)
            status = bet.get("status", "Pending")

            # Calculate total payout.
            # A winning payout includes the original wager.

            if status == "Won":
                total_payout = round(
                    wager + profit, 2
                )

            elif status == "Push":
                total_payout = round(wager, 2)

            elif status == "Lost":
                total_payout = 0.0

            else:
                total_payout = None

            # Build a readable betting history.

            history.append({
                "Bet ID": bet.get("id"),
                "Sport": bet.get("sport"),
                "Bet": bet.get("bet_description"),
                "Market": bet.get("betting_market"),
                "Odds": bet.get("odds"),
                "Wager": round(wager, 2),
                "Status": status,
                "Profit / Loss": (
                    round(profit, 2)
                    if status != "Pending"
                    else None
                ),
                "Total Payout": total_payout,
            })

        # Display complete betting history.

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Wager": st.column_config.NumberColumn(
                    "Wager",
                    format="$%.2f",
                ),
                "Profit / Loss": st.column_config.NumberColumn(
                    "Profit / Loss",
                    format="$%.2f",
                ),
                "Total Payout": st.column_config.NumberColumn(
                    "Total Payout",
                    format="$%.2f",
                ),
            },
        )

    st.divider()
        # -----------------------------------
        # Update bet results
        # -----------------------------------


    # ==========================================
    # LOAD BETTING HISTORY
    # ==========================================

    bets = get_all_bets()

    # ==========================================
    # UPDATE A BET RESULT
    # ==========================================

    st.subheader("Update a Bet Result")

    pending_bets = [
        bet for bet in bets
        if bet["status"] == "Pending"
    ]

    if pending_bets:

        bet_options = {
            (
                f"#{bet['id']} - "
                f"{bet['bet_description']}"
            ): bet
            for bet in pending_bets
        }

        selected_bet = st.selectbox(
            "Select a Pending Bet",
            list(bet_options.keys()),
        )

        selected_result = st.selectbox(
            "What was the result?",
            [
                "Won",
                "Lost",
                "Push",
            ],
        )

        if st.button("Save Bet Result"):

            bet = bet_options[selected_bet]

            wager_amount = float(
                bet["wager"]
            )

            american_odds = int(
                bet["odds"]
            )

            if selected_result == "Won":

                if american_odds > 0:

                    profit_loss = (
                        wager_amount
                        * american_odds / 100
                    )

                else:

                    profit_loss = (
                        wager_amount
                        * 100 / abs(american_odds)
                    )

            elif selected_result == "Lost":

                profit_loss = -wager_amount

            else:

                profit_loss = 0

            success, result = update_bet_result(
                bet["id"],
                selected_result,
                round(profit_loss, 2),
            )

            if success:

                st.success(
                    "Your betting result was updated!"
                )

                st.rerun()

            else:

                st.error(
                    "Unable to update your bet."
                )

    else:

        st.info(
            "You have no pending bets to update."
        )




if page == "📊 Performance":
    st.title("📊 Betting Performance")
    st.caption("Recorded wager results, not model validation or future accuracy.")
    try:
        recorded_bets = get_all_bets() or []
        settled = [b for b in recorded_bets if b.get("status") in ("Won", "Lost", "Push")]
        decided = [b for b in settled if b.get("status") in ("Won", "Lost")]
        wins = sum(b.get("status") == "Won" for b in decided)
        pnl = sum(float(b.get("profit_loss") or 0) for b in settled)
        pending = sum(b.get("status") == "Pending" for b in recorded_bets)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Settled Bets", len(settled))
        c2.metric("Win Rate (excludes pushes)", f"{wins / len(decided):.1%}" if decided else "—")
        c3.metric("Realized Profit / Loss", f"${pnl:,.2f}")
        c4.metric("Pending Bets", pending)
        if recorded_bets:
            st.dataframe(pd.DataFrame(recorded_bets), use_container_width=True, hide_index=True)
        else:
            st.info("No recorded bets yet. Use My Bets to add one.")
    except Exception as e:
        st.error(f"Could not load betting performance: {e}")


if page == "📈 Trading Center":
    render_trading_center(
        latest_scan=latest_scan,
        stock_decision=stock_decision,
        universe=default,
        clean_symbols=clean_symbols,
        model=M,
    )



def render_research_lab():
    if page == "🧪 Research Lab":
        st.subheader("🧪 Research Lab - Optional Revalidation")

        # ============================================================
        # NFL V1 VS V2 MODEL BENCHMARK
        # ============================================================

        st.divider()

        st.subheader("🏈 NFL V1 vs V2 Model Benchmark")

        st.caption(
            "Compare the current NFL winner model against the "
            "expanded V2 model using the same leakage-safe "
            "historical walk-forward validation."
        )

        st.info(
            "This test does not replace the live NFL model. "
            "V1 remains active while V2 is evaluated."
        )

        if st.button(
            "Run NFL V1 vs V2 Benchmark",
            key="run_nfl_v1_v2_benchmark",
            type="primary",
        ):

            try:

                feature_games = st.session_state.get(
                    "nfl_feature_games"
                )

        
                # ----------------------------------------------------
                # BUILD HISTORICAL FEATURES IF THEY ARE NOT LOADED
                # ----------------------------------------------------

                if (
                    feature_games is None
                    or (
                        hasattr(
                            feature_games,
                            "empty",
                        )
                        and feature_games.empty
                    )
                ):

                    with st.spinner(
                        "Loading NFL historical data..."
                    ):

                        season_ids = [
                            "sr:season:115087",
                            "sr:season:127985",
                        ]

                        multi_nfl = (
                            get_multiple_nfl_seasons(
                                season_ids
                            )
                        )

                        historical_games = (
                            multi_nfl["games"]
                        )

                        if (
                            historical_games is None
                            or historical_games.empty
                        ):

                            raise ValueError(
                                "NFL historical download "
                                "returned no games."
                            )

                        feature_games = (
                            build_nfl_pregame_features(
                                historical_games
                            )
                        )

                        if feature_games.empty:

                            raise ValueError(
                                "NFL historical feature "
                                "generation returned no data."
                            )

                        st.session_state[
                            "nfl_feature_games"
                        ] = feature_games

                # ----------------------------------------------------
                # RUN V1 VS V2
                # ----------------------------------------------------

                with st.spinner(
                    "Running NFL V1 and V2 walk-forward "
                    "validation..."
                ):

                    benchmark = compare_nfl_v1_v2(
                        feature_games=feature_games,
                        min_train_games=100,
                        retrain_every=25,
                    )

                st.session_state[
                    "nfl_v1_v2_benchmark"
                ] = benchmark

                st.success(
                    "NFL V1 vs V2 benchmark completed."
                )

            except Exception as error:

                st.error(
                    "NFL V1 vs V2 benchmark failed."
                )

                st.exception(error)

        # ============================================================
        # DISPLAY SAVED BENCHMARK
        # ============================================================

        nfl_benchmark = st.session_state.get(
            "nfl_v1_v2_benchmark"
        )

        if nfl_benchmark:

            v1 = nfl_benchmark["v1"]
            v2 = nfl_benchmark["v2"]

            st.markdown(
                "### 📊 Model Comparison"
            )

            # --------------------------------------------------------
            # MODEL SUMMARY
            # --------------------------------------------------------

            (
                v1_col,
                v2_col,
            ) = st.columns(2)

            with v1_col:

                st.markdown(
                    "#### V1 — Current Model"
                )

                st.metric(
                    "Accuracy",
                    f"{v1['accuracy']:.1%}",
                )

                st.metric(
                    "AUC",
                    f"{v1['auc']:.3f}",
                )

                st.metric(
                    "Brier Score",
                    f"{v1['brier']:.3f}",
                )

                st.metric(
                    "Log Loss",
                    f"{v1['log_loss']:.3f}",
                )

                st.metric(
                    "Features",
                    len(
                        v1["feature_columns"]
                    ),
                )

                st.metric(
                    "Predictions",
                    v1["prediction_count"],
                )

            with v2_col:

                st.markdown(
                    "#### V2 — Expanded Model"
                )

                st.metric(
                    "Accuracy",
                    f"{v2['accuracy']:.1%}",
                )

                st.metric(
                    "AUC",
                    f"{v2['auc']:.3f}",
                )

                st.metric(
                    "Brier Score",
                    f"{v2['brier']:.3f}",
                )

                st.metric(
                    "Log Loss",
                    f"{v2['log_loss']:.3f}",
                )

                st.metric(
                    "Features",
                    len(
                        v2["feature_columns"]
                    ),
                )

                st.metric(
                    "Predictions",
                    v2["prediction_count"],
                )

            # --------------------------------------------------------
            # CHANGE FROM V1 TO V2
            # --------------------------------------------------------

            st.markdown(
                "### 🔬 V2 Change vs V1"
            )

            (
                change_col1,
                change_col2,
                change_col3,
                change_col4,
            ) = st.columns(4)

            accuracy_change = (
                nfl_benchmark[
                    "accuracy_change"
                ]
            )

            auc_change = (
                nfl_benchmark[
                    "auc_change"
                ]
            )

            brier_change = (
                nfl_benchmark[
                    "brier_change"
                ]
            )

            log_loss_change = (
                nfl_benchmark[
                    "log_loss_change"
                ]
            )

            change_col1.metric(
                "Accuracy Change",
                f"{accuracy_change:+.2%}",
            )

            if pd.notna(auc_change):

                change_col2.metric(
                    "AUC Change",
                    f"{auc_change:+.3f}",
                )

            else:

                change_col2.metric(
                    "AUC Change",
                    "—",
                )

            change_col3.metric(
                "Brier Change",
                f"{brier_change:+.3f}",
            )

            change_col4.metric(
                "Log Loss Change",
                f"{log_loss_change:+.3f}",
            )

            # --------------------------------------------------------
            # HIGH CONFIDENCE PERFORMANCE
            # --------------------------------------------------------

            st.markdown(
                "### 🔥 High-Confidence Performance"
            )

            (
                confidence_col1,
                confidence_col2,
            ) = st.columns(2)

            with confidence_col1:

                st.markdown(
                    "**V1**"
                )

                v1_high_count = (
                    v1.get(
                        "high_confidence_count",
                        0,
                    )
                )

                v1_high_accuracy = (
                    v1.get(
                        "high_confidence_accuracy"
                    )
                )

                st.metric(
                    "70%+ Predictions",
                    v1_high_count,
                )

                if (
                    v1_high_accuracy
                    is not None
                    and pd.notna(
                        v1_high_accuracy
                    )
                ):

                    st.metric(
                        "70%+ Accuracy",
                        f"{v1_high_accuracy:.1%}",
                    )

                else:

                    st.metric(
                        "70%+ Accuracy",
                        "—",
                    )

            with confidence_col2:

                st.markdown(
                    "**V2**"
                )

                v2_high_count = (
                    v2.get(
                        "high_confidence_count",
                        0,
                    )
                )

                v2_high_accuracy = (
                    v2.get(
                        "high_confidence_accuracy"
                    )
                )

                st.metric(
                    "70%+ Predictions",
                    v2_high_count,
                )

                if (
                    v2_high_accuracy
                    is not None
                    and pd.notna(
                        v2_high_accuracy
                    )
                ):

                    st.metric(
                        "70%+ Accuracy",
                        f"{v2_high_accuracy:.1%}",
                    )

                else:

                    st.metric(
                        "70%+ Accuracy",
                        "—",
                    )

            # --------------------------------------------------------
            # FULL COMPARISON TABLE
            # --------------------------------------------------------

            comparison_df = (
                nfl_benchmark[
                    "comparison"
                ].copy()
            )

            st.markdown(
                "### 📋 Full Benchmark"
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Higher accuracy and AUC are better. "
                "Lower Brier score and log loss are better. "
                "V2 is experimental and has not replaced V1."
            )

                # ============================================================
        # NFL PLAYER PROP V1 VS V2A BENCHMARK
        # ============================================================

        st.divider()

        st.subheader(
            "🏈 NFL Player Prop V1 vs V2A Benchmark"
        )

        st.caption(
            "Leakage-safe historical comparison of the existing "
            "mean player-prop projection against the experimental "
            "V2A recency-weighted projection."
        )

        st.info(
            "This benchmark does not change the live Player Prop "
            "engine. Lower MAE and RMSE are better. Bias closer "
            "to zero is better."
        )

        if st.button(
            "Run NFL Player Prop V1 vs V2A Benchmark",
            key="run_nfl_prop_v1_v2a_benchmark",
            type="primary",
        ):

            try:

                # ----------------------------------------------------
                # LOAD PLAYER HISTORY
                # ----------------------------------------------------

                with st.spinner(
                    "Loading historical NFL player data..."
                ):

                    current_nfl_season = pd.Timestamp.now().year

                    prop_history = load_player_history(
                        [
                            current_nfl_season - 2,
                            current_nfl_season - 1,
                            current_nfl_season,
                        ]
                    )

                if (
                    prop_history is None
                    or prop_history.empty
                ):

                    raise ValueError(
                        "NFL player history returned no data."
                    )

                # ----------------------------------------------------
                # RUN PAIRED WALK-FORWARD TEST
                # ----------------------------------------------------

                with st.spinner(
                    "Running NFL player-prop V1 vs V2A "
                    "walk-forward benchmark..."
                ):

                    prop_benchmark = (
                        compare_v1_v2_walkforward(
                            history=prop_history,
                            window=12,
                            min_games=6,
                            decay=0.88,
                        )
                    )

                if (
                    prop_benchmark is None
                    or prop_benchmark.get(
                        "comparison"
                    ) is None
                    or prop_benchmark[
                        "comparison"
                    ].empty
                ):

                    raise ValueError(
                        "Player-prop benchmark produced "
                        "no comparison results."
                    )

                st.session_state[
                    "nfl_prop_v1_v2a_benchmark"
                ] = prop_benchmark

                st.success(
                    "NFL player-prop V1 vs V2A "
                    "benchmark completed."
                )

            except Exception as error:

                st.error(
                    "NFL player-prop benchmark failed."
                )

                st.exception(
                    error
                )

        # ============================================================
        # DISPLAY SAVED PLAYER PROP BENCHMARK
        # ============================================================

        prop_benchmark = st.session_state.get(
            "nfl_prop_v1_v2a_benchmark"
        )

        if prop_benchmark:

            prop_v1 = prop_benchmark.get(
                "v1"
            )

            prop_v2 = prop_benchmark.get(
                "v2"
            )

            if (
                prop_v1 is not None
                and prop_v2 is not None
            ):

                st.markdown(
                    "### 📊 Player Prop Projection Comparison"
                )

                (
                    prop_v1_col,
                    prop_v2_col,
                ) = st.columns(2)

                # ----------------------------------------------------
                # V1
                # ----------------------------------------------------

                with prop_v1_col:

                    st.markdown(
                        "#### V1 — Mean Baseline"
                    )

                    st.metric(
                        "Predictions",
                        f"{prop_v1['prediction_count']:,}",
                    )

                    st.metric(
                        "MAE",
                        f"{prop_v1['mae']:.3f}",
                    )

                    st.metric(
                        "RMSE",
                        f"{prop_v1['rmse']:.3f}",
                    )

                    st.metric(
                        "Bias",
                        f"{prop_v1['bias']:+.3f}",
                    )

                # ----------------------------------------------------
                # V2A
                # ----------------------------------------------------

                with prop_v2_col:

                    st.markdown(
                        "#### V2A — Recency Weighted"
                    )

                    st.metric(
                        "Predictions",
                        f"{prop_v2['prediction_count']:,}",
                    )

                    st.metric(
                        "MAE",
                        f"{prop_v2['mae']:.3f}",
                    )

                    st.metric(
                        "RMSE",
                        f"{prop_v2['rmse']:.3f}",
                    )

                    st.metric(
                        "Bias",
                        f"{prop_v2['bias']:+.3f}",
                    )

                # ----------------------------------------------------
                # CHANGE
                # ----------------------------------------------------

                st.markdown(
                    "### 🔬 V2A Change vs V1"
                )

                (
                    prop_change_1,
                    prop_change_2,
                ) = st.columns(2)

                prop_mae_change = (
                    prop_benchmark.get(
                        "mae_change"
                    )
                )

                prop_rmse_change = (
                    prop_benchmark.get(
                        "rmse_change"
                    )
                )

                if prop_mae_change is not None:

                    prop_change_1.metric(
                        "MAE Change",
                        f"{prop_mae_change:+.3f}",
                    )

                else:

                    prop_change_1.metric(
                        "MAE Change",
                        "—",
                    )

                if prop_rmse_change is not None:

                    prop_change_2.metric(
                        "RMSE Change",
                        f"{prop_rmse_change:+.3f}",
                    )

                else:

                    prop_change_2.metric(
                        "RMSE Change",
                        "—",
                    )

                # ----------------------------------------------------
                # FULL OVERALL COMPARISON
                # ----------------------------------------------------

                st.markdown(
                    "### 📋 Overall Benchmark"
                )

                prop_comparison_df = (
                    prop_benchmark[
                        "comparison"
                    ].copy()
                )

                st.dataframe(
                    prop_comparison_df,
                    use_container_width=True,
                    hide_index=True,
                )

                # ----------------------------------------------------
                # MARKET-BY-MARKET ANALYSIS
                # ----------------------------------------------------

                paired_prop_predictions = (
                    prop_benchmark.get(
                        "paired_predictions"
                    )
                )

                if (
                    paired_prop_predictions
                    is not None
                    and not paired_prop_predictions.empty
                ):

                    market_rows = []

                    for (
                        prop_market,
                        prop_market_df,
                    ) in paired_prop_predictions.groupby(
                        "market"
                    ):

                        v1_market_mae = (
                            prop_market_df[
                                "v1_absolute_error"
                            ].mean()
                        )

                        v2_market_mae = (
                            prop_market_df[
                                "v2_absolute_error"
                            ].mean()
                        )

                        v1_market_rmse = (
                            np.sqrt(
                                prop_market_df[
                                    "v1_squared_error"
                                ].mean()
                            )
                        )

                        v2_market_rmse = (
                            np.sqrt(
                                prop_market_df[
                                    "v2_squared_error"
                                ].mean()
                            )
                        )

                        v1_market_bias = (
                            prop_market_df[
                                "v1_error"
                            ].mean()
                        )

                        v2_market_bias = (
                            prop_market_df[
                                "v2_error"
                            ].mean()
                        )

                        market_rows.append(
                            {
                                "Market":
                                    prop_market,

                                "Predictions":
                                    len(
                                        prop_market_df
                                    ),

                                "V1 MAE":
                                    round(
                                        v1_market_mae,
                                        3,
                                    ),

                                "V2A MAE":
                                    round(
                                        v2_market_mae,
                                        3,
                                    ),

                                "MAE Change":
                                    round(
                                        v2_market_mae
                                        - v1_market_mae,
                                        3,
                                    ),

                                "V1 RMSE":
                                    round(
                                        v1_market_rmse,
                                        3,
                                    ),

                                "V2A RMSE":
                                    round(
                                        v2_market_rmse,
                                        3,
                                    ),

                                "RMSE Change":
                                    round(
                                        v2_market_rmse
                                        - v1_market_rmse,
                                        3,
                                    ),

                                "V1 Bias":
                                    round(
                                        v1_market_bias,
                                        3,
                                    ),

                                "V2A Bias":
                                    round(
                                        v2_market_bias,
                                        3,
                                    ),
                            }
                        )

                    market_comparison_df = (
                        pd.DataFrame(
                            market_rows
                        )
                    )

                    if not market_comparison_df.empty:

                        market_comparison_df = (
                            market_comparison_df
                            .sort_values(
                                "MAE Change"
                            )
                            .reset_index(
                                drop=True
                            )
                        )

                        st.markdown(
                            "### 🧬 Performance by Prop Market"
                        )

                        st.caption(
                            "Negative MAE/RMSE change means "
                            "V2A improved over V1 for that "
                            "specific market."
                        )

                        st.dataframe(
                            market_comparison_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # --------------------------------------------
                        # MARKET SUMMARY
                        # --------------------------------------------

                        improved_markets = (
                            market_comparison_df[
                                market_comparison_df[
                                    "MAE Change"
                                ] < 0
                            ]
                        )

                        worse_markets = (
                            market_comparison_df[
                                market_comparison_df[
                                    "MAE Change"
                                ] > 0
                            ]
                        )

                        (
                            improved_col,
                            worse_col,
                        ) = st.columns(2)

                        improved_col.metric(
                            "Markets Improved",
                            len(
                                improved_markets
                            ),
                        )

                        worse_col.metric(
                            "Markets Worse",
                            len(
                                worse_markets
                            ),
                        )

                st.caption(
                    "This test evaluates statistical projection "
                    "accuracy only. It does not yet measure "
                    "historical sportsbook ROI because historical "
                    "prop lines and prices are not currently part "
                    "of the player-history dataset."
                )

# ==========================================
# NFL PLAYER PROP V1 VS V2A VS V2B BENCHMARK
# ==========================================

st.divider()

st.subheader(
    "🏈 NFL Player Prop V1 vs V2A vs V2B Benchmark"
)

st.caption(
    "Leakage-safe historical comparison of the mean baseline, "
    "recency-weighted model, and opponent matchup-adjusted model."
)

st.info(
    "V2B adds opponent defensive context to V2A. "
    "Lower MAE and RMSE are better. "
    "Bias closer to zero is better."
)

if st.button(
    "Run NFL Player Prop V2B Benchmark",
    key="run_nfl_prop_v2b_benchmark",
):

    with st.spinner(
        "Running V1 vs V2A vs V2B walk-forward benchmark..."
    ):

        current_nfl_season = pd.Timestamp.now().year

        prop_history_v2b = load_player_history(
            [
                current_nfl_season - 2,
                current_nfl_season - 1,
                current_nfl_season,
            ]
        )

        v2b_benchmark = (
            compare_v1_v2a_v2b_walkforward(
                history=prop_history_v2b,
                window=12,
                min_games=6,
                decay=0.88,
            )
        )

        st.session_state[
            "nfl_prop_v2b_benchmark"
        ] = v2b_benchmark

        st.success(
            "NFL Player Prop V2B benchmark completed."
        )


v2b_benchmark = st.session_state.get(
    "nfl_prop_v2b_benchmark"
)

if v2b_benchmark:

    v1 = v2b_benchmark.get("v1")
    v2a = v2b_benchmark.get("v2a")
    v2b = v2b_benchmark.get("v2b")

    if (
        v1 is not None
        and v2a is not None
        and v2b is not None
    ):

        st.subheader(
            "📊 V1 vs V2A vs V2B"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                "### V1 — Mean Baseline"
            )

            st.metric(
                "Predictions",
                f"{v1['prediction_count']:,}",
            )

            st.metric(
                "MAE",
                f"{v1['mae']:.3f}",
            )

            st.metric(
                "RMSE",
                f"{v1['rmse']:.3f}",
            )

            st.metric(
                "Bias",
                f"{v1['bias']:+.3f}",
            )

        with col2:

            st.markdown(
                "### V2A — Recency Weighted"
            )

            st.metric(
                "Predictions",
                f"{v2a['prediction_count']:,}",
            )

            st.metric(
                "MAE",
                f"{v2a['mae']:.3f}",
            )

            st.metric(
                "RMSE",
                f"{v2a['rmse']:.3f}",
            )

            st.metric(
                "Bias",
                f"{v2a['bias']:+.3f}",
            )

        with col3:

            st.markdown(
                "### V2B — Matchup Adjusted"
            )

            st.metric(
                "Predictions",
                f"{v2b['prediction_count']:,}",
            )

            st.metric(
                "MAE",
                f"{v2b['mae']:.3f}",
            )

            st.metric(
                "RMSE",
                f"{v2b['rmse']:.3f}",
            )

            st.metric(
                "Bias",
                f"{v2b['bias']:+.3f}",
            )

        st.subheader(
            "🔬 V2B Change"
        )

        change_col1, change_col2 = (
            st.columns(2)
        )

        with change_col1:

            st.metric(
                "MAE vs V2A",
                (
                    f"{v2b_benchmark['v2b_vs_v2a_mae_change']:+.3f}"
                ),
            )

            st.metric(
                "MAE vs V1",
                (
                    f"{v2b_benchmark['v2b_vs_v1_mae_change']:+.3f}"
                ),
            )

        with change_col2:

            st.metric(
                "RMSE vs V2A",
                (
                    f"{v2b_benchmark['v2b_vs_v2a_rmse_change']:+.3f}"
                ),
            )

            st.metric(
                "RMSE vs V1",
                (
                    f"{v2b_benchmark['v2b_vs_v1_rmse_change']:+.3f}"
                ),
            )

        st.metric(
            "Matchup Coverage",
            (
                f"{v2b_benchmark['matchup_coverage'] * 100:.1f}%"
            ),
        )

        st.subheader(
            "📋 Overall Benchmark"
        )

        comparison = (
            v2b_benchmark[
                "comparison"
            ].copy()
        )

        numeric_columns = [
            "mae",
            "rmse",
            "bias",
            "average_forecast",
        ]

        for column in numeric_columns:

            if column in comparison.columns:

                comparison[column] = (
                    comparison[column]
                    .round(4)
                )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        paired = (
            v2b_benchmark[
                "paired_predictions"
            ].copy()
        )

        if not paired.empty:

            st.subheader(
                "🧬 Performance by Prop Market"
            )

            market_rows = []

            for (
                market,
                market_data,
            ) in paired.groupby(
                "market"
            ):

                market_rows.append(
                    {
                        "Market":
                            market,

                        "Predictions":
                            len(
                                market_data
                            ),

                        "V1 MAE":
                            market_data[
                                "v1_absolute_error"
                            ].mean(),

                        "V2A MAE":
                            market_data[
                                "v2a_absolute_error"
                            ].mean(),

                        "V2B MAE":
                            market_data[
                                "v2b_absolute_error"
                            ].mean(),

                        "V2B vs V2A MAE":
                            (
                                market_data[
                                    "v2b_absolute_error"
                                ].mean()
                                - market_data[
                                    "v2a_absolute_error"
                                ].mean()
                            ),

                        "V1 RMSE":
                            np.sqrt(
                                market_data[
                                    "v1_squared_error"
                                ].mean()
                            ),

                        "V2A RMSE":
                            np.sqrt(
                                market_data[
                                    "v2a_squared_error"
                                ].mean()
                            ),

                        "V2B RMSE":
                            np.sqrt(
                                market_data[
                                    "v2b_squared_error"
                                ].mean()
                            ),

                        "V2B vs V2A RMSE":
                            (
                                np.sqrt(
                                    market_data[
                                        "v2b_squared_error"
                                    ].mean()
                                )
                                - np.sqrt(
                                    market_data[
                                        "v2a_squared_error"
                                    ].mean()
                                )
                            ),

                        "Matchup Coverage":
                            market_data[
                                "matchup_available"
                            ].mean(),
                    }
                )

            market_table = pd.DataFrame(
                market_rows
            )

            market_table[
                "Matchup Coverage"
            ] = (
                market_table[
                    "Matchup Coverage"
                ]
                * 100.0
            )

            numeric_market_columns = [
                "V1 MAE",
                "V2A MAE",
                "V2B MAE",
                "V2B vs V2A MAE",
                "V1 RMSE",
                "V2A RMSE",
                "V2B RMSE",
                "V2B vs V2A RMSE",
                "Matchup Coverage",
            ]

            market_table[
                numeric_market_columns
            ] = (
                market_table[
                    numeric_market_columns
                ].round(3)
            )

            market_table = (
                market_table
                .sort_values(
                    "V2B vs V2A MAE"
                )
                .reset_index(
                    drop=True
                )
            )

            st.dataframe(
                market_table,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Negative V2B-vs-V2A MAE/RMSE changes mean "
            "the matchup adjustment improved projection accuracy. "
            "This remains a projection benchmark, not a historical "
            "sportsbook ROI test."
        )

        # ==========================================
        # STOCK MODEL TRAINING CENTER
        # ==========================================

        st.divider()
        st.subheader("📈 Stock Model Training Center")

        st.caption(
            "Train and evaluate historical stock predictions "
            "using Alpaca market data. No trades are placed."
        )

        stock_symbols = st.text_input(
            "Stocks to train",
            value="AAPL,MSFT,NVDA,AMZN,META",
            key="stock_training_symbols",
        )

        symbols = [
            symbol.strip().upper()
            for symbol in stock_symbols.split(",")
            if symbol.strip()
        ]

        st.write(
            "Prediction horizons: 1, 5, and 20 trading days."
        )

        if st.button(
            "Train Stock Prediction Models",
            key="train_stock_models_button",
            type="primary",
        ):

            if not symbols:
                st.warning(
                    "Enter at least one stock symbol."
                )

            else:
                try:
                    with st.spinner(
                        "Downloading historical data and "
                        "training stock prediction models..."
                    ):

                        results = train_stock_models(symbols)

                    st.session_state[
                        "stock_training_results"
                    ] = results

                    st.success(
                        "Stock model training completed."
                    )

                except Exception as e:
                    st.error(
                        f"Stock model training failed: {e}"
                    )

        results = st.session_state.get(
            "stock_training_results"
        )

        
        if results:

            st.subheader("📊 Stock Model Performance")

            st.caption(
                "Historical out-of-sample validation. "
                "Results do not guarantee future performance."
            )

            for horizon, metrics in results.items():

                st.divider()

                st.subheader(
                    f"📈 {horizon}-Day Prediction Model"
                )

                passed = metrics.get(
                    "passes_benchmarks", False
                )

                if passed:
                    st.success(
                        "Validation benchmarks passed."
                    )
                else:
                    st.error(
                        "Validation benchmarks not passed. "
                        "Model requires further evaluation."
                    )

                auc = metrics.get("auc")
                test_rows = metrics.get("test_rows", 0)

                model_brier = metrics.get("model_brier")
                baseline_brier = metrics.get(
                    "baseline_brier"
                )

                model_log_loss = metrics.get(
                    "model_log_loss"
                )

                baseline_log_loss = metrics.get(
                    "baseline_log_loss"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "AUC Score",
                        f"{auc:.3f}"
                        if auc is not None else "N/A"
                    )

                    st.metric(
                        "Model Brier Score",
                        f"{model_brier:.4f}"
                        if model_brier is not None
                        else "N/A",
                        delta=(
                            f"{baseline_brier - model_brier:+.4f}"
                            if model_brier is not None
                            and baseline_brier is not None
                            else None
                        ),
                        delta_color="normal",
                    )

                with col2:
                    st.metric(
                        "Test Observations",
                        f"{test_rows:,}"
                    )

                    st.metric(
                        "Model Log Loss",
                        f"{model_log_loss:.4f}"
                        if model_log_loss is not None
                        else "N/A",
                        delta=(
                            f"{baseline_log_loss - model_log_loss:+.4f}"
                            if model_log_loss is not None
                            and baseline_log_loss is not None
                            else None
                        ),
                        delta_color="normal",
                    )


                # -----------------------------------------
                # Prediction Diagnostics
                # -----------------------------------------

                st.markdown("### Prediction Diagnostics")

                diagnostics = metrics.get(
                    "prediction_diagnostics"
                )

                if diagnostics is not None:
                    st.json(diagnostics)
                else:
                    st.warning(
                        "Prediction diagnostics are not available. "
                        "Retrain the stock models to generate them."
                    )

                # -----------------------------------------
                # Historical Data Summary
                # -----------------------------------------

                st.markdown(
                    "#### Historical Data Summary"
                )

                col3, col4, col5 = st.columns(3)

                with col3:
                    st.metric(
                        "Training Observations",
                        f"{metrics.get('train_rows', 0):,}"
                    )

                with col4:
                    st.metric(
                        "Calibration Observations",
                        f"{metrics.get('calibration_rows', 0):,}"
                    )

                with col5:
                    base_rate = metrics.get("base_rate")

                    st.metric(
                        "Positive Outcome Rate",
                        f"{base_rate:.2%}"
                        if base_rate is not None
                        else "N/A"
                    )

                with st.expander(
                    "View Detailed Validation Data"
                ):
                    st.json(metrics)

        # -----------------------------------------
        # Saved Stock Model Results
        # -----------------------------------------

        st.divider()

        if st.button(
            "View Saved Stock Model Results",
            key="view_stock_model_results",
        ):
            try:
                saved_metrics = stock_model_metrics()

                if saved_metrics:
                    st.json(saved_metrics)
                else:
                    st.info(
                        "No saved stock model results found."
                    )

            except Exception as e:
                st.error(
                    f"Unable to load model results: {e}"
                )

        # ==========================================
        # EXISTING NBA RESEARCH CONTINUES BELOW
        # ==========================================
        st.subheader("🧪 Research Lab – Optional Revalidation")

        st.markdown("### 🏀 NBA Data Connection Test")

        if st.button("Test NBA Data Sources"):
            with st.spinner("Testing SportsDataIO and The Odds API..."):
                nba_status = test_nba_connections()

            for provider, info in nba_status.items():
                if info["working"]:
                    st.success(f"✅ {provider}: {info['message']}")
                elif info["configured"]:
                    st.error(f"❌ {provider}: {info['message']}")
                else:
                    st.warning(f"⚠️ {provider}: {info['message']}")

        st.divider()
        # Betting performance


    # ============================================================
    # MULTI-SEASON NBA VALIDATION
    # ============================================================

        st.markdown("### 🏀 BALLDONTLIE Historical Data Test")

        if st.button("Test BALLDONTLIE Historical Data"):
            try:
                with st.spinner("Downloading 2023 NBA games from BALLDONTLIE..."):
                    bdl_test = test_balldontlie_connection()

                st.success(
                    f"BALLDONTLIE connection successful — "
                    f"{bdl_test['games_returned']} historical games returned."
                )

                st.write(
                    "Season:",
                    bdl_test["season"],
                )

                st.write(
                    "Date range:",
                    bdl_test["first_game"],
                    "to",
                    bdl_test["last_game"],
                )

                st.dataframe(
                    bdl_test["sample"],
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as e:
                st.error(
                    f"BALLDONTLIE historical data error: {e}"
                )
        if st.button("Download Full 2023 NBA Season"):
            try:
                with st.spinner(
                    "Downloading the full 2023 NBA season. "
                    "This may take a couple of minutes..."
                ):
                    full_season_test = test_full_historical_season(2023)

                st.success(
                    f"Full season downloaded — "
                    f"{full_season_test['games_returned']} games."
                )

                st.write(
                    "API requests used:",
                    full_season_test["requests_used"],
                )

                st.write(
                    "Date range:",
                    full_season_test["first_game"],
                    "to",
                    full_season_test["last_game"],
                )

                st.write(
                    "Home win rate:",
                    f"{full_season_test['home_win_rate']:.1%}",
                )

                st.dataframe(
                    full_season_test["sample"],
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as e:
                st.error(
                    f"Full-season download error: {e}"
                )



        if st.button("Test 2022 + 2023 NBA Seasons"):
            try:
                with st.spinner(
                    "Downloading 2022 and 2023 NBA seasons. "
                    "This will take several minutes..."
                ):
                    multi_bdl_test = test_multiple_historical_seasons(
                        [2022, 2023, 2024, 2025]
                    )

                st.success(
                    f"Multi-season download successful - "
                    f"{multi_bdl_test['games_returned']} total games."
                )

                st.write(
                    "API requests used:",
                    multi_bdl_test["total_requests"],
                )

                st.write(
                    "Overall date range:",
                    multi_bdl_test["first_game"],
                    "to",
                    multi_bdl_test["last_game"],
                )

                st.write(
                    "Overall home win rate:",
                    f"{multi_bdl_test['home_win_rate']:.1%}",
                )

                st.markdown("#### Season Summary")

                st.dataframe(
                    multi_bdl_test["season_summary"],
                    use_container_width=True,
                    hide_index=True,
                )

                # ==================================================
                # RESEARCH PIPELINE BRIDGE
                # ==================================================

                st.markdown("#### 🧠 Research Pipeline Bridge Test")

                research_games = prepare_balldontlie_games_for_research(
                    multi_bdl_test["games"]
                )

                st.success(
                    f"Research bridge successful - "
                    f"{len(research_games)} games ready for modeling."
                )

                bridge_col1, bridge_col2, bridge_col3 = st.columns(3)

                bridge_col1.metric(
                    "Model-Ready Games",
                    len(research_games),
                )

                bridge_col2.metric(
                    "First Game",
                    research_games["game_date"]
                    .min()
                    .strftime("%Y-%m-%d"),
                )

                bridge_col3.metric(
                    "Last Game",
                    research_games["game_date"]
                    .max()
                    .strftime("%Y-%m-%d"),
                )

                st.write(
                    "Verified home-win rate:",
                    f"{research_games['home_win'].mean():.1%}",
                )

                # ==================================================
                # PRE-GAME FEATURE ENGINE
                # ==================================================

                st.markdown("#### 🧮 Pre-Game Feature Engine Test")

                feature_games = build_balldontlie_pregame_features(
                    multi_bdl_test["games"],
                    min_games=5,
                )

                st.success(
                    f"Feature engine successful - "
                    f"{len(feature_games)} model-ready rows created."
                )

                feature_col1, feature_col2, feature_col3 = st.columns(3)

                feature_col1.metric(
                    "Feature Rows",
                    len(feature_games),
                )

                feature_col2.metric(
                    "First Feature Date",
                    feature_games["game_date"]
                    .min()
                    .strftime("%Y-%m-%d"),
                )

                feature_col3.metric(
                    "Last Feature Date",
                    feature_games["game_date"]
                    .max()
                    .strftime("%Y-%m-%d"),
                )

                st.write(
                    "Feature columns:",
                    len(feature_games.columns),
                )

                st.dataframe(
                    feature_games.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                # ==================================================
                # LEAKAGE AUDIT
                # ==================================================

                st.markdown("#### 🔒 Pre-Game Leakage Audit")

                leakage_audit = audit_balldontlie_pregame_features(
                    feature_games
                )

                leak_col1, leak_col2, leak_col3 = st.columns(3)

                leak_col1.metric(
                    "Rows Audited",
                    leakage_audit["total_rows"],
                )

                leak_col2.metric(
                    "Model Features",
                    leakage_audit["model_feature_count"],
                )

                leak_col3.metric(
                    "Suspicious Features",
                    len(
                        leakage_audit[
                            "suspicious_model_features"
                        ]
                    ),
                )

                st.write(
                    "Model feature columns:",
                    leakage_audit["model_features"],
                )

                if leakage_audit["suspicious_model_features"]:
                    st.error(
                        "Possible leakage detected: "
                        + ", ".join(
                            leakage_audit[
                                "suspicious_model_features"
                            ]
                        )
                    )
                else:
                    st.success(
                        "No obvious current-game outcome leakage "
                        "detected in the proposed model features."
                    )

                if len(
                    leakage_audit["missing_feature_values"]
                ) > 0:
                    st.warning(
                        "Missing values found in model features."
                    )

                    st.dataframe(
                        leakage_audit[
                            "missing_feature_values"
                        ]
                        .rename("missing_values")
                        .reset_index()
                        .rename(
                            columns={
                                "index": "feature"
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success(
                        "No missing values found in model features."
                    )

                # ==================================================
                # WALK-FORWARD MODEL
                # ==================================================

                st.markdown("#### 🧠 Walk-Forward Model Test")

                walkforward_results = (
                    run_balldontlie_walkforward_model(
                        feature_games
                    )
                )

                st.success(
                    f"Walk-forward validation successful - "
                    f"{walkforward_results['games_predicted']} "
                    f"unseen games predicted."
                )
                st.markdown("#### 📊 Walk-Forward Model Performance")
    
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
                perf_col1.metric(
                    "Accuracy",
                    f"{walkforward_results['accuracy']:.1%}",
                )
    
                perf_col2.metric(
                    "AUC",
                    f"{walkforward_results['auc']:.3f}",
                )
    
                perf_col3.metric(
                    "Brier Score",
                    f"{walkforward_results['brier']:.4f}",
                )
    
                perf_col4.metric(
                    "Log Loss",
                    f"{walkforward_results['log_loss']:.4f}",
                )
    
                st.markdown("##### Baseline Comparison")
    
                base_col1, base_col2, base_col3 = st.columns(3)
    
                base_col1.metric(
                    "Baseline Accuracy",
                    f"{walkforward_results['baseline_accuracy']:.1%}",
                )
    
                base_col2.metric(
                    "Baseline Brier",
                    f"{walkforward_results['baseline_brier']:.4f}",
                )
    
                base_col3.metric(
                    "Baseline Log Loss",
                    f"{walkforward_results['baseline_log_loss']:.4f}",
                )
                st.markdown("##### 🎚️ Walk-Forward Confidence Calibration")
    
                calibration_predictions = (
                    walkforward_results["predictions"].copy()
                )
    
                calibration_predictions["confidence"] = (
                    calibration_predictions[
                        "probability"
                    ]
                    .where(
                        calibration_predictions["prediction"] == 1,
                        1.0
                        - calibration_predictions[
                            "probability"
                        ],
                    )
                )
    
                calibration_predictions["correct"] = (
                    calibration_predictions["prediction"]
                    == calibration_predictions["actual"]
                ).astype(int)
    
                calibration_predictions["confidence_band"] = pd.cut(
                    calibration_predictions["confidence"],
                    bins=[
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
                    ],
                    labels=[
                        "50–55%",
                        "55–60%",
                        "60–65%",
                        "65–70%",
                        "70–75%",
                        "75–80%",
                        "80–85%",
                        "85–90%",
                        "90–95%",
                        "95–100%",
                    ],
                    include_lowest=True,
                )
    
                calibration_table = (
                    calibration_predictions
                    .groupby(
                        "confidence_band",
                        observed=False,
                    )
                    .agg(
                        games=("correct", "size"),
                        actual_accuracy=("correct", "mean"),
                        average_confidence=("confidence", "mean"),
                    )
                    .reset_index()
                )
    
                calibration_table["actual_accuracy"] = (
                    calibration_table["actual_accuracy"]
                    .map(lambda value: f"{value:.1%}")
                )
    
                calibration_table["average_confidence"] = (
                    calibration_table["average_confidence"]
                    .map(lambda value: f"{value:.1%}")
                )
    
                st.dataframe(
                    calibration_table,
                    use_container_width=True,
                    hide_index=True,
                )

            
                st.markdown("#### 🔮 Future Matchup Feature Test")
    
                historical_test_game = feature_games.iloc[-1]
    
                future_test_features = (
                    build_balldontlie_future_matchup_features(
                        multi_bdl_test["games"],
                        historical_test_game["home_team"],
                        historical_test_game["away_team"],
                        historical_test_game["game_date"],
                    )
                )
    
                st.success(
                    "Future matchup feature builder successful."
                )
    
                future_col1, future_col2, future_col3 = st.columns(3)
    
                future_col1.metric(
                    "Home Team",
                    historical_test_game["home_team"],
                )
    
                future_col2.metric(
                    "Away Team",
                    historical_test_game["away_team"],
                )
    
                future_col3.metric(
                    "Feature Count",
                    len(future_test_features.columns),
                )
    
                st.write(
                    "Prediction date:",
                    historical_test_game["game_date"].strftime("%Y-%m-%d"),
                )
    
                st.dataframe(
                    future_test_features,
                    use_container_width=True,
                    hide_index=True,
                )

                st.write(
                    "DEBUG future columns:",
                    list(future_test_features.columns),
                )

                st.markdown("#### 🎯 NBA Matchup Prediction")
    
                live_prediction = predict_balldontlie_matchup(
                    feature_games,
                    future_test_features,
                )
    
                pred_col1, pred_col2, pred_col3 = st.columns(3)
    
                pred_col1.metric(
                    "Home Win Probability",
                    f"{live_prediction['home_win_probability']:.1%}",
                )
    
                pred_col2.metric(
                    "Away Win Probability",
                    f"{live_prediction['away_win_probability']:.1%}",
                )
    
                pred_col3.metric(
                    "Model Confidence",
                    f"{live_prediction['confidence']:.1%}",
                )
    
                st.write(
                    "Predicted side:",
                    live_prediction["predicted_side"],
                )
    
                st.caption(
                    f"Model trained on "
                    f"{live_prediction['training_games']} historical games "
                    f"using {live_prediction['feature_count']} features."
                )
                st.markdown("##### 🧪 No-Vig Edge Test")

                test_edge = calculate_no_vig_model_edge(
                    home_model_probability=0.71,
                    home_american_odds=-150,
                    away_american_odds=130,
                )
            
                edge_col1, edge_col2, edge_col3 = st.columns(3)
            
                with edge_col1:
                    st.metric(
                        "Model Home Probability",
                        f"{test_edge['home_model_probability']:.1%}",
                    )
            
                with edge_col2:
                    st.metric(
                        "No-Vig Market Probability",
                        f"{test_edge['home_market_probability']:.1%}",
                    )
            
                with edge_col3:
                    st.metric(
                        "Model Edge",
                        f"{test_edge['home_edge']:+.1%}",
                    )
            
                st.write(
                    f"Best model side: **{test_edge['best_side']}**"
                )
            
                st.write(
                    f"Sportsbook hold: "
                    f"**{test_edge['sportsbook_hold']:.2%}**"
                )
            
                # ==================================================
                # HISTORICAL DATASET AUDIT
                # ==================================================

                st.markdown("#### 🔎 Historical Dataset Audit")

                audit = audit_historical_games(
                    multi_bdl_test["games"]
                )

                audit_col1, audit_col2, audit_col3 = st.columns(3)

                audit_col1.metric(
                    "Total Rows",
                    audit["total_rows"],
                )

                audit_col2.metric(
                    "Unique Game IDs",
                    audit["unique_game_ids"],
                )

                audit_col3.metric(
                    "Duplicate Game IDs",
                    audit["duplicate_game_ids"],
                )

                audit_col4, audit_col5, audit_col6 = st.columns(3)

                audit_col4.metric(
                    "Missing Scores",
                    audit["missing_scores"],
                )

                audit_col5.metric(
                    "Tied Games",
                    audit["tied_games"],
                )

                audit_col6.metric(
                    "Teams Found",
                    audit["teams_found"],
                )

                st.write(
                    "Unusual teams:",
                    audit["unusual_teams"]
                    if audit["unusual_teams"]
                    else "None",
                )

                st.markdown("##### Games by Month")

                st.dataframe(
                    audit["games_by_month"],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("##### Game Status")

                st.dataframe(
                    audit["status_summary"],
                    use_container_width=True,
                    hide_index=True,
                )

                if len(audit["suspicious_scores"]):
                    st.warning(
                        f"{len(audit['suspicious_scores'])} "
                        f"games have suspicious scores."
                    )
                else:
                    st.success(
                        "No suspicious zero or missing scores found."
                    )

            except Exception as e:
                st.error(
                    f"Multi-season BALLDONTLIE error: {e}"
                )
    
        st.divider()
    


    
        st.markdown("### 🧪 NBA.com Historical Data Test")

        if st.button("Test NBA.com Historical Data"):
            try:
                with st.spinner("Downloading 2024-25 NBA games from NBA.com..."):
                    history_test = test_nba_history("2024-25")

                st.success(
                    f"NBA.com connection successful — "
                    f"{history_test['games']} games loaded."
                )

                st.write(
                    "Date range:",
                    history_test["first_game"],
                    "to",
                    history_test["last_game"],
                )

                st.dataframe(
                    history_test["sample"],
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as e:
                st.error(f"NBA.com historical data error: {e}")

        st.divider()



    st.markdown("### 🏀 Multi-Season NBA Validation")

    nba_seasons_text = st.text_input(
        "NBA seasons to validate",
        value="2022,2023,2024,2025",
        help="Enter SportsDataIO NBA seasons separated by commas.",
    )

    if st.button("Run Multi-Season NBA Validation"):

        seasons = [
            int(x.strip())
            for x in nba_seasons_text.split(",")
            if x.strip()
        ]

        try:
            with st.spinner(
                "Downloading multiple NBA seasons and running walk-forward validation..."
            ):
                multi_bdl_result = test_multiple_historical_seasons(seasons)

                historical_games = multi_bdl_result["games"]

                prepared_games = prepare_balldontlie_games_for_research(
                    historical_games
                )

                feature_games = build_balldontlie_pregame_features(
                    prepared_games
                )

                multi_nba_result = run_balldontlie_walkforward_model(
                    feature_games
                )

                st.session_state["multi_nba_research_result"] = multi_nba_result

            st.success("Multi-season NBA validation complete.")

        except Exception as e:
            st.error(f"Multi-season NBA validation error: {e}")
    if "multi_nba_research_result" in st.session_state:

        mr = st.session_state["multi_nba_research_result"]

        st.markdown("#### 🏀 Multi-Season Dataset")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Games predicted",
            mr["games_predicted"],
        )

        m2.metric(
            "Accuracy",
            f"{mr['accuracy']:.1%}",
        )

        m3.metric(
            "AUC",
            f"{mr['auc']:.3f}",
        )

        m4.metric(
            "Features",
            mr["feature_count"],
        )

        st.markdown("#### 📊 Walk-Forward Performance")

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Accuracy",
            f"{mr['accuracy']:.1%}",
        )

        p2.metric(
            "AUC",
            f"{mr['auc']:.3f}",
        )

        p3.metric(
            "Brier Score",
            f"{mr['brier']:.4f}",
        )

        p4.metric(
            "Log Loss",
            f"{mr['log_loss']:.4f}",
        )

        st.markdown("#### ⚖️ Model vs Baseline")

        comparison_df = pd.DataFrame(
            [
                {
                    "Model": "NBA Walk-Forward Model",
                    "Accuracy": mr["accuracy"],
                    "Brier Score": mr["brier"],
                    "Log Loss": mr["log_loss"],
                },
                {
                    "Model": "Home Win Baseline",
                    "Accuracy": mr["baseline_accuracy"],
                    "Brier Score": mr["baseline_brier"],
                    "Log Loss": mr["baseline_log_loss"],
                },
            ]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 🎯 Probability Calibration")

        calibration_result = nba_calibration_summary(
            mr["predictions"]
        )

        calibration_table = calibration_result["calibration"]

        if len(calibration_table):

            calibration_display = calibration_table.copy()

            for column in [
                "avg_predicted_probability",
                "actual_win_rate",
                "calibration_gap",
                "absolute_calibration_error",
            ]:
                if column in calibration_display.columns:
                    calibration_display[column] = (
                        calibration_display[column]
                        .map(lambda x: f"{x:.1%}")
                    )

            st.dataframe(
                calibration_display,
                use_container_width=True,
                hide_index=True,
            )

            weighted_error = calibration_result.get(
                "weighted_calibration_error"
            )

            if weighted_error is not None:
                st.metric(
                    "Weighted Calibration Error",
                    f"{weighted_error:.1%}",
                )

        else:
            st.info(
                "No calibration results were generated."
            )

        st.markdown("#### 🔬 High-Confidence Threshold Analysis")

        high_confidence = calibration_result.get(
            "high_confidence"
        )

        if (
            high_confidence is not None
            and len(high_confidence)
        ):
            threshold_display = high_confidence.copy()

            for column in [
                "minimum_probability",
                "accuracy",
                "avg_model_probability",
                "calibration_gap",
            ]:
                if column in threshold_display.columns:
                    threshold_display[column] = (
                        threshold_display[column]
                        .map(lambda x: f"{x:.1%}")
                    )

            st.dataframe(
                threshold_display,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No high-confidence threshold results were generated."
            )

        st.divider()

        st.markdown("### 🏀 NBA Historical Validation")

        nba_season = st.text_input(
            "NBA season",
            value="2025",
            help="SportsDataIO season to use for historical NBA research.",
            key="single_nba_season",
        )

        if st.button("Run NBA Historical Validation", key="run_single_nba_validation"):
            try:
                with st.spinner(
                    "Building historical NBA features and running walk-forward validation..."
                ):
                    nba_result = run_nba_research(
                        nba_season,
                        minimum_training_games=250,
                        test_block_size=100,
                    )

                st.session_state["nba_research_result"] = nba_result
                st.success("NBA historical validation complete.")

            except Exception as e:
                st.error(f"NBA validation error: {e}")

        if "nba_research_result" in st.session_state:
            r = st.session_state["nba_research_result"]

            st.markdown("#### Dataset")

            n1, n2, n3, n4 = st.columns(4)

            n1.metric("Raw games", r["raw_games"])
            n2.metric("Completed games", r["completed_games"])
            n3.metric("Feature rows", r["feature_rows"])
            n4.metric("Home win rate", f"{r['home_win_rate']:.1%}")

            st.markdown("#### Walk-Forward Performance")

            model = r["overall"]
            baseline = r["baseline"]

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("AUC", f"{model['auc']:.3f}")
            m2.metric("Accuracy", f"{model['accuracy']:.1%}")
            m3.metric("Brier Score", f"{model['brier']:.4f}")
            m4.metric("Log Loss", f"{model['logloss']:.4f}")

            st.markdown("#### Model vs Baseline")

            comparison = pd.DataFrame(
                [
                    {
                        "Model": "NBA Model",
                        "Accuracy": model["accuracy"],
                        "Brier": model["brier"],
                        "Log Loss": model["logloss"],
                        "AUC": model["auc"],
                    },
                    {
                        "Model": "Home Win Base Rate",
                        "Accuracy": baseline["accuracy"],
                        "Brier": baseline["brier"],
                        "Log Loss": baseline["logloss"],
                        "AUC": baseline["auc"],
                    },
                ]
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Confidence Analysis")

            confidence = r["confidence"]

            if len(confidence):
                st.dataframe(
                    confidence,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No confidence-band results were generated.")

            with st.expander("Walk-Forward Folds"):
                st.dataframe(
                    r["folds"],
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("### 🏀 NBA Betting Center")

    st.caption(
        "Live NBA moneylines, sportsbook odds, "
        "and potential betting payouts."
    )

    try:
        live_moneylines = get_live_nba_moneylines()

        if live_moneylines:

            moneyline_df = pd.DataFrame(live_moneylines)

            moneyline_df["home_team_code"] = (
                moneyline_df["home_team"].apply(
                    normalize_nba_team_name
                )
            )

            moneyline_df["away_team_code"] = (
                moneyline_df["away_team"].apply(
                    normalize_nba_team_name
                )
            )

            unmapped_home = moneyline_df[
                moneyline_df["home_team_code"].isna()
            ]["home_team"].unique()

            unmapped_away = moneyline_df[
                moneyline_df["away_team_code"].isna()
            ]["away_team"].unique()

            unmapped_teams = sorted(
                set(unmapped_home) | set(unmapped_away)
            )

            if unmapped_teams:
                st.warning(
                    "Unmapped NBA teams: "
                    + ", ".join(unmapped_teams)
                )
            else:
                st.success(
                    "NBA team mapping successful — "
                    "all live teams recognized."
                )

            st.success(
                f"Live NBA odds retrieved — "
                f"{len(moneyline_df)} sportsbook lines found."
            )

            st.subheader("Today's NBA Games")

            st.dataframe(
                moneyline_df,
                width="stretch",
                hide_index=True,
            )

            st.divider()

            st.subheader("NBA Betting Payout Calculator")

            st.caption(
                "Select a sportsbook line to calculate "
                "your potential return."
            )

            selected_index = st.selectbox(
                "Select NBA Game",
                options=list(moneyline_df.index),
                format_func=lambda i: (
                    f"{moneyline_df.loc[i, 'away_team']} "
                    f"vs {moneyline_df.loc[i, 'home_team']}"
                ),
            )

            selected_game = moneyline_df.loc[selected_index]


            
            # ========================================
            # RETRIEVE SELECTED NBA GAME INFORMATION
            # ========================================

            selected_game = moneyline_df.loc[selected_index]

            # Retrieve team names
            home_team = str(selected_game["home_team"])
            away_team = str(selected_game["away_team"])

            # Retrieve sportsbook information
            sportsbook = str(
                selected_game.get("bookmaker", "Sportsbook")
            )

            # Retrieve American betting odds
            home_odds = selected_game.get("home_odds")
            away_odds = selected_game.get("away_odds")

            # Format American betting odds
            def format_moneyline(odds):

                if pd.isna(odds):
                    return "N/A"

                odds = int(odds)

                return f"{odds:+d}"

            home_odds_display = format_moneyline(home_odds)
            away_odds_display = format_moneyline(away_odds)


            # ========================================
            # MARKET EDGE AI - NBA MATCHUP CARD
            # ========================================

            st.markdown("### NBA Moneyline")

            away_col, home_col = st.columns(
                2,
                gap="medium"
            )

            # ========================================
            # AWAY TEAM
            # ========================================

            with away_col:
                with st.container(border=True):

                    st.caption("AWAY TEAM")

                    st.subheader(away_team)

                    st.metric(
                        label="Moneyline Odds",
                        value=away_odds_display,
                    )

            # ========================================
            # HOME TEAM
            # ========================================

            with home_col:
                with st.container(border=True):

                    st.caption("HOME TEAM")

                    st.subheader(home_team)

                    st.metric(
                        label="Moneyline Odds",
                        value=home_odds_display,
                    )

            # ========================================
            # TECHNICAL GAME DATA
            # ========================================

            

            # ==========================================
            # MARKET EDGE AI - GAME INFORMATION PANEL
            # ==========================================

            with st.expander(
                "Game Information",
                expanded=True,
            ):

                game_info = selected_game

                # Retrieve game information
                event_id = str(
                    game_info.get("event_id", "N/A")
                )

                game_date = pd.to_datetime(
                    game_info.get("commence_time"),
                    utc=True,
                    errors="coerce",
                )

                if pd.notna(game_date):

                    game_date = game_date.tz_convert(
                        "America/Chicago"
                    )

                    date_display = game_date.strftime(
                        "%B %d, %Y"
                    )

                    time_display = game_date.strftime(
                        "%I:%M %p %Z"
                    )

                else:

                    date_display = "Not available"
                    time_display = "Not available"

                sportsbook_name = str(
                    game_info.get("bookmaker", "N/A")
                )

                home_code = str(
                    game_info.get("home_team_code", "N/A")
                )

                away_code = str(
                    game_info.get("away_team_code", "N/A")
                )

                # ======================================
                # GAME DETAILS
                # ======================================

                st.markdown("### Game Details")

                col1, col2 = st.columns(2)

                with col1:

                    st.caption("GAME DATE")
                    st.markdown(f"**{date_display}**")

                    st.caption("AWAY TEAM")
                    st.markdown(
                        f"**{away_team} ({away_code})**"
                    )

                with col2:

                    st.caption("GAME TIME")
                    st.markdown(f"**{time_display}**")

                    st.caption("HOME TEAM")
                    st.markdown(
                        f"**{home_team} ({home_code})**"
                    )

                st.divider()

                # ======================================
                # SPORTSBOOK INFORMATION
                # ======================================

                st.markdown(
                    "### Sportsbook Information"
                )

                col3, col4 = st.columns(2)

                with col3:

                    st.caption("SPORTSBOOK")
                    st.markdown(
                        f"**{sportsbook_name}**"
                    )

                with col4:

                    st.caption("MARKET")
                    st.markdown("**NBA Moneyline**")

                st.divider()

                st.caption("EVENT ID")
                st.code(
                    event_id,
                    language=None,
                )
    
            st.divider()

            # ========================================
            # NBA PAYOUT CALCULATOR
            # ========================================

            st.subheader("Calculate Potential Payout")

            wager_amount = st.number_input(
                "Wager Amount ($)",
                min_value=1.0,
                value=10.0,
                step=5.0,
                key="nba_wager_amount",
            )

            american_odds = st.number_input(
                "American Odds",
                value=100,
                step=10,
                key="nba_american_odds",
                help=(
                    "Enter the odds shown by your sportsbook. "
                    "For example, +120 or -150."
                ),
            )

            if american_odds == 0:

                st.error(
                    "American odds cannot be zero."
                )

            else:

                if american_odds > 0:

                    potential_profit = (
                        wager_amount
                        * american_odds / 100
                    )

                else:

                    potential_profit = (
                        wager_amount
                        * 100 / abs(american_odds)
                    )

                total_payout = (
                    wager_amount + potential_profit
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Your Wager",
                        f"${wager_amount:,.2f}",
                    )

                with col2:
                    st.metric(
                        "Potential Profit",
                        f"${potential_profit:,.2f}",
                    )

                with col3:
                    st.metric(
                        "Total Payout",
                        f"${total_payout:,.2f}",
                    )

                st.info(
                    "Total payout includes your original "
                    "wager plus potential profit. "
                    "This is a calculation, not a prediction."
                )

        else:

            st.info(
                "The Odds API connection worked, "
                "but no NBA moneylines are currently available."
            )

    except Exception as e:

        st.error(
            f"Live NBA odds test failed: {e}"
        )

        st.divider()

    st.markdown("---")
    st.markdown("### 🏈 Football API Connection")

    if st.button("Test Football API"):
        try:
            with st.spinner("Connecting to Sportradar..."):
                football_test = test_sportradar_connection()

            st.success(
                f"Sportradar connection successful — "
                f"{football_test['competition_count']} competitions found."
            )

            competitions_df = football_test["competitions"]

            if not competitions_df.empty:
                st.dataframe(
                    competitions_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(
                    "Connection succeeded, but no competitions were returned."
                )

        except Exception as e:
            st.error(f"Sportradar connection error: {e}")

    st.markdown("### 🏈 NFL Historical Seasons")

    if st.button("Get NFL Seasons"):
        try:
            with st.spinner("Checking available NFL seasons..."):
                nfl_season_test = get_nfl_seasons()

            st.success(
                f"NFL season lookup successful — "
                f"{nfl_season_test['season_count']} seasons found."
            )

            nfl_seasons_df = nfl_season_test["seasons"]

            if not nfl_seasons_df.empty:
                st.dataframe(
                    nfl_seasons_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(
                    "Connection succeeded, but no NFL seasons were returned."
                )

        except Exception as e:
            st.error(f"NFL season lookup error: {e}")

    st.markdown("### 🏈 NFL Historical Games")

    if st.button("Load 2024–25 NFL Games"):
        try:
            with st.spinner("Downloading 2024–25 NFL games..."):
                nfl_games_test = get_nfl_season_games(
                    "sr:season:115087"
                )

            st.success(
                f"NFL historical game download successful — "
                f"{nfl_games_test['game_count']} games found."
            )

            nfl_games_df = nfl_games_test["games"]

            if not nfl_games_df.empty:
                st.dataframe(
                    nfl_games_df,
                    use_container_width=True,
                    hide_index=True,
                )

                nfl_audit = audit_nfl_games(nfl_games_df)

                st.markdown("#### 🔍 NFL Dataset Audit")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Games",
                        nfl_audit["total_games"],
                    )

                with col2:
                    st.metric(
                        "Unique Games",
                        nfl_audit["unique_game_ids"],
                    )

                with col3:
                    st.metric(
                        "Duplicates",
                        nfl_audit["duplicate_games"],
                    )

                with col4:
                    st.metric(
                        "Teams",
                        nfl_audit["team_count"],
                    )

                col5, col6, col7, col8 = st.columns(4)

                with col5:
                    st.metric(
                        "Completed Games",
                        nfl_audit["completed_games"],
                    )

                with col6:
                    st.metric(
                        "Missing Scores",
                        nfl_audit["missing_home_scores"]
                        + nfl_audit["missing_away_scores"],
                    )

                with col7:
                    st.metric(
                        "Ties",
                        nfl_audit["ties"],
                    )

                with col8:
                    home_win_rate = nfl_audit["home_win_rate"]

                    st.metric(
                        "Home Win Rate",
                        (
                            f"{home_win_rate:.1%}"
                            if home_win_rate is not None
                            else "N/A"
                        ),
                    )

                st.write(
                    "Date range:",
                    nfl_audit["start_date"],
                    "→",
                    nfl_audit["end_date"],
                ) 
            else:
                st.warning(
                    "Connection succeeded, but no NFL games were returned."
                )

        except Exception as e:
            st.error(f"NFL historical games error: {e}")

    st.markdown("---")
    st.markdown("### 🏈 Multi-Season NFL Dataset")

    if st.button("Load Multi-Season NFL Dataset"):

        try:
            with st.spinner("Downloading multiple NFL seasons..."):

                season_ids = [
                    "sr:season:115087",  # 2024-25
                    "sr:season:127985",  # 2025-26
                ]

                multi_nfl = get_multiple_nfl_seasons(
                    season_ids
                )

                st.session_state["multi_nfl_games"] = multi_nfl["games"]

            st.success(
                f"Multi-season NFL download successful — "
                f"{multi_nfl['game_count']} games found."
            )

            st.markdown("#### Season Summary")
            st.dataframe(
                multi_nfl["season_summary"],
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(f"Multi-season NFL error: {e}")


    if "multi_nfl_games" in st.session_state:

        multi_games = st.session_state["multi_nfl_games"]

        st.markdown("#### Combined NFL Games")

        st.dataframe(
            multi_games,
            use_container_width=True,
            hide_index=True,
        )

        multi_audit = {
        "total_games": len(multi_games),
        "unique_games": multi_games["game_id"].nunique(),
        "duplicate_games": multi_games["game_id"].duplicated().sum(),
        "team_count": len(
            set(multi_games["home_team"].dropna())
            | set(multi_games["away_team"].dropna())
        ),
        "completed_games": (
            multi_games["status"].astype(str).str.lower() == "closed"
        ).sum(),
        "missing_scores": (
            multi_games["home_score"].isna()
            | multi_games["away_score"].isna()
        ).sum(),
        "ties": (
            multi_games["home_score"] == multi_games["away_score"]
        ).sum(),
        "home_win_rate": (
            multi_games["home_score"] > multi_games["away_score"]
        ).mean(),
    }

        st.markdown("### 🔍 Multi-Season NFL Audit")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Games",
            multi_audit["total_games"],
        )

        c2.metric(
            "Unique Games",
            multi_audit["unique_games"],
        )

        c3.metric(
            "Duplicates",
            multi_audit["duplicate_games"],
        )

        c4.metric(
            "Teams",
            multi_audit["team_count"],
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Completed Games",
            multi_audit["completed_games"],
        )

        c2.metric(
            "Missing Scores",
            multi_audit["missing_scores"],
        )

        c3.metric(
            "Ties",
            multi_audit["ties"],
        )

        c4.metric(
            "Home Win Rate",
            f"{multi_audit['home_win_rate']:.1%}",
        )

    # ============================================================
    # NFL PREGAME FEATURE ENGINE
    # ============================================================

    if "multi_nfl_games" in st.session_state:

        st.markdown("---")
        st.markdown("### 🧠 NFL Pregame Feature Engine")

        if st.button("Build NFL Pregame Features"):

            try:
                with st.spinner(
                    "Building leakage-safe NFL pregame features..."
                ):
                    nfl_feature_games = build_nfl_pregame_features(
                        st.session_state["multi_nfl_games"]
                    )

                    st.session_state[
                        "nfl_feature_games"
                    ] = nfl_feature_games

                st.success(
                    f"NFL feature build complete — "
                    f"{len(nfl_feature_games)} games processed."
                )

            except Exception as e:
                st.error(
                    f"NFL feature engine error: {e}"
                )


    if "nfl_feature_games" in st.session_state:

        nfl_features = st.session_state[
            "nfl_feature_games"
        ]

        st.markdown("#### 🧠 NFL Predictive Feature Dataset")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Games",
            len(nfl_features),
        )

        c2.metric(
            "Predictive Features",
            max(len(nfl_features.columns) - 6, 0),
        )

        c3.metric(
            "Training Games",
            nfl_features["home_win"].notna().sum(),
        )

        c4.metric(
            "Ties Excluded",
            nfl_features["home_win"].isna().sum(),
        )

        st.dataframe(
            nfl_features,
            use_container_width=True,
            hide_index=True,
        )

    # ============================================================
    # NFL WALK-FORWARD MODEL VALIDATION
    # ============================================================

    st.markdown("## 🧠 NFL Predictive Model Validation")

    if "multi_nfl_games" not in st.session_state:
        st.info(
            "Load the multi-season NFL dataset first before "
            "running predictive validation."
        )

    else:
        nfl_model_games = st.session_state["multi_nfl_games"]

        st.write(
            f"Historical games available for modeling: "
            f"{len(nfl_model_games):,}"
        )

        if st.button("Run NFL Walk-Forward Model"):

            try:
                with st.spinner(
                    "Building leakage-safe features and "
                    "running NFL walk-forward validation..."
                ):

                    nfl_features = build_nfl_pregame_features(
                        nfl_model_games
                    )

                    nfl_model_result = run_nfl_walkforward_model(
                        nfl_features
                    )

                    st.session_state[
                        "nfl_walkforward_result"
                    ] = nfl_model_result

                    st.session_state[
                        "nfl_feature_games"
                    ] = nfl_features

                st.success(
                    "NFL walk-forward validation complete."
                )

            except Exception as e:
                st.error(
                    f"NFL walk-forward validation error: {e}"
                )


    if "nfl_walkforward_result" in st.session_state:

        result = st.session_state[
            "nfl_walkforward_result"
        ]

        st.markdown("### 🏈 NFL Model Performance")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Predictions",
            f"{result['prediction_count']:,}",
        )

        c2.metric(
            "Accuracy",
            f"{result['accuracy']:.1%}",
        )

        c3.metric(
            "AUC",
            f"{result['auc']:.3f}",
        )

        c4.metric(
            "Brier Score",
            f"{result['brier']:.3f}",
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Log Loss",
            f"{result['log_loss']:.3f}",
        )

        c2.metric(
            "Home-Team Baseline",
            f"{result['baseline_home_accuracy']:.1%}",
        )

        c3.metric(
            "Accuracy vs Baseline",
            f"{result['accuracy_vs_baseline']:+.1%}",
        )


        # ------------------------------------------
        # NFL PROBABILITY-BAND VALIDATION
        # ------------------------------------------

        st.markdown("### 🎯 NFL Probability-Band Validation")

        probability_bands = result.get("probability_bands")

        if (
            probability_bands is not None
            and not probability_bands.empty
        ):

            display_bands = probability_bands.copy()

            # Format percentages for dashboard display.

            percentage_columns = [
                "average_confidence",
                "actual_win_rate",
                "calibration_gap",
                "win_rate_lower_95",
                "win_rate_upper_95",
            ]

            for column in percentage_columns:
                display_bands[column] = (
                    display_bands[column] * 100
                ).round(1)

            display_bands = display_bands.rename(
                columns={
                    "confidence_band": "Confidence Band",
                    "total_predictions": "Games",
                    "correct_predictions": "Correct",
                    "average_confidence": "Average Confidence (%)",
                    "actual_win_rate": "Actual Win Rate (%)",
                    "calibration_gap": "Calibration Gap (pp)",
                    "win_rate_lower_95": "95% Lower Bound (%)",
                    "win_rate_upper_95": "95% Upper Bound (%)",
                }
            )

            st.dataframe(
                display_bands,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Historical out-of-sample prediction accuracy "
                "by model-confidence range. Confidence intervals "
                "reflect sampling uncertainty and do not guarantee "
                "future betting performance."
            )

        else:

            st.info(
                "Run the NFL Walk-Forward Model to generate "
                "probability-band validation results."
            )


    
        st.markdown("### 🔎 Walk-Forward Predictions")

        prediction_table = result[
            "predictions"
        ].copy()

        prediction_table[
            "home_win_probability"
        ] = prediction_table[
            "probability"
        ].map(
            lambda x: f"{x:.1%}"
        )

        st.dataframe(
            prediction_table[
                [
                    "start_time",
                    "home_team",
                    "away_team",
                    "home_win_probability",
                    "prediction",
                    "actual",
                    "correct",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    
# =================================================
# NFL ACCURACY AUDIT DASHBOARD
# =================================================

st.markdown("---")

st.header("NFL Prediction Accuracy Audit")

st.caption(
    "Historical out-of-sample model evaluation"
)

if st.button(
    "Run NFL Accuracy Audit",
    key="run_nfl_accuracy_audit",
):

    walkforward = st.session_state.get(
        "nfl_walkforward_result"
    )

    if walkforward is None:
        st.warning(
            "Run the NFL walk-forward model first."
        )

    else:

        try:

            if isinstance(walkforward, dict):

                predictions = walkforward.get(
                    "predictions"
                )

            else:

                predictions = walkforward

            if predictions is None:

                raise ValueError(
                    "Walk-forward predictions are missing."
                )

            audit_df = pd.DataFrame(
                predictions
            ).copy()

            # Normalize probability column.

            if (
                "probability" not in audit_df.columns
                and "home_win_probability"
                in audit_df.columns
            ):

                audit_df["probability"] = (
                    audit_df["home_win_probability"]
                )

            # Normalize actual outcome column.

            if (
                "actual" not in audit_df.columns
                and "home_win" in audit_df.columns
            ):

                audit_df["actual"] = (
                    audit_df["home_win"]
                )

            # Never audit unfinished games.

            audit_df = audit_df.dropna(
                subset=["probability", "actual"]
            )

            audit = run_nfl_accuracy_audit(
                audit_df,
                min_samples=30,
            )

            st.session_state[
                "nfl_accuracy_audit"
            ] = audit

        except Exception as e:

            st.error(
                f"NFL accuracy audit failed: {e}"
            )


# =================================================
# DISPLAY AUDIT RESULTS
# =================================================

audit = st.session_state.get(
    "nfl_accuracy_audit"
)

if audit is not None:

    overall = audit["overall"]

    st.subheader(
        "Overall Historical Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predictions",
        overall["total_predictions"],
    )

    col2.metric(
        "Correct",
        overall["correct_predictions"],
    )

    col3.metric(
        "Accuracy",
        f'{overall["accuracy"]:.1%}',
    )

    col4.metric(
        "Brier Score",
        f'{overall["brier_score"]:.4f}',
    )

    if not audit["sufficient_sample"]:

        st.warning(
            "Limited historical sample. "
            "Interpret results cautiously."
        )

    st.markdown("---")

    st.subheader(
        "Accuracy by Confidence Threshold"
    )

    threshold_df = audit[
        "confidence_thresholds"
    ].copy()

    st.dataframe(
        threshold_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader(
        "Probability Calibration"
    )

    calibration_df = audit[
        "calibration"
    ].copy()

    st.dataframe(
        calibration_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader(
        "Historical Performance Over Time"
    )

    performance_df = audit[
        "performance_over_time"
    ]

    if not performance_df.empty:

        st.line_chart(
            performance_df.set_index(
                "month"
            )["accuracy"]
        )

        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Historical prediction dates are "
            "not available."
        )

    st.markdown("---")

    st.subheader(
        "Baseline Comparison"
    )

    baseline = audit[
        "baseline_comparison"
    ]

    if baseline is not None:

        col1, col2 = st.columns(2)

        col1.metric(
            "Model Accuracy",
            f'{baseline["model_accuracy"]:.1%}',
        )

        col2.metric(
            "Baseline Accuracy",
            f'{baseline["baseline_accuracy"]:.1%}',
        )

    else:

        st.info(
            "No pregame baseline probabilities "
            "were supplied."
        )

    # ============================================================
    # NFL CONFIDENCE CALIBRATION
    # ============================================================

    if "nfl_walkforward_result" in st.session_state:

        st.markdown("## 🎯 NFL Confidence Calibration")

        calibration_result = st.session_state[
            "nfl_walkforward_result"
        ]

        calibration_df = calibration_result[
            "predictions"
        ].copy()

        # Confidence = probability assigned to the predicted winner.
        calibration_df["confidence"] = (
            calibration_df["probability"].where(
                calibration_df["prediction"] == 1,
                1.0 - calibration_df["probability"],
            )
        )

        calibration_df["correct"] = (
            calibration_df["prediction"]
            == calibration_df["actual"]
        ).astype(int)

        # Confidence bands.
        calibration_df["confidence_band"] = pd.cut(
            calibration_df["confidence"],
            bins=[
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
            ],
            labels=[
                "50–55%",
                "55–60%",
                "60–65%",
                "65–70%",
                "70–75%",
                "75–80%",
                "80–85%",
                "85–90%",
                "90–95%",
                "95–100%",
            ],
            include_lowest=True,
        )

        calibration_summary = (
            calibration_df
            .groupby(
                "confidence_band",
                observed=False,
            )
            .agg(
                predictions=("correct", "size"),
                correct=("correct", "sum"),
                actual_accuracy=("correct", "mean"),
                avg_confidence=("confidence", "mean"),
            )
            .reset_index()
        )

        calibration_summary["calibration_gap"] = (
            calibration_summary["actual_accuracy"]
            - calibration_summary["avg_confidence"]
        )


        # ------------------------------------------
        # Store NFL historical confidence calibration
        # for use by the live opportunity engine.
        # ------------------------------------------

        st.session_state["nfl_calibration_summary"] = (
            calibration_summary.copy()
        )
    

        display_calibration = calibration_summary.copy()

        display_calibration["Average Confidence"] = (
            display_calibration["avg_confidence"]
            .map(
                lambda x: f"{x:.1%}"
                if pd.notna(x)
                else "—"
            )
        )

        display_calibration["Actual Accuracy"] = (
            display_calibration["actual_accuracy"]
            .map(
                lambda x: f"{x:.1%}"
                if pd.notna(x)
                else "—"
            )
        )

        display_calibration["Calibration Gap"] = (
            display_calibration["calibration_gap"]
            .map(
                lambda x: f"{x:+.1%}"
                if pd.notna(x)
                else "—"
            )
        )

        display_calibration = display_calibration[
            [
                "confidence_band",
                "predictions",
                "correct",
                "Average Confidence",
                "Actual Accuracy",
                "Calibration Gap",
            ]
        ]

        display_calibration = display_calibration.rename(
            columns={
                "confidence_band": "Confidence Band",
                "predictions": "Predictions",
                "correct": "Correct",
            }
        )

        st.dataframe(
            display_calibration,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------------------
        # HIGH-CONFIDENCE PERFORMANCE
        # --------------------------------------------------------

        st.markdown("### 🔥 High-Confidence Performance")

        confidence_thresholds = [
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
        ]

        threshold_rows = []

        for threshold in confidence_thresholds:

            subset = calibration_df[
                calibration_df["confidence"] >= threshold
            ]

            if len(subset) == 0:
                continue

            threshold_rows.append(
                {
                    "Minimum Confidence": threshold,
                    "Predictions": len(subset),
                    "Correct": int(
                        subset["correct"].sum()
                    ),
                    "Accuracy": subset["correct"].mean(),
                    "Average Confidence": subset[
                        "confidence"
                    ].mean(),
                }
            )
    
        threshold_df = pd.DataFrame(
            threshold_rows
        )

        if not threshold_df.empty:

            threshold_display = threshold_df.copy()

            threshold_display[
                "Minimum Confidence"
            ] = threshold_display[
                "Minimum Confidence"
            ].map(
                lambda x: f"{x:.0%}+"
            )

            threshold_display[
                "Accuracy"
            ] = threshold_display[
                "Accuracy"
            ].map(
                lambda x: f"{x:.1%}"
            )

            threshold_display[
                "Average Confidence"
            ] = threshold_display[
                "Average Confidence"
            ].map(
                lambda x: f"{x:.1%}"
            )

            st.dataframe(
                threshold_display,
                use_container_width=True,
                hide_index=True,
            )

    # Save model predictions for later decision-engine/live-engine work.
    calibration_result = st.session_state.get(
        "nfl_walkforward_result"
    )

    if calibration_result is not None:
        nfl_model_predictions = calibration_result[
            "predictions"
        ].copy()

        if (
            "home_win_probability" not in nfl_model_predictions.columns
            and "probability" in nfl_model_predictions.columns
        ):
            nfl_model_predictions["home_win_probability"] = (
                nfl_model_predictions["probability"]
            )

        st.session_state[
        "nfl_historical_predictions"
    ] = nfl_model_predictions

    # Store historical calibration data separately
    # from predictions for upcoming NFL games.


    # ==========================================
    # STORE NFL HISTORICAL PREDICTIONS
    # ==========================================

    nfl_model_predictions = st.session_state.get(
        "nfl_model_predictions",
        pd.DataFrame()
    )

    if (
        isinstance(nfl_model_predictions, pd.DataFrame)
        and not nfl_model_predictions.empty
    ):

        st.session_state[
            "nfl_historical_predictions"
        ] = nfl_model_predictions.copy()

        st.session_state[
            "nfl_calibration_predictions"
        ] = nfl_model_predictions.copy()

    
    # ---------------------------------------------------------
    # NFL DECISION ENGINE
    # ---------------------------------------------------------

    st.markdown("### 🧠 NFL Decision Engine")


    try:
        nfl_predictions = st.session_state.get(
            "nfl_historical_predictions",
            pd.DataFrame()
        ).copy()

        if nfl_predictions.empty:
            raise ValueError(
                "Run the NFL historical model first to load NFL model predictions."
            )
        if "home_win_probability" not in nfl_predictions.columns and "probability" in nfl_predictions.columns:
            nfl_predictions["home_win_probability"] = nfl_predictions["probability"]
    
        if (
            "home_win_probability" not in nfl_predictions.columns
            and "probability" in nfl_predictions.columns
        ):
            nfl_predictions["home_win_probability"] = nfl_predictions["probability"]

        decision_profile = build_nfl_confidence_profile(
            nfl_predictions
        )

        decision_rows = []

        for _, game in nfl_predictions.iterrows():

            decision = get_nfl_decision(
                home_team=game["home_team"],
                away_team=game["away_team"],
                home_win_probability=game["home_win_probability"],
                confidence_profile=decision_profile,
            )

            decision_rows.append(
                {
                    "start_time": game["start_time"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "predicted_team": decision["predicted_team"],
                    "confidence": decision["confidence"],
                    "decision": decision["decision"],
                    "historical_accuracy": decision["historical_accuracy"],
                    "historical_sample": decision["historical_sample"],
                    "reason": decision["reason"],
                    "actual": game["actual"],
                    "correct": game["correct"],
                }
            )

        nfl_decisions = pd.DataFrame(decision_rows)

        st.session_state["nfl_decisions"] = nfl_decisions
        st.session_state["nfl_confidence_profile"] = decision_profile

        bet_count = (nfl_decisions["decision"] == "BET").sum()
        lean_count = (nfl_decisions["decision"] == "LEAN").sum()
        pass_count = (nfl_decisions["decision"] == "PASS").sum()

        c1, c2, c3 = st.columns(3)

        c1.metric("BET", int(bet_count))
        c2.metric("LEAN", int(lean_count))
        c3.metric("PASS", int(pass_count))

        bet_games = nfl_decisions[
            nfl_decisions["decision"] == "BET"
        ].copy()

        if not bet_games.empty:

            bet_accuracy = bet_games["correct"].mean()

            st.markdown("#### 🔥 Historical BET Performance")

            b1, b2, b3 = st.columns(3)

            b1.metric("Qualified Bets", len(bet_games))
            b2.metric("Correct", int(bet_games["correct"].sum()))
            b3.metric("Accuracy", f"{bet_accuracy:.1%}")

            display_bets = bet_games.copy()

            display_bets["confidence"] = (
                display_bets["confidence"]
                .map(lambda x: f"{x:.1%}")
            )

            display_bets["historical_accuracy"] = (
                display_bets["historical_accuracy"]
                .map(
                    lambda x: f"{x:.1%}"
                    if pd.notna(x)
                    else "—"
                )
            )

            st.dataframe(
                display_bets,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No historical predictions currently "
                "qualify as BET decisions."
            )

    except Exception as e:
        st.error(f"NFL Decision Engine error: {e}")

    # --------------------------------------------------
    # NFL MARKET EDGE TEST
    # --------------------------------------------------

    st.markdown("### 💰 NFL Market Edge Test")

    test_home_probability = st.number_input(
        "Home model probability",
        min_value=0.01,
        max_value=0.99,
        value=0.70,
        step=0.01,
    )

    test_home_odds = st.number_input(
        "Home moneyline",
        value=-150,
        step=5,
    )

    test_away_odds = st.number_input(
        "Away moneyline",
        value=130,
        step=5,
    )

    if st.button("Test NFL Market Edge"):

        try:
            edge_result = calculate_nfl_market_edge(
                home_model_probability=test_home_probability,
                home_odds=test_home_odds,
                away_odds=test_away_odds,
            )

            if edge_result["best_side"] == "HOME":
                selected_probability = edge_result[
                    "home_model_probability"
                ]
                selected_market_probability = edge_result[
                    "home_no_vig_probability"
                ]
            else:
                selected_probability = edge_result[
                    "away_model_probability"
                ]
                selected_market_probability = edge_result[
                    "away_no_vig_probability"
                ]

            classification = classify_nfl_market_edge(
                model_probability=selected_probability,
                market_probability=selected_market_probability,
                american_odds=edge_result["best_odds"],
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Best Side",
                edge_result["best_side"],
            )

            col2.metric(
                "Model Probability",
                f"{selected_probability:.1%}",
            )

            col3.metric(
                "Market No-Vig Probability",
                f"{selected_market_probability:.1%}",
            )

            col4.metric(
                "Model Edge",
                f"{edge_result['best_edge']:+.1%}",
            )

            st.metric(
                "Expected Value / $1",
                f"{classification['expected_value']:+.3f}",
            )

            decision = classification["decision"]

            if decision == "BET":
                st.success(
                    f"BET — {classification['reason']}"
                )
            elif decision == "LEAN":
                st.warning(
                    f"LEAN — {classification['reason']}"
                )
            else:
                st.info(
                    f"PASS — {classification['reason']}"
                )

        except Exception as e:
            st.error(
                f"NFL Market Edge Test error: {e}"
            )

    # --------------------------------------------------
    # LIVE NFL MONEYLINES
    # --------------------------------------------------

    st.markdown("### 📡 Live NFL Moneylines")

    if st.button("Load Live NFL Moneylines"):
        try:
            with st.spinner("Loading current NFL moneylines..."):
                live_nfl_odds = get_live_nfl_moneylines()

            if not live_nfl_odds:
                st.info(
                    "No current NFL moneyline markets were returned."
                )
            else:
                live_nfl_odds_df = pd.DataFrame(live_nfl_odds)

                st.success(
                    f"Loaded {len(live_nfl_odds_df)} "
                    "NFL sportsbook moneyline markets."
                )

                
            # NFL SPORTSBOOK ODDS DASHBOARD

            from zoneinfo import ZoneInfo

            def display_nfl_moneyline(odds):
                if pd.isna(odds):
                    return "N/A"
                return f"{int(odds):+d}"

            st.markdown("### 🏈 NFL Sportsbook Odds Center")

            st.caption(
                "Compare moneyline odds across sportsbooks "
                "and find the best available price for each team."
            )

            nfl_games = live_nfl_odds_df.groupby(
                "game_id", sort=False
            )

            st.info(
                f"Displaying {len(nfl_games)} NFL games "
                f"across {len(live_nfl_odds_df)} sportsbook markets."
            )

            for game_id, game_lines in nfl_games:

                game = game_lines.iloc[0]

                home_team = str(game["home_team"])
                away_team = str(game["away_team"])

                game_time = pd.to_datetime(
                    game["commence_time"],
                    utc=True,
                    errors="coerce",
                )

                if pd.notna(game_time):
                    game_time_display = (
                        game_time.tz_convert(
                            ZoneInfo("America/Chicago")
                        ).strftime("%a, %b %d · %I:%M %p CT")
                    )
                else:
                    game_time_display = "Time unavailable"

                home_lines = game_lines.dropna(
                    subset=["home_moneyline"]
                )

                away_lines = game_lines.dropna(
                    subset=["away_moneyline"]
                )

                best_home = (
                    home_lines.loc[
                        home_lines["home_moneyline"].idxmax()
                    ]
                    if not home_lines.empty
                    else None
                )

                best_away = (
                    away_lines.loc[
                        away_lines["away_moneyline"].idxmax()
                    ]
                    if not away_lines.empty
                    else None
                )

                with st.container(border=True):

                    st.caption(game_time_display)

                    st.subheader(
                        f"{away_team} @ {home_team}"
                    )

                    away_col, home_col = st.columns(2)

                    with away_col:
                        st.caption("AWAY TEAM")
                        st.markdown(f"**{away_team}**")

                        if best_away is not None:
                            st.metric(
                                "Best Moneyline",
                                display_nfl_moneyline(
                                    best_away["away_moneyline"]
                                ),
                            )

                            st.caption(
                                f"Sportsbook: {best_away['sportsbook']}"
                            )
                        else:
                            st.info("Odds unavailable")

                    with home_col:
                        st.caption("HOME TEAM")
                        st.markdown(f"**{home_team}**")

                        if best_home is not None:
                            st.metric(
                                "Best Moneyline",
                                display_nfl_moneyline(
                                    best_home["home_moneyline"]
                                ),
                            )

                            st.caption(
                                f"Sportsbook: {best_home['sportsbook']}"
                            )
                        else:
                            st.info("Odds unavailable")

                    with st.expander(
                        "View All Sportsbook Odds"
                    ):

                        comparison = game_lines[
                            [
                                "sportsbook",
                                "away_moneyline",
                                "home_moneyline",
                            ]
                        ].copy()

                        comparison["away_moneyline"] = (
                            comparison["away_moneyline"].apply(
                                display_nfl_moneyline
                            )
                        )

                        comparison["home_moneyline"] = (
                            comparison["home_moneyline"].apply(
                                display_nfl_moneyline
                            )
                        )

                        comparison.columns = [
                            "Sportsbook",
                            away_team,
                            home_team,
                        ]

                        st.dataframe(
                            comparison,
                            use_container_width=True,
                            hide_index=True,
                        )
                st.session_state["live_nfl_odds"] = (
                    live_nfl_odds_df
                )

        except Exception as e:
            st.error(f"Live NFL odds error: {e}")

    # --------------------------------------------------
    # NFL BEST-LINE SHOPPER
    # --------------------------------------------------

    st.markdown("### 🛒 NFL Best-Line Shopper")

    if "live_nfl_odds" not in st.session_state:
        st.info(
            "Load Live NFL Moneylines first."
        )

    else:
        if st.button("Find Best NFL Moneylines"):
            try:
                odds_rows = (
                    st.session_state["live_nfl_odds"]
                    .to_dict("records")
                )

                best_lines = get_best_nfl_moneylines(
                    odds_rows
                )

                best_lines_df = pd.DataFrame(best_lines)

                st.session_state["nfl_best_lines"] = (
                    best_lines_df

                )

                st.success(
                    f"Found best available lines for "
                    f"{len(best_lines_df)} NFL games."
                )

                st.dataframe(
                    best_lines_df,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as e:
                st.error(
                    f"NFL best-line error: {e}"
                )

    
    # ============================================================
    # LIVE NFL OPPORTUNITY ENGINE
    # ============================================================

    st.markdown("### 🧠 Live NFL Opportunity Engine")

    best_lines = st.session_state.get("nfl_best_lines")
    feature_games = st.session_state.get("nfl_feature_games")
    historical_result = st.session_state.get(
        "nfl_walkforward_result"
    )

    # ------------------------------------------------------------
    # ENGINE STATUS
    # ------------------------------------------------------------

    st.caption("NFL Engine Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sportsbook Lines",
            "Ready" if (
                isinstance(best_lines, pd.DataFrame)
                and not best_lines.empty
            ) else "Not Loaded"
        )

    with col2:
        st.metric(
            "Historical Features",
            "Ready" if (
                isinstance(feature_games, pd.DataFrame)
                and not feature_games.empty
            ) else "Not Loaded"
        )

    with col3:
        st.metric(
            "Walk-Forward Validation",
            "Ready" if historical_result else "Not Loaded"
        )

    # ------------------------------------------------------------
    # GENERATE UPCOMING NFL PREDICTIONS
    # ------------------------------------------------------------

    if st.button(
        "Generate Live NFL Predictions",
        key="generate_live_nfl_predictions"
    ):

        try:

            if (
                best_lines is None
                or best_lines.empty
            ):
                st.error(
                    "Load NFL moneylines and run the "
                    "Best-Line Shopper first."
                )

            elif (
                feature_games is None
                or feature_games.empty
            ):
                st.error(
                    "Build NFL pregame features first."
                )


            else:

                prediction_rows = []
                prediction_errors = []

                current_time = pd.Timestamp.now(tz="UTC")

                with st.spinner(
                    "Generating predictions for upcoming NFL games..."
                ):

                    for _, game in best_lines.iterrows():

                        home_team = game["home_team"]
                        away_team = game["away_team"]

                        game_time = pd.to_datetime(
                            game["commence_time"],
                            utc=True,
                            errors="coerce"
                        )

                        if pd.isna(game_time):
                            continue

                        # Only predict games that have not started.
                        if game_time <= current_time:
                            continue

                        try:

                            # Build matchup features using
                            # historical games before kickoff.

                            future_features = (
                                build_nfl_future_matchup_features(
                                    feature_games,
                                    home_team,
                                    away_team,
                                    game_time
                                )
                            )

                            # Generate an actual model prediction.

                            prediction = predict_nfl_matchup(
                                feature_games,
                                future_features
                            )

                            prediction_rows.append(
                                {
                                    "game_id": game.get("game_id"),
                                    "commence_time": game_time,
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "home_win_probability": prediction[
                                        "home_win_probability"
                                    ],
                                    "away_win_probability": prediction[
                                        "away_win_probability"
                                    ],
                                    "predicted_team": prediction[
                                        "predicted_team"
                                    ],
                                    "confidence": prediction[
                                        "confidence"
                                    ],
                                    "training_games": prediction[
                                        "training_games"
                                    ]
                                }
                            )

                        except Exception as game_error:

                            prediction_errors.append(
                                f"{away_team} at {home_team}: "
                                f"{game_error}"
                            )

                # ------------------------------------------------
                # SAVE LIVE PREDICTIONS
                # ------------------------------------------------

                live_predictions = pd.DataFrame(
                    prediction_rows
                )

                st.session_state[
                    "nfl_live_predictions"
                ] = live_predictions

                # Clear previously generated opportunities
                # so stale predictions cannot be displayed.

                st.session_state.pop(
                    "nfl_live_opportunities",
                    None
                )

                if live_predictions.empty:

                    st.warning(
                        "No upcoming NFL predictions could "
                        "be generated from the available data."
                    )

                else:

                    st.success(
                        f"Generated predictions for "
                        f"{len(live_predictions)} upcoming NFL games."
                    )

                if prediction_errors:

                    with st.expander(
                        "Games that could not be predicted"
                    ):

                        for error in prediction_errors:
                            st.write(error)

        except Exception as e:

            st.error(
                f"NFL prediction error: {e}"
            )

    # ------------------------------------------------------------
    # DISPLAY LIVE PREDICTIONS
    # ------------------------------------------------------------

    live_predictions = st.session_state.get(
        "nfl_live_predictions"
    )

    
    if (
        isinstance(live_predictions, pd.DataFrame)
        and not live_predictions.empty
        and isinstance(best_lines, pd.DataFrame)
        and not best_lines.empty
        and isinstance(historical_result, dict)
        and historical_result.get("success") is True
    ):

        st.markdown("### 🏈 Upcoming NFL Predictions")

        display_predictions = live_predictions.copy()

        display_predictions["commence_time"] = (
            pd.to_datetime(
                display_predictions["commence_time"],
                utc=True
            )
            .dt.tz_convert("America/Chicago")
            .dt.strftime("%b %d, %I:%M %p CT")
        )

        for column in [
            "home_win_probability",
            "away_win_probability",
            "confidence"
        ]:

            display_predictions[column] = (
                display_predictions[column]
                .map(lambda x: f"{x:.1%}")
            )

        st.dataframe(
            display_predictions[
                [
                    "commence_time",
                    "away_team",
                    "home_team",
                    "predicted_team",
                    "confidence",
                    "home_win_probability",
                    "away_win_probability",
                    "training_games"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Model probabilities are estimates, "
            "not guaranteed outcomes."
        )

    # ------------------------------------------------------------
    # LIVE NFL MARKET OPPORTUNITY ANALYSIS
    # ------------------------------------------------------------

    st.markdown("### 💰 NFL Market Opportunities")

    if (
        isinstance(live_predictions, pd.DataFrame)
        and not live_predictions.empty
        and isinstance(best_lines, pd.DataFrame)
        and not best_lines.empty
        and historical_result
    ):

        if st.button(
            "Analyze Live NFL Opportunities",
            key="analyze_live_nfl_opportunities"
        ):

            try:

                with st.spinner(
                    "Comparing NFL predictions against sportsbook odds..."
                ):

                    # Use live predictions, not historical
                    # calibration predictions.

                    live_opportunities = (
                        build_live_nfl_opportunities(
                            model_predictions=live_predictions,
                            best_lines=best_lines,
                            historical_accuracy=historical_result.get(
                                "accuracy"
                            ),
                            historical_sample=historical_result.get(
                                "prediction_count"
                            )
                        )
                    )

                    st.session_state[
                        "nfl_live_opportunities"
                    ] = live_opportunities

                if live_opportunities.empty:

                    st.warning(
                        "No upcoming NFL games matched "
                        "the available sportsbook markets."
                    )

                else:

                    st.success(
                        f"Analyzed {len(live_opportunities)} "
                        "NFL market opportunities."
                    )

            except Exception as e:

                st.error(
                    f"NFL opportunity analysis error: {e}"
                )

    
    else:

        if (
            isinstance(live_predictions, pd.DataFrame)
            and not live_predictions.empty
            and isinstance(best_lines, pd.DataFrame)
            and not best_lines.empty
        ):
            st.info(
                "Live NFL predictions are available. "
                "Run the NFL Walk-Forward Model to enable "
                "validated market opportunity analysis."
            )

        else:
            st.info(
                "Generate live NFL predictions and load "
                "the Best-Line Shopper before analyzing opportunities."
            )
    # ------------------------------------------------------------
    # DISPLAY NFL OPPORTUNITIES
    # ------------------------------------------------------------

    live_opportunities = st.session_state.get(
        "nfl_live_opportunities"
    )

    if (
        isinstance(live_opportunities, pd.DataFrame)
        and not live_opportunities.empty
    ):

        st.markdown("### 📊 NFL Opportunity Results")

        live_display = live_opportunities.copy()

        for column in [
            "home_win_probability",
            "model_probability",
            "market_no_vig_probability",
            "model_edge"
        ]:

            if column in live_display.columns:

                live_display[column] = (
                    live_display[column]
                    .map(
                        lambda x: f"{x:.1%}"
                        if pd.notna(x)
                        else "—"
                    )
                )

        if "expected_value" in live_display.columns:

            live_display["expected_value"] = (
                live_display["expected_value"]
                .map(
                    lambda x: f"{x:+.3f}"
                    if pd.notna(x)
                    else "—"
                )
            )

        st.dataframe(
            live_display,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Expected value depends on model probability "
            "accuracy and the sportsbook odds available "
            "when the analysis was generated."
        )

if page == "🧪 Research Lab":
    render_research_lab()



# ==========================================
# MARKET EDGE AI V5
# MLB RESEARCH LAB - HISTORICAL DATA TEST
# ==========================================

if page == "🧪 Research Lab":

    st.divider()

    st.subheader("⚾ MLB Research Lab")

    st.caption(
        "Historical MLB data retrieval and "
        "pregame feature validation."
    )

    with st.expander(
        "MLB Historical Data Test",
        expanded=False,
    ):

        st.info(
            "This test retrieves completed MLB games "
            "and calculates historical team statistics. "
            "It does not generate betting predictions."
        )

        if st.button(
            "Run MLB Historical Data Test",
            key="run_mlb_historical_test",
        ):

            try:

                from dual_agent.mlb_research import (
                    fetch_mlb_games,
                    build_mlb_pregame_features,
                    summarize_mlb_dataset,
                )

                with st.spinner(
                    "Retrieving historical MLB games..."
                ):

                    games = fetch_mlb_games(
                        "2025-04-01",
                        "2025-04-15",
                    )

                if games.empty:

                    st.error(
                        "No historical MLB games were returned."
                    )

                else:

                    st.success(
                        "MLB historical data retrieved!"
                    )

                    with st.spinner(
                        "Building pregame team statistics..."
                    ):

                        features = (
                            build_mlb_pregame_features(
                                games
                            )
                        )

                    summary = summarize_mlb_dataset(
                        features
                    )

                    # ------------------------------
                    # HISTORICAL DATA METRICS
                    # ------------------------------

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Historical Games",
                            len(games),
                        )

                    with col2:

                        st.metric(
                            "Pregame Feature Rows",
                            len(features),
                        )

                    with col3:

                        st.metric(
                            "Labeled Games",
                            summary.get(
                                "labeled_games",
                                0,
                            ),
                        )

                    # ------------------------------
                    # DATASET SUMMARY
                    # ------------------------------

                    st.markdown(
                        "### MLB Dataset Summary"
                    )

                    st.json(summary)

                    # ------------------------------
                    # SAMPLE PREGAME FEATURES
                    # ------------------------------

                    st.markdown(
                        "### Sample Pregame Matchups"
                    )

                    display_columns = [
                        "home_team",
                        "away_team",
                        "home_win",
                        "home_win_pct",
                        "away_win_pct",
                        "win_pct_diff",
                    ]

                    st.dataframe(
                        features[
                            display_columns
                        ].head(15),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # ------------------------------
                    # DATA INTEGRITY CHECK
                    # ------------------------------

                    st.markdown(
                        "### Data Integrity Check"
                    )

                    if (
                        len(games) == len(features)
                        and summary.get(
                            "labeled_games",
                            0,
                        ) > 0
                    ):

                        st.success(
                            "Historical game retrieval "
                            "and feature generation passed "
                            "the basic integrity checks."
                        )

                    else:

                        st.warning(
                            "The historical dataset requires "
                            "additional investigation."
                        )

                    st.caption(
                        "This preliminary test does not "
                        "verify historical result-availability "
                        "timestamps or establish that the "
                        "features are leakage-free."
                    )

            except Exception as error:

                st.error(
                    "MLB historical data test failed."
                )

                
# ==========================================
# MLB MODEL TRAINING AND VALIDATION
# ==========================================

st.divider()

st.subheader("⚾ MLB Prediction Model Training")

st.caption(
    "Train and evaluate the MLB game-winner model "
    "using historical regular-season games."
)

if st.button(
    "Train MLB Prediction Model",
    key="train_mlb_prediction_model_button",
    type="primary",
):

    try:
        from dual_agent.mlb_research import (
            fetch_mlb_games,
            build_mlb_pregame_features,
            train_mlb_prediction_model,
        )

        with st.spinner(
            "Downloading MLB history and training model..."
        ):

            games = fetch_mlb_games(
                "2025-03-01",
                "2025-09-28",
            )

            if games.empty:
                raise ValueError(
                    "No historical MLB games were retrieved."
                )

            features = build_mlb_pregame_features(
                games
            )

            if features.empty:
                raise ValueError(
                    "MLB feature generation returned no data."
                )

            results = train_mlb_prediction_model(
                features
            )

            st.session_state[
                "mlb_training_metrics"
            ] = results["metrics"]

        st.success(
            "MLB model training and validation completed."
        )

    except Exception as e:

        st.error(
            f"MLB model training failed: {e}"
        )

# ==========================================
# MLB V2A PERMANENT PITCHER HISTORY
# ==========================================

if "mlb_v2a_pitcher_logs" not in st.session_state:

    with st.spinner(
        "Loading permanent MLB pitcher history..."
    ):

        permanent_pitcher_logs = (
            load_mlb_pitcher_history()
        )

        st.session_state[
            "mlb_v2a_pitcher_logs"
        ] = permanent_pitcher_logs

        if (
            permanent_pitcher_logs is not None
            and not permanent_pitcher_logs.empty
        ):

            st.session_state[
                "mlb_pitcher_history_loaded"
            ] = True

        else:

            st.session_state[
                "mlb_pitcher_history_loaded"
            ] = False
            
# ==========================================
# MLB V1 WALK-FORWARD BENCHMARK
# ==========================================

st.divider()

st.subheader("⚾ MLB V1 Walk-Forward Benchmark")

st.caption(
    "Tests the current MLB V1 model using expanding-history "
    "walk-forward validation. Every prediction is generated "
    "using only games that occurred before that matchup."
)

if st.button(
    "Run MLB V1 Walk-Forward Benchmark",
    key="run_mlb_v1_walkforward",
):

    try:

        with st.spinner(
            "Building MLB historical dataset and running "
            "walk-forward predictions..."
        ):

            current_year = pd.Timestamp.now().year

            start_date = (
                pd.Timestamp(
                    year=current_year - 3,
                    month=3,
                    day=1,
                )
                .strftime("%Y-%m-%d")
            )

            end_date = (
                pd.Timestamp.now()
                .strftime("%Y-%m-%d")
            )

            mlb_games = fetch_mlb_games(
                start_date=start_date,
                end_date=end_date,
            )

            st.session_state[
                "mlb_v2a_games"
            ] = mlb_games
            

            # ----------------------------------
            # STARTING PITCHER COVERAGE CHECK
            # ----------------------------------
            
            pitcher_columns = [
                "home_starting_pitcher_id",
                "away_starting_pitcher_id",
            ]
            
            if all(
                column in mlb_games.columns
                for column in pitcher_columns
            ):
            
                pitcher_coverage = mlb_games.copy()
            
                pitcher_coverage["season"] = (
                    pd.to_datetime(
                        pitcher_coverage["start_time"],
                        utc=True,
                        errors="coerce",
                    )
                    .dt.year
                )
            
                pitcher_coverage["both_starters"] = (
                    pitcher_coverage[
                        "home_starting_pitcher_id"
                    ].notna()
                    &
                    pitcher_coverage[
                        "away_starting_pitcher_id"
                    ].notna()
                )
            
                total_pitcher_games = len(
                    pitcher_coverage
                )
            
                games_with_both_starters = int(
                    pitcher_coverage[
                        "both_starters"
                    ].sum()
                )
            
                overall_pitcher_coverage = (
                    games_with_both_starters
                    / total_pitcher_games
                    if total_pitcher_games
                    else 0.0
                )
            
                pitcher_coverage_by_season = (
                    pitcher_coverage
                    .groupby(
                        "season",
                        as_index=False,
                    )
                    .agg(
                        games=(
                            "game_id",
                            "count",
                        ),
                        games_with_both_starters=(
                            "both_starters",
                            "sum",
                        ),
                    )
                )
            
                pitcher_coverage_by_season[
                    "coverage"
                ] = (
                    pitcher_coverage_by_season[
                        "games_with_both_starters"
                    ]
                    /
                    pitcher_coverage_by_season[
                        "games"
                    ]
                )
            
            else:
            
                total_pitcher_games = len(
                    mlb_games
                )
            
                games_with_both_starters = 0
            
                overall_pitcher_coverage = 0.0
            
                pitcher_coverage_by_season = (
                    pd.DataFrame()
                )

            if mlb_games.empty:
                raise ValueError(
                    "No completed MLB games were returned."
                )

            mlb_features = build_mlb_pregame_features(
                mlb_games
            )

            if mlb_features.empty:
                raise ValueError(
                    "No MLB pregame features were generated."
                )

            mlb_v1_predictions = (
                run_mlb_walkforward_v1(
                    mlb_features,
                    min_train_games=500,
                    retrain_every=100,
                )
            )

            mlb_v1_results = (
                summarize_mlb_walkforward_v1(
                    mlb_v1_predictions
                )
            )

            st.success(
                "MLB V1 walk-forward benchmark completed."
            )

    except Exception as exc:
        st.error(
            f"MLB V1 benchmark failed: {exc}"
        )
        st.exception(exc)


# ==========================================
# MLB V2A PITCHER RESEARCH
# ==========================================

mlb_v2a_games = st.session_state.get(
    "mlb_v2a_games"
)

cached_pitcher_logs = st.session_state.get(
    "mlb_v2a_pitcher_logs",
    pd.DataFrame(),
)

if (
    mlb_v2a_games is not None
    and not mlb_v2a_games.empty
):
    st.divider()

    st.subheader(
        "⚾ MLB V2A Pitcher Research"
    )

    missing_pitcher_games = (
        get_missing_mlb_pitcher_log_games(
            mlb_v2a_games,
            cached_pitcher_logs,
        )
    )

    total_games = len(mlb_v2a_games)

    coverage_games = (
        total_games
        - len(missing_pitcher_games)
    )

    coverage_pct = (
        coverage_games / total_games
        if total_games
        else 0.0
    )

    cache_col1, cache_col2, cache_col3 = (
        st.columns(3)
    )

    cache_col1.metric(
        "Pitcher Rows Collected",
        f"{len(cached_pitcher_logs):,}",
    )

    cache_col2.metric(
        "Games Remaining",
        f"{len(missing_pitcher_games):,}",
    )

    cache_col3.metric(
        "V2A Data Progress",
        f"{coverage_pct:.1%}",
    )

    # ==========================================
    # PERMANENT PITCHER HISTORY BACKUP
    # ==========================================

    if (
        cached_pitcher_logs is not None
        and not cached_pitcher_logs.empty
    ):

        if st.button(
            "💾 Save Pitcher History Permanently",
            key="save_mlb_pitcher_history_permanently",
        ):

            with st.spinner(
                "Saving MLB pitcher history permanently..."
            ):

                save_result = (
                    save_mlb_pitcher_history(
                        cached_pitcher_logs
                    )
                )

            if save_result.get(
                "success",
                False,
            ):

                st.success(
                    "MLB pitcher history permanently saved — "
                    f"{save_result['rows_saved']:,} "
                    "pitcher rows."
                )

            else:

                st.error(
                    "Permanent pitcher-history save failed: "
                    f"{save_result.get('error')}"
                )

    # ==========================================
    # HISTORICAL PITCHER DATA STATUS
    # ==========================================

    if missing_pitcher_games.empty:

        st.success(
            "Historical MLB starting-pitcher "
            "dataset is complete."
        )

    else:

        st.caption(
            "Historical pitcher data only needs to be "
            "built once. The automated builder works "
            "in 250-game checkpoints and permanently "
            "saves each successful checkpoint."
        )

        if st.button(
            "Build Remaining MLB Pitcher History",
            key="build_remaining_mlb_pitcher_history",
            type="primary",
        ):

            working_logs = (
                cached_pitcher_logs.copy()
            )

            total_requested = 0
            total_new_rows = 0
            completed_batches = 0

            progress_bar = st.progress(
                coverage_pct
            )

            status_box = st.empty()

            try:

                while True:

                    remaining_before = (
                        get_missing_mlb_pitcher_log_games(
                            mlb_v2a_games,
                            working_logs,
                        )
                    )

                    if remaining_before.empty:
                        break

                    completed_batches += 1

                    batch_target = min(
                        250,
                        len(remaining_before),
                    )

                    status_box.info(
                        "Building historical pitcher data — "
                        f"checkpoint {completed_batches:,} | "
                        f"{len(remaining_before):,} games "
                        "remaining..."
                    )

                    collection = (
                        collect_mlb_pitcher_logs_batch(
                            games=mlb_v2a_games,
                            existing_logs=working_logs,
                            batch_size=batch_target,
                        )
                    )

                    batch_requested = int(
                        collection.get(
                            "requested_games",
                            0,
                        )
                    )

                    batch_new_rows = int(
                        collection.get(
                            "new_pitcher_rows",
                            0,
                        )
                    )

                    updated_logs = collection.get(
                        "logs",
                        working_logs,
                    )

                    # ------------------------------
                    # CHECKPOINT SUCCESSFUL WORK
                    # ------------------------------

                    working_logs = (
                        updated_logs.copy()
                    )

                    # Keep the current Streamlit
                    # session updated.
                    st.session_state[
                        "mlb_v2a_pitcher_logs"
                    ] = working_logs

                    # Permanently save every
                    # successful checkpoint.
                    if (
                        working_logs is not None
                        and not working_logs.empty
                    ):

                        save_result = (
                            save_mlb_pitcher_history(
                                working_logs
                            )
                        )

                        if not save_result.get(
                            "success",
                            False,
                        ):

                            raise RuntimeError(
                                "Pitcher checkpoint was "
                                "collected but could not be "
                                "permanently saved: "
                                f"{save_result.get('error')}"
                            )

                    total_requested += (
                        batch_requested
                    )

                    total_new_rows += (
                        batch_new_rows
                    )

                    # ------------------------------
                    # RECALCULATE REMAINING GAMES
                    # ------------------------------

                    remaining_after = (
                        get_missing_mlb_pitcher_log_games(
                            mlb_v2a_games,
                            working_logs,
                        )
                    )

                    completed_games = (
                        total_games
                        - len(remaining_after)
                    )

                    current_progress = (
                        completed_games
                        / total_games
                        if total_games
                        else 0.0
                    )

                    progress_bar.progress(
                        min(
                            max(
                                current_progress,
                                0.0,
                            ),
                            1.0,
                        )
                    )

                    status_box.info(
                        "Building historical pitcher data — "
                        f"{len(working_logs):,} pitcher rows "
                        f"cached | "
                        f"{len(remaining_after):,} games "
                        "remaining | "
                        f"{current_progress:.1%} complete"
                    )

                    # ------------------------------
                    # COMPLETE
                    # ------------------------------

                    if remaining_after.empty:
                        break

                    if collection.get(
                        "complete",
                        False,
                    ):
                        break

                    # ------------------------------
                    # SAFETY: NO PROGRESS
                    # ------------------------------

                    if (
                        len(remaining_after)
                        >= len(remaining_before)
                    ):

                        st.warning(
                            "Historical pitcher build stopped "
                            "because the latest checkpoint made "
                            "no additional progress. Completed "
                            "data was preserved."
                        )

                        break

                    if batch_requested == 0:
                        break

            except Exception as exc:

                st.session_state[
                    "mlb_v2a_pitcher_logs"
                ] = working_logs

                status_box.error(
                    "Historical pitcher build encountered "
                    "an error. All completed checkpoints "
                    "were preserved."
                )

                st.exception(exc)

            # ======================================
            # FINAL COLLECTION STATUS
            # ======================================

            final_missing = (
                get_missing_mlb_pitcher_log_games(
                    mlb_v2a_games,
                    working_logs,
                )
            )

            final_coverage_games = (
                total_games
                - len(final_missing)
            )

            final_coverage_pct = (
                final_coverage_games
                / total_games
                if total_games
                else 0.0
            )

            st.session_state[
                "mlb_v2a_last_collection"
            ] = {
                "requested_games":
                    total_requested,
                "new_pitcher_rows":
                    total_new_rows,
                "cached_pitcher_rows":
                    len(working_logs),
                "remaining_games":
                    len(final_missing),
                "coverage":
                    final_coverage_pct,
            }

            if final_missing.empty:

                progress_bar.progress(1.0)

                status_box.success(
                    "Historical MLB pitcher dataset "
                    "build complete."
                )

                st.success(
                    "V2A historical pitcher dataset is "
                    f"complete — {len(working_logs):,} "
                    "pitcher rows cached and permanently "
                    "saved. We can now run the full "
                    "leakage-safe V2A walk-forward "
                    "validation."
                )

            else:

                st.info(
                    "Historical build checkpoint finished — "
                    f"{total_requested:,} games processed "
                    "during this run, "
                    f"{total_new_rows:,} pitcher rows added, "
                    f"{len(final_missing):,} games remain, "
                    f"{final_coverage_pct:.1%} complete."
                )

    # ==========================================
    # MLB V2A WALK-FORWARD VALIDATION
    # ==========================================

    st.divider()

    st.subheader(
        "⚾ MLB V2A Walk-Forward Validation"
    )

    st.caption(
        "Tests the MLB model with starting-pitcher "
        "intelligence and compares it directly with "
        "the original V1 model."
    )

    mlb_v2a_games = st.session_state.get(
        "mlb_v2a_games"
    )

    mlb_v2a_pitcher_logs = st.session_state.get(
        "mlb_v2a_pitcher_logs",
        pd.DataFrame(),
    )

    v2a_ready = (
        mlb_v2a_games is not None
        and not mlb_v2a_games.empty
        and mlb_v2a_pitcher_logs is not None
        and not mlb_v2a_pitcher_logs.empty
    )

    if not v2a_ready:

        st.info(
            "V2A validation will become available after "
            "the MLB historical games and pitcher history "
            "have been collected."
        )

    else:

        remaining_v2a_games = (
            get_missing_mlb_pitcher_log_games(
                mlb_v2a_games,
                mlb_v2a_pitcher_logs,
            )
        )

        validation_col1, validation_col2 = (
            st.columns(2)
        )

        validation_col1.metric(
            "Historical Games",
            f"{len(mlb_v2a_games):,}",
        )

        validation_col2.metric(
            "Pitcher Rows",
            f"{len(mlb_v2a_pitcher_logs):,}",
        )

        if len(remaining_v2a_games) > 0:

            st.caption(
                f"{len(remaining_v2a_games):,} historical "
                "games do not have complete pitcher records. "
                "They will not prevent V2A validation."
            )

        if st.button(
            "Run MLB V2A Walk-Forward Validation",
            key="run_mlb_v2a_walkforward_validation",
            type="primary",
        ):

            try:

                with st.spinner(
                    "Building leakage-safe pitcher features "
                    "and running V1 vs V2A walk-forward "
                    "validation..."
                ):

                    # ======================================
                    # BUILD ORIGINAL V1 TEAM FEATURES
                    # ======================================

                    mlb_v1_features = (
                        build_mlb_pregame_features(
                            mlb_v2a_games
                        )
                    )

                    if mlb_v1_features.empty:

                        raise ValueError(
                            "MLB V1 feature generation "
                            "returned no data."
                        )

                    # ======================================
                    # BUILD V2A PITCHER FEATURES
                    # ======================================

                    mlb_v2a_features = (
                        build_mlb_v2a_features(
                            games=mlb_v2a_games,
                            team_features=mlb_v1_features,
                            pitcher_logs=mlb_v2a_pitcher_logs,
                        )
                    )

                    if mlb_v2a_features.empty:

                        raise ValueError(
                            "MLB V2A feature generation "
                            "returned no data."
                        )

                    # ======================================
                    # RUN V1 ON SAME DATASET
                    # ======================================

                    mlb_v1_comparison_predictions = (
                        run_mlb_walkforward_v1(
                            mlb_v1_features,
                            min_train_games=500,
                            retrain_every=100,
                        )
                    )

                    mlb_v1_comparison_results = (
                        summarize_mlb_walkforward_v1(
                            mlb_v1_comparison_predictions
                        )
                    )

                    # ======================================
                    # RUN V2A
                    # ======================================

                    mlb_v2a_predictions = (
                        run_mlb_walkforward_v2a(
                            mlb_v2a_features,
                            min_train_games=500,
                            retrain_every=100,
                        )
                    )

                    mlb_v2a_results = (
                        summarize_mlb_walkforward_v2a(
                            mlb_v2a_predictions
                        )
                    )

                    # ======================================
                    # STORE RESULTS
                    # ======================================

                    st.session_state[
                        "mlb_v1_comparison_results"
                    ] = mlb_v1_comparison_results

                    st.session_state[
                        "mlb_v2a_results"
                    ] = mlb_v2a_results

                    st.session_state[
                        "mlb_v2a_features"
                    ] = mlb_v2a_features

                st.success(
                    "MLB V2A walk-forward validation "
                    "completed."
                )

            except Exception as exc:

                st.error(
                    "MLB V2A walk-forward validation failed."
                )

                st.exception(exc)


    # ==========================================
    # MLB V1 VS V2A MODEL COMPARISON
    # ==========================================

    v1_compare = st.session_state.get(
        "mlb_v1_comparison_results"
    )

    v2a_compare = st.session_state.get(
        "mlb_v2a_results"
    )

    if v1_compare and v2a_compare:

        st.divider()

        st.subheader(
            "📊 MLB V1 vs V2A Model Comparison"
        )

        v1_accuracy = float(
            v1_compare["accuracy"]
        )

        v2a_accuracy = float(
            v2a_compare["accuracy"]
        )

        accuracy_change = (
            v2a_accuracy
            - v1_accuracy
        )

        v1_auc = v1_compare.get("auc")
        v2a_auc = v2a_compare.get("auc")

        v1_brier = float(
            v1_compare["brier"]
        )

        v2a_brier = float(
            v2a_compare["brier"]
        )

        v1_log_loss = float(
            v1_compare["log_loss"]
        )

        v2a_log_loss = float(
            v2a_compare["log_loss"]
        )

        comparison_data = pd.DataFrame(
            {
                "Metric": [
                    "Predictions",
                    "Accuracy",
                    "AUC",
                    "Brier Score",
                    "Log Loss",
                ],
                "V1": [
                    f"{v1_compare['prediction_count']:,}",
                    f"{v1_accuracy:.2%}",
                    (
                        f"{v1_auc:.3f}"
                        if v1_auc is not None
                        else "N/A"
                    ),
                    f"{v1_brier:.4f}",
                    f"{v1_log_loss:.4f}",
                ],
                "V2A": [
                    f"{v2a_compare['prediction_count']:,}",
                    f"{v2a_accuracy:.2%}",
                    (
                        f"{v2a_auc:.3f}"
                        if v2a_auc is not None
                        else "N/A"
                    ),
                    f"{v2a_brier:.4f}",
                    f"{v2a_log_loss:.4f}",
                ],
            }
        )

        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True,
        )

        # ======================================
        # ACCURACY IMPACT
        # ======================================

        st.markdown(
            "### Starting-Pitcher Impact"
        )

        impact_col1, impact_col2, impact_col3 = (
            st.columns(3)
        )

        impact_col1.metric(
            "V1 Accuracy",
            f"{v1_accuracy:.2%}",
        )

        impact_col2.metric(
            "V2A Accuracy",
            f"{v2a_accuracy:.2%}",
        )

        impact_col3.metric(
            "Accuracy Change",
            f"{accuracy_change:+.2%}",
        )

        # ======================================
        # MODEL QUALITY CHECK
        # ======================================

        accuracy_improved = (
            v2a_accuracy > v1_accuracy
        )

        brier_improved = (
            v2a_brier < v1_brier
        )

        log_loss_improved = (
            v2a_log_loss < v1_log_loss
        )

        auc_improved = (
            v1_auc is not None
            and v2a_auc is not None
            and v2a_auc > v1_auc
        )

        improvements = sum(
            [
                accuracy_improved,
                brier_improved,
                log_loss_improved,
                auc_improved,
            ]
        )

        if improvements == 4:

            st.success(
                "V2A improved all four major validation "
                "metrics: accuracy, AUC, Brier score, "
                "and log loss."
            )

        elif accuracy_improved:

            st.info(
                "V2A improved prediction accuracy, but "
                "the probability-quality metrics are mixed. "
                "Review the detailed results before deciding "
                "whether V2A should replace V1."
            )

        else:

            st.warning(
                "V2A did not improve overall prediction "
                "accuracy. V1 remains the benchmark while "
                "we investigate the pitcher features."
            )

        # ======================================
        # CONFIDENCE COMPARISON
        # ======================================

        st.markdown(
            "### Accuracy by Confidence"
        )

        confidence_col1, confidence_col2 = (
            st.columns(2)
        )

        with confidence_col1:

            st.markdown("#### V1")

            st.dataframe(
                v1_compare[
                    "confidence_summary"
                ],
                use_container_width=True,
                hide_index=True,
            )

        with confidence_col2:

            st.markdown("#### V2A")

            st.dataframe(
                v2a_compare[
                    "confidence_summary"
                ],
                use_container_width=True,
                hide_index=True,
            )

        # ======================================
        # SEASON COMPARISON
        # ======================================

        st.markdown(
            "### Season-by-Season Accuracy"
        )

        v1_seasons = (
            v1_compare[
                "season_summary"
            ]
            .rename(
                columns={
                    "predictions":
                        "v1_predictions",
                    "accuracy":
                        "v1_accuracy",
                }
            )
        )

        v2a_seasons = (
            v2a_compare[
                "season_summary"
            ]
            .rename(
                columns={
                    "predictions":
                        "v2a_predictions",
                    "accuracy":
                        "v2a_accuracy",
                }
            )
        )

        season_comparison = (
            v1_seasons.merge(
                v2a_seasons,
                on="season_id",
                how="outer",
            )
        )

        season_comparison[
            "accuracy_change"
        ] = (
            season_comparison[
                "v2a_accuracy"
            ]
            - season_comparison[
                "v1_accuracy"
            ]
        )

        for column in [
            "v1_accuracy",
            "v2a_accuracy",
            "accuracy_change",
        ]:

            season_comparison[column] = (
                season_comparison[column]
                .map(
                    lambda value:
                        f"{value:+.2%}"
                        if (
                            column
                            == "accuracy_change"
                            and pd.notna(value)
                        )
                        else (
                            f"{value:.2%}"
                            if pd.notna(value)
                            else "—"
                        )
                )
            )

        st.dataframe(
            season_comparison,
            use_container_width=True,
            hide_index=True,
        )

    # ==========================================
    # DISPLAY MLB VALIDATION RESULTS
    # ==========================================

    mlb_metrics = st.session_state.get(
        "mlb_training_metrics"
    )

    if mlb_metrics:
        st.subheader("MLB Model Validation Results")

        if mlb_metrics["passes_benchmarks"]:
            st.success(
                "Historical validation benchmarks passed."
            )
        else:
            st.warning(
                "Historical validation benchmarks not passed. "
                "Further model evaluation is required."
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Model Accuracy",
                f"{mlb_metrics['model_accuracy']:.2%}",
            )
            st.metric(
                "Baseline Accuracy",
                f"{mlb_metrics['baseline_accuracy']:.2%}",
            )

        with col2:
            st.metric(
                "Model Brier Score",
                f"{mlb_metrics['model_brier']:.4f}",
            )
            st.metric(
                "Baseline Brier Score",
                f"{mlb_metrics['baseline_brier']:.4f}",
            )

        with col3:
            st.metric(
                "Model Log Loss",
                f"{mlb_metrics['model_log_loss']:.4f}",
            )
            st.metric(
                "AUC Score",
                (
                    f"{mlb_metrics['auc']:.3f}"
                    if mlb_metrics["auc"] is not None
                    else "N/A"
                ),
            )

        st.subheader("Historical Training Summary")
        st.json(mlb_metrics)

# ==========================================
# MLB V3 HISTORICAL PLAYER RESEARCH
# ==========================================

st.divider()
st.subheader("⚾ MLB V3 Historical Player Research")
st.caption(
    "Build the permanent player-game warehouse used by the V3 "
    "leakage-safe historical reconstruction. Each successful "
    "250-game checkpoint is saved permanently to Supabase."
)

# V3 must also work before any V1 benchmark or V2A action has run.
mlb_v3_games = st.session_state.get("mlb_v3_games")
if mlb_v3_games is None or mlb_v3_games.empty:
    mlb_v3_games = st.session_state.get("mlb_v2a_games")

v3_history_ready = False
try:
    if mlb_v3_games is None or mlb_v3_games.empty:
        with st.spinner("Loading MLB historical games for V3..."):
            mlb_v3_games = fetch_mlb_games(
                start_date="2023-03-01",
                end_date="2026-09-30",
            )

    if mlb_v3_games is None or mlb_v3_games.empty:
        st.warning(
            "No completed MLB historical games were loaded. "
            "Reload the page to retry before building V3 player history."
        )
    else:
        st.session_state["mlb_v3_games"] = mlb_v3_games
        if st.session_state.get("mlb_v3_player_logs") is None:
            with st.spinner("Loading permanent MLB player history..."):
                st.session_state["mlb_v3_player_logs"] = load_mlb_player_history()

        cached_player_logs = st.session_state.get("mlb_v3_player_logs")
        if cached_player_logs is None:
            raise RuntimeError("Permanent MLB player history did not load. Reload the page to retry.")

        missing_player_games = get_missing_mlb_player_log_games(
            mlb_v3_games,
            existing_logs=cached_player_logs,
        )
        v3_history_ready = True
except Exception as exc:
    st.error("MLB V3 history could not be initialized. Reload the page to retry.")
    st.exception(exc)

if v3_history_ready:
    player_total_games = len(mlb_v3_games)
    player_completed_games = (
        player_total_games
        - len(missing_player_games)
    )
    player_coverage_pct = (
        player_completed_games / player_total_games
        if player_total_games
        else 0.0
    )

    player_count = (
        int(cached_player_logs["player_id"].nunique())
        if (
            cached_player_logs is not None
            and not cached_player_logs.empty
            and "player_id" in cached_player_logs.columns
        )
        else 0
    )

    player_col1, player_col2, player_col3, player_col4 = (
        st.columns(4)
    )

    player_col1.metric(
        "Player Rows Collected",
        f"{len(cached_player_logs):,}",
    )
    player_col2.metric(
        "Unique Players",
        f"{player_count:,}",
    )
    player_col3.metric(
        "Games Remaining",
        f"{len(missing_player_games):,}",
    )
    player_col4.metric(
        "V3 Data Progress",
        f"{player_coverage_pct:.1%}",
    )

    st.progress(
        min(max(player_coverage_pct, 0.0), 1.0)
    )

    if missing_player_games.empty:

        st.success(
            "Historical MLB player-game warehouse is 100% complete. "
            "The permanent V3 cache is ready for historical feature "
            "reconstruction and challenger-model validation."
        )

    else:

        st.caption(
            "The builder resumes from the permanent Supabase checkpoint, "
            "fetches up to 10 MLB boxscores concurrently, skips games "
            "already collected, and saves after every successful "
            "250-game checkpoint."
        )

        if st.button(
            "Build Remaining MLB Player History",
            key="build_remaining_mlb_player_history",
            type="primary",
        ):

            working_player_logs = cached_player_logs.copy()
            total_player_requested = 0
            total_player_new_rows = 0
            player_batches = 0

            player_progress_bar = st.progress(
                player_coverage_pct
            )
            player_status_box = st.empty()

            try:

                while True:

                    remaining_before = (
                        get_missing_mlb_player_log_games(
                            mlb_v3_games,
                            existing_logs=working_player_logs,
                        )
                    )

                    if remaining_before.empty:
                        break

                    player_batches += 1
                    batch_target = min(
                        250,
                        len(remaining_before),
                    )

                    player_status_box.info(
                        "Building MLB V3 player history — "
                        f"checkpoint {player_batches:,} | "
                        f"{len(remaining_before):,} games remaining..."
                    )

                    collection = collect_mlb_player_logs_batch(
                        games=mlb_v3_games,
                        existing_logs=working_player_logs,
                        batch_size=batch_target,
                    )

                    batch_requested = int(
                        collection.get("games_requested", 0)
                    )
                    new_logs = collection.get(
                        "new_logs",
                        pd.DataFrame(),
                    )
                    updated_logs = collection.get(
                        "combined_logs",
                        working_player_logs,
                    )

                    working_player_logs = updated_logs.copy()
                    st.session_state[
                        "mlb_v3_player_logs"
                    ] = working_player_logs

                    if (
                        working_player_logs is not None
                        and not working_player_logs.empty
                    ):

                        save_result = save_mlb_player_history(
                            working_player_logs
                        )

                        if not save_result.get("success", False):
                            raise RuntimeError(
                                "Player checkpoint was collected but could "
                                "not be permanently saved: "
                                f"{save_result.get('error')}"
                            )

                    batch_new_rows = (
                        len(new_logs)
                        if new_logs is not None
                        else 0
                    )
                    total_player_requested += batch_requested
                    total_player_new_rows += batch_new_rows

                    remaining_after = (
                        get_missing_mlb_player_log_games(
                            mlb_v3_games,
                            existing_logs=working_player_logs,
                        )
                    )

                    completed_games = (
                        player_total_games
                        - len(remaining_after)
                    )
                    current_progress = (
                        completed_games / player_total_games
                        if player_total_games
                        else 0.0
                    )

                    current_players = (
                        int(working_player_logs["player_id"].nunique())
                        if (
                            not working_player_logs.empty
                            and "player_id" in working_player_logs.columns
                        )
                        else 0
                    )

                    player_progress_bar.progress(
                        min(max(current_progress, 0.0), 1.0)
                    )
                    player_status_box.info(
                        "Building MLB V3 player history — "
                        f"{len(working_player_logs):,} rows | "
                        f"{current_players:,} players | "
                        f"{len(remaining_after):,} games remaining | "
                        f"{current_progress:.1%} complete"
                    )

                    if remaining_after.empty:
                        break

                    if collection.get("complete", False):
                        break

                    if len(remaining_after) >= len(remaining_before):
                        st.warning(
                            "Player-history build stopped because the latest "
                            "checkpoint made no additional progress. All "
                            "completed data was preserved in Supabase."
                        )
                        break

                    if batch_requested == 0:
                        break

            except Exception as exc:

                st.session_state[
                    "mlb_v3_player_logs"
                ] = working_player_logs

                player_status_box.error(
                    "MLB V3 player-history build encountered an error. "
                    "All successfully saved checkpoints were preserved."
                )
                st.exception(exc)

            final_missing_players = (
                get_missing_mlb_player_log_games(
                    mlb_v3_games,
                    existing_logs=working_player_logs,
                )
            )
            final_player_coverage = (
                (player_total_games - len(final_missing_players))
                / player_total_games
                if player_total_games
                else 0.0
            )
            final_unique_players = (
                int(working_player_logs["player_id"].nunique())
                if (
                    not working_player_logs.empty
                    and "player_id" in working_player_logs.columns
                )
                else 0
            )

            st.session_state[
                "mlb_v3_last_player_collection"
            ] = {
                "requested_games": total_player_requested,
                "new_player_rows": total_player_new_rows,
                "cached_player_rows": len(working_player_logs),
                "unique_players": final_unique_players,
                "remaining_games": len(final_missing_players),
                "coverage": final_player_coverage,
            }

            if final_missing_players.empty:
                player_progress_bar.progress(1.0)
                player_status_box.success(
                    "MLB V3 historical player warehouse is 100% complete."
                )
                st.success(
                    f"V3 player history complete — "
                    f"{len(working_player_logs):,} player-game rows across "
                    f"{final_unique_players:,} players are permanently saved."
                )
            else:
                st.info(
                    "Player-history checkpoint finished — "
                    f"{total_player_requested:,} games processed this run, "
                    f"{total_player_new_rows:,} player rows added, "
                    f"{len(final_missing_players):,} games remain, "
                    f"{final_player_coverage:.1%} complete."
                )



# ==========================================
# MLB V3 WAREHOUSE AUDIT (READ ONLY)
# ==========================================
def _audit_mlb_v3_warehouse(games, player_logs):
    """Screen stored history without modifying it or requesting boxscores."""
    checks = []
    details = {}

    def check(name, count, note, warning=False):
        checks.append({"Check": name, "Status": ("REVIEW" if warning else "FAIL") if count else "PASS",
                       "Issues": int(count), "Meaning": note})

    required_games = ["game_id", "season_id", "start_time", "home_team_id", "away_team_id"]
    required_logs = ["game_id", "season_id", "start_time", "player_id", "side", "team_id",
                     "opponent_team_id", "batting_order", "pitching_games"]
    stats = ["batting_games", "plate_appearances", "at_bats", "hits", "doubles", "triples",
             "batting_home_runs", "batting_runs", "rbi", "batting_walks", "batting_strikeouts",
             "stolen_bases", "pitching_games", "games_started", "pitching_outs", "batters_faced",
             "pitcher_strikeouts", "pitcher_walks", "pitcher_hits", "pitcher_home_runs",
             "earned_runs", "pitches", "strikes"]
    absent = ["games." + c for c in required_games if c not in games.columns]
    absent += ["players." + c for c in dict.fromkeys(required_logs + stats) if c not in player_logs.columns]
    check("Required columns", len(absent), ", ".join(absent) or "Expected warehouse fields exist.")
    if absent:
        return {"checks": pd.DataFrame(checks), "details": details, "passed": False}

    g, p = games.copy(), player_logs.copy()
    for frame, cols in [(g, ["game_id", "season_id", "home_team_id", "away_team_id"]),
                        (p, ["game_id", "season_id", "player_id", "team_id", "opponent_team_id"])]:
        invalid = pd.Series(False, index=frame.index)
        for c in cols:
            values = pd.to_numeric(frame[c], errors="coerce")
            invalid |= values.isna() | values.le(0) | values.mod(1).ne(0)
            frame[c] = values
        frame["_invalid_id"] = invalid
        frame["start_time"] = pd.to_datetime(frame["start_time"], utc=True, errors="coerce")
    check("Historical game identifiers", g._invalid_id.sum(), "IDs and seasons must be positive whole numbers.")
    check("Player identifiers", p._invalid_id.sum(), "IDs and seasons must be positive whole numbers.")
    check("Historical game timestamps", g.start_time.isna().sum(), "Each scheduled game needs a valid timestamp.")
    check("Player timestamps", p.start_time.isna().sum(), "Each history row needs a valid timestamp.")
    check("Duplicate historical games", g.duplicated("game_id").sum(), "One schedule row per game.")
    dup = p.duplicated(["game_id", "player_id"], keep=False)
    check("Duplicate player-game rows", dup.sum(), "Every player-game key must be unique; all affected rows counted.")
    details["Duplicate player-game rows"] = p.loc[dup].drop(columns="_invalid_id").head(250)
    check("Invalid home/away labels", (~p.side.isin(["home", "away"])).sum(), "Every row must belong to home or away.")
    expected = g.drop_duplicates("game_id")
    known = p.game_id.isin(expected.game_id)
    check("Rows outside historical schedule", (~known).sum(), "Review rows from games outside this collection window.", warning=True)
    represented = set(p.loc[~p._invalid_id, "game_id"])
    missing = expected.loc[~expected.game_id.isin(represented)]
    check("Games without player history", len(missing), "Every expected completed game must have history.")
    details["Games without player history"] = missing.drop(columns="_invalid_id").head(250)
    joined = p.loc[known].merge(expected, on="game_id", suffixes=("_player", "_game"), how="left")
    home = joined.side.eq("home")
    team = joined.home_team_id.where(home, joined.away_team_id)
    opponent = joined.away_team_id.where(home, joined.home_team_id)
    mismatch = joined.team_id.ne(team) | joined.opponent_team_id.ne(opponent)
    check("Team/opponent mismatches", mismatch.sum(), "Player teams must match the game's home/away teams.")
    check("Season mismatches", joined.season_id_player.ne(joined.season_id_game).sum(), "Player and schedule seasons must match.")
    check("Timestamp mismatches", joined.start_time_player.ne(joined.start_time_game).sum(),
          "Different stored timestamps need review before chronological reconstruction.", warning=True)

    numeric = p[stats].apply(pd.to_numeric, errors="coerce")
    field_quality = pd.DataFrame({"Field": stats, "Missing or nonnumeric": numeric.isna().sum().values,
                                  "Negative values": numeric.lt(0).sum().values})
    check("Missing/nonnumeric statistics", numeric.isna().sum().sum(), "Zero is valid; missing and nonnumeric values require review.")
    check("Negative count statistics", numeric.lt(0).sum().sum(), "Stored boxscore counts must be nonnegative.")
    details["Statistic field quality"] = field_quality
    # batting_order can legitimately be absent for pitchers or unused players.
    # Minimum lineup/pitcher coverage is a screening rule, not proof of a full boxscore.
    valid = p.loc[known & ~p._invalid_id & p.side.isin(["home", "away"])].drop_duplicates(["game_id", "player_id"]).copy()
    order = pd.to_numeric(valid.batting_order, errors="coerce")
    valid["_lineup"] = order.between(1, 9)
    valid["_pitcher"] = pd.to_numeric(valid.pitching_games, errors="coerce").gt(0)
    sides = valid.groupby(["game_id", "side"]).agg(
        players=("player_id", "nunique"), lineup_players=("_lineup", "sum"), pitchers=("_pitcher", "sum"))
    side_index = pd.MultiIndex.from_product([expected.game_id.dropna().unique(), ["home", "away"]], names=["game_id", "side"])
    sides = sides.reindex(side_index, fill_value=0).reset_index()
    suspect = sides.loc[sides.lineup_players.lt(9) | sides.pitchers.lt(1)]
    check("Possible incomplete game sides", len(suspect), "Review sides with fewer than nine lineup players or no pitching line; unusual games may be legitimate.", warning=True)
    details["Possible incomplete game sides"] = suspect
    season = expected.groupby("season_id").agg(expected_games=("game_id", "nunique"),
                                                   first_game=("start_time", "min"), last_game=("start_time", "max"))
    covered = expected.loc[expected.game_id.isin(represented)].groupby("season_id").game_id.nunique()
    season["covered_games"] = covered.reindex(season.index, fill_value=0)
    season["missing_games"] = season.expected_games - season.covered_games
    season["player_rows"] = p.groupby("season_id").size().reindex(season.index, fill_value=0)
    season["unique_players"] = p.groupby("season_id").player_id.nunique().reindex(season.index, fill_value=0)
    details["Season coverage"] = season.reset_index()
    dates = expected.start_time.dt.strftime("%Y-%m")
    months = expected.assign(month=dates).groupby("month").agg(expected_games=("game_id", "nunique"))
    months["covered_games"] = expected.loc[expected.game_id.isin(represented)].assign(
        month=lambda x: x.start_time.dt.strftime("%Y-%m")).groupby("month").game_id.nunique().reindex(months.index, fill_value=0)
    months["missing_games"] = months.expected_games - months.covered_games
    details["Monthly coverage"] = months.reset_index()
    result = pd.DataFrame(checks)
    return {"checks": result, "details": details, "passed": result.Status.eq("PASS").all()}


if v3_history_ready:
    st.divider()
    st.subheader("⚾ MLB V3 Warehouse Audit")
    st.caption(
        "Check saved player history before building V3 features. This audit reads existing data "
        "and makes no boxscore requests. Coverage is measured against the loaded historical schedule."
    )
    if st.button("Run MLB V3 Warehouse Audit", key="run_mlb_v3_warehouse_audit", type="primary"):
        try:
            with st.spinner("Auditing MLB player history..."):
                audit = _audit_mlb_v3_warehouse(
                    mlb_v3_games, st.session_state["mlb_v3_player_logs"]
                )
            # Do not keep stale audit results across changes to the warehouse.
            st.session_state.pop("mlb_v3_warehouse_audit", None)
            st.dataframe(audit["checks"], use_container_width=True, hide_index=True)
            if audit["passed"]:
                st.success("Warehouse screening passed. Next: verify historical feature timing before the V3 model comparison.")
            else:
                st.warning("The audit found issues to inspect before V3 modeling. Review the tables below.")
            for title, table in audit["details"].items():
                if not table.empty:
                    with st.expander(title, expanded=title == "Season coverage"):
                        st.dataframe(table, use_container_width=True, hide_index=True)
            st.caption(
                "Duplicate samples are limited to 250 rows. Missing batting order for non-lineup players is normal. "
                "Stored zeroes may also represent source fields the collector did not receive; this audit cannot detect that. "
                "Passing these checks does not prove every boxscore is complete or that model features avoid future information."
            )
        except Exception as exc:
            st.error("The warehouse audit could not finish.")
            st.exception(exc)


# ==========================================
# MLB V3 HISTORICAL FEATURE TIMING CHECK
# ==========================================
if v3_history_ready:
    st.divider()
    st.subheader("⚾ MLB V3 Historical Feature Check")
    st.caption(
        "Build history-only features after the warehouse audit passes. Lineup uses the last "
        "observed lineup from an earlier UTC date; bullpen uses prior relief appearances. "
        "Current-game participants, starter identities and weather are not reconstructed as pregame facts."
    )
    if st.button("Check Timing and Build MLB V3 Features", key="check_build_mlb_v3_features", type="primary"):
        try:
            from dual_agent.mlb_research import (
                build_mlb_v3_historical_features,
                summarize_mlb_v3_historical_readiness,
            )
            with st.spinner("Checking warehouse and feature timing..."):
                current_logs = st.session_state["mlb_v3_player_logs"]
                current_audit = _audit_mlb_v3_warehouse(mlb_v3_games, current_logs)
                if not current_audit["passed"]:
                    raise ValueError("Warehouse audit has unresolved issues. Run the audit and review its tables first.")
                sample_games = mlb_v3_games.copy()
                sample_games["start_time"] = pd.to_datetime(sample_games.start_time, utc=True)
                sample_games = sample_games.sort_values(["start_time", "game_id"])
                sample_dates = sample_games.start_time.dt.floor("D").drop_duplicates().head(6)
                if len(sample_dates) < 3:
                    raise ValueError("At least three historical UTC dates are needed for the timing check.")
                sample_games = sample_games.loc[sample_games.start_time.dt.floor("D").isin(sample_dates)].copy()
                sample_logs = current_logs.loc[current_logs.game_id.isin(sample_games.game_id)].copy()
                baseline = build_mlb_v3_historical_features(sample_games, sample_logs)
                cutoff = sample_dates.iloc[-2]
                altered_games = sample_games.copy()
                altered_logs = sample_logs.copy()
                affected = altered_games.start_time.dt.floor("D").ge(cutoff)
                altered_games.loc[affected, "home_score"] = 0
                altered_games.loc[affected, "away_score"] = 99
                affected_logs = altered_logs.game_id.isin(altered_games.loc[affected, "game_id"])
                for field in ["hits", "earned_runs", "pitcher_hits", "batting_order"]:
                    altered_logs.loc[affected_logs, field] = 99
                altered_logs.loc[affected_logs, "player_id"] = 999999999
                altered_logs.loc[affected_logs, "is_bullpen"] = False
                altered = build_mlb_v3_historical_features(altered_games, altered_logs)
                columns = [c for c in baseline.columns if c != "home_win"]
                ids = sample_games.loc[sample_games.start_time.dt.floor("D").le(cutoff), "game_id"]
                def same_features(left, right, game_ids):
                    pd.testing.assert_frame_equal(
                        left.loc[left.game_id.isin(game_ids), columns].reset_index(drop=True),
                        right.loc[right.game_id.isin(game_ids), columns].reset_index(drop=True),
                    )
                same_features(baseline, altered, ids)
                prefix = sample_games.loc[sample_games.start_time.dt.floor("D").le(cutoff)]
                truncated = build_mlb_v3_historical_features(
                    prefix, sample_logs.loc[sample_logs.game_id.isin(prefix.game_id)]
                )
                same_features(baseline, truncated, ids)
                st.success("Sample timing checks passed: current/future results and participants did not change earlier or same-day features.")
            with st.spinner("Building the full MLB V3 historical feature matrix. This may take several minutes..."):
                features = build_mlb_v3_historical_features(mlb_v3_games, current_logs)
                readiness = summarize_mlb_v3_historical_readiness(features)
                st.session_state["mlb_v3_historical_features"] = features
                st.session_state["mlb_v3_historical_feature_readiness"] = readiness
            st.success(f"Built {len(features):,} MLB V3 historical feature rows. No model has been trained or promoted.")
            st.dataframe(readiness["feature_coverage"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download MLB V3 Historical Features CSV",
                data=features.to_csv(index=False).encode("utf-8"),
                file_name="mlb_v3_historical_features.csv",
                mime="text/csv",
                key="download_mlb_v3_historical_features",
            )
            st.info(
                "Unavailable starter and weather fields remain missing. Next, define a historical model feature set "
                "that excludes unavailable fields and evaluate V1/V3 on matching dates. "
                "The date rule assumes earlier-date results were available; suspended-game timing needs separate evidence."
            )
        except Exception as exc:
            st.error("V3 feature checking/building stopped. Review the error before modeling.")
            st.exception(exc)


# ==========================================
# MLB V3 HISTORY-ONLY MODEL COMPARISON
# ==========================================
if v3_history_ready:
    st.divider()
    st.subheader("⚾ MLB V3 Model Comparison")
    st.caption(
        "All three V3 models and a V1-feature control use identical evaluation games and earlier-UTC-date training. "
        "The frozen V1 benchmark remains 54.99%; this stricter control is a separate experiment. No automatic promotion."
    )
    v3_interval = st.selectbox("Retrain after approximately this many predictions", [500, 250, 100], key="v3_compare_interval")
    if st.button("Run MLB V3 Model Comparison", key="run_v3_history_comparison", type="primary"):
        try:
            from dual_agent.mlb_research import run_mlb_v3_history_only_comparison
            matrix = st.session_state.get("mlb_v3_historical_features")
            if matrix is None or matrix.empty:
                raise ValueError("Click Check Timing and Build MLB V3 Features first in this session.")
            progress = st.progress(0.0)
            status = st.empty()
            def update_v3_comparison(value, model):
                progress.progress(min(max(float(value), 0.0), 1.0))
                status.info(f"Testing {model}. Larger retraining intervals finish faster.")
            result = run_mlb_v3_history_only_comparison(matrix, min_train_games=500,
                retrain_every=int(v3_interval), progress_callback=update_v3_comparison)
            st.session_state["mlb_v3_history_comparison"] = result
            status.success("Model comparison finished. V1 and V2A are unchanged.")
        except Exception as exc:
            st.error("V3 model comparison stopped.")
            st.exception(exc)
    result = st.session_state.get("mlb_v3_history_comparison")
    if result is not None:
        st.dataframe(result["comparison"], use_container_width=True, hide_index=True)
        st.caption("Accuracy/AUC: higher is better. Brier/log loss: lower is better. Rates are shown as decimals (0.55 = 55%).")
        with st.expander("Season results and calibration"):
            st.dataframe(result["season_results"], use_container_width=True, hide_index=True)
            st.dataframe(result["calibration"], use_container_width=True, hide_index=True)
            st.dataframe(result["baseline"], use_container_width=True, hide_index=True)
        st.caption(f"Settings used: minimum training games {result['min_train_games']}; retrain interval {result['retrain_every']}.")
        combined_predictions = pd.concat(result["predictions"].values(), ignore_index=True)
        st.download_button("Download V3 Comparison Predictions CSV", combined_predictions.to_csv(index=False).encode("utf-8"),
            file_name="mlb_v3_comparison_predictions.csv", mime="text/csv", key="download_v3_comparison")
        st.info("These results test historical proxies. Confirmed live lineups/starters require a separately validated feature/model path before use in predictions.")



# ==========================================
# FOCUSED MLB MODEL IMPROVEMENT EXPERIMENT
# ==========================================
if v3_history_ready:
    st.divider()
    st.subheader("⚾ Focused MLB Model Improvement Test")
    st.caption("One smaller candidate versus the baseline. Uses your existing historical feature matrix, selects on an earlier season, and evaluates on the latest season. No new API calls.")
    if st.button("Test Focused MLB Candidate", key="mlb_focused_model_test", type="primary"):
        try:
            import importlib
            import dual_agent.mlb_research as research
            if getattr(research, "MLB_MODEL_STRENGTHENING_VERSION", None) != 1:
                importlib.invalidate_caches()
                research = importlib.reload(research)
            if getattr(research, "MLB_MODEL_STRENGTHENING_VERSION", None) != 1:
                raise RuntimeError("Deploy the matching mlb_research.py model update.")
            matrix = st.session_state.get("mlb_v3_historical_features")
            if matrix is None or matrix.empty:
                raise ValueError("Click Check Timing and Build MLB V3 Features first in this session.")
            with st.spinner("Training the focused candidate and evaluating both models..."):
                st.session_state["mlb_focused_model_result"] = research.run_mlb_focused_model_test(matrix)
        except Exception as exc:
            st.error("Focused model test stopped: " + str(exc))
    focused = st.session_state.get("mlb_focused_model_result")
    if focused:
        st.write(f"Selected on {focused['development_season']}: {focused['selected_on_development']}. Evaluation season: {focused['evaluation_season']}.")
        st.dataframe(focused["scores"], use_container_width=True, hide_index=True)
        low, high = focused["accuracy_change_interval"]
        st.write(f"Candidate accuracy change versus control: {100*focused['accuracy_change']:+.2f} percentage points. Paired date-bootstrap 95% interval: {100*low:+.2f} to {100*high:+.2f}.")
        st.info(focused["recommendation"])
        st.caption("Accuracy/AUC: higher is better. Brier/log loss: lower is better. Live models remain unchanged. " + focused["limitations"])
        with st.expander("Candidate features and learned weights"):
            st.dataframe(focused["coefficients"], use_container_width=True, hide_index=True)
        st.download_button("Download Focused Model Predictions CSV", focused["predictions"].to_csv(index=False).encode("utf-8"),
            file_name="mlb_focused_model_predictions.csv", mime="text/csv", key="mlb_focused_predictions_csv")


def _mlb_pregame_storage_module():
    import importlib
    import dual_agent.supabase_db as storage
    if getattr(storage, "MLB_PREGAME_STORAGE_VERSION", None) != 2:
        importlib.invalidate_caches()
        storage = importlib.reload(storage)
    if getattr(storage, "MLB_PREGAME_STORAGE_VERSION", None) != 2:
        raise RuntimeError("Deploy the matching dual_agent/supabase_db.py before saving captures.")
    return storage


def _save_mlb_captured_information(package):
    try:
        return _mlb_pregame_storage_module().save_mlb_pregame_intelligence(
            package, snapshot_date=package.get("game_date")
        )
    except Exception as exc:
        return {"success": False, "archive_saved": False, "latest_saved": False, "message": str(exc)}


def _display_mlb_player_update_table(package):
    rows = []
    for game in package.get("games", []):
        updates = game.get("current_information", {}) or {}
        enriched = (game.get("enrichment", {}) or {}).get("players", {}) or {}
        news = {str(x.get("player_id")): x for x in updates.get("player_news", [])}
        context = game.get("team_context", {}) or {}
        for side, team in updates.get("teams", {}).items():
            lineup = set((context.get(side, {}) or {}).get("batting_order", []) or [])
            transactions = team.get("transactions", [])
            for entry in team.get("roster", []):
                person = entry.get("person", {}) or {}; pid = person.get("id")
                player = enriched.get(str(pid), {}) or {}
                mentions = news.get(str(pid), {})
                related = [t for t in transactions if (t.get("person", {}) or {}).get("id")==pid]
                recent = player.get("recent_stats", {}) or {}
                rows.append({"Game ID": game.get("game_id"), "Side": side, "Player": person.get("fullName"),
                    "Roster status": (entry.get("status", {}) or {}).get("description") or "Unknown",
                    "Listed in feed lineup": pid in lineup, "Recent transactions": len(related),
                    "Latest transaction": max((str(t.get("date") or t.get("effectiveDate") or "") for t in related), default=None),
                    "Recent stats available": bool(recent), "Profile available": bool(player.get("profile")),
                    "Injury report": "Mention found — review" if mentions.get("report_match") else "Unverified",
                    "Roster fetched UTC": team.get("roster_fetched_at"), "Stats collected in capture UTC": game.get("capture_finished_at") if recent else None})
    return pd.DataFrame(rows)

st.divider()
st.subheader("⚾ Complete Pregame Prediction Model")
st.caption("Connect team performance, probable pitchers, feed lineups, roster availability, recent workload, weather, and dated stadium observations to learned model coefficients. Compare feature groups on identical games.")
if st.button("Build Player-Information Candidate", key="mlb_saved_player_candidate", type="primary"):
    try:
        import importlib
        import dual_agent.mlb_research as research
        storage = _mlb_pregame_storage_module()
        if getattr(research, "MLB_SAVED_PLAYER_MODEL_VERSION", None) != 2:
            research = importlib.reload(research)
        if getattr(storage, "MLB_SAVED_PLAYER_MODEL_STORAGE_VERSION", None) != 2:
            storage = importlib.reload(storage)
        missing_modules = []
        if getattr(research, "MLB_SAVED_PLAYER_MODEL_VERSION", None) != 2:
            missing_modules.append("dual_agent/mlb_research.py (player-model version 2)")
        if getattr(storage, "MLB_SAVED_PLAYER_MODEL_STORAGE_VERSION", None) != 2:
            missing_modules.append("dual_agent/supabase_db.py (player-model storage version 2)")
        if missing_modules:
            raise RuntimeError("The loaded code is missing: " + "; ".join(missing_modules) + ". Replace those exact repository files and reboot Streamlit.")
        with st.spinner("Checking saved game results and connecting player features..."):
            dataset = storage.load_mlb_capture_learning_dataset()
            archived = storage.load_mlb_pregame_intelligence()
            batch = research.build_mlb_capture_learning_rows(archived["data"]) if archived and isinstance(archived.get("data"), dict) else {"rows": [], "errors": []}
            keys = {row["capture_key"] for row in batch["rows"]}
            pending = research.refresh_mlb_capture_learning_results([row for row in dataset["rows"] if row["capture_key"] not in keys])
            batch["rows"] = pending["rows"] + batch["rows"]
            checks = batch.get("errors", []) + pending.get("errors", [])
            merged = storage.update_mlb_capture_learning_dataset(batch)
            if not merged.get("success"):
                raise RuntimeError("Unable to update saved results: " + merged.get("message", "Unknown error"))
            result = research.run_mlb_saved_player_model(merged["data"])
            result["result_checks"] = checks
            saved = storage.save_mlb_saved_player_model(result)
        st.session_state["mlb_saved_player_candidate_result"] = result
        st.session_state["mlb_saved_player_candidate_save"] = saved
    except Exception as exc:
        st.error("Player-information model stopped: " + str(exc))
player_model = st.session_state.get("mlb_saved_player_candidate_result")
if player_model:
    st.info(player_model["message"])
    st.write(f"Eligible games: {player_model['eligible_games']} | Usable completed games: {player_model['completed_games']}")
    save_status = st.session_state.get("mlb_saved_player_candidate_save", {})
    if save_status.get("success"):
        st.success("Player-model research record permanently saved.")
    else:
        st.error("Research record could not be saved: " + save_status.get("message", "Unknown error"))
    if player_model["scores"]:
        st.dataframe(pd.DataFrame(player_model["scores"]), use_container_width=True, hide_index=True)
        st.caption("Accuracy: higher is better. Brier/log loss: lower is better. These are captured-data research comparisons, not a live model promotion.")
    if player_model["upcoming_predictions"]:
        st.dataframe(pd.DataFrame(player_model["upcoming_predictions"]), use_container_width=True, hide_index=True)
    if player_model.get("comparison_note"):
        st.caption(player_model["comparison_note"])
    if player_model.get("feature_coverage"):
        with st.expander("Weather, stadium, availability, and workload coverage"):
            st.dataframe(pd.DataFrame(player_model["feature_coverage"]), use_container_width=True, hide_index=True)
    if player_model.get("equation"):
        st.code(player_model["equation"], language="text")
        st.caption(player_model.get("context_note", ""))
    with st.expander("Player model coverage and saved weights"):
        st.json({"excluded": player_model["excluded"], "result_checks": player_model.get("result_checks", []), "models": player_model["models"]})


st.divider()
st.subheader("MLB historical accuracy comparison")
st.caption("Select the player candidate on 2025, refit on 2024–2025, and compare on identical 2026 games sampled across May–September. Tests ERA, WHIP, strikeout/walk rates, and lineup OPS, OBP, and SLG. Weather, injuries, workload, and stadium updates require separately verified historical captures.")
if st.button("Run historical MLB accuracy test", key="mlb_integrated_accuracy"):
    try:
        from dual_agent.mlb_historical_accuracy_test import run_historical_accuracy_test
        progress_bar = st.progress(0, text="Loading historical pregame archives...")
        with st.spinner("Running paired accuracy test; the first run checks up to 1,800 archived games..."):
            report = run_historical_accuracy_test(progress=lambda done, total: progress_bar.progress(done / total, text=f"Checked {done} of {total} games"))
        st.session_state["mlb_integrated_accuracy_report"] = report
        progress_bar.empty()
    except Exception as exc:
        st.error("Historical accuracy test stopped: " + str(exc))
report = st.session_state.get("mlb_integrated_accuracy_report")
if report:
    st.dataframe(pd.DataFrame(report["scores"]), use_container_width=True, hide_index=True)
    low, high = report["accuracy_change_95_interval"]
    st.write(f"Accuracy change: {report['accuracy_change'] * 100:+.2f} percentage points. 95% interval: {low * 100:+.2f} to {high * 100:+.2f}.")
    if low <= 0:
        st.warning("This test does not establish improved accuracy. Keep the candidate experimental.")
    st.caption(report["scope"])
    with st.expander("Test coverage and limitations"):
        st.json({"usable_games": report["usable_games"], "seasons": report["counts_by_season"], "excluded": report["excluded_reasons"], "limitations": report["limitations"], "selected_candidate": report.get("selected_candidate"), "development_scores": report.get("development_scores"), "feature_coverage": report.get("feature_coverage")})

st.divider()
st.subheader("MLB matchup + Elo — goal: 7 correct out of 10 recommended games")
st.caption("Estimate each team's scoring from its lineup and the opposing starter, then calculate win probability. Add pregame Elo and learn a logistic win layer from earlier-block forecasts. Choose a recommendation cutoff on 2025 targeting 70% with at least 50 picks, then report 2026 results. If no cutoff meets 70%, display the strongest supported development group as research picks and show its actual accuracy. Bullpen membership comes from frozen feeds; immediate three-day relief workload uses saved player logs with verified completed-game timestamps; quality retains a 48-hour delay. Handedness splits use prior completed plate appearances; confirmed bullpen availability and weather remain incomplete.")
handedness_enabled = st.checkbox("Include prior plate-appearance handedness splits", value=True, key="mlb_matchup_handedness")
st.caption("Starter rest, recent workload/form, and prior park scoring are connected to this matchup model. Weather and roof inputs use dated saved captures when available. First handedness build downloads prior play histories; successful downloads are cached.")
matchup_scope = st.selectbox("Matchup test coverage", ["Cached June pilot — verify model", "May–September — up to 1,800 games"], index=1, key="mlb_matchup_scope_v7")
if st.button("Build and evaluate MLB matchup model", key="mlb_matchup_run", type="primary"):
    # A failed new build must not display an earlier run as its result.
    st.session_state.pop("mlb_matchup_result", None)
    try:
        import ast
        from pathlib import Path
        required_matchup_files = {
            "mlb_historical_accuracy_test.py": {"run_historical_accuracy_test"},
            "mlb_matchup_model.py": {"run_mlb_matchup_model", "save_mlb_matchup_record"},
            "mlb_matchup_elo.py": {"attach_elo", "select_small_slates", "summarize_small_slates"},
            "mlb_bullpen_freshness.py": {"collect_bullpen_completion_times"},
            "mlb_handedness.py": {"collect_prior_appearances", "attach_handedness"},
            "mlb_matchup_context.py": {"attach_matchup_context"},
        }
        for filename, expected_functions in required_matchup_files.items():
            module_path = Path(__file__).resolve().parent / "dual_agent" / filename
            if not module_path.is_file():
                raise RuntimeError("Missing dual_agent/" + filename + ". Upload the exact .py file to that folder.")
            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(module_path))
            actual_functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if "test_alpaca_connection" in source or not expected_functions.issubset(actual_functions):
                raise RuntimeError("Wrong or outdated content in dual_agent/" + filename + ". Replace it with the matching downloaded file; app.py belongs only in the repository root.")
            if filename == "mlb_matchup_model.py":
                runner = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == "run_mlb_matchup_model")
                if "stage_callback" not in {arg.arg for arg in runner.args.args}:
                    raise RuntimeError("dual_agent/mlb_matchup_model.py is an older version. Replace it with the supplied version 7 file containing small-slate evaluation and stage_callback.")
        import importlib
        import inspect
        import dual_agent.mlb_historical_accuracy_test as matchup_archive_module
        import dual_agent.mlb_matchup_model as matchup_module
        # Streamlit reruns app.py while imported modules may retain earlier code.
        importlib.invalidate_caches()
        matchup_archive_module = importlib.reload(matchup_archive_module)
        import dual_agent.mlb_handedness as matchup_handedness_module
        import dual_agent.mlb_matchup_context as matchup_context_module
        importlib.reload(matchup_handedness_module)
        importlib.reload(matchup_context_module)
        import dual_agent.mlb_bullpen_freshness as matchup_freshness_module
        importlib.reload(matchup_freshness_module)
        import dual_agent.mlb_matchup_elo as matchup_elo_module
        importlib.reload(matchup_elo_module)
        matchup_module = importlib.reload(matchup_module)
        run_mlb_matchup_model = matchup_module.run_mlb_matchup_model
        if not {"player_logs", "include_handedness", "include_context"}.issubset(inspect.signature(run_mlb_matchup_model).parameters):
            raise RuntimeError("The deployed matchup module is outdated: " + str(matchup_module.__file__) + ". Its run_mlb_matchup_model function must accept player_logs, include_handedness, and include_context. Check the repository branch used by this deployment.")
        stage_display = st.empty()
        stage_name = {"value":"Checking pregame archives"}
        progress_bar = st.progress(0, text="Checking pregame archives...")
        def matchup_stage(message):
            stage_name["value"]=message
            stage_display.info(message + "…")
            progress_bar.progress(0.0,text=message)
        def matchup_progress(done,total):
            progress_bar.progress(done/total if total else 0.0,text=f"{stage_name['value']}: {done:,} / {total:,}")
        with st.spinner("Learning expected runs and checking matchup predictions..."):
            bullpen_logs = st.session_state.get("mlb_v3_player_logs")
            if bullpen_logs is None or bullpen_logs.empty:
                from dual_agent.supabase_db import load_mlb_player_history
                bullpen_logs = load_mlb_player_history()
            if bullpen_logs is None or bullpen_logs.empty:
                raise ValueError("Historical player warehouse is empty; load it before bullpen modeling")
            from dual_agent.supabase_db import load_mlb_capture_learning_dataset
            matchup_context = load_mlb_capture_learning_dataset()
            result = run_mlb_matchup_model(stage_callback=matchup_stage, include_context=True, context_dataset=matchup_context, include_handedness=handedness_enabled, player_logs=bullpen_logs, cohort="june_pilot" if matchup_scope.startswith("Cached") else "seasonwide", progress=matchup_progress)
        matchup_stage("Saving completed model and features")
        result["archive_status"] = matchup_module.save_mlb_matchup_record(result)
        st.session_state["mlb_matchup_result"] = result
        stage_display.success("Model build complete.")
        progress_bar.empty()
    except Exception as exc:
        st.error("Matchup model stopped: " + str(exc))
def audit_saved_mlb_engine(archive):
    """Replay the saved engine, without fitting, fetching data, or saving picks."""
    import numpy as audit_np
    from scipy.stats import skellam as audit_skellam
    from scipy.special import expit as audit_expit
    from dual_agent.mlb_matchup_model import matchup_rows, FEATURES, BULLPEN_FEATURES
    from dual_agent.mlb_handedness import HAND_FEATURES
    from dual_agent.mlb_matchup_context import FORM_FEATURES, ENVIRONMENT_FEATURES
    from dual_agent.mlb_matchup_elo import WIN_COLUMNS
    stored = archive.get("predictions", [])
    history = archive.get("retained_feature_rows", [])
    if not stored or not history:
        raise ValueError("The full saved model archive must contain predictions and retained_feature_rows.")
    by_id = {int(g["game_id"]):g for g in history}
    ids = [int(g["game_id"]) for g in stored]
    if len(set(ids)) != len(ids) or len(by_id) != len(history):
        raise ValueError("Duplicate game IDs in saved archive.")
    if any(gid not in by_id for gid in ids):
        raise ValueError("Saved feature rows are missing evaluation games.")
    games = [by_id[gid] for gid in ids]
    flags = [bool(archive.get(k, False)) for k in ["bullpen_selected", "handedness_connected", "context_connected"]]
    names = FEATURES + (BULLPEN_FEATURES if flags[0] else []) + (HAND_FEATURES if flags[1] else []) + (FORM_FEATURES+ENVIRONMENT_FEATURES if flags[2] else [])
    # Targets supplied to the row builder are unused during inference.
    rows = [dict(g, home_runs=0, away_runs=0) for g in games]
    matrix, _ = matchup_rows(rows, *flags)
    bundle = archive["model_bundle"]
    columns = bundle["columns"]
    if len(set(columns)) != len(columns) or any(name not in names for name in columns):
        raise ValueError("Saved model has an unsupported or duplicate feature schema.")
    vectors = {key:audit_np.asarray(bundle[key],dtype=float) for key in ["median","mean","scale","weights"]}
    if any(value.shape != (len(columns),) or not audit_np.isfinite(value).all() for value in vectors.values()) or (vectors["scale"] <= 0).any():
        raise ValueError("Saved scoring weights/scaler are invalid.")
    X = matrix[:,[names.index(name) for name in columns]]
    missing = ~audit_np.isfinite(X)
    filled = audit_np.where(missing, vectors["median"], X)
    contributions = ((filled-vectors["mean"])/vectors["scale"])*vectors["weights"]
    z = contributions.sum(axis=1)+float(bundle["intercept"])
    rates = audit_np.exp(z).reshape(-1,2)
    if not audit_np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Saved model produced invalid scoring rates.")
    raw = audit_np.clip(audit_skellam.sf(0,rates[:,0],rates[:,1])+.5*audit_skellam.pmf(0,rates[:,0],rates[:,1]),1e-6,1-1e-6)
    wb = archive["win_layer_bundle"]
    if wb["columns"] != WIN_COLUMNS:
        raise ValueError("Unsupported Elo win-layer schema.")
    WX = audit_np.column_stack([audit_np.log(raw/(1-raw)),[g["elo_difference"] for g in games],rates[:,0]-rates[:,1]])
    wm,ws,ww = [audit_np.asarray(wb[key],dtype=float) for key in ["mean","scale","weights"]]
    if any(v.shape != (3,) or not audit_np.isfinite(v).all() for v in [wm,ws,ww]) or (ws <= 0).any() or not audit_np.isfinite(WX).all():
        raise ValueError("Invalid Elo layer inputs/scaler.")
    win_terms = (WX-wm)/ws*ww
    hybrid = audit_expit(audit_np.clip(win_terms.sum(axis=1)+float(wb["intercept"]),-40,40))
    selected = hybrid if archive.get("win_layer_selected",True) else raw
    saved = audit_np.asarray([g["Home win probability"] for g in stored],dtype=float)
    saved_rates = audit_np.asarray([[g["Projected home runs"],g["Projected away runs"]] for g in stored],dtype=float)
    probability_error = float(audit_np.max(audit_np.abs(selected-saved)))
    rate_error = float(audit_np.max(audit_np.abs(rates-saved_rates)))
    if not audit_np.isfinite(saved).all() or not audit_np.isfinite(saved_rates).all() or probability_error > 1e-7 or rate_error > 1e-7:
        raise ValueError(f"Saved inference does not reproduce the archive: probability difference={probability_error:.8g}, run difference={rate_error:.8g}. Investigate this mismatch before comparing accuracy.")
    outcomes = audit_np.asarray([g["home_win"] for g in games],dtype=int)
    def summary(label,probability,mask):
        count=int(mask.sum());correct=int(((probability[mask]>=.5)==outcomes[mask]).sum())
        return {"Group":label,"Games":count,"Correct":correct,"Accuracy (%)":round(100*correct/count,2) if count else None}
    all_games=audit_np.ones(len(games),dtype=bool)
    scores=[]
    for label,probability in [("Runs model",raw),("Runs + Elo layer",hybrid),("Selected engine",selected)]:
        row=summary(label,probability,all_games)
        row["Brier"]=float(audit_np.mean((probability-outcomes)**2))
        scores.append(row)
    flips=(raw>=.5)!=(hybrid>=.5)
    coverage=[{"Feature":name,"Missing scoring rows":int(missing[:,i].sum()),"Scoring rows":len(X),"Missing (%)":round(100*float(missing[:,i].mean()),2)} for i,name in enumerate(columns)]
    for name in names:
        if name not in columns:coverage.append({"Feature":name,"Missing scoring rows":None,"Scoring rows":len(X),"Missing (%)":None,"Status":"Not retained in training"})
    counts=missing.reshape(len(games),2,-1).sum(axis=(1,2))
    groups=[summary("Elo changed winner",hybrid,flips),summary("Elo kept winner",hybrid,~flips),summary("Selected: no imputed inputs",selected,counts==0),summary("Selected: imputed inputs",selected,counts>0)]
    for cutoff in [.60,.70]:groups.append(summary(f"Selected: confidence >= {cutoff:.0%}",selected,audit_np.maximum(selected,1-selected)>=cutoff))
    detail=[]
    for i,g in enumerate(games):
        difference=contributions[2*i]-contributions[2*i+1]
        strongest=audit_np.argsort(-audit_np.abs(difference))[:3]
        detail.append({"game_id":g["game_id"],"Home":g["home_team"],"Away":g["away_team"],"Runs home probability":float(raw[i]),"Elo-layer home probability":float(hybrid[i]),"Selected home probability":float(selected[i]),"Confidence change (pp)":float(100*(max(hybrid[i],1-hybrid[i])-max(raw[i],1-raw[i]))),"Winner changed":bool(flips[i]),"Runs correct":bool((raw[i]>=.5)==outcomes[i]),"Elo correct":bool((hybrid[i]>=.5)==outcomes[i]),"Imputed inputs":int(counts[i]),"Largest scoring contributions":", ".join(f"{columns[j]}: {difference[j]:+.3f}" for j in strongest),"Elo log-odds contribution":float(win_terms[i,1])})
    return {"archive_id":str(archive.get("created_at","")),"verification":{"Games replayed":len(games),"Max probability difference":probability_error,"Max projected-run difference":rate_error,"Inference matches saved predictions":True},"layer_scores":scores,"coverage":coverage,"change_groups":groups,"games":detail}


# Read an existing archive without retraining or downloading historical games.
with st.expander("Load saved MLB results — no model rerun"):
    saved_mlb_upload = st.file_uploader("Saved matchup model JSON", type=["json"], key="mlb_confidence_saved_upload")
    if st.button("Open saved MLB results", key="mlb_confidence_saved_open"):
        try:
            if saved_mlb_upload is None:
                raise ValueError("Choose your saved matchup model JSON first.")
            import json as mlb_report_json
            saved_mlb_report = mlb_report_json.load(saved_mlb_upload)
            required = {"predictions", "scores", "cohort", "confidence_selection", "accuracy_change", "accuracy_change_95_interval", "run_MAE"}
            if not isinstance(saved_mlb_report, dict) or not required.issubset(saved_mlb_report):
                raise ValueError("This is not a complete matchup model result archive.")
            if not isinstance(saved_mlb_report["predictions"], list):
                raise ValueError("The archive has invalid prediction rows.")
            st.session_state["mlb_matchup_result"] = saved_mlb_report
            st.success("Saved results loaded. No model training was started.")
        except Exception as exc:
            st.error("Unable to open saved MLB results: " + str(exc))
matchup = st.session_state.get("mlb_matchup_result")
if matchup:
    st.caption("Displayed result cohort: " + matchup["cohort"])
    if matchup.get("archive_status", {}).get("success"):
        st.success("Matchup model, weights, and collected feature inputs permanently archived.")
    elif matchup.get("archive_status"):
        st.warning("Model built, but permanent archive failed: " + matchup["archive_status"].get("message", "Unknown error"))
    st.dataframe(pd.DataFrame(matchup["scores"]), use_container_width=True, hide_index=True)
    low,high = matchup["accuracy_change_95_interval"]
    st.write(f"Accuracy change: {matchup['accuracy_change']*100:+.2f} percentage points; 95% interval {low*100:+.2f} to {high*100:+.2f}. Average run error: {matchup['run_MAE']:.2f} runs per team.")
    st.info("Experimental matchup model. A result here does not replace the live model.")
    st.caption(matchup.get("bullpen_note", ""))
    if "bullpen_selected" in matchup:
        st.write("Bullpen feature group selected on 2025: " + ("Yes" if matchup["bullpen_selected"] else "No — the simpler matchup scored better on development games"))
    slates=matchup.get("small_slates")
    if slates:
        st.subheader("Fixed small-slate results — goal: 7 out of 10")
        slate=slates["evaluation"]
        s1,s2,s3=st.columns(3)
        s1.metric("Small-slate accuracy",f"{slate['accuracy']:.1%}" if slate['accuracy'] is not None else "No eligible picks")
        s2.metric("Correct / selected",f"{slate['correct']} / {slate['games']}")
        s3.metric("Available games selected",f"{slate['coverage']:.1%}")
        st.caption("Fixed rule: at least 55% model confidence; one pick per hourly start window, up to three per Chicago calendar day. Rank only captures available before the window's first scheduled start. No daily quota is forced.")
        st.warning("Historical slates are incomplete: missing or late archived captures can exclude candidates. These results do not establish sustained live accuracy.")
        with st.expander("Every week and every ten-pick block",expanded=True):
            st.dataframe(pd.DataFrame(slate["weeks"]),use_container_width=True,hide_index=True)
            st.dataframe(pd.DataFrame(slate["ten_pick_blocks"]),use_container_width=True,hide_index=True)
            st.caption("A final block with fewer than ten picks is partial. Count all selected losses; never remove a weak week.")
        with st.expander("Frozen slate policy and decision-time coverage"):
            st.json(slates["details"]["policy"])
            st.dataframe(pd.DataFrame(slates["details"]["decisions"]),use_container_width=True,hide_index=True)
            st.json(slates["details"]["exclusion_reasons"])
    metrics=matchup.get("recommendation_metrics")
    if metrics:
        c1,c2,c3=st.columns(3)
        c1.metric("Recommended-pick accuracy", f"{metrics['accuracy']:.1%}" if metrics['accuracy'] is not None else "No qualifying picks")
        c2.metric("Correct / recommended", f"{metrics['correct']} / {metrics['games']}")
        c3.metric("Games recommended", f"{metrics['coverage']:.1%}")
        st.caption("Goal: 70% across every recommended game. These are historical results, not a future recommendation ledger.")
        if metrics.get("wilson_95_interval"):
            lo,hi=metrics["wilson_95_interval"]
            st.caption(f"Selected-pick 95% Wilson interval: {lo:.1%}–{hi:.1%}. This interval treats picks as independent; shared game dates can add uncertainty.")
        if not metrics.get("target_demonstrated"):
            st.info("Sustained 70% accuracy has not been established. Keep gathering future evidence before live promotion.")
    if "win_layer_selected" in matchup:
        st.write("Recommendation model chosen using 2025: " + ("Matchup + Elo + logistic layer" if matchup["win_layer_selected"] else "Matchup runs — the Elo layer did not improve development log loss"))
    with st.expander("Trace saved MLB predictions — no retraining", expanded=True):
        st.caption("Reconstruct the scoring and Elo layers from archived weights and inputs. Outcomes are used only to grade the reconstructed predictions.")
        if st.button("Audit saved prediction engine", key="mlb_saved_engine_audit_run"):
            try:
                st.session_state["mlb_saved_engine_audit_result"] = audit_saved_mlb_engine(matchup)
            except Exception as exc:
                st.session_state.pop("mlb_saved_engine_audit_result", None)
                st.error("Saved engine audit stopped: " + str(exc))
        saved_audit = st.session_state.get("mlb_saved_engine_audit_result")
        if saved_audit and saved_audit["archive_id"] == str(matchup.get("created_at", "")):
            st.json(saved_audit["verification"])
            st.dataframe(pd.DataFrame(saved_audit["layer_scores"]), use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame(saved_audit["coverage"]), use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame(saved_audit["change_groups"]), use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame(saved_audit["games"]), use_container_width=True, hide_index=True)
            st.caption("Imputation counts describe availability, not source freshness. Layer comparisons diagnose this archive; they do not authorize choosing the best layer on evaluation outcomes. All-missing training features were not learned.")
    with st.expander("Strengthen MLB probability layers from saved features", expanded=True):
        st.caption("Fits a consistent frozen-base comparison from saved features only. Selects runs settings on 2024, fits probability layers on early 2025, chooses the layer and cutoff on late 2025, then grades 2026. No historical downloads. The live engine is unchanged.")
        probability_odds_upload = st.file_uploader("Optional historical sportsbook consensus JSON", type=["json"], key="mlb_probability_odds_upload")
        st.caption('Odds format: {"rows": [{"game_id": 123, "captured_at": "2025-06-01T18:00:00Z", "bookmaker": "book-name", "home_decimal": 1.5, "away_decimal": 2.8}]}. At least two distinct books per game; quotes must be at or before the saved prediction and no more than 30 minutes old. No API requests or purchases are made.')
        if st.button("Build consistent MLB probability comparison", key="mlb_probability_upgrade_run"):
            try:
                import json as mlb_upgrade_json
                import importlib as mlb_upgrade_importlib
                import dual_agent.mlb_probability_upgrade as mlb_upgrade_module
                mlb_upgrade_module = mlb_upgrade_importlib.reload(mlb_upgrade_module)
                odds_payload = mlb_upgrade_json.load(probability_odds_upload) if probability_odds_upload is not None else None
                with st.spinner("Fitting probability layers from saved features..."):
                    upgraded = mlb_upgrade_module.run_saved_probability_upgrade(matchup, odds_payload)
                from dual_agent.mlb_matchup_model import save_mlb_matchup_record
                upgraded["archive_status"] = save_mlb_matchup_record(upgraded)
                st.session_state["mlb_probability_upgrade_result"] = upgraded
            except Exception as exc:
                st.session_state.pop("mlb_probability_upgrade_result", None)
                st.error("Probability upgrade stopped: " + str(exc))
        upgraded = st.session_state.get("mlb_probability_upgrade_result")
        if upgraded and upgraded.get("source_archive") == matchup.get("created_at"):
            st.write(upgraded["protocol"])
            st.write("Layer selected on late 2025: " + upgraded["selected_layer"])
            st.json(upgraded["counts"])
            st.dataframe(pd.DataFrame([{k:v for k,v in row.items() if k != "confidence"} for row in upgraded["evaluation"]]), use_container_width=True, hide_index=True)
            st.write("Selected recommendation rule and measured results")
            st.json({"threshold":upgraded["threshold"], **upgraded["recommendations"]})
            selected_report = next(row for row in upgraded["evaluation"] if row["Model"] == upgraded["selected_layer"])
            st.dataframe(pd.DataFrame(selected_report["confidence"]["bands"]), use_container_width=True, hide_index=True)
            st.info(upgraded["market"]["status"])
            if upgraded["market"].get("evaluation"):
                st.write("Market comparison — identical odds-covered games")
                st.dataframe(pd.DataFrame([{k:v for k,v in row.items() if k != "confidence"} for row in upgraded["market"]["evaluation"]]), use_container_width=True, hide_index=True)
                st.json({"selected_on_late_2025":upgraded["market"]["selected_on_late_2025"],"covered_games":upgraded["market"]["covered_games"],"recommendations":upgraded["market"]["recommendations"]})
            if upgraded["archive_status"].get("success"):
                st.success("Probability comparison and learned weights saved.")
            else:
                st.warning("Comparison completed, but permanent saving failed: " + upgraded["archive_status"].get("message","Unknown error"))
            st.download_button("Download probability comparison JSON", data=__import__("json").dumps(upgraded, default=str), file_name="mlb_probability_comparison.json", mime="application/json", key="mlb_probability_upgrade_download")
            with st.expander("Probability training choices and limitations"):
                st.json({key:upgraded[key] for key in ["run_selection","development","threshold_development","limitations"]})
    st.subheader("MLB confidence selection — same comparison as football")
    confidence_rows = pd.DataFrame(matchup.get("predictions", []))
    if not confidence_rows.empty and {"Home win probability", "Correct"}.issubset(confidence_rows.columns):
        probability = pd.to_numeric(confidence_rows["Home win probability"], errors="coerce")
        valid = probability.between(0, 1) & confidence_rows["Correct"].isin([True, False, 0, 1])
        confidence_rows = confidence_rows.loc[valid].copy()
        confidence_rows["winner_confidence"] = probability.loc[valid].map(lambda value: max(value, 1-value))
        confidence_rows["correct_numeric"] = confidence_rows["Correct"].astype(int)
        def mlb_confidence_summary(group, label):
            count = len(group)
            correct = int(group["correct_numeric"].sum())
            rate = correct/count if count else None
            average = float(group["winner_confidence"].mean()) if count else None
            lower = upper = None
            if count:
                z = 1.96
                denominator = 1 + z*z/count
                center = (rate + z*z/(2*count))/denominator
                half = z*((rate*(1-rate)/count + z*z/(4*count*count))**.5)/denominator
                lower, upper = max(0, center-half), min(1, center+half)
            return {"Picks": label, "Games": count, "Correct": correct,
                    "Accuracy (%)": round(100*rate, 1) if count else None,
                    "Average confidence (%)": round(100*average, 1) if count else None,
                    "Calibration gap (pp)": round(100*(rate-average), 1) if count else None,
                    "95% lower bound (%)": round(100*lower, 1) if count else None,
                    "95% upper bound (%)": round(100*upper, 1) if count else None}
        threshold_rows = [mlb_confidence_summary(confidence_rows, "All predictions")]
        for threshold in [.60, .70]:
            threshold_rows.append(mlb_confidence_summary(confidence_rows[confidence_rows.winner_confidence >= threshold], f"{threshold:.0%} confidence or higher"))
        st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True, hide_index=True)
        band_rows = []
        for low, high in [(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.000001)]:
            group = confidence_rows[(confidence_rows.winner_confidence >= low) & (confidence_rows.winner_confidence < high)]
            band_rows.append(mlb_confidence_summary(group, f"{low:.0%} to below {high:.0%}" if high < 1 else "90%–100%"))
        st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)
        st.caption("These fixed 60% and 70% comparisons use saved historical predictions. They do not change the model or choose a cutoff from evaluation outcomes. Confidence is the predicted winner's probability. Intervals assume independent games.")
        st.info("The recommendation cutoff remains the rule selected on development games. A confidence label alone does not establish a 70% win rate; future picks must confirm the rule.")
        st.download_button("Download saved matchup results", data=__import__("json").dumps(matchup, default=str), file_name="mlb_matchup_saved_results.json", mime="application/json", key="mlb_confidence_report_download")
    diagnostics=matchup.get("confidence_diagnostics",{})
    with st.expander("Confidence ranking — where strong picks succeed or fail", expanded=True):
        for label,key in [("Development games — used to choose the rule","development"),("Evaluation games — diagnostic only","evaluation")]:
            report=diagnostics.get(key)
            if report:
                st.write(label)
                st.dataframe(pd.DataFrame(report["bands"]),use_container_width=True,hide_index=True)
                st.dataframe(pd.DataFrame(report["ranked_groups"]),use_container_width=True,hide_index=True)
        st.caption("Top groups rank games across the entire period. Their outcomes do not set the cutoff and do not guarantee ten daily recommendations.")
    confidence = matchup["confidence_selection"]
    if confidence["threshold"] is not None:
        st.write(f"Confidence cutoff chosen on 2025: {confidence['threshold']:.0%}. Selected 2026 games: {confidence['test_games']}.")
        if confidence["test_accuracy"] is not None:
            st.write(f"Selected-game accuracy: {confidence['test_accuracy']:.2%}.")
    else:
        st.caption("No research recommendation rule is available in this older result. Rebuild using the updated module.")
    with st.expander("Game predictions and misses"):
        st.dataframe(pd.DataFrame(matchup["predictions"]), use_container_width=True, hide_index=True)
    with st.expander("Learned weights, development results, and missing inputs"):
        st.json({key:matchup.get(key) for key in ["small_slates","bullpen_freshness_source","win_layer_selected","confidence_diagnostics","elo_source","win_layer_bundle","win_layer_development","recommendation_policy","recommendation_metrics","selected_alpha","development_scores","coefficients","missing_inputs","limitations","confidence_selection","bullpen_selected","bullpen_note","bullpen_coverage_by_season","handedness_connected","handedness_source","handedness_note","context_connected","context_note","context_coverage","unlearned_features"]})
