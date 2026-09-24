import pandas as pd
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

import streamlit as st

from dual_agent.supabase_db import (
    test_connection,
    save_bet,
    get_all_bets,
    update_bet_result,
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


if page == "🏀 Sports Center":

    st.title("🏈 Sports Betting Center")

    st.write(
        "Find opportunities, review predictions, "
        "and track your betting performance."
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
                "NFL Football",
                "NBA Basketball",
                "College Football",
                "MLB Baseball",
                "Other Sports",
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


    # ==========================================
    # NBA PREDICTION CENTER
    # ==========================================

    st.divider()
    st.subheader("🏀 NBA Prediction Center")

    st.caption(
        "NBA game predictions, sportsbook comparison, "
        "and potential wager payouts."
    )

    
    # ==========================================
    # NBA TEAM SELECTION
    # ==========================================

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


    # ==========================================
    # AUTOMATIC NBA PREDICTION CENTER
    # ==========================================

    st.divider()
    st.subheader("🏀 Automatic NBA Prediction Center")

    st.caption(
        "Analyze upcoming NBA games using the existing "
        "prediction model and live sportsbook odds."
    )

    if st.button(
        "Analyze Upcoming NBA Games",
        key="auto_nba_predictions",
        type="primary",
    ):

        try:

            with st.spinner(
                "Retrieving upcoming NBA games..."
            ):

                live_games = get_live_nba_moneylines()

                if not live_games:
                    st.warning(
                        "No upcoming NBA moneylines "
                        "are currently available."
                    )
                    st.stop()

                odds_df = pd.DataFrame(live_games)

                # Convert game times to UTC.

                odds_df["game_time"] = pd.to_datetime(
                    odds_df["commence_time"],
                    utc=True,
                    errors="coerce",
                )

                current_time = pd.Timestamp.now(
                    tz="UTC"
                )

                # Exclude games that have already started.

                odds_df = odds_df[
                    odds_df["game_time"] > current_time
                ].copy()

                if odds_df.empty:
                    st.warning(
                        "No upcoming NBA games were found."
                    )
                    st.stop()

                # Each sportsbook can return the same game.
                # Predict each unique matchup only once.

                unique_games = (
                    odds_df
                    .sort_values("game_time")
                    .drop_duplicates(subset=["event_id"])
                    .copy()
                )

                st.success(
                    f"Found {len(unique_games)} "
                    "upcoming NBA games."
                )

            # ======================================
            # LOAD HISTORICAL DATA ONCE
            # ======================================

            with st.spinner(
                "Loading historical NBA data..."
            ):

                latest_game = unique_games[
                    "game_time"
                ].max()

                season_year = (
                    latest_game.year
                    if latest_game.month >= 9
                    else latest_game.year - 1
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

                # Never use games occurring on or
                # after the earliest prediction date.

                first_game_date = (
                    unique_games["game_time"]
                    .min()
                    .date()
                )

                historical_games = historical_games[
                    pd.to_datetime(
                        historical_games["game_date"],
                        utc=True,
                    ).dt.date < first_game_date
                ].copy()

                if historical_games.empty:
                    raise ValueError(
                        "No historical NBA games "
                        "are available before "
                        "the upcoming matchups."
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

            # ======================================
            # GENERATE AUTOMATIC PREDICTIONS
            # ======================================

            results = []

            progress = st.progress(0)

            total_games = len(unique_games)

            for game_number, (_, game) in enumerate(
                unique_games.iterrows(),
                start=1,
            ):

                home_team = game["home_team"]
                away_team = game["away_team"]

                try:

                    home_code = normalize_nba_team_name(
                        home_team
                    )

                    away_code = normalize_nba_team_name(
                        away_team
                    )

                    if not home_code or not away_code:
                        raise ValueError(
                            "Unable to identify NBA team."
                        )

                    game_date = game["game_time"].date()

                    # Build features for this matchup.

                    matchup_features = (
                        build_balldontlie_future_matchup_features(
                            prepared_games,
                            home_code,
                            away_code,
                            game_date,
                        )
                    )

                    # Run the existing NBA model.

                    prediction = (
                        predict_balldontlie_matchup(
                            feature_games,
                            matchup_features,
                        )
                    )

                    home_probability = float(
                        prediction["home_win_probability"]
                    )

                    away_probability = float(
                        prediction["away_win_probability"]
                    )

                    predicted_side = prediction[
                        "predicted_side"
                    ]

                    predicted_team = (
                        home_team
                        if predicted_side == "HOME"
                        else away_team
                    )

                    # ==================================
                    # COMPARE AVAILABLE SPORTSBOOKS
                    # ==================================

                    game_odds = odds_df[
                        odds_df["event_id"]
                        == game["event_id"]
                    ]

                    for _, sportsbook in game_odds.iterrows():

                        for side, team, probability, odds_key in [
                            (
                                "HOME",
                                home_team,
                                home_probability,
                                "home_odds",
                            ),
                            (
                                "AWAY",
                                away_team,
                                away_probability,
                                "away_odds",
                            ),
                        ]:

                            american_odds = float(
                                sportsbook[odds_key]
                            )

                            if abs(american_odds) < 100:
                                continue

                            if american_odds > 0:

                                implied_probability = (
                                    100
                                    / (american_odds + 100)
                                )

                                profit_per_dollar = (
                                    american_odds / 100
                                )

                            else:

                                implied_probability = (
                                    abs(american_odds)
                                    / (abs(american_odds) + 100)
                                )

                                profit_per_dollar = (
                                    100 / abs(american_odds)
                                )

                            model_edge = (
                                probability
                                - implied_probability
                            )

                            expected_value = (
                                probability
                                * profit_per_dollar
                                - (1 - probability)
                            )

                            results.append({

                                "Game Date":
                                    str(game_date),

                                "Matchup":
                                    f"{away_team} at {home_team}",

                                "Predicted Winner":
                                    predicted_team,

                                "Betting Side":
                                    team,

                                "Model Probability":
                                    probability,

                                "Sportsbook":
                                    sportsbook["bookmaker"],

                                "American Odds":
                                    american_odds,

                                "Implied Probability":
                                    implied_probability,

                                "Model Edge":
                                    model_edge,

                                "Expected ROI":
                                    expected_value,

                                "Training Games":
                                    prediction["training_games"],

                            })


                except Exception as game_error:
                    st.warning(
                        f"Could not analyze "
                        f"{away_team} at {home_team}: "
                        f"{game_error}"
                    )
                    st.exception(game_error)
    
                progress.progress(
                    game_number / total_games
                )

            # ======================================
            # DISPLAY AUTOMATIC PREDICTIONS
            # ======================================

            if results:

                results_df = pd.DataFrame(results)

                st.session_state[
                    "automatic_nba_results"
                ] = results_df

                st.success(
                    "Automatic NBA analysis completed!"
                )

            else:

                st.warning(
                    "No NBA predictions could be generated."
                )

        except Exception as e:

            st.error(
                "Automatic NBA analysis failed."
            )

            st.error(
                f"Error details: {e}"
            )

    # ==========================================
    # DISPLAY SAVED RESULTS
    # ==========================================

    if "automatic_nba_results" in st.session_state:

        results_df = st.session_state[
            "automatic_nba_results"
        ]

        st.subheader("Upcoming NBA Predictions")

        st.caption(
            "Estimated probabilities and model-based "
            "expected returns. These are not verified "
            "profitable betting opportunities."
        )

        minimum_edge = st.slider(
            "Minimum model edge (%)",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
            key="auto_nba_edge_filter",
        )

        filtered_results = results_df[
            results_df["Model Edge"]
            >= minimum_edge / 100
        ].copy()

        filtered_results = (
            filtered_results
            .sort_values(
                "Expected ROI",
                ascending=False,
            )
        )

        display_df = filtered_results.copy()

        for column in [
            "Model Probability",
            "Implied Probability",
            "Model Edge",
            "Expected ROI",
        ]:

            display_df[column] = (
                display_df[column]
                .map(lambda x: f"{x:.1%}")
            )

        if display_df.empty:

            st.info(
                "No predictions match the selected "
                "model-edge threshold."
            )

        else:

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Model edge is the difference between "
            "estimated win probability and the "
            "sportsbook's implied probability. "
            "Expected ROI assumes the model's "
            "probabilities are accurate. Neither "
            "metric guarantees a profitable wager."
        )




elif page == "🎟️ My Bets":

    st.title("My Bets")
    st.write(
        "Record your wagers, track wins and losses, "
        "and monitor your betting performance."
    )

    st.divider()

    # -----------------------------------
    # Record a new bet
    # -----------------------------------

    st.subheader("Record a New Bet")

    with st.form("new_bet_form"):

        sport = st.selectbox(
            "Sport",
            [
                "NFL",
                "NBA",
                "College Football",
                "MLB",
                "Other",
            ],
        )

        betting_market = st.selectbox(
            "Betting Market",
            [
                "Game Winner",
                "Player Points",
                "Player Rebounds",
                "Player Assists",
                "Passing Yards",
                "Rushing Yards",
                "Receiving Yards",
                "Three-Pointers",
                "Other",
            ],
        )

        player_name = st.text_input(
            "Player Name (optional)"
        )

        team_name = st.text_input(
            "Team Name (optional)"
        )

        bet_description = st.text_input(
            "Describe Your Bet",
            placeholder="Example: Player over 25.5 points",
        )

        bet_type = st.selectbox(
            "Bet Direction",
            [
                "Over",
                "Under",
                "Moneyline",
                "Spread",
                "Other",
            ],
        )

        betting_line = st.number_input(
            "Betting Line",
            value=0.0,
            step=0.5,
        )

        odds = st.number_input(
            "American Odds",
            value=-110,
            step=1,
        )

        wager = st.number_input(
            "Amount Wagered ($)",
            min_value=0.01,
            value=10.00,
            step=1.00,
        )

        sportsbook = st.text_input(
            "Sportsbook (optional)"
        )

        notes = st.text_area(
            "Notes (optional)"
        )

        submitted = st.form_submit_button(
            "Save My Bet"
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
