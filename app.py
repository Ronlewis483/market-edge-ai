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
    get_missing_mlb_player_log_games,
    collect_mlb_player_logs_batch,
)
st.set_page_config(page_title="Market Edge AI V5", page_icon="📊", layout="wide")

# ==========================================
# MARKET EDGE AI V5 - ALPACA CONNECTION TEST
# ==========================================

import requests

def render_stock_connection():
    with st.expander("Alpaca Connection"):

        if st.button("Test Alpaca API", key="test_alpaca_connection"):

            try:
                api_key = st.secrets["ALPACA_API_KEY"]
                secret_key = prediction_api_key("ALPACA_SECRET_KEY", "ALPACA_API_SECRET")
                if not secret_key:
                    raise KeyError("ALPACA_SECRET_KEY or ALPACA_API_SECRET")

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


/* Prediction workspace: responsive master/detail panels with native controls. */
.me-workspace-header { border: 1px solid #284868; background: linear-gradient(120deg, #112d48, #152039 70%); border-radius: 18px; padding: 22px 26px; margin: 4px 0 20px; box-shadow: 0 8px 24px #0002; }
.me-workspace-header h2 { color: #f1f7ff; margin: 8px 0; }
.me-workspace-header p { color: #bed0e4; margin: 0; }
.me-eyebrow { color: #70d6ff; font-size: .75rem; font-weight: 800; letter-spacing: .13em; }
/* Presentation polish: keep native controls and all labels accessible. */
:root { --me-panel: #121e31; --me-border: #2a3b55; --me-muted: #b6c4d8; }
.stApp { background: radial-gradient(ellipse at top right, #152a43 0, #0b1220 42%); }
.block-container { padding-top: 1.6rem; max-width: 1440px; }
h1 { font-size: 2.05rem !important; line-height: 1.2 !important; }
h2 { font-size: 1.55rem !important; line-height: 1.3 !important; }
h3 { font-size: 1.18rem !important; line-height: 1.4 !important; }
[data-testid="stCaptionContainer"] p { color: var(--me-muted) !important; line-height: 1.5; }
section[data-testid="stSidebar"] { background: #101b2d; }
section[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; min-height: 44px; border-radius: 9px; background: transparent; border-color: transparent; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #133d60; border: 1px solid #328ec4; border-left: 3px solid #54c5fa; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { margin-top: .55rem; }
.stButton > button { transition: border-color .12s ease, background .12s ease; }
.stButton > button[kind="primary"] { background: #0369a1; border: 1px solid #3199cf; box-shadow: 0 4px 16px #0003; }
.stButton > button[kind="primary"]:hover { background: #075985; color: #fff; border-color: #7dd3fc; }
button:focus-visible, input:focus-visible, summary:focus-visible { outline: 2px solid #7dd3fc !important; outline-offset: 3px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--me-border) !important; border-radius: 14px !important; background: #101b2bcc; }
[data-testid="stMetric"] { padding: 14px 16px; border-radius: 12px; background: #16243a; }
[data-testid="stMetricLabel"] { color: #c1cee0; }
[data-testid="stMetricValue"] { font-size: 1.75rem; }
[data-testid="stExpander"] { border-radius: 11px; background: #101b2b; }
[data-testid="stExpander"] summary { min-height: 46px; color: #e7effc; font-weight: 600; }
[data-baseweb="tab-list"] { gap: .5rem; border-bottom: 1px solid var(--me-border); }
[data-baseweb="tab"] { color: #c4d1e4; font-weight: 600; padding: .7rem 1rem; }
[data-baseweb="tab"][aria-selected="true"] { color: #7dd3fc; background: #183650; border-radius: 8px 8px 0 0; }
[data-testid="stAlert"] { border-radius: 10px; }
hr { margin: 1.1rem 0; opacity: .7; }
.me-confidence { display: inline-flex; align-items: center; gap: .7rem; font-size: .78rem; font-weight: 700; letter-spacing: .035em; padding: .45rem .7rem; border: 1px solid; border-radius: 8px; margin: .25rem 0 .7rem; }
.me-confidence span { font-size: .9rem; letter-spacing: 0; }
.me-strong { color: #9ce9c3; background: #113b31; border-color: #296b55; }
.me-moderate { color: #a9daff; background: #173754; border-color: #2c638c; }
.me-close { color: #f5d78d; background: #3b3020; border-color: #74603a; }
@media (max-width: 760px) {
 .block-container { padding: 1.1rem .9rem 2rem; }
 h1 { font-size: 1.7rem !important; }
 h2 { font-size: 1.3rem !important; }
 [data-testid="stMetric"] { padding: 10px; }
 [data-testid="stMetricValue"] { font-size: 1.35rem; }
 .me-confidence { font-size: .7rem; gap: .4rem; }
}
@media (prefers-reduced-motion: reduce) { .stButton > button { transition: none; } }

    </style>
    """,
    unsafe_allow_html=True,
)

def prediction_date_groups(predictions, now=None):
    """Rank the full visible slate: strongest ten first, then every remaining pick."""
    if predictions is None or predictions.empty:
        return []
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    times = pd.to_datetime(predictions["commence_time"], utc=True, errors="coerce")
    # Retention already removes confirmed finals. Keep saved active games as well.
    visible = predictions.loc[times.notna() & (times <= current + pd.Timedelta(days=7))]
    ranked = visible.sort_values("confidence", ascending=False, kind="stable")
    return [
        ("⭐ TOP 10 STRONGEST PICKS", "The highest-confidence predictions across the league's seven-day slate, ranked strongest first. Saved active picks remain visible until final.", ranked.iloc[:10]),
        ("🏟️ ALL REMAINING GAME PICKS", "Every other available prediction, ranked by confidence. Games above are not repeated.", ranked.iloc[10:]),
    ]


def visible_prediction_rows(predictions):
    if predictions is None or predictions.empty:
        return predictions
    groups = prediction_date_groups(predictions)
    return pd.concat([rows for _, _, rows in groups], ignore_index=True)


def render_confidence_badge(label, confidence):
    from html import escape
    tone = "strong" if confidence >= .70 else "moderate" if confidence >= .58 else "close"
    st.markdown(f'<div class="me-confidence me-{tone}">{escape(label)} <span>{confidence:.1%}</span></div>', unsafe_allow_html=True)


def render_prediction_workspace(league, predictions, result):
    """Interactive game selector with a dedicated panel for the selected game's bets."""
    from html import escape
    import hashlib
    if league == 'NFL':
        render_nfl_freshness_monitor()
        if result.get('context_issue'):st.warning(result['context_issue'])
    st.markdown(f'<div class="me-workspace-header"><div class="me-eyebrow">PREDICTION WORKSPACE · {escape(league)}</div><h2>Find your next pick</h2><p>Select a game. Compare its winner, spread and strongest player props.</p></div>', unsafe_allow_html=True)
    controls = st.columns([2, 2, 3])
    with controls[0]:
        period = st.selectbox("Game window", ["Next 7 days + active", "Today"], key=f"workspace_window_{league}")
    with controls[1]:
        threshold = st.selectbox("Winner confidence", ["All predictions", "70% or higher", "80% or higher"], key=f"workspace_confidence_{league}")
    with controls[2]:
        search = st.text_input("Find a team", key=f"workspace_search_{league}", placeholder="Search either team…")
    frame = predictions.copy()
    if period == 'Today':
        dates = pd.to_datetime(frame.commence_time, utc=True, errors='coerce').dt.tz_convert('America/Chicago').dt.date
        frame = frame.loc[dates == pd.Timestamp.now(tz='America/Chicago').date()]
    if threshold != 'All predictions':
        frame = frame.loc[frame.confidence >= (.8 if threshold.startswith('80') else .7)]
    if search.strip():
        mask = frame.home_team.astype(str).str.contains(search.strip(), case=False, regex=False) | frame.away_team.astype(str).str.contains(search.strip(), case=False, regex=False)
        frame = frame.loc[mask]
    frame = frame.sort_values('confidence', ascending=False, kind='stable')
    if frame.empty:
        st.info('No games match these filters. Try another team or confidence level.')
        return
    games = frame.to_dict('records')
    def identity(game):
        value = '|'.join(str(game.get(c,'')) for c in ('home_team','away_team','commence_time'))
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    ids = {identity(g): g for g in games}
    selection_key = f'workspace_selected_{league}'
    if st.session_state.get(selection_key) not in ids:
        st.session_state[selection_key] = identity(games[0])
    left, right = st.columns([1, 1.25], gap='large')
    with left:
        st.subheader(f'{league} game picks')
        st.caption(f'{len(games)} games · strongest ten first, then the remaining slate')
        with st.container(height=650, border=True):
            for rank, game in enumerate(games, 1):
                if rank == 1:
                    st.markdown('**⭐ STRONGEST TEN**')
                elif rank == 11:
                    st.markdown('**ALL REMAINING GAMES**')
                gid = identity(game)
                with st.container(border=True):
                    stamp = pd.to_datetime(game.get('commence_time'), utc=True, errors='coerce')
                    date_text = stamp.tz_convert('America/Chicago').strftime('%b %d, %Y · %-I:%M %p CT') if pd.notna(stamp) else 'Time pending'
                    st.caption(date_text + ' · ' + str(game.get('pick_game_status', 'Upcoming')))
                    st.markdown(f"**#{rank} · {game['away_team']} @ {game['home_team']}**")
                    confidence = float(game.get('confidence',0))
                    render_confidence_badge(str(game.get('predicted_team','Winner pending')), confidence)
                    if st.button('Selected game' if st.session_state[selection_key] == gid else 'View game & player props', key=f'workspace_pick_{league}_{gid}', type='primary' if st.session_state[selection_key] == gid else 'secondary', use_container_width=True):
                        st.session_state[selection_key] = gid
    game = ids[st.session_state[selection_key]]
    with right:
        st.subheader('Game spotlight')
        with st.container(border=True):
            st.markdown(f"### {game['away_team']} @ {game['home_team']}")
            st.caption(str(game.get('pick_game_status','Upcoming')))
            st.markdown(f"**🏆 Predicted winner: {game.get('predicted_team','Pending')}**")
            confidence = float(game.get('confidence',0))
            render_confidence_badge('Estimated win chance', confidence)
            st.progress(min(1.,max(0.,confidence)))
            if game.get('input_status'):
                st.caption(game['input_status'])
            if game.get('input_notes'):
                notes = game['input_notes']
                st.warning(' · '.join(notes) if isinstance(notes,list) else str(notes))
                st.caption('Early forecasts with missing inputs need an update before treating them as recommended picks.')
            with st.expander('View model details'):
                st.write('Home win chance:', f"{float(game.get('home_win_probability',0)):.1%}")
                st.write('Away win chance:', f"{float(game.get('away_win_probability',0)):.1%}")
                if league == 'MLB':
                    st.caption('Equation: ' + str(game.get('Equation', 'Saved matchup model')))
                    missing = game.get('Missing model inputs', [])
                    if missing:
                        st.warning('Training medians used for: ' + ', '.join(missing))
                if game.get('training_games') is not None:
                    st.write('Historical training games:', game['training_games'])
            if league == 'NFL':
                render_nfl_game_context(game)
            elif league == 'MLB':
                render_mlb_game_context(game)
            render_game_addons(league, game, result)


def decimal_to_american_label(price):
    return f"{100*(price-1):+.0f}" if price >= 2 else f"{-100/(price-1):.0f}"


def render_mlb_game_context(game):
    context = game.get('game_context')
    with st.expander('Game inputs · starters, lineups & conditions', expanded=True):
        if not isinstance(context, dict):
            st.info('Generate again to collect the current MLB game inputs.')
            return
        st.caption('Captured UTC: ' + str(context.get('captured_at', 'Unknown')))
        for side in ['away', 'home']:
            team = context.get('teams', {}).get(side, {})
            st.markdown('**' + str(game.get(side+'_team', side.title())) + '**')
            st.write('Starting pitcher: ' + str(team.get('starter') or 'Unannounced') + ' · ' + str(team.get('starter_status', 'Unknown')))
            with st.container(border=True):
                st.markdown('**Pitcher forecast**')
                forecasts = team.get('pitcher_forecasts', [])
                for forecast in forecasts:
                    st.markdown(f"**{forecast['Pick']} {forecast['Line']:g} {forecast['Market']} · {forecast['Estimated chance']:.1%} estimated cover chance**")
                    st.caption(f"Projection {forecast['Projected stat']:.1f} · {forecast['Prior games']} prior starts · {forecast['Best sportsbook']} ({decimal_to_american_label(forecast['Best decimal odds'])})")
                    if forecast.get('Non-push games', 0) < 10:
                        st.caption('Limited history · fewer than 10 non-push starts. Informational forecast only.')
                    elif forecast['Estimated chance'] < .65:
                        st.caption('Below the 65% betting minimum · informational forecast only.')
                    else:
                        st.caption('Meets the cover-chance minimum; participation and quote checks still apply.')
                    if 'neutral' in str(forecast.get('Estimate method','')):
                        st.caption('One-sided sportsbook quote · estimate uses neutral shrinkage.')
                covered = {f['Market'] for f in forecasts}
                for projection in team.get('pitcher_projections', []):
                    if projection['Market'] not in covered:
                        st.write(f"{projection['Market']}: projected {projection['Projected stat']:.1f} · {projection['Prior games']} prior starts")
                available_stats = covered | {p['Market'] for p in team.get('pitcher_projections', [])}
                for label in ['Innings pitched','Strikeouts','Projected game ERA','Walks','Total pitches','Runs allowed']:
                    if label not in available_stats:
                        st.caption(label + ': unavailable in saved prior starts.')
                st.caption('Innings use decimal innings (5.5 = 5½ innings on average). Projected game ERA uses prior earned runs per nine innings, not total runs.')
                if not forecasts:
                    st.caption('No usable fresh sportsbook line and cover estimate. Projections remain visible when history is available.')
                st.caption(team.get('pitcher_forecast_note', 'Generate again to collect pitcher forecasts.'))
                st.caption('Cover percentages exclude pushes and assume the pitcher starts. They are estimates, not measured prediction accuracy.')
            if team.get('lineup_confirmed'):
                with st.expander('Posted batting order'):
                    for rank, name in enumerate(team.get('lineup', []), 1):
                        st.write(f'{rank}. {name}')
            else:
                st.caption('Full batting order is not posted. Provisional players are labeled on prop cards.')
        updates = context.get('current_information', {})
        weather = updates.get('stadium_forecast') or {}
        venue = (updates.get('stadium_metadata') or {}).get('data', {})
        if venue.get('name'):
            st.write('Stadium: ' + venue['name'])
        if weather:
            for key, label in [('temperature_2m','Temperature'), ('wind_speed_10m','Wind'), ('precipitation_probability','Rain chance')]:
                if weather.get(key) is not None:
                    st.caption(f"{label}: {weather[key]} {weather.get('units', {}).get(key, '')}")
            st.caption(str(weather.get('roof_status', 'Roof status unverified')))
            st.caption('Outdoor stadium forecast; enclosed-field conditions may differ.')
        else:
            st.caption('Game-time weather is unavailable.')
        news = [n for n in updates.get('player_news', []) if n.get('report_match')]
        if news:
            with st.expander('Player report mentions · availability requires verification'):
                for item in news:
                    st.write(str(item.get('player_name')) + ': ' + str(item.get('excerpt', '')))
        market = context.get('market_home_probability')
        if market is not None:
            st.caption(f"Sportsbook comparison · home {market:.1%} / away {1-market:.1%}")
        else:
            st.caption('Fresh sportsbook winner comparison is unavailable.')
        with st.expander('Inputs used by the saved equation'):
            st.write(', '.join(context.get('model_columns', [])))
            st.caption('Only inputs with saved coefficients affect the numerical forecast. Report mentions do not imply confirmed injury or medical clearance.')
        for issue in updates.get('errors', []):
            st.caption('Unavailable source: ' + str(issue.get('scope', 'Game update')))


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

    predictions = pipeline_result.get("all_predictions", pipeline_result.get("predictions"))
    opportunities = pipeline_result.get("opportunities")
    if league_name == "NFL":
        predictions = retain_nfl_picks(predictions, "nfl_saved_team_cards")

    if league_name in ("NBA", "MLB"):
        predictions = retain_sport_picks(league_name, predictions, limit=None)
    predictions = visible_prediction_rows(predictions)
    if predictions is None or predictions.empty:
        st.info(
            f"No upcoming {league_name} predictions are currently available."
        )
        return

    if league_name in ("NBA", "MLB"):
        if predictions.empty:
            st.info(f"No upcoming or active {league_name} picks available.")
            return
    prediction_count = len(predictions)

    # ============================================
    # UPCOMING GAME PREDICTIONS
    # ============================================

    with st.expander(
        f"{league_icon} Strongest {league_name} Team Picks "
        f"({prediction_count})",
        expanded=True,
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

        render_prediction_workspace(league_name, sorted_predictions, pipeline_result)

        st.markdown("---")

        st.caption(
            "The strongest ten picks appear first; the remaining games are ranked by "
            "model confidence. Confidence "
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

# Connection diagnostics live in Research Lab below the sidebar routing.


# ==========================================
# MARKET EDGE AI — NAVIGATION V2
# ==========================================

def select_main_page(destination):
    st.session_state["main_navigation"] = destination


with st.sidebar:
    st.markdown("## ⚡ MARKET EDGE AI")
    st.caption("Your sports & market dashboard")
    navigation_groups = [
        ("START", ["🏠 Home"]),
        ("SPORTS", ["🏈 NFL", "🏀 NBA", "⚾ MLB"]),
        ("MARKETS", ["📈 Stocks"]),
        ("WORKSPACE", ["🎟️ My Bets", "📊 Performance", "🧪 Research Lab"]),
    ]
    navigation_options = [item for _, items in navigation_groups for item in items]
    if st.session_state.get("main_navigation") not in navigation_options:
        st.session_state["main_navigation"] = "🏠 Home"
    for group, destinations in navigation_groups:
        st.caption(group)
        for destination in destinations:
            st.button(destination, key="nav_" + destination, use_container_width=True,
                type="primary" if st.session_state["main_navigation"] == destination else "secondary",
                on_click=select_main_page, args=(destination,))
    page = st.session_state["main_navigation"]
    st.divider()
    st.caption("Calendar dates & game times shown in Central time")

if page == "🧪 Research Lab":
    with st.expander("Database Connection Status"):
        if st.button("Test Supabase Connection"):
            success, message = test_connection()


            if success:
                st.success(message)
            else:
                st.error("Database connection failed.")
                st.error(f"Error details: {message}")



default=clean_symbols(DEFAULT_UNIVERSE)


def prediction_api_key(name, alternate=None):
    import os
    value = os.environ.get(name) or (os.environ.get(alternate) if alternate else None)
    if value:
        return value
    try:
        return st.secrets.get(name) or (st.secrets.get(alternate) if alternate else None)
    except Exception:
        return None


def render_player_prop_picks(league, result):
    props = (result or {}).get("player_props", {})
    now = pd.Timestamp.now(tz="UTC")
    picks = retain_sport_picks(league, props.get("picks", []), kind="props").to_dict("records")
    with st.expander(f"🎯 Strongest {league} Player Prop Picks ({len(picks)})", expanded=True):
        if not picks:
            st.info(props.get("message", "Generate predictions to collect available player-prop lines."))
        else:
            st.caption("Ranked by estimated confidence after eligibility checks: a fresh exact bet from at least one sportsbook, verified player identity and at least 10 non-push prior appearances. Provisional participation is labeled." if league == "MLB" else "Historical hit-rate estimates with sportsbook shrinkage. These estimates are not calibrated win probabilities.")
            for offset in range(0, len(picks), 2):
                for column, pick in zip(st.columns(2), picks[offset:offset+2]):
                    with column:
                        with st.container(border=True):
                            st.caption(pick["Game"])
                            pick_time = pd.to_datetime(pick.get("start_time"), utc=True, errors="coerce")
                            if pd.notna(pick_time):
                                st.caption(pick_time.tz_convert("America/Chicago").strftime("%b %d, %Y · %I:%M %p CT"))
                            st.caption(pick.get("pick_game_status", "Upcoming"))
                            st.markdown("### " + pick["Player"])
                            st.markdown(f"**{pick['Pick']} {pick['Line']:g} {pick['Market']}**")
                            if pick.get("Event match") == "Unique same-day matchup; start times differ":
                                st.caption("MLB and sportsbook start times differ; matched a unique same-day game.")
                            if pick.get("Participation status"):
                                st.caption(pick["Participation status"])
                                if pick["Participation status"] != "Confirmed lineup" and pick["Participation status"] != "Recorded starter":
                                    st.warning("Participation is provisional. Recheck the lineup or starter before using this pick.")
                            st.metric("Confidence · excludes pushes", f"{pick['Estimated chance']:.1%}")
                            st.progress(min(1.0, max(0.0, pick["Estimated chance"])))
                            st.caption(f"Projection {pick['Projected stat']:.1f} · {pick['Prior games']} prior appearances · {pick['Books']} books")
                            st.caption(f"Historical wins: {pick['Historical wins']} · pushes: {pick['Historical pushes']}")
            st.caption(props.get("note", "Player participation is required; availability is not confirmed by an offered line."))
        issues = list(props.get("errors", [])) + list(props.get("exclusions", []))
        if issues:
            with st.expander(f"🔎 Why some props aren't available ({len(issues)})"):
                st.caption("These games or players were left out of the prop picks. Here's why.")
                def friendly_time(value):
                    stamp = pd.to_datetime(value, utc=True, errors="coerce")
                    return stamp.tz_convert("America/Chicago").strftime("%b %d, %Y · %-I:%M %p CT") if pd.notna(stamp) else None
                for offset in range(0, len(issues), 2):
                    for column, issue in zip(st.columns(2), issues[offset:offset+2]):
                        with column:
                            with st.container(border=True):
                                if isinstance(issue, dict):
                                    st.markdown("**" + str(issue.get("Player") or issue.get("Game") or "Player-prop update") + "**")
                                    if issue.get("Player") and issue.get("Game"):
                                        st.caption(str(issue["Game"]))
                                    reason = str(issue.get("Reason") or issue.get("error") or issue.get("error_type") or "Information is unavailable.")
                                    messages = {
                                        "No event returned for these teams and home/away order.": "The odds provider hasn't returned a matching game for this matchup.",
                                        "Start-time mismatch or multiple same-day games; automatic match withheld.": "The game dates or times don't line up, or there is more than one possible game. This prop was left out until the match is clear.",
                                        "Odds feed has the opposite home/away order; automatic match withheld.": "The two sources disagree about which team is at home. This prop was left out until the matchup is clear.",
                                        "Multiple sportsbook events match this start time.": "More than one sportsbook game matches. We couldn't identify the correct one.",
                                        "One sportsbook event would map to multiple MLB games.": "This sportsbook game matches multiple scheduled games. We couldn't identify the correct one.",
                                        "Event matched, but no sportsbook prop lines were returned.": "The game was found, but the provider returned no player-prop lines.",
                                        "Fewer than two books offer this same two-sided line.": "We need matching over/under lines from two sportsbooks. This candidate doesn't have enough coverage.",
                                        "No unique player identity found in this game feed.": "We couldn't confidently identify this player in the game information.",
                                        "No saved prior appearances for this player.": "Saved player history is unavailable for this estimate.",
                                        "Player is outside the posted lineup or is not a recorded/probable starter.": "This player isn't in the posted lineup or listed as a starting pitcher.",
                                    }
                                    st.write(messages.get(reason, reason))
                                    scheduled = friendly_time(issue.get("MLB start UTC"))
                                    if scheduled:
                                        st.caption("MLB schedule: " + scheduled)
                                    odds_times = [friendly_time(t) for t in issue.get("Odds starts UTC", [])]
                                    odds_times = [t for t in odds_times if t]
                                    if odds_times:
                                        st.caption("Sportsbook schedule: " + " · ".join(odds_times))
                                else:
                                    st.markdown("**Player-prop update**")
                                    st.write(str(issue))


def run_nba_prediction_with_props(progress_callback=None):
    previous = st.session_state.get("nba_prediction_pipeline_result") or {}
    retain_sport_picks("NBA", previous.get("predictions"))
    retain_sport_picks("NBA", previous.get("player_props", {}).get("picks", []), kind="props")
    result = run_nba_prediction_pipeline(progress_callback=progress_callback)
    now = pd.Timestamp.now(tz="UTC")
    predictions = result.get("predictions")
    if predictions is not None and not predictions.empty:
        predictions = predictions.copy()
        times = pd.to_datetime(predictions["commence_time"], utc=True, errors="coerce")
        eligible_predictions = predictions.loc[(times>now)&(times<=now+pd.Timedelta(days=7))].sort_values("confidence", ascending=False)
        result["all_predictions"] = eligible_predictions.copy()
        predictions = eligible_predictions.head(10)
        result["predictions"] = predictions
        try:
            from dual_agent.sports_player_prop_predictions import generate_player_prop_picks
            result["player_props"] = generate_player_prop_picks("NBA", eligible_predictions.to_dict("records"),
                prediction_api_key("ODDS_API_KEY", "THE_ODDS_API_KEY"), nba_key=prediction_api_key("BALLDONTLIE_API_KEY"))
        except Exception:
            result["player_props"] = {"picks": [], "message": "Player-prop data unavailable. Team predictions are preserved.", "errors": []}
        try:
            import dual_agent.supabase_db as storage
            ready = storage.ensure_market_edge_storage_bucket()
            if not ready.get("success"):
                raise RuntimeError("Prediction storage unavailable")
            saved_at = pd.Timestamp.now(tz="UTC")
            props = result["player_props"]
            props["picks"] = [r for r in props.get("picks", []) if pd.Timestamp(r["start_time"])>saved_at and saved_at-pd.Timestamp(r["Captured UTC"])<=pd.Timedelta(minutes=15)]
            payload = {"created_at": saved_at.isoformat(), "predictions": predictions.to_dict("records"), "player_props": props}
            path = "nba/generated_picks/" + saved_at.strftime("%Y%m%dT%H%M%S%fZ") + ".json"
            storage.get_supabase_client().storage.from_(storage.MLB_STORAGE_BUCKET).upload(path,
                __import__("json").dumps(storage._json_safe(payload), allow_nan=False).encode(), {"content-type": "application/json", "upsert": "false"})
        except Exception:
            result["player_props"].setdefault("errors", []).append("Unable to save this NBA prediction snapshot.")
    else:
        result["player_props"] = {"picks": [], "message": "No eligible NBA games in the next seven days.", "errors": []}
    return result


def load_mlb_prediction_pipeline():
    import ast
    import importlib
    from pathlib import Path
    # Validate the file before import: importing misplaced app.py code creates
    # a second set of Streamlit widgets before a module-version check can run.
    spec = importlib.util.find_spec("dual_agent.mlb_prediction_pipeline")
    if spec is None or not spec.origin:
        raise RuntimeError("Add the supplied mlb_prediction_pipeline.py inside dual_agent/.")
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "run_mlb_prediction_pipeline" not in definitions or "render_mlb_prediction_center" in definitions:
        raise RuntimeError("dual_agent/mlb_prediction_pipeline.py contains the wrong code. Replace it with the supplied pipeline file; app.py belongs only in the repository root.")
    pipeline = importlib.import_module("dual_agent.mlb_prediction_pipeline")
    if getattr(pipeline, "MLB_PIPELINE_VERSION", None) != 13:
        importlib.invalidate_caches()
        pipeline = importlib.reload(pipeline)
    if getattr(pipeline, "MLB_PIPELINE_VERSION", None) != 13:
        raise RuntimeError("Deploy the matching dual_agent/mlb_prediction_pipeline.py from this update. The running MLB module is older than the seven-day forecast fix.")
    return pipeline


@st.cache_data(ttl=180, show_spinner=False)
def cached_nfl_pick_statuses(api_key):
    """Check completion, never infer final status from elapsed time."""
    if not api_key:
        return []
    try:
        response = requests.get(
            "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores",
            params={"apiKey": api_key, "daysFrom": 3, "dateFormat": "iso"},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []
    except (requests.RequestException, ValueError):
        return []


def nfl_pick_identity(row, time_column):
    kickoff = pd.to_datetime(row.get(time_column), utc=True, errors="coerce")
    if pd.isna(kickoff):
        return None
    return (str(row.get("home_team", "")).strip(),
            str(row.get("away_team", "")).strip(), kickoff.isoformat())


def merge_nfl_saved_picks(previous, fresh, time_column, identity_columns, statuses, now, completed=()):
    """Freeze existing started picks; update future picks and retire confirmed finals."""
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    final_games = set(completed)
    status_by_game = {}
    for status in statuses:
        game = nfl_pick_identity(status, "commence_time")
        if game is not None:
            status_by_game[game] = status
            if status.get("completed") is True:
                final_games.add(game)
    def records(value):
        return value.to_dict("records") if isinstance(value, pd.DataFrame) else []
    saved = {}
    # Existing started predictions take priority over a regenerated prediction.
    for row in records(previous) + records(fresh):
        game = nfl_pick_identity(row, time_column)
        if game is None or game in final_games:
            continue
        kickoff = pd.Timestamp(game[2])
        if kickoff > now + pd.Timedelta(days=7):
            continue
        identity = game + tuple(str(row.get(c, "")) for c in identity_columns)
        if identity in saved and kickoff <= now:
            continue
        row = dict(row)
        status = status_by_game.get(game, {})
        row["pick_game_status"] = (
            "Upcoming" if kickoff > now else
            "In progress · saved pregame pick" if status.get("scores") else
            "Started · awaiting game status · saved pregame pick"
        )
        saved[identity] = row
    return pd.DataFrame(saved.values()), final_games


def retain_nfl_picks(rows, archive_key, time_column="commence_time", identity_columns=(), now=None):
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    previous = st.session_state.get(archive_key)
    needs_status = False
    for frame in (previous, rows):
        if isinstance(frame, pd.DataFrame) and time_column in frame:
            times = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
            needs_status = needs_status or bool((times <= current).any())
    statuses = cached_nfl_pick_statuses(st.secrets.get("ODDS_API_KEY", st.secrets.get("THE_ODDS_API_KEY", ""))) if needs_status else []
    final_key = archive_key + "_completed"
    visible, completed = merge_nfl_saved_picks(previous, rows, time_column, identity_columns,
        statuses, current, st.session_state.get(final_key, set()))
    st.session_state[archive_key] = visible
    st.session_state[final_key] = completed
    return visible


@st.cache_data(ttl=180, show_spinner=False)
def cached_sport_pick_statuses(sport, api_key):
    if not api_key:
        return []
    try:
        response = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport}/scores",
            params={"apiKey": api_key, "daysFrom": 3, "dateFormat": "iso"}, timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []
    except (requests.RequestException, ValueError):
        return []


def retain_sport_picks(league, rows, kind="teams", now=None, limit=10):
    """Keep saved pregame cards through kickoff; retire only confirmed finals."""
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows or [])
    if not frame.empty:
        if "commence_time" not in frame and "start_time" in frame:
            frame["commence_time"] = frame["start_time"]
        if "home_team" not in frame:
            if "Home" in frame:
                frame["home_team"] = frame["Home"]
                frame["away_team"] = frame["Away"]
            elif "Game" in frame:
                teams = frame["Game"].astype(str).str.split(" @ ", n=1, expand=True)
                if teams.shape[1] == 2:
                    frame["away_team"], frame["home_team"] = teams[0], teams[1]
        if "Captured UTC" in frame:
            kickoff = pd.to_datetime(frame["commence_time"], utc=True, errors="coerce")
            captured = pd.to_datetime(frame["Captured UTC"], utc=True, errors="coerce")
            # A stale line is retained only as an existing pregame pick, never a new recommendation.
            frame = frame.loc[(kickoff <= current) | ((captured <= current) & (current-captured <= pd.Timedelta(minutes=15)))].copy()
    archive_key = f"{league.lower()}_saved_{kind}_cards"
    previous = st.session_state.get(archive_key)
    if isinstance(previous, pd.DataFrame) and "Captured UTC" in previous:
        kickoff = pd.to_datetime(previous["commence_time"], utc=True, errors="coerce")
        captured = pd.to_datetime(previous["Captured UTC"], utc=True, errors="coerce")
        previous = previous.loc[(kickoff <= current) | ((captured <= current) & (current-captured <= pd.Timedelta(minutes=15)))].copy()
    started = any(isinstance(f, pd.DataFrame) and "commence_time" in f and
        bool((pd.to_datetime(f["commence_time"], utc=True, errors="coerce") <= current).any())
        for f in (previous, frame))
    sport = {"MLB": "baseball_mlb", "NBA": "basketball_nba"}[league]
    statuses = cached_sport_pick_statuses(sport, prediction_api_key("ODDS_API_KEY", "THE_ODDS_API_KEY")) if started else []
    columns = ("Player", "Market", "Pick", "Line") if kind == "props" else ()
    done_key = archive_key + "_completed"
    visible, completed = merge_nfl_saved_picks(previous, frame, "commence_time", columns,
        statuses, current, st.session_state.get(done_key, set()))
    st.session_state[archive_key] = visible
    st.session_state[done_key] = completed
    if visible.empty:
        return visible
    kickoff = pd.to_datetime(visible["commence_time"], utc=True, errors="coerce")
    live = visible.loc[kickoff <= current]
    future = visible.loc[kickoff > current]
    score = next((c for c in ("Estimated chance", "confidence", "Winner probability") if c in future), None)
    if score:
        future = future.sort_values(score, ascending=False, kind="stable")
    # Saved live cards cannot be displaced by the next batch of upcoming top-ten picks.
    return pd.concat([live, future if limit is None else future.head(limit)], ignore_index=True)


def nfl_week_rows(rows, now=None):
    """Keep unstarted games in the next seven days, one row per matchup."""
    if rows is None:
        return None
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "commence_time" not in frame:
        return frame.iloc[:0].copy()
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    kickoff = pd.to_datetime(frame["commence_time"], utc=True, errors="coerce")
    frame = frame.loc[(kickoff > current) & (kickoff <= current + pd.Timedelta(days=7))].copy()
    frame["commence_time"] = kickoff.loc[frame.index]
    if "confidence" in frame:
        frame = frame.sort_values("confidence", ascending=False, kind="stable")
    identity = [c for c in ("home_team", "away_team", "commence_time") if c in frame]
    if len(identity) == 3:
        frame = frame.drop_duplicates(identity, keep="first")
    return frame.reset_index(drop=True)


def run_nfl_prediction_with_addons(feature_games, historical_accuracy=None, historical_sample=None):
    feature_games = nfl_prediction_features()
    previous = st.session_state.get("nfl_prediction_pipeline_result") or {}
    remember_game_addons("NFL", previous)
    result = run_nfl_prediction_pipeline(feature_games=feature_games,
        historical_accuracy=historical_accuracy, historical_sample=historical_sample)
    try:
        from dual_agent.nfl_live_context import attach_context
        result['predictions'] = attach_context(result.get('predictions'), result.get('live_odds'))
    except Exception as exc:
        result['context_issue'] = 'NFL context retrieval unavailable (' + type(exc).__name__ + ')'
    try:
        from dual_agent.game_prediction_addons import enhance_result
        result = enhance_result("NFL", result, prediction_api_key("ODDS_API_KEY", "THE_ODDS_API_KEY"), feature_games)
        st.session_state["nfl_auto_prop_predictions"] = result.get("nfl_player_props", pd.DataFrame())
        st.session_state["nfl_prop_scan_scope"] = "next_7_days_v1"
    except ImportError:
        result["prop_issues"] = ["Deploy dual_agent/game_prediction_addons.py to enable automatic props and spreads."]
    remember_game_addons("NFL", result)
    return result


def remember_game_addons(league, result):
    """Seed frozen pregame attachments before a rescan replaces the result."""
    if league == "NFL":
        retain_nfl_picks(result.get("nfl_player_props"), "nfl_saved_prop_cards", "game_time", ("player", "market", "model_pick", "line"))
        spread_rows = pd.DataFrame(result.get("spreads", {}).get("picks", []))
        retain_nfl_picks(spread_rows, "nfl_saved_spread_cards")
    else:
        props = result.get("player_props", {})
        retain_sport_picks(league, props.get("game_picks", props.get("picks", [])), kind="props", limit=None)
        if league == "MLB":
            retain_sport_picks(league, result.get("spreads", {}).get("picks", []), kind="spreads", limit=None)


def render_game_addons(league, game, result):
    """Place independently estimated spreads and eligible props inside their game."""
    if league not in ("NFL", "MLB", "NBA"):
        return
    try:
        from dual_agent.game_prediction_addons import matching_game_rows
    except ImportError:
        st.caption("Deploy game_prediction_addons.py to show this game's spreads and props.")
        return
    remember_game_addons(league, result or {})
    spread_key = "nfl_saved_spread_cards" if league == "NFL" else "mlb_saved_spreads_cards"
    if league in ("NFL", "MLB"):
        spreads = matching_game_rows(game, st.session_state.get(spread_key))
        with st.expander("📏 Spread prediction" if league == "NFL" else "📏 Run-line prediction", expanded=False):
            if spreads:
                spread = spreads[0]
                kickoff = pd.to_datetime(spread["commence_time"], utc=True, errors="coerce")
                captured = pd.to_datetime(spread.get("captured_at"), utc=True, errors="coerce")
                current = pd.Timestamp.now(tz="UTC")
                fresh = pd.notna(captured) and pd.Timedelta(0) <= current-captured <= pd.Timedelta(minutes=30)
                if kickoff <= current or fresh:
                    st.markdown(f"**{spread['spread_team']} {spread['spread_line']:+g}**")
                    st.metric("Estimated cover chance · excludes pushes", f"{spread['cover_chance']:.1%}")
                    st.caption(f"Push chance {spread['push_chance']:.1%} · {spread['books']} sportsbooks at this line")
                    st.caption(f"Projected home margin: {spread['projected_home_margin']:+.1f}")
                    st.caption("New spread estimate; accuracy is not established. " + spread["method"] + ".")
                else:
                    st.info("Spread line is stale. Generate again for a fresh estimate.")
            else:
                reasons = (result or {}).get("spreads", {}).get("issues", [])
                text = next((r for r in reasons if str(game.get("away_team", "")) in r and str(game.get("home_team", "")) in r), None)
                st.info(text or (reasons[0] if reasons and len(reasons)==1 else "No eligible spread estimate: two fresh matching books and complete model inputs are required."))
    props_key = "nfl_saved_prop_cards" if league == "NFL" else f"{league.lower()}_saved_props_cards"
    props = matching_game_rows(game, st.session_state.get(props_key))
    if league == "NFL":
        current = pd.Timestamp.now(tz="UTC")
        props = [p for p in props if pd.to_datetime(p.get("game_time"), utc=True, errors="coerce") <= current
                 or len(p.get("sportsbook_offers", [])) >= 1]
    score = "historical_support" if league == "NFL" else "Estimated chance"
    value_key = "estimated_return_per_unit" if league == "NFL" else "Estimated return per unit"
    props = [p for p in props if float(p.get(score, 0)) >= .65]
    props = sorted(props, key=lambda row: (-float(row.get(score, 0)), -float(row[value_key])))[:10]
    with st.expander(f"🎯 Player props for this game ({len(props)})", expanded=True):
        st.caption("Minimum 65% estimated cover chance · exact bet offered by a sportsbook")
        if not props:
            issues = (result or {}).get("prop_issues", []) if league == "NFL" else (result or {}).get("player_props", {}).get("errors", [])
            st.info("No eligible player props for this matchup yet. A listed exact bet, at least 65% estimated cover chance, usable history are required.")
            if league == "MLB":
                from collections import Counter
                exclusions = (result or {}).get('player_props', {}).get('exclusions', [])
                matchup = str(game.get('away_team','')) + ' @ ' + str(game.get('home_team',''))
                reasons = Counter(str(e.get('Reason','')) for e in exclusions if e.get('Game') == matchup)
                for reason, count in reasons.most_common(3):
                    st.caption(f'{count} candidates: {reason}')
            if issues and league == "NFL":
                matching = [issue for issue in issues if str(game.get("home_team", "")) in issue and str(game.get("away_team", "")) in issue]
                for issue in matching or issues[:1]:
                    st.caption(str(issue))
        for rank, prop in enumerate(props, 1):
            with st.container(border=True):
                if league == "NFL":
                    side = str(prop["model_pick"])
                    pick = side if side in ("YES", "NO") else f"{side} {float(prop['line']):g}"
                    st.markdown(f"**#{rank} · {prop['player']} · {pick} {prop['market']}**")
                    st.caption(f"Model ranking score {float(prop['prediction_score'])*100:.1f}/100 · historical support {float(prop['historical_support']):.1%}")
                    st.caption(f"Projection {float(prop['projected_value']):.1f} · {int(prop.get('sample_size', 0))} prior games")
                    st.caption(f"Estimated cover chance {float(prop['historical_support']):.1%} · historical estimate, not calibrated")
                    if prop.get("best_sportsbook"):
                        st.caption(f"Best available odds: {prop['best_sportsbook']} · {float(prop['best_american_odds']):+g}")
                    offers = prop.get("sportsbook_offers", [])
                    if len(offers) == 1:
                        st.caption("Single-book listing · no cross-book confirmation")
                    if offers:
                        st.caption("Offered at: " + ", ".join(str(o['bookmaker_key']) + " (" + format(float(o['american_odds']), '+g') + ")" for o in offers))
                    # Keep the existing manual bet-record workflow available.
                    import hashlib
                    identity = "|".join(str(prop.get(c, "")) for c in ("event_id", "game_time", "player", "market", "model_pick", "line"))
                    button_key = "attached_nfl_prop_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
                    kickoff = pd.to_datetime(prop.get("game_time"), utc=True, errors="coerce")
                    if st.button("Record this player prop", key=button_key, disabled=pd.isna(kickoff) or kickoff <= pd.Timestamp.now(tz="UTC")):
                        st.session_state.update({"bet_builder_mode": "Single", "single_sport": "NFL",
                            "single_market": {"Passing yards":"Passing Yards", "Rushing yards":"Rushing Yards", "Receiving yards":"Receiving Yards", "Receptions":"Receptions", "Anytime touchdown":"Anytime Touchdown"}.get(prop["market"], "Other"),
                            "single_player": str(prop["player"]), "single_team": "", "single_direction": {"OVER":"Over", "UNDER":"Under", "YES":"Yes", "NO":"No"}.get(side.upper(), "Other"),
                            "single_line": float(prop["line"]), "single_description": f"{prop['player']} {pick} {prop['market']} — {prop.get('away_team', '')} @ {prop.get('home_team', '')}", "main_navigation": "🎟️ My Bets"})
                        st.rerun()
                else:
                    st.markdown(f"**#{rank} · {prop['Player']} · {prop['Pick']} {prop['Line']:g} {prop['Market']}**")
                    st.caption(f"Estimated chance {prop['Estimated chance']:.1%} · projection {prop['Projected stat']:.1f} · {prop['Prior games']} prior appearances · {prop['Books']} books")
                    if prop.get("Best sportsbook"):
                        st.caption(f"Best available odds: {prop['Best sportsbook']} · odds {decimal_to_american_label(float(prop['Best decimal odds']))}")
                    offers = prop.get("Sportsbook offers", [])
                    if offers:
                        st.caption("Offered at: " + ", ".join(str(o['book']) + " (" + decimal_to_american_label(float(o['decimal_odds'])) + ")" for o in offers))
                    if "neutral" in str(prop.get("Estimate method", "")):
                        st.caption("One-sided listing · estimate uses prior appearances and a neutral prior; a margin-free market probability is unavailable.")
                    participation = prop.get("Participation status")
                    if participation:
                        st.caption(participation)
                        if participation not in ("Confirmed lineup", "Recorded starter"):
                            st.warning("Provisional participation—recheck the lineup or starter.")
                st.caption(prop.get("pick_game_status", "Upcoming"))
        if props:
            st.caption("Ordered from strongest to weakest within this game. Prop estimates and ranking scores are not calibrated future probabilities.")


def nfl_prediction_features():
    """Rebuild from refreshed final results; session history is never the authority."""
    from dual_agent.nfl_live_context import fresh_schedule, completed_history
    schedule, fetched = fresh_schedule()
    history = completed_history(schedule)
    if history.empty:
        raise ValueError('Current NFL completed-game history unavailable; stale features were not silently reused.')
    signature = pd.util.hash_pandas_object(history[['game_id','home_score','away_score']],index=False).sum().item()
    if st.session_state.get('nfl_history_signature') != signature or st.session_state.get('nfl_live_feature_games') is None:
        features = build_nfl_pregame_features(history)
        if features is None or features.empty:
            raise ValueError('Fresh NFL completed results produced no usable features.')
        st.session_state['nfl_live_feature_games'] = features
        st.session_state['nfl_feature_games'] = features
        st.session_state['multi_nfl_games'] = history
        st.session_state['nfl_history_signature'] = signature
    st.session_state['nfl_history_fetched_at'] = fetched
    st.session_state['nfl_latest_completed_game'] = history.start_time.max().isoformat()
    return st.session_state['nfl_live_feature_games']


@st.fragment(run_every='180s')
def render_nfl_freshness_monitor():
    try:
        nfl_prediction_features()
        stamp = pd.Timestamp(st.session_state['nfl_history_fetched_at']).tz_convert('America/Chicago')
        latest = pd.Timestamp(st.session_state['nfl_latest_completed_game']).tz_convert('America/Chicago')
        st.caption(f"History checked {stamp.strftime('%b %d · %I:%M %p CT')} · latest completed game {latest.strftime('%b %d, %Y')} · checks every 3 minutes while this page is open")
    except Exception as exc:
        st.warning('NFL history refresh unavailable: ' + str(exc))


def render_nfl_game_context(game):
    context = game.get('game_context')
    with st.expander('Quarterbacks, injuries & game conditions', expanded=True):
        if not isinstance(context,dict):
            st.info('Generate fresh NFL predictions to retrieve current game context.')
            return
        st.caption('Context fetched ' + str(context.get('fetched_at','Not available')))
        for side in ['away','home']:
            team = context.get(side,{})
            st.markdown('**' + str(team.get('team',game.get(side+'_team','Team'))) + '**')
            qb=team.get('qb')
            if qb:
                st.write(f"Expected QB: **{qb['name']}**")
                st.caption(f"Injury: {qb['injury_status']} · Practice: {qb['practice_status']}")
                st.caption(qb['designation'] + ' · depth chart ' + str(team.get('depth_updated_at','Not verified')))
                if qb.get('report_updated_at'):st.caption('QB report updated ' + qb['report_updated_at'])
            else:
                st.warning('Expected starting QB not verified from the retrieved depth chart.')
            with st.expander('Injury & practice report · ' + str(team.get('team',side))):
                reports=team.get('injuries',[])
                if reports:st.dataframe(pd.DataFrame(reports).drop(columns=['gsis_id'],errors='ignore'),hide_index=True,use_container_width=True)
                else:st.caption('No current-week report returned. This does not establish that the team is healthy.')
            with st.expander('Depth chart · ' + str(team.get('team',side))):
                if team.get('depth'):st.dataframe(pd.DataFrame(team['depth']),hide_index=True,use_container_width=True)
                else:st.caption('Current depth chart unavailable.')
        weather=context.get('weather',{})
        if weather.get('source'):
            st.markdown('**Kickoff weather · ' + str(weather['venue']) + '**')
            st.caption(f"{weather['temperature_f']}°F · wind {weather['wind_mph']} mph · rain chance {weather['rain_chance']}%")
            st.caption(weather['source'] + ' · ' + weather['fetched_at'])
            st.caption('Roof type: ' + str(weather['roof_type']) + ' · ' + weather['roof_status'] + '. Outdoor forecast may not affect an enclosed field.')
        market=context.get('market',{})
        if market.get('books'):
            st.markdown('**Sportsbook comparison · margin removed**')
            st.caption(f"{game['home_team']}: {market['home_probability']:.1%} · {game['away_team']}: {market['away_probability']:.1%}")
            st.caption(f"Model–market gap: {market['gap']*100:.1f} percentage points · {len(market['books'])} books")
            st.dataframe(pd.DataFrame(market['books'])[['sportsbook','home_moneyline','away_moneyline']],hide_index=True,use_container_width=True)
        elif market.get('issue'):st.caption(market['issue'])
        for issue in dict.fromkeys(context.get('issues',[])):st.warning(issue)
        st.caption(context.get('probability_use',''))



@st.fragment(run_every="180s")
def render_mlb_prediction_center(location):
    st.subheader("⚾ MLB Prediction Center")
    st.caption("Current lineups, starters, team form and available game conditions feed the saved matchup equation.")
    with st.expander("Saved MLB model setup", expanded=False):
        st.caption("Activate your full saved matchup model JSON once. Predictions reuse it after app restarts.")
        model_upload = st.file_uploader("Saved matchup model", type=["json"], key=location + "_model_upload")
        if model_upload is not None and st.button("Use this MLB model", key=location + "_activate"):
            try:
                activated = load_mlb_prediction_pipeline().activate_model(__import__("json").load(model_upload))
                st.success("MLB model activated.")
                st.session_state.pop("mlb_live_pipeline_result", None)
            except Exception as exc:
                st.error("Unable to activate MLB model: " + str(exc))
    st.caption("Scores the next seven days of games. The strongest ten lead the slate; ranked sportsbook props appear inside each game.")
    team_cards = st.empty()
    def show_team_cards(rows):
        winners = pd.DataFrame([{
            **r,
            "home_team": r.get("Home", r.get("home_team")),
            "away_team": r.get("Away", r.get("away_team")),
            "predicted_team": r.get("Predicted winner", r.get("predicted_team")),
            "confidence": r.get("Winner probability", r.get("confidence")),
            "home_win_probability": r.get("Home win probability", r.get("home_win_probability")),
            "away_win_probability": r.get("away_win_probability", 1-r.get("Home win probability", r.get("home_win_probability", .5))),
            "commence_time": r.get("start_time", r.get("commence_time")),
            "input_status": r.get("Input status", r.get("input_status")),
            "input_notes": r.get("Input notes", r.get("input_notes", [])),
            "pick_game_status": r.get("pick_game_status", "Upcoming"),
            "Captured UTC": r.get("Captured UTC"),
        } for r in rows])
        with team_cards.container():
            attached = dict(st.session_state.get("mlb_live_pipeline_result") or {})
            attached.update(predictions=winners, all_predictions=winners, opportunities=None)
            render_league_prediction_results("MLB", "⚾", attached)
    generated = st.button("⚾ Generate MLB Predictions · Next 7 Days", key=location + "_generate", type="primary", use_container_width=True)
    previous = st.session_state.get("mlb_live_pipeline_result")
    if previous:
        retain_sport_picks("MLB", previous.get("all_predictions", previous.get("predictions", [])), limit=None)
        remember_game_addons("MLB", previous)
    checked_at = pd.to_datetime(st.session_state.get("mlb_last_refresh_attempt"), utc=True, errors="coerce")
    refresh_due = bool(previous and (pd.isna(checked_at) or pd.Timestamp.now(tz="UTC")-checked_at >= pd.Timedelta(seconds=180)))
    st.caption("After generation, lineups, starters and available props refresh every three minutes while this screen stays open.")
    if generated or refresh_due:
        st.session_state["mlb_last_refresh_attempt"] = pd.Timestamp.now(tz="UTC").isoformat()
        try:
            import os
            pipeline = load_mlb_prediction_pipeline()
            odds_key = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
            if not odds_key:
                try:
                    odds_key = st.secrets.get("ODDS_API_KEY") or st.secrets.get("THE_ODDS_API_KEY")
                except Exception:
                    odds_key = None
            status = st.empty()
            result = pipeline.run_mlb_prediction_pipeline(api_key=odds_key, progress=status.info, on_team_predictions=show_team_cards, force_refresh=True)
            try:
                from dual_agent.game_prediction_addons import enhance_result
                result = enhance_result("MLB", result, odds_key)
            except ImportError:
                result["spreads"] = {"picks": [], "issues": ["Deploy dual_agent/game_prediction_addons.py for run-line predictions."]}
            result["pipeline_version"] = pipeline.MLB_PIPELINE_VERSION
            st.session_state["mlb_live_pipeline_result"] = result
            status.success(result["message"])
        except Exception as exc:
            st.error("MLB prediction pipeline stopped: " + str(exc))
    result = st.session_state.get("mlb_live_pipeline_result")
    if result and result.get("pipeline_version") != 13:
        st.session_state.pop("mlb_live_pipeline_result", None)
        result = None
        st.info("The MLB pipeline was updated. Generate again to search the upcoming seven-day schedule.")
    if result:
        now = pd.Timestamp.now(tz="UTC")
        rows = retain_sport_picks("MLB", result.get("all_predictions", result.get("predictions", [])), limit=None).to_dict("records")
        if rows:
            show_team_cards(rows)
            st.caption("Picks are saved before first pitch. Confidence reflects the saved model; the 70% accuracy goal is not established.")
        else:
            st.info(result.get("message", "No fresh upcoming predictions. Generate again to collect current game information."))
            if result.get("scheduled_games"):
                st.dataframe(pd.DataFrame(result["scheduled_games"]), use_container_width=True, hide_index=True)
        st.caption("Player props and run-line estimates are attached to their matching game cards.")
        last_capture = pd.to_datetime(result.get("created_at"), utc=True, errors="coerce")
        if pd.notna(last_capture):
            st.caption("Last successful refresh: " + last_capture.tz_convert("America/Chicago").strftime("%b %d · %-I:%M %p CT"))


if page == "🏠 Home":

    st.title("Your prediction hub")

    st.write(
        "Generate the next seven days of picks. Review winners, spreads and "
        "player props together, then track your results in My Bets."
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
        "Rank NFL games in the next seven days, shop available "
        "moneylines, and analyze model-vs-market opportunities."
    )
    
    if st.button(
        "🏈 Generate NFL Predictions · Next 7 Days",
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
                    run_nfl_prediction_with_addons(
                        feature_games=feature_games,
                        historical_accuracy=st.session_state.get(
                            "nfl_historical_accuracy"
                        ),
                        historical_sample=st.session_state.get(
                            "nfl_historical_sample"
                        ),
                    )
                )
    
                retain_nfl_picks(st.session_state.get("nfl_prediction_pipeline_result", {}).get("predictions"), "nfl_saved_team_cards")
                retain_nfl_picks(nfl_pipeline_result.get("predictions"), "nfl_saved_team_cards")
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
    
        predictions = visible_prediction_rows(retain_nfl_picks(nfl_pipeline_result.get("predictions"), "nfl_saved_team_cards"))
    
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
            expanded=True,
        ):

            st.caption(
                "Strongest model picks first, for games in the next seven days."
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

            render_prediction_workspace("NFL", sorted_predictions, nfl_pipeline_result)

            st.markdown("---")

            st.caption(
                "The strongest ten picks appear first; the remaining games are ranked by "
                "model confidence. Confidence "
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
    st.caption("NFL props and spread estimates load with the game predictions and appear inside each game card.")


# ============================================
# SPORTS CENTER — LEAGUE PREDICTION HUB
# ============================================

if page in ["🏈 NFL", "🏀 NBA", "⚾ MLB"]:
    selected_league = page
    st.title(page + " Center")
    st.caption("Game predictions, player props, and current game information.")

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

        if st.button("🏈 Generate NFL Predictions · Next 7 Days", key="nfl_sidebar_generate", type="primary"):
            try:
                with st.spinner("Generating NFL predictions..."):
                    result = run_nfl_prediction_with_addons(
                        feature_games=nfl_prediction_features(),
                        historical_accuracy=st.session_state.get("nfl_historical_accuracy"),
                        historical_sample=st.session_state.get("nfl_historical_sample"),
                    )
                    retain_nfl_picks(st.session_state.get("nfl_prediction_pipeline_result", {}).get("predictions"), "nfl_saved_team_cards")
                    retain_nfl_picks(result.get("predictions"), "nfl_saved_team_cards")
                    st.session_state["nfl_prediction_pipeline_result"] = result
                    for session_key, result_key in [("live_nfl_moneylines", "live_odds"),
                        ("best_nfl_moneylines", "best_lines"), ("live_nfl_predictions", "predictions"),
                        ("live_nfl_opportunities", "opportunities")]:
                        st.session_state[session_key] = result.get(result_key)
            except Exception as exc:
                st.error("NFL prediction pipeline failed: " + str(exc))

        nfl_result = st.session_state.get(
            "nfl_prediction_pipeline_result"
        )

        render_league_prediction_results(
            league_name="NFL",
            league_icon="🏈",
            pipeline_result=nfl_result,
        )

        st.caption("Player props load automatically inside their matching game cards.")

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
            "🏀 Generate NBA Predictions · Next 7 Days",
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
                    run_nba_prediction_with_props(
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
        # NBA props are attached to matching game cards.

    # ========================================
    # MLB
    # ========================================

    elif selected_league == "⚾ MLB":

        render_mlb_prediction_center("mlb_sidebar")


# ============================================
# NBA ONE-CLICK PREDICTION CENTER
# ============================================

if page == "🏠 Home":

    st.divider()
    st.subheader("🏀 NBA Prediction Center")
    
    
    st.caption(
        "Rank NBA games in the next seven days and review "
        "player props beneath each matching game."
    )
    
    if st.button(
        "🏀 Generate NBA Predictions · Next 7 Days",
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
                run_nba_prediction_with_props(
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

if page == "🏠 Home" and nba_pipeline_result is not None:
    render_league_prediction_results("NBA", "🏀", nba_pipeline_result)
    # NBA props are attached to matching game cards.


if page == "🏈 NFL":
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
        st.caption("Use Generate NFL Predictions above for winners, spreads and automatic player props.")
        st.divider()
        st.subheader("Today's Picks")

    

        st.divider()

        # Sports selection

        sport = "NFL Football"

        # Betting market selection

        st.subheader("Choose Your Betting Market")

        market = st.selectbox(
            "What type of bet are you interested in?",
            ["Game Winner", "Player Passing Yards", "Player Rushing Yards",
             "Player Receiving Yards", "Player Receptions", "Player Passing Touchdowns",
             "Player Anytime Touchdown", "Player Rushing + Receiving Yards",
             "Player Passing Completions", "Player Interceptions Thrown",
             "Game Total Points", "Point Spread", "Other"],
            key="nfl_betting_market",
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
                if market == "Game Winner":
                    nfl_picks = (st.session_state.get("nfl_prediction_pipeline_result") or {}).get("predictions")
                    if nfl_picks is not None and not pd.DataFrame(nfl_picks).empty:
                        choices = pd.DataFrame(nfl_picks).to_dict("records")
                        now = pd.Timestamp.now(tz="UTC")
                        choices = [r for r in choices if pd.to_datetime(r.get("commence_time"), utc=True, errors="coerce") > now]
                        if choices:
                            selected_pick = st.selectbox("Choose an NFL winner to record", range(len(choices)),
                                format_func=lambda i: f"{choices[i]['away_team']} @ {choices[i]['home_team']} — {choices[i]['predicted_team']}",
                                key="nfl_winner_to_record")
                            if st.button("Record this NFL bet", key="nfl_record_winner", type="primary"):
                                pick = choices[selected_pick]
                                st.session_state["bet_builder_mode"] = "Single"
                                st.session_state["single_sport"] = "NFL"
                                st.session_state["single_market"] = "Game Winner"
                                st.session_state["single_team"] = pick["predicted_team"]
                                st.session_state["single_player"] = ""
                                st.session_state["single_description"] = f"{pick['predicted_team']} moneyline — {pick['away_team']} @ {pick['home_team']}"
                                st.session_state["single_direction"] = "Moneyline"
                                st.session_state["single_line"] = 0.0
                                st.session_state["main_navigation"] = "🎟️ My Bets"
                                st.rerun()
                        else:
                            st.info("Generate NFL predictions above to get upcoming game winners.")
                    else:
                        st.info("Generate NFL predictions above to view game winners, then record your bet here or in My Bets.")
                else:
                    st.info("Record this market through My Bets. Player forecasts are available for the listed NFL prop markets.")
                if st.button("Open NFL bet entry", key="nfl_open_bet_entry"):
                    st.session_state["single_sport"] = "NFL"
                    st.session_state["main_navigation"] = "🎟️ My Bets"
                    st.rerun()

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

                    # Display saved forecasts from highest estimated chance to lowest.
                    matching_forecasts = sorted(
                        matching_forecasts,
                        key=lambda item: (float(item.get("estimated_chance",0)),item.get("generated_at","")),
                        reverse=True,
                    )

                    for forecast_index, forecast in enumerate(matching_forecasts):

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
                            if st.button("Record this player prop", key=f"nfl_record_prop_{forecast_index}"):
                                st.session_state["bet_builder_mode"] = "Single"
                                st.session_state["single_sport"] = "NFL"
                                st.session_state["single_market"] = {"Passing yards": "Passing Yards", "Rushing yards": "Rushing Yards",
                                    "Receiving yards": "Receiving Yards", "Receptions": "Receptions",
                                    "Anytime touchdown": "Anytime Touchdown"}.get(selected_prop_market, "Other")
                                st.session_state["single_player"] = player_name
                                st.session_state["single_team"] = ""
                                st.session_state["single_description"] = f"{player_name} {prop_side} {prop_line:g} {selected_prop_market} — {game_name}"
                                st.session_state["single_direction"] = prop_side if prop_side in ["Over", "Under", "Yes", "No"] else "Other"
                                st.session_state["single_line"] = float(prop_line)
                                st.session_state["main_navigation"] = "🎟️ My Bets"
                                st.rerun()


        else:

            st.info(
                "NFL player prop forecasts are "
                "currently available through "
                "this dashboard connection. "
                "Other sports will be connected "
                "separately."
            )

    # The NFL My Bets tab renders the shared bet builder below.

    if show_performance:
        st.subheader("Model Performance")
        st.info("Open Performance in the sidebar for recorded betting results. Model validation remains in Research Lab.")
        if st.button("Open Performance", key="sports_open_performance"):
            st.session_state["main_navigation"] = "📊 Performance"
            st.rerun()

if page == "🏠 Home":
    st.divider()
    render_mlb_prediction_center("mlb_home")

if page == "📈 Stocks":
    from dual_agent.stock_opportunities_ui import render_stock_opportunities
    render_stock_opportunities()


if page == "🏀 NBA":
    with st.expander("Advanced NBA matchup tools", expanded=False):

        # ============================================
        # NBA PREDICTION CENTER
        # ============================================

        if page == "🏀 NBA":

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


elif page == "🎟️ My Bets" or (page == "🏈 NFL" and show_bets):

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


if page == "📈 Stocks":
    render_stock_connection()
    with st.expander("Trade planning and existing five-day research tools", expanded=False):
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

if page == "🏈 NFL":
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

if page == "🏈 NFL" and st.session_state.get("nfl_prop_v2b_benchmark"):

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

if page == "🏈 NFL":
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

if page == "🏈 NFL" and st.session_state.get("nfl_accuracy_audit") is not None:

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
                        ).strftime("%b %d, %Y · %I:%M %p CT")
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

                    for _, game in nfl_week_rows(best_lines, now=current_time).iterrows():

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

    live_predictions = retain_nfl_picks(st.session_state.get("nfl_live_predictions"), "nfl_saved_research_cards")

    
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
# MLB V3 HISTORICAL PLAYER RESEARCH
# ==========================================

if page == "🧪 Research Lab":
    st.divider()
    st.subheader("⚾ MLB Historical Player Data")
    st.caption(
        "Build the permanent player-game warehouse used by the V3 "
        "prediction engine. Each successful "
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


if page == "🧪 Research Lab":
    st.divider()
    st.subheader("MLB historical odds data")
    st.caption("Collect and cache pregame sportsbook quotes for model setup. This section does not run accuracy tests or model comparisons.")
    saved_model_file = st.file_uploader("Load saved matchup model for odds collection", type=["json"], key="mlb_odds_data_model_file")
    if saved_model_file is not None:
        try:
            import json as mlb_saved_data_json
            matchup = mlb_saved_data_json.load(saved_model_file)
            if not isinstance(matchup, dict) or not matchup.get("retained_feature_rows"):
                raise ValueError("Use the full saved matchup model JSON with retained features.")
            odds_batch_limit = st.number_input("Maximum new historical odds requests per batch", min_value=1, max_value=200, value=10, step=1, key="mlb_historical_odds_limit")
            st.caption(f"This batch can use up to {int(odds_batch_limit)*10:,} Odds API credits for US moneylines. Cached snapshots are reused. The first uncached request checks historical access; an access error stops the batch.")
            if st.button("Collect historical odds for this saved model", key="mlb_historical_odds_collect"):
                try:
                    import os as mlb_odds_os
                    import importlib as mlb_odds_importlib
                    import dual_agent.mlb_historical_odds as mlb_odds_module
                    mlb_odds_module = mlb_odds_importlib.reload(mlb_odds_module)
                    odds_key = mlb_odds_os.environ.get("ODDS_API_KEY") or mlb_odds_os.environ.get("THE_ODDS_API_KEY")
                    if not odds_key:
                        odds_key = st.secrets.get("ODDS_API_KEY") or st.secrets.get("THE_ODDS_API_KEY")
                    odds_progress = st.progress(0, text="Collecting historical moneyline snapshots...")
                    odds_result = mlb_odds_module.collect_historical_odds(matchup, odds_key, max_requests=int(odds_batch_limit), progress=lambda done,total: odds_progress.progress(done/total, text=f"Checked {done} of {total} snapshots"))
                    odds_result["source_archive"] = matchup.get("created_at")
                    st.session_state["mlb_historical_odds_dataset"] = odds_result
                    odds_progress.empty()
                except Exception as exc:
                    st.error("Historical odds collection stopped: " + str(exc))
            collected_odds = st.session_state.get("mlb_historical_odds_dataset")
            if collected_odds and collected_odds.get("source_archive") == matchup.get("created_at"):
                st.json({key:collected_odds.get(key) for key in ["requests_used","credits_used_reported","credits_remaining","cached_snapshots","pending_snapshots","historical_access_verified","errors"]})
                odds_coverage = pd.DataFrame(collected_odds.get("coverage", []))
                if not odds_coverage.empty:
                    st.dataframe(odds_coverage.groupby("season").agg(games=("game_id","count"),games_with_consensus=("eligible_books",lambda counts:int((counts>=2).sum()))).reset_index(), use_container_width=True, hide_index=True)
                st.caption("Quotes are cached for model setup. Repeat collection to resume pending snapshots.")
                st.download_button("Download collected historical odds", data=__import__("json").dumps(collected_odds), file_name="mlb_historical_odds.json", mime="application/json", key="mlb_historical_odds_download")
        except Exception as exc:
            st.error("Unable to open model data: " + str(exc))
