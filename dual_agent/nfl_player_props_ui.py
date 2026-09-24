"""Standalone Streamlit panel. Never mutates existing NFL/stock session keys."""
import pandas as pd
import requests
import streamlit as st
from dual_agent.nfl_player_props import (MARKETS, analyze, event_props, events,
                                          normalize_props, validate_history, walkforward)

from dual_agent.nfl_player_history import load_player_history
from dual_agent.nfl_player_props import canonical_name

@st.cache_data(ttl=600, show_spinner=False)
def cached_events(key):
    return events(key)

@st.cache_data(ttl=300, show_spinner=False)
def cached_props(key, event_id, market_keys):
    return event_props(key,event_id,list(market_keys))


@st.cache_data(ttl=21600, show_spinner=False)
def cached_player_history(seasons):
    return validate_history(
        load_player_history(seasons)
    )



def render_nfl_player_props():
    st.header('🏈 NFL Player Props Research')
    st.caption('Independent module. Historical frequencies are not calibrated predictions or betting recommendations.')
    key=st.secrets.get('ODDS_API_KEY','')
    if not key:
        st.warning('Add ODDS_API_KEY to Streamlit Community Cloud → App settings → Secrets.')
        return
    selected=st.multiselect('Markets', list(MARKETS), default=list(MARKETS), key='props_markets')
    if st.button('Load upcoming NFL games', key='props_load_games'):
        try: st.session_state['props_events']=cached_events(key)
        except (requests.RequestException,ValueError) as exc: st.error(f'Games unavailable: {exc}')
    games=st.session_state.get('props_events',[])
    if not games:
        st.info('Load games to begin. No upcoming events may be available outside the season.')
        return
    choices={f"{g.get('away_team')} at {g.get('home_team')} — {g.get('commence_time')}":g['id'] for g in games}
    chosen=st.selectbox('Game',list(choices),key='props_event')
    if st.button('Load player prop lines', key='props_load_lines'):
        if not selected: st.warning('Select at least one market.')
        else:
            try:
                event,remaining=cached_props(key,choices[chosen],tuple(MARKETS[m] for m in selected))
                st.session_state['props_lines']=normalize_props(event)
                st.session_state['props_loaded_event']=choices[chosen]
                st.session_state['props_quota']=remaining
            except (requests.RequestException,ValueError) as exc: st.error(f'Prop lines unavailable: {exc}')
    if st.session_state.get('props_loaded_event')!=choices[chosen]:
        st.info('Load lines for the selected game.'); return
    lines=st.session_state.get('props_lines',pd.DataFrame())
    st.caption(f"API requests remaining (provider-reported): {st.session_state.get('props_quota','?')}")
    if lines.empty:
        st.warning('No player prop lines returned for this game and selected markets.'); return
    
    # ==========================================================
    # NFL PLAYER PROPS — PLAYER INTELLIGENCE
    # ==========================================================

    st.markdown("## 🏈 NFL Player Intelligence")

    st.caption(
        "Explore live player markets, compare sportsbook "
        "prices, and review the available betting lines."
    )

    # Normalize display columns without changing source data.

    display_lines = lines.copy()

    player_col = next(
        (
            col for col in ["player", "description"]
            if col in display_lines.columns
        ),
        None
    )

    
    market_col = next(
        (
            col for col in [
                "market_label",
                "market",
                "market_key"
            ]
            if col in display_lines.columns
        ),
        None
    )

    odds_col = next(
        (
            col for col in ["american_odds", "price"]
            if col in display_lines.columns
        ),
        None
    )

    line_col = next(
        (
            col for col in ["line", "point"]
            if col in display_lines.columns
        ),
        None
    )

    side_col = next(
        (
            col for col in ["side", "outcome"]
            if col in display_lines.columns
        ),
        None
    )

  
    # Identify the sportsbook column.
    book_col = next(
        (
            col for col in [
                "bookmaker",
                "sportsbook",
                "bookmaker_key"
            ]
            if col in display_lines.columns
        ),
        None
    )

    if not all([
        player_col,
        market_col,
        book_col,
        odds_col,
        side_col,
    ]):

        st.error(
            "The player props data is missing one or more "
            "required display columns."
        )

        with st.expander("View available data columns"):
            st.write(list(display_lines.columns))

    else:

        # ------------------------------------------------------
        # DASHBOARD SUMMARY
        # ------------------------------------------------------

        total_players = (
            display_lines[player_col].nunique()
        )

        total_markets = (
            display_lines[market_col].nunique()
        )

        total_books = (
            display_lines[book_col].nunique()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Players",
            total_players
        )

        c2.metric(
            "Prop Markets",
            total_markets
        )

        c3.metric(
            "Sportsbooks",
            total_books
        )

        st.divider()

        # ------------------------------------------------------
        # PLAYER SELECTION
        # ------------------------------------------------------

        players = sorted(
            display_lines[player_col]
            .dropna()
            .astype(str)
            .unique()
        )

        if not players:

            st.warning(
                "No player names were returned for this game."
            )

        else:

            selected_player = st.selectbox(
                "Select Player",
                players,
                key="nfl_props_player_intelligence"
            )

            player_data = display_lines.loc[
                display_lines[player_col].astype(str)
                == selected_player
            ].copy()

            st.markdown(
                f"### 👤 {selected_player}"
            )

            st.caption(
                f"{player_data[market_col].nunique()} "
                "available prop markets"
            )

            st.divider()

            # --------------------------------------------------
            # MARKET SELECTION
            # --------------------------------------------------

            markets = sorted(
                player_data[market_col]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_market = st.selectbox(
                "Select Prop Market",
                markets,
                key="nfl_props_market_intelligence"
            )

            market_data = player_data.loc[
                player_data[market_col].astype(str)
                == selected_market
            ].copy()

            st.markdown(
                f"#### 🎯 {selected_market}"
            )

            st.caption(
                "Available sportsbook lines and prices "
                "for the selected player."
            )

            # --------------------------------------------------
            # SPORTSBOOK COMPARISON CARDS
            # --------------------------------------------------

            if line_col:

                grouped_lines = market_data.groupby(
                    line_col,
                    dropna=False,
                    sort=False
                )

            else:

                grouped_lines = [
                    (None, market_data)
                ]

            for line_value, line_group in grouped_lines:

                if pd.notna(line_value):

                    st.markdown(
                        f"##### Prop Line: {line_value}"
                    )

                for side in ["Over", "Under", "Yes", "No"]:

                    side_data = line_group.loc[
                        line_group[side_col]
                        .astype(str)
                        .str.casefold()
                        == side.casefold()
                    ].copy()

                    if side_data.empty:
                        continue

                    st.markdown(
                        f"**{side.upper()}**"
                    )

                    cols = st.columns(2)

                    for index, (_, row) in enumerate(
                        side_data.iterrows()
                    ):

                        with cols[index % 2]:

                            with st.container(border=True):

                                st.markdown(
                                    f"#### {row[book_col]}"
                                )

                                st.caption(
                                    f"{selected_player} · "
                                    f"{selected_market}"
                                )

                                odds = pd.to_numeric(
                                    row[odds_col],
                                    errors="coerce"
                                )

                                if pd.notna(odds):

                                    odds_text = (
                                        f"+{int(odds)}"
                                        if odds > 0
                                        else str(int(odds))
                                    )

                                    st.metric(
                                        "Sportsbook Odds",
                                        odds_text
                                    )

                                    if odds > 0:

                                        implied = (
                                            100 / (odds + 100)
                                        )

                                    elif odds < 0:

                                        implied = (
                                            abs(odds)
                                            / (abs(odds) + 100)
                                        )

                                    else:

                                        implied = None

                                    if implied is not None:

                                        st.metric(
                                            "Implied Probability",
                                            f"{implied:.1%}"
                                        )

                                else:

                                    st.metric(
                                        "Sportsbook Odds",
                                        "N/A"
                                    )

                                st.caption(
                                    "Sportsbook-implied probability "
                                    "before removing bookmaker margin."
                                )

            st.info(
                "These are live sportsbook prices, not "
                "independently validated player projections. "
                "Historical performance analysis is available "
                "in the research section below."
            )

    # ==========================================================
    # END PLAYER INTELLIGENCE
    # ==========================================================
    st.divider()
    st.subheader('Optional historical player-statistics CSV')
    st.caption('CSV columns: player, market, game_time (ISO timestamp), value. One player/game/market per row. '
               'Use completed games only. This module does not invent historical statistics.')
    upload=st.file_uploader('Upload player game logs',type=['csv'],key='props_history_upload')
    if upload is None:
        st.info('Historical data provider is unconfirmed. Live odds work independently; research estimates require an uploaded history CSV.')
        return
    try:
        history=validate_history(pd.read_csv(upload))
        if history.empty: st.warning('No valid historical rows.'); return
        with st.expander('Walk-forward historical baseline evaluation'):
            if st.button('Run player-statistics audit',key='props_audit_run'):
                wf=walkforward(history)
                if wf.empty: st.warning('Insufficient historical games for validation.')
                else:
                    st.metric('Out-of-sample observations',len(wf))
                    st.dataframe(wf.groupby('market').agg(observations=('absolute_error','size'),
                        mean_absolute_error=('absolute_error','mean')).reset_index(),hide_index=True)
                    st.download_button('Download validation rows',wf.to_csv(index=False),'player_props_walkforward.csv',key='props_audit_download')
        st.subheader('Historical frequency comparison — NOT a calibrated model')
        result=analyze(lines,history)
        if result.empty: st.warning('No matching players/markets with at least six prior games.'); return
        st.dataframe(result,hide_index=True,use_container_width=True)
        st.download_button('Download research rows',result.to_csv(index=False),'player_props_research.csv',key='props_research_download')
    except (ValueError,KeyError,pd.errors.ParserError) as exc:
        st.error(f'Historical CSV could not be processed: {exc}')
