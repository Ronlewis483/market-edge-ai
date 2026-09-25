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


    # ==========================================================
    # HISTORICAL PLAYER PERFORMANCE
    # ==========================================================

    st.divider()

    st.markdown("## 📊 Historical Player Performance")

    st.caption(
        "Review completed NFL game statistics, recent player "
        "performance, and historical results against sportsbook lines."
    )

    
    # ==================================================
    # HISTORICAL PLAYER DATA UPLOADER
    # ==================================================

    st.markdown("### 📂 Historical Player Data")

    st.caption(
        "Upload completed NFL player game statistics "
        "to activate historical performance analysis."
    )

    
    # ==========================================================
    # AUTOMATIC NFL HISTORICAL PLAYER DATA
    # ==========================================================

    st.markdown("### 📚 Automatic Historical NFL Data")

    st.caption(
        "Load completed NFL player game statistics automatically. "
        "No CSV upload required."
    )

    available_seasons = list(
        range(2026, 2021, -1)
    )

    selected_seasons = st.multiselect(
        "Select NFL seasons",
        options=available_seasons,
        default=[2025, 2026],
        max_selections=4,
        key="nfl_props_history_seasons"
    )

    if st.button(
        "Load Historical Player Statistics",
        key="nfl_props_load_history"
    ):

        if not selected_seasons:

            st.warning(
                "Select at least one NFL season."
            )

        else:

            try:

                with st.spinner(
                    "Downloading historical NFL player statistics..."
                ):

                    player_history = cached_player_history(
                        tuple(selected_seasons)
                    )

                    st.session_state[
                        "nfl_props_historical_data"
                    ] = player_history

                    st.session_state[
                        "nfl_props_loaded_seasons"
                    ] = tuple(selected_seasons)

                st.success(
                    f"Loaded {len(player_history):,} "
                    "historical player-statistic records."
                )

            except Exception as history_error:

                st.error(
                    "Unable to load historical NFL statistics: "
                    f"{type(history_error).__name__}: "
                    f"{history_error}"
                )

    player_history = st.session_state.get(
        "nfl_props_historical_data"
    )

    loaded_seasons = st.session_state.get(
        "nfl_props_loaded_seasons"
    )

    if (
        not isinstance(player_history, pd.DataFrame)
        or player_history.empty
    ):

        st.info(
            "Select your NFL seasons and click "
            "'Load Historical Player Statistics' "
            "to activate automatic player performance analysis."
        )

    elif loaded_seasons != tuple(selected_seasons):

        st.info(
            "Your season selection has changed. "
            "Click 'Load Historical Player Statistics' "
            "to refresh the historical dataset."
        )

    else:

        try:

            st.caption(
                f"Historical data loaded: "
                f"{', '.join(map(str, loaded_seasons))}"
            )

            st.write("Historical data diagnostic")
            
            st.write(
                "Historical data type:",
                type(player_history).__name__
            )
            
            if isinstance(player_history, pd.DataFrame):
                st.write(
                    "Historical records:",
                    len(player_history)
                )
            
                st.write(
                    "Available columns:",
                    list(player_history.columns)
                )
            
                st.write(
                    "Sample historical records:"
                )
            
                st.dataframe(
                    player_history.head(5),
                    use_container_width=True
                )
          

            if not isinstance(player_history, pd.DataFrame) or player_history.empty:

                st.warning(
                    "No historical player statistics are currently available. "
                    "Select your NFL seasons and click "
                    "'Load Historical Player Statistics' "
                    "to download the data automatically."
                )

            elif "selected_player" not in locals() or "selected_market" not in locals():

                st.info(
                    "Select a player and prop market above "
                    "to view historical performance."
                )

            else:

                market_lookup = {
                    value: name
                    for name, value in MARKETS.items()
                }

                historical_market = market_lookup.get(
                    selected_market,
                    selected_market
                )

                
                player_key = canonical_name(
                    selected_player
                )

              
                st.write("Player matching diagnostic")
                
                st.write(
                    "Selected sportsbook player:",
                    selected_player
                )
                
                st.write(
                    "Canonical player key:",
                    player_key
                )
                
                st.write(
                    "Historical market:",
                    historical_market
                )
                
                matching_players = player_history.loc[
                    player_history["player_key"] == player_key
                ]

                
            # ==========================================
            # HISTORICAL PLAYER NAME MATCHING DIAGNOSTIC
            # ==========================================
            
            import re
            
            def abbreviated_player_key(name):
                name = str(name).strip()
            
                parts = re.findall(
                    r"[A-Za-z]+",
                    name
                )
            
                if len(parts) < 2:
                    return canonical_name(name)
            
                return (
                    parts[0][0] + parts[-1]
                ).lower()
            
            
            abbreviated_key = abbreviated_player_key(
                selected_player
            )
            
            st.write(
                "Expected historical player key:",
                abbreviated_key
            )
            
            abbreviated_matches = player_history.loc[
                player_history["player_key"] == abbreviated_key
            ]
            
            st.write(
                "Abbreviated-name historical matches:",
                len(abbreviated_matches)
            )
            
            if not abbreviated_matches.empty:
            
                st.write(
                    "Historical records found for:",
                    selected_player
                )
            
                st.dataframe(
                    abbreviated_matches.head(10),
                    use_container_width=True
                )
                
                st.write(
                    "Matching historical records:",
                    len(matching_players)
                )
                
                if not matching_players.empty:
                    st.dataframe(
                        matching_players.head(10),
                        use_container_width=True
                    )

                player_games = player_history.loc[
                    (
                        player_history["player_key"]
                        == player_key
                    )
                    &
                    (
                        player_history["market"]
                        == historical_market
                    )
                ].copy()

                player_games = player_games.sort_values(
                    "game_time"
                )

                if player_games.empty:

                    st.info(
                        f"No historical {historical_market} "
                        f"statistics are available for "
                        f"{selected_player}."
                    )

                else:

                    st.markdown(
                        f"### 👤 {selected_player}"
                    )

                    st.caption(
                        f"Historical performance: "
                        f"{historical_market}"
                    )

                    # ------------------------------------------
                    # RECENT GAME SELECTION
                    # ------------------------------------------

                    recent_count = st.selectbox(
                        "Historical sample",
                        [5, 10, 12, 15, 20],
                        index=1,
                        key="props_history_sample"
                    )

                    player_games = player_games.tail(
                        recent_count
                    )

                    total_games = len(player_games)

                    average_value = (
                        player_games["value"].mean()
                    )

                    highest_value = (
                        player_games["value"].max()
                    )

                    lowest_value = (
                        player_games["value"].min()
                    )

                    # ------------------------------------------
                    # PERFORMANCE SUMMARY
                    # ------------------------------------------

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Games Analyzed",
                        total_games
                    )

                    c2.metric(
                        "Historical Average",
                        f"{average_value:.1f}"
                    )

                    c3.metric(
                        "Highest",
                        f"{highest_value:.1f}"
                    )

                    c4.metric(
                        "Lowest",
                        f"{lowest_value:.1f}"
                    )

                    st.divider()

                    # ------------------------------------------
                    # SPORTSBOOK LINE COMPARISON
                    # ------------------------------------------

                    available_lines = pd.to_numeric(
                        market_data["line"],
                        errors="coerce"
                    ).dropna()

                    if not available_lines.empty:

                        selected_line = st.selectbox(
                            "Compare against sportsbook line",
                            sorted(
                                available_lines.unique().tolist()
                            ),
                            key="props_history_line"
                        )

                        over_count = int(
                            (
                                player_games["value"]
                                > selected_line
                            ).sum()
                        )

                        under_count = int(
                            (
                                player_games["value"]
                                < selected_line
                            ).sum()
                        )

                        push_count = int(
                            (
                                player_games["value"]
                                == selected_line
                            ).sum()
                        )

                        eligible_games = (
                            over_count + under_count
                        )

                        over_rate = (
                            over_count / eligible_games
                            if eligible_games
                            else 0
                        )

                        under_rate = (
                            under_count / eligible_games
                            if eligible_games
                            else 0
                        )

                        st.markdown(
                            f"### 🎯 Historical Results vs {selected_line}"
                        )

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Games Over",
                            f"{over_count}/{eligible_games}",
                            f"{over_rate:.1%}"
                            if eligible_games
                            else "N/A"
                        )

                        c2.metric(
                            "Games Under",
                            f"{under_count}/{eligible_games}",
                            f"{under_rate:.1%}"
                            if eligible_games
                            else "N/A"
                        )

                        c3.metric(
                            "Pushes",
                            push_count
                        )

                        st.caption(
                            "Historical frequencies exclude pushes. "
                            "These results describe completed games "
                            "and are not calibrated predictions "
                            "of future performance."
                        )

                        # --------------------------------------
                        # GAME-BY-GAME PERFORMANCE CHART
                        # --------------------------------------

                        st.markdown(
                            "### 📈 Recent Game Performance"
                        )

                        chart_data = (
                            player_games[
                                ["game_time", "value"]
                            ]
                            .copy()
                            .set_index("game_time")
                        )

                        chart_data["Sportsbook Line"] = (
                            selected_line
                        )

                        chart_data = chart_data.rename(
                            columns={
                                "value": "Player Performance"
                            }
                        )

                        st.line_chart(
                            chart_data
                        )

                        # --------------------------------------
                        # RECENT GAME LOG
                        # --------------------------------------

                        st.markdown(
                            "### 🏈 Recent Game Log"
                        )

                        game_log = player_games[
                            ["game_time", "value"]
                        ].copy()

                        game_log["Result"] = (
                            game_log["value"].apply(
                                lambda value:
                                "OVER"
                                if value > selected_line
                                else (
                                    "UNDER"
                                    if value < selected_line
                                    else "PUSH"
                                )
                            )
                        )

                        game_log = game_log.rename(
                            columns={
                                "game_time": "Game Date",
                                "value": historical_market
                            }
                        )

                        st.dataframe(
                            game_log.sort_values(
                                "Game Date",
                                ascending=False
                            ),
                            hide_index=True,
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "No valid sportsbook line is "
                            "available for historical comparison."
                        )

        except (
            ValueError,
            KeyError,
            pd.errors.ParserError
        ) as history_error:

            st.error(
                "Historical performance could not be "
                f"loaded: {history_error}"
            )

    # ==========================================================
    # END HISTORICAL PLAYER PERFORMANCE
    # ==========================================================
    

    st.divider()
    
