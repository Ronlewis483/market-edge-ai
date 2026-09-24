"""Standalone Streamlit panel. Never mutates existing NFL/stock session keys."""
import pandas as pd
import requests
import streamlit as st
from dual_agent.nfl_player_props import (MARKETS, analyze, event_props, events,
                                          normalize_props, validate_history, walkforward)

@st.cache_data(ttl=600, show_spinner=False)
def cached_events(key):
    return events(key)

@st.cache_data(ttl=300, show_spinner=False)
def cached_props(key, event_id, market_keys):
    return event_props(key,event_id,list(market_keys))

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
    st.subheader('Live sportsbook lines')
    st.dataframe(lines,hide_index=True,use_container_width=True)
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
