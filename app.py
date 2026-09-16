
import streamlit as st
st.set_page_config(page_title="Market Edge AI", page_icon="📊", layout="wide")

from dual_agent.research import DEFAULT_UNIVERSE, run_research, latest_scan
from dual_agent.signal_engine import clean_symbols, select_research_model, stock_decision, sports_decision

st.title("📊 Market Edge AI")
st.caption("Validated signal engine • research/paper mode")

with st.sidebar:
    page = st.radio("Navigation", ["Command Center","Research Lab"])
    st.info("Research/paper mode. Signals are withheld unless validation gates are met.")

# Critical fix: normalize the V4 multiline DEFAULT_UNIVERSE.
default = clean_symbols(DEFAULT_UNIVERSE)

if page == "Command Center":
    st.subheader("🎯 Command Center")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 📈 Best Stock Signal")
        if "research" not in st.session_state:
            st.warning("NO QUALIFYING TRADE")
            st.caption("Run the Research Lab first in this session.")
        else:
            chosen = select_research_model(st.session_state["research"])
            scan_txt = st.text_input("Stock universe to scan",
                "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AVGO,AMD,JPM,LLY,XOM")
            scan_syms = clean_symbols(scan_txt)

            if st.button("Find Best Trade", type="primary", use_container_width=True):
                try:
                    with st.spinner("Scanning validated stock model..."):
                        # Both lists are sanitized before reaching research/latest_scan.
                        scan = latest_scan(clean_symbols(default), clean_symbols(scan_syms))
                    st.session_state["stock_decision"] = stock_decision(scan, chosen)
                except Exception as e:
                    st.error(f"Scan failed: {e}")

            d = st.session_state.get("stock_decision")
            if d:
                if d["status"] == "MAKE THIS TRADE":
                    st.success("MAKE THIS TRADE")
                    st.markdown(f"## {d['symbol']}")
                    st.metric("Model probability", f"{d['probability']:.1%}")
                    st.metric("Model edge vs historical base", f"+{d['edge']:.1%}")
                    st.write(f"**Direction:** {d['direction']}")
                    st.write(f"**Horizon:** {d['horizon']}")
                    st.write(f"**Model target:** {d['target']}")
                    st.caption(f"Research AUC {d['research_auc']:.3f} • features: {d['feature_set']}")
                else:
                    st.warning(d["status"])
                    st.caption(d["reason"])
    with c2:
        st.markdown("### 🏀 Best Sports Signal")
        sd=sports_decision()
        st.warning(sd["status"])
        st.caption(sd["reason"])
        st.code("PICK THIS TEAM / PICK THIS PLAYER PROP\nor\nNO QUALIFYING SPORTS PICK")

    st.divider()
    st.caption("The Command Center is intentionally selective: no qualifying signal is a valid outcome.")

else:
    st.subheader("🧪 Stock Research Lab")
    st.write("Tests multiple 5-day targets and feature sets using repeated walk-forward out-of-sample periods.")
    txt = st.text_area("Training universe", ",".join(default), height=150)
    universe = clean_symbols(txt)
    st.metric("Training symbols", len(universe))

    if st.button("Run Walk-Forward Research", type="primary"):
        try:
            with st.spinner("Downloading history and running repeated historical simulations..."):
                st.session_state["research"] = run_research(clean_symbols(universe))
            st.success("Research complete.")
        except Exception as e:
            st.error(str(e))

    if "research" in st.session_state:
        df=st.session_state["research"].copy()
        st.dataframe(df,use_container_width=True,hide_index=True)
        chosen=select_research_model(df)
        if chosen:
            st.success(f"Validated candidate: Beat SPY in 5d • {chosen['Features']} • AUC {chosen['auc']:.3f}. Command Center is unlocked for this research session.")
        else:
            st.warning("No stock model currently clears the promotion gates. Command Center will return NO QUALIFYING TRADE.")
