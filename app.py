
import streamlit as st
st.set_page_config(page_title="Market Edge AI V5", page_icon="📊", layout="wide")

from dual_agent.research import DEFAULT_UNIVERSE, latest_scan, run_research
from dual_agent.signal_engine import clean_symbols, stock_decision, sports_decision
st.title("📊 Market Edge AI — V5")
st.caption("Persistent validated model • lightweight daily inference • research/paper mode")

with st.sidebar:
    page=st.radio("Navigation",["Command Center","Saved Model","Research Lab"])
    st.success("Persistence enabled")
    st.caption("Validated configuration loads automatically. Daily use does not require walk-forward retraining.")

default=clean_symbols(DEFAULT_UNIVERSE)

if page=="Command Center":
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

elif page=="Saved Model":
    st.subheader("💾 Saved Validated Model")
    st.success("This configuration is bundled with the app and survives Streamlit restarts.")
    st.json(M)
    st.write("Daily Command Center scans load this approved configuration automatically; they do not rerun the 90-symbol walk-forward experiment.")

else:
    st.subheader("🧪 Research Lab — Optional Revalidation")
    st.warning("CPU-intensive. This is for periodic model research, not daily use.")
    txt=st.text_area("Training universe",",".join(default),height=150)
    universe=clean_symbols(txt)
    st.metric("Training symbols",len(universe))
    confirm=st.checkbox("I understand this is CPU-intensive")
    if st.button("Run Full Walk-Forward Revalidation",disabled=not confirm):
        try:
            with st.spinner("Running full historical revalidation..."):
                result=run_research(universe)
                st.session_state["research_result"]=result
            st.success("Revalidation complete. Review results before changing the persisted production candidate.")
        except Exception as e:
            st.error(str(e))
    if "research_result" in st.session_state:
        st.dataframe(st.session_state["research_result"],use_container_width=True,hide_index=True)
