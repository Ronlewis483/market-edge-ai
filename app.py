import streamlit as st
st.set_page_config(page_title="Market Edge AI V4",page_icon="📊",layout="wide")
from dual_agent.research import DEFAULT_UNIVERSE,run_research,latest_scan
st.title("📊 Market Edge AI V4 — Research Lab")
st.caption("Walk-forward validation • larger cross-sector universe • target testing • feature ablation")
page=st.sidebar.radio("Navigation",["Research Lab","Experimental Scanner"])
default=[x for x in DEFAULT_UNIVERSE.split(",") if x]
if page=="Research Lab":
    st.subheader("🧪 5-Day Model Research")
    st.write("This tests multiple prediction targets and feature sets using repeated walk-forward out-of-sample periods.")
    txt=st.text_area("Training universe",",".join(default),height=150)
    universe=[x.strip().upper() for x in txt.split(",") if x.strip()]
    st.metric("Training symbols",len(universe))
    if st.button("Run Walk-Forward Research",type="primary"):
        try:
            with st.spinner("Downloading history and running repeated historical simulations..."):
                df=run_research(universe)
            st.session_state["research"]=df
        except Exception as e: st.error(str(e))
    if "research" in st.session_state:
        df=st.session_state["research"].copy()
        show=df.copy()
        for c in ["base_rate","auc","model_brier","baseline_brier","model_log_loss","baseline_log_loss","high_conf_accuracy"]:
            if c in show: show[c]=show[c].map(lambda x:"" if x!=x else f"{x:.4f}")
        st.dataframe(show,use_container_width=True,hide_index=True)
        st.info("Lower Brier/log loss is better. AUC above .50 indicates discrimination, but promotion to the live scanner should require repeated benchmark wins—not AUC alone.")
elif page=="Experimental Scanner":
    st.subheader("🔬 Experimental 5-Day Relative-Performance Scanner")
    st.warning("This scanner is experimental. Validate the corresponding research configuration before interpreting its probabilities.")
    scan_txt=st.text_input("Stocks to scan","AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA")
    scan=[x.strip().upper() for x in scan_txt.split(",") if x.strip()]
    if st.button("Run Experimental Scan",type="primary"):
        try:
            with st.spinner("Training on broad universe and scanning..."):
                df=latest_scan(default,scan)
            d=df.copy(); d["P"]=d["P"].map(lambda x:f"{x:.1%}")
            st.dataframe(d,use_container_width=True,hide_index=True)
        except Exception as e: st.error(str(e))
