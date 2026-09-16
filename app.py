import streamlit as st
st.set_page_config(page_title="Market Edge AI",page_icon="📊",layout="wide")
from src.dual_agent.stock import train,scan as stock_scan
from src.dual_agent.sports import scan as sports_scan
from src.dual_agent.db import history
st.title("📊 Market Edge AI")
st.caption("Stocks + sports probability dashboard • research/paper mode")
with st.sidebar:
    page=st.radio("Navigation",["Overview","Stocks","Sports","Prediction History"])
    st.info("This version analyzes and tracks predictions. It does not place real-money trades or wagers.")
if page=="Overview":
    h=history(); a,b,c=st.columns(3)
    a.metric("Predictions logged",len(h)); b.metric("Stock predictions",int((h.domain=="stock").sum()) if len(h) else 0); c.metric("Sports scans",int((h.domain=="sports_market").sum()) if len(h) else 0)
    st.markdown("### Command Center\nThe stock engine trains on historical price/volume features. The sports screen currently computes no-vig consensus probabilities across bookmakers. Every scan is logged for later grading.")
elif page=="Stocks":
    st.subheader("📈 Stock Probability Scanner")
    txt=st.text_input("Symbols","SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA")
    syms=[x.strip().upper() for x in txt.split(",") if x.strip()]; h=st.selectbox("Horizon",[1,5,10,20],index=1)
    a,b=st.columns(2)
    if a.button("Train / Retrain",use_container_width=True):
        try:
            with st.spinner("Training model..."): st.json(train(syms,h))
        except Exception as e: st.error(str(e))
    if b.button("Run Scan",type="primary",use_container_width=True):
        try:
            with st.spinner("Scanning..."): df,metrics=stock_scan(syms)
            d=df.copy(); d["Probability higher"]=d["Probability higher"].map(lambda x:f"{x:.1%}"); d["Strength"]=d["Strength"].map(lambda x:f"{x:.1%}")
            st.dataframe(d,use_container_width=True,hide_index=True)
            with st.expander("Model test metrics"): st.json(metrics)
        except Exception as e: st.error(str(e))
elif page=="Sports":
    st.subheader("🏀 Sports Market Scanner")
    leagues={"NBA":"basketball_nba","NFL":"americanfootball_nfl","MLB":"baseball_mlb","NHL":"icehockey_nhl"}
    league=st.selectbox("League",list(leagues))
    st.warning("Current sports probabilities are bookmaker consensus, not an independent AI edge yet.")
    if st.button("Scan Current Moneylines",type="primary"):
        try:
            with st.spinner("Scanning books..."): df=sports_scan(leagues[league])
            d=df.copy()
            for c in ["Market P(Home)","Market P(Away)","Book disagreement"]: d[c]=d[c].map(lambda x:f"{x:.1%}")
            st.dataframe(d,use_container_width=True,hide_index=True)
        except Exception as e: st.error(str(e))
else:
    st.subheader("🧾 Prediction History"); df=history()
    if df.empty: st.info("No predictions logged yet.")
    else:
        st.dataframe(df.drop(columns=["metadata"],errors="ignore"),use_container_width=True,hide_index=True)
        st.download_button("Download CSV",df.to_csv(index=False),"prediction_history.csv","text/csv")
