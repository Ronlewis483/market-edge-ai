import streamlit as st
st.set_page_config(page_title="Market Edge AI V3",page_icon="📊",layout="wide")
from dual_agent.stock import train_all,scan as stock_scan,metrics_all
from dual_agent.sports import scan as sports_scan
from dual_agent.db import history
st.title("📊 Market Edge AI V3")
st.caption("Calibrated multi-horizon stock models + sports market scanner • research/paper mode")
with st.sidebar:
    page=st.radio("Navigation",["Overview","Stocks","Sports","Prediction History"])
    st.info("Research/paper mode. No real-money orders or wagers.")
if page=="Overview":
    st.markdown("### Stronger V3 engine\nSeparate **1-day, 5-day and 20-day** models; date-separated train/calibration/test periods; calibrated probabilities; SPY/QQQ regime context; relative strength; RSI; ATR; volatility; drawdown; and benchmark comparisons.")
elif page=="Stocks":
    st.subheader("📈 Multi-Horizon Stock Scanner")
    txt=st.text_input("Symbols","SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA")
    syms=[x.strip().upper() for x in txt.split(",") if x.strip()]
    a,b=st.columns(2)
    if a.button("Train All Horizons",use_container_width=True):
        try:
            with st.spinner("Training 1d, 5d and 20d models..."): result=train_all(syms)
            st.success("Training complete."); st.json(result)
        except Exception as e: st.error(str(e))
    if b.button("Run Multi-Horizon Scan",type="primary",use_container_width=True):
        try:
            with st.spinner("Scanning..."): df=stock_scan(syms)
            show=df.copy()
            for c in ["Raw P","Calibrated P","Historical base","Model vs base"]: show[c]=show[c].map(lambda x:f"{x:.1%}")
            st.dataframe(show,use_container_width=True,hide_index=True)
            st.subheader("Calibrated probability by horizon")
            st.dataframe(df.pivot(index="Symbol",columns="Horizon",values="Calibrated P").style.format("{:.1%}"),use_container_width=True)
        except Exception as e: st.error(str(e))
    ms=metrics_all()
    if ms:
        with st.expander("Out-of-sample validation"):
            for h,m in ms.items():
                st.markdown(f"**{h}-day — {'PASSES benchmark' if m['passes_benchmarks'] else 'does NOT yet pass benchmark'}**"); st.json(m)
elif page=="Sports":
    st.subheader("🏀 Sports Market Scanner")
    leagues={"NBA":"basketball_nba","NFL":"americanfootball_nfl","MLB":"baseball_mlb","NHL":"icehockey_nhl"}; league=st.selectbox("League",list(leagues))
    st.warning("Sports remains a no-vig bookmaker consensus scanner until we add historical pregame data.")
    if st.button("Scan Current Moneylines",type="primary"):
        try:
            df=sports_scan(leagues[league]); d=df.copy()
            for c in ["Market P(Home)","Market P(Away)","Book disagreement"]: d[c]=d[c].map(lambda x:f"{x:.1%}")
            st.dataframe(d,use_container_width=True,hide_index=True)
        except Exception as e: st.error(str(e))
else:
    df=history(); st.subheader("🧾 Prediction History")
    if df.empty: st.info("No predictions logged yet.")
    else: st.dataframe(df,use_container_width=True,hide_index=True)
