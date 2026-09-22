
import streamlit as st
import pandas as pd

from datetime import datetime, timedelta, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from dual_agent.config import secret


st.set_page_config(
    page_title="Alpaca Connection Test",
    layout="wide",
)

st.title("Market Edge AI V5")
st.subheader("Alpaca Market Data Verification")

st.info(
    "This test checks historical intraday market data. "
    "No trades will be placed."
)


def test_alpaca_connection():

    api_key = secret("ALPACA_API_KEY")
    api_secret = secret("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        st.error("Alpaca credentials are missing.")
        return

    client = StockHistoricalDataClient(
        api_key,
        api_secret,
    )

    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=7)

    request = StockBarsRequest(
        symbol_or_symbols=["AAPL"],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    try:

        response = client.get_stock_bars(request)

        df = response.df.reset_index()

        if df.empty:
            st.warning(
                "Connection succeeded, but no "
                "market data was returned."
            )
            return

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

        latest = df["timestamp"].max()

        st.success(
            "Alpaca intraday data retrieved successfully!"
        )

        st.metric(
            "Candles Retrieved",
            len(df),
        )

        st.metric(
            "Latest Candle",
            str(latest),
        )

        st.dataframe(
            df.tail(10),
            use_container_width=True,
        )

    except Exception as error:

        st.error(
            "Alpaca market data request failed."
        )

        st.code(
            f"{type(error).__name__}: {error}"
        )


if st.button("Test Alpaca Connection"):

    test_alpaca_connection()
