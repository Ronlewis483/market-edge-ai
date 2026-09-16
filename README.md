# Market Edge AI — Streamlit-ready

## Deploy
1. Create a GitHub repository.
2. Upload the **contents** of this ZIP (not the ZIP itself).
3. In Streamlit Community Cloud choose **Deploy a public app from GitHub**.
4. Select the repo and branch `main`.
5. Main file path: `app.py`.

## Add secrets
In Streamlit app settings → Secrets, paste:

```toml
ALPACA_API_KEY = "YOUR_ALPACA_KEY"
ALPACA_API_SECRET = "YOUR_ALPACA_SECRET"
ODDS_API_KEY = "YOUR_ODDS_API_KEY"
```

Never commit API keys to GitHub.

## Use
Stocks → Train / Retrain → Run Scan.
Sports → choose league → Scan Current Moneylines.

This is a research/paper build and does not place real-money trades or wagers.
