import numpy as np,pandas as pd,requests
from .config import secret
from .db import log_prediction
def scan(sport):
    key=secret("ODDS_API_KEY")
    if not key: raise RuntimeError("Add ODDS_API_KEY in Streamlit Secrets.")
    r=requests.get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds",params={"apiKey":key,"regions":"us","markets":"h2h","oddsFormat":"decimal"},timeout=30)
    r.raise_for_status(); out=[]
    for game in r.json():
        home,away=game["home_team"],game["away_team"]; hp=[]; ap=[]
        for book in game.get("bookmakers",[]):
            m=next((x for x in book.get("markets",[]) if x["key"]=="h2h"),None)
            if not m: continue
            prices={o["name"]:o["price"] for o in m["outcomes"]}
            if home not in prices or away not in prices: continue
            ih,ia=1/prices[home],1/prices[away]; total=ih+ia; hp.append(ih/total); ap.append(ia/total)
        if hp:
            ph,pa=float(np.mean(hp)),float(np.mean(ap)); dis=float(np.std(hp)) if len(hp)>1 else 0
            log_prediction("sports_market",f"{away} @ {home}",ph,"moneyline",ph,metadata={"books":len(hp)})
            out.append({"Start":game["commence_time"],"Away":away,"Home":home,"Books":len(hp),"Market P(Home)":ph,"Market P(Away)":pa,"Book disagreement":dis})
    return pd.DataFrame(out)
