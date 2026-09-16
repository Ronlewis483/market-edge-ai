import sqlite3, json
from datetime import datetime, timezone
from .config import DB_PATH
def connect():
    con=sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, domain TEXT, target TEXT, horizon TEXT, probability REAL, market_probability REAL, estimated_edge REAL, metadata TEXT, outcome REAL)")
    con.commit(); return con
def log_prediction(domain,target,probability,horizon=None,market_probability=None,estimated_edge=None,metadata=None):
    con=connect()
    con.execute("INSERT INTO predictions (created_at,domain,target,horizon,probability,market_probability,estimated_edge,metadata) VALUES (?,?,?,?,?,?,?,?)",
      (datetime.now(timezone.utc).isoformat(),domain,target,horizon,float(probability),market_probability,estimated_edge,json.dumps(metadata) if isinstance(metadata,dict) else metadata))
    con.commit(); con.close()
def history():
    import pandas as pd
    con=connect(); df=pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC",con); con.close(); return df
