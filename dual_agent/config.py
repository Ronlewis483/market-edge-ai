from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
DATA_DIR=ROOT/"data"; MODEL_DIR=ROOT/"models"; DB_PATH=DATA_DIR/"agent.db"
DATA_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
def secret(name, default=""):
    try:
        import streamlit as st
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)
