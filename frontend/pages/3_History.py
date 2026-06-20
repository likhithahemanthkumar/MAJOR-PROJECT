import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Audit History Logs", layout="wide")
st.title("🗄️ Patient Longitudinal History Logs")

st.write("This secure view displays historical outputs across the deployment workspace.")

if st.button("Refresh Central Database Logs"):
    try:
        res = requests.get("http://127.0.0.1:8000/history")
        logs = res.json()["history"]
        
        if logs:
            # Structuring row array into clean dataframes
            df = pd.DataFrame(logs, columns=["Patient ID", "Modality Type", "Predicted Condition Stage", "Confidence", "Timestamp"])
            df["Confidence"] = df["Confidence"].map(lambda val: f"{val * 100:.1f}%")
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Database active but currently clear of logging entries.")
    except Exception:
        st.error("Unable to execute query against database API systems.")