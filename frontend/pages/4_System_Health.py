import streamlit as st
from frontend.api_client import get_health

st.set_page_config(page_title="System Health", page_icon="🩺", layout="wide")
st.title("🩺 System Health Monitor")

st.markdown("Monitor backend latency and API status.")

if st.button("Check Health", type="primary"):
    with st.spinner("Pinging API..."):
        data, latency = get_health()
        
        if data.get("status") == "ok":
            st.success("API is Online")
            col1, col2 = st.columns(2)
            if latency:
                col1.metric("API Latency", f"{float(latency):.4f}s")
            col2.metric("Models Loaded", str(data.get("models_loaded", False)))
        else:
            st.error("API is Offline or Unreachable")
            st.json(data)
