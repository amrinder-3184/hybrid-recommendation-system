import pandas as pd
import streamlit as st

from frontend.api_client import get_popular

st.set_page_config(page_title="Popular Products", page_icon="🔥", layout="wide")
st.title("🔥 Globally Popular Products")
st.markdown("These are the most interacted-with products across the entire dataset. This serves as our cold-user fallback.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K", min_value=1, max_value=50, value=10)
    fetch = st.button("Fetch Popular")

if fetch:
    with st.spinner("Fetching..."):
        try:
            data, latency = get_popular(top_k)
            if latency:
                st.success(f"Retrieved popular items in {float(latency):.4f}s")
            
            recs = data.get("recommendations", [])
            if not recs:
                st.info("No popular items found.")
            else:
                st.dataframe(pd.DataFrame(recs), use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
