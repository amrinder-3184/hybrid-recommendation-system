import streamlit as st
import pandas as pd
from frontend.api_client import get_user_recommendations

st.set_page_config(page_title="User Recommendations", page_icon="👤", layout="wide")
st.title("👤 Personalized User Recommendations")

with st.sidebar:
    st.header("Settings")
    user_id = st.text_input("User ID", value="A123")
    top_k = st.slider("Top K Recommendations", min_value=1, max_value=50, value=10)
    fetch = st.button("Fetch Recommendations")

if fetch:
    with st.spinner("Fetching from API..."):
        try:
            data, latency = get_user_recommendations(user_id, top_k)
            if latency:
                st.success(f"Successfully retrieved recommendations in {float(latency):.4f}s")
            
            recs = data.get("recommendations", [])
            if not recs:
                st.info("No recommendations found.")
            else:
                df = pd.DataFrame(recs)
                # Reorder columns
                cols = ["product_id", "title", "score", "cf_contribution", "cb_contribution", "source"]
                # Keep only existing columns
                cols = [c for c in cols if c in df.columns]
                st.dataframe(df[cols], use_container_width=True)
                
        except Exception as e:
            st.error(f"Error fetching data. Is the FastAPI backend running? Details: {e}")
