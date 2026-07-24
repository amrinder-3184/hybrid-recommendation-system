import streamlit as st
import pandas as pd
from frontend.api_client import get_item_cb_similar, get_item_cf_similar

st.set_page_config(page_title="Similar Products", page_icon="🛍️", layout="wide")
st.title("🛍️ Similar Products Explorer")

with st.sidebar:
    st.header("Settings")
    product_id = st.text_input("Product ID", value="B00004U9V6")
    top_k = st.slider("Top K", min_value=1, max_value=50, value=10)
    method = st.radio("Similarity Method", ["Content-Based (Metadata)", "Collaborative (Latent Embeddings)"])
    fetch = st.button("Find Similar")

if fetch:
    with st.spinner("Fetching..."):
        try:
            if "Content" in method:
                data, latency = get_item_cb_similar(product_id, top_k)
            else:
                data, latency = get_item_cf_similar(product_id, top_k)
                
            if latency:
                st.success(f"Retrieved similar items in {float(latency):.4f}s")
            
            recs = data.get("recommendations", [])
            if not recs:
                st.info("No similar items found.")
            else:
                st.dataframe(pd.DataFrame(recs), use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
