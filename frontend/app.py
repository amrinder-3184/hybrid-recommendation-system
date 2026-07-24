import streamlit as st

st.set_page_config(
    page_title="Hybrid Recommendation System",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Hybrid Recommendation System Dashboard")

st.markdown("""
Welcome to the End-to-End Hybrid Recommendation System!

This dashboard serves as the frontend client for our FastAPI backend inference service. It is designed to evaluate and interpret our production Machine Learning models in real-time.

### 🌟 Features
- **User Recommendations**: Explore personalized hybrid recommendations for any user.
- **Recommendation Explanations**: Deep dive into *why* an item was recommended, featuring score breakdowns and keyword matching.
- **Similar Products**: Discover items utilizing Content-Based and Collaborative metrics independently.
- **System Health**: Monitor backend latency and model loading status.

Select a page from the sidebar to begin!
""")
