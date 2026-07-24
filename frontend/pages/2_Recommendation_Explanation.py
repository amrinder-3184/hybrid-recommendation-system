import streamlit as st
from frontend.api_client import get_user_recommendations

st.set_page_config(page_title="Recommendation Explanation", page_icon="🔍", layout="wide")
st.title("🔍 Recommendation Explainability")
st.markdown("Interviewers love explainability. This page breaks down exactly **why** the Hybrid model recommended an item.")

with st.sidebar:
    user_id = st.text_input("User ID", value="A123")
    top_k = st.number_input("Top K", value=5, min_value=1, max_value=20)
    fetch = st.button("Analyze Recommendations")

if fetch:
    with st.spinner("Analyzing..."):
        try:
            data, _ = get_user_recommendations(user_id, top_k)
            recs = data.get("recommendations", [])
            
            if not recs:
                st.warning("No recommendations found to analyze.")
            else:
                for i, rec in enumerate(recs, 1):
                    with st.expander(f"{i}. {rec['title']} (Score: {rec['score']:.4f})", expanded=(i==1)):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric("Final Hybrid Score", f"{rec['score']:.4f}")
                            if rec.get("cf_contribution") is not None:
                                st.metric("CF Contribution", f"{rec['cf_contribution']:.4f}")
                                st.metric("CB Contribution", f"{rec['cb_contribution']:.4f}")
                                
                        with col2:
                            st.subheader("Why was this recommended?")
                            if str(rec.get("source")).startswith("Fallback"):
                                st.info("Users similar to you did not have enough data. This item is globally popular.")
                            else:
                                st.success("Users with similar latent behaviors to you highly interacted with this item (Collaborative Filtering), AND it shares textual metadata characteristics with items you previously interacted with (Content-Based Filtering).")
                                
                            if rec.get("explanation_terms"):
                                st.markdown("**Matched Keywords:**")
                                for term in rec["explanation_terms"]:
                                    st.markdown(f"- ✓ {term}")
        except Exception as e:
            st.error(f"Failed to fetch explanations. Is the FastAPI backend running? Details: {e}")
