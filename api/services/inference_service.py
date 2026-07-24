import os
import pandas as pd
from src.hybrid import HybridRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.utils import get_logger, load_config

logger = get_logger(__name__)

class InferenceService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing Inference Service & Loading Models...")
        self.config = load_config()
        
        # We only really need to load Hybrid model since it encapsulates CB and CF
        self.hybrid = HybridRecommender(self.config)
        self.hybrid.load()
        
        # Load reviews dataframe for history exclusion
        reviews_path = os.path.join(self.config['data']['paths']['processed_dir'], "reviews_processed.parquet")
        self.reviews_df = pd.read_parquet(reviews_path)
        
        self.initialized = True
        logger.info("Models loaded successfully.")

    def reload(self):
        self.initialized = False
        self.initialize()

    def recommend_for_user(self, user_id: str, top_k: int = 10):
        recs = self.hybrid.recommend_for_user(user_id, self.reviews_df, top_k=top_k)
        formatted_recs = []
        for r in recs:
            formatted_recs.append({
                "product_id": r["product_id"],
                "title": r["title"],
                "score": r.get("hybrid_score", r.get("score", 0.0)),
                "cf_contribution": r.get("cf_contribution"),
                "cb_contribution": r.get("cb_contribution"),
                "source": r.get("source", "Hybrid")
            })
        return formatted_recs

    def recommend_for_item(self, product_id: str, top_k: int = 10):
        # Fallback to pure CB model inside hybrid for item-item similarity
        recs = self.hybrid.cb_model.recommend_similar_items(product_id, top_k=top_k)
        formatted_recs = []
        for r in recs:
            formatted_recs.append({
                "product_id": r["product_id"],
                "title": r["title"],
                "score": r["similarity_score"],
                "explanation_terms": r.get("explanation_terms"),
                "source": "Content-Based"
            })
        return formatted_recs

    def similar_items_cf(self, product_id: str, top_k: int = 10):
        recs = self.hybrid.cf_model.similar_items(product_id, top_k=top_k)
        formatted_recs = []
        for r in recs:
            formatted_recs.append({
                "product_id": r["product_id"],
                "title": r["title"],
                "score": r["similarity_score"],
                "source": "Collaborative"
            })
        return formatted_recs

    def similar_users(self, user_id: str, top_k: int = 10):
        return self.hybrid.cf_model.similar_users(user_id, top_k=top_k)

    def popular_items(self, top_k: int = 10):
        pop = self.hybrid.popular_items[:top_k]
        formatted_recs = []
        for pid in pop:
            title = self.hybrid.cf_model.meta_df[self.hybrid.cf_model.meta_df['product_id'] == pid].iloc[0]['title']
            formatted_recs.append({
                "product_id": pid,
                "title": title,
                "score": 1.0,
                "source": "Fallback (Popular)"
            })
        return formatted_recs

# Global singleton instance
inference_service = InferenceService()
