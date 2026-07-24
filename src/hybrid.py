import os
import json
import time
import numpy as np
import pandas as pd
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.evaluation import Evaluator
from src.utils import get_logger, load_config

logger = get_logger(__name__)

class HybridRecommender:
    def __init__(self, config: dict):
        self.config = config
        self.artifacts_dir = config['data']['paths']['artifacts_dir']
        self.processed_dir = config['data']['paths']['processed_dir']
        
        self.cb_model = ContentBasedRecommender(config)
        self.cf_model = CollaborativeRecommender(config)
        
        self.best_alpha = 0.5
        self.cf_min = 0.0
        self.cf_max = 1.0
        self.cb_min = 0.0
        self.cb_max = 1.0
        
        self.popular_items = [] # Fallback for completely cold users
        
    def _normalize(self, scores, min_val, max_val):
        """Global Min-Max Normalization"""
        if max_val == min_val:
            return np.zeros_like(scores)
        norm = (scores - min_val) / (max_val - min_val)
        return np.clip(norm, 0, 1)

    def fit(self, reviews_df: pd.DataFrame):
        logger.info("Initializing Hybrid Recommender Tuning...")
        self.cb_model.load()
        self.cf_model.load()
        
        # Calculate Global Bounds for Normalization
        logger.info("Calculating global score distributions for normalization...")
        cf_u_emb = self.cf_model.model.user_embeddings
        cf_i_emb = self.cf_model.model.item_embeddings
        
        # Sample to prevent OOM
        np.random.seed(42)
        sample_u = cf_u_emb[np.random.choice(cf_u_emb.shape[0], min(500, cf_u_emb.shape[0]), replace=False)]
        sample_i = cf_i_emb[np.random.choice(cf_i_emb.shape[0], min(500, cf_i_emb.shape[0]), replace=False)]
        dot_prods = np.dot(sample_u, sample_i.T)
        
        self.cf_min = float(dot_prods.min())
        self.cf_max = float(dot_prods.max())
        self.cb_min = 0.0
        self.cb_max = 1.0 # Cosine Sim bound
        
        # Calculate global popular items for cold-start fallback
        self.popular_items = reviews_df['parent_asin'].value_counts().head(20).index.tolist()
        
        logger.info(f"CF Bounds: [{self.cf_min:.2f}, {self.cf_max:.2f}]")
        
        # Grid Search for best Alpha
        alphas = self.config['hybrid']['alpha_grid']
        val_size = self.config['hybrid']['validation_subset_size']
        primary_metric = self.config['hybrid']['primary_metric']
        
        # Create validation set
        users = reviews_df['user_id'].unique()
        val_users = np.random.choice(users, min(val_size, len(users)), replace=False)
        
        # For evaluation, we pretend the user has interacted with 80% of their items, and we predict the 20%
        actuals = []
        train_histories = {}
        
        for u in val_users:
            u_items = reviews_df[reviews_df['user_id'] == u]['parent_asin'].tolist()
            if len(u_items) < 3:
                continue
            split_idx = int(len(u_items) * 0.8)
            train_histories[u] = u_items[:split_idx]
            actuals.append(u_items[split_idx:])
            
        valid_val_users = list(train_histories.keys())
        logger.info(f"Running evaluation on {len(valid_val_users)} users...")
        
        evaluator = Evaluator(k_values=[10])
        best_metric_val = -1.0
        
        # Precompute CF and CB raw scores for efficiency during tuning
        n_items = len(self.cf_model.item_mapping)
        item_ids_array = np.array([self.cf_model.idx_to_item[i] for i in range(n_items)])
        
        user_cf_scores = {}
        user_cb_scores = {}
        
        for u in valid_val_users:
            # CF
            u_idx = self.cf_model.user_mapping[u]
            cf_raw = self.cf_model.model.predict(u_idx, np.arange(n_items))
            user_cf_scores[u] = self._normalize(cf_raw, self.cf_min, self.cf_max)
            
            # CB
            hist = train_histories[u]
            hist_idx = [self.cb_model.product_mapping[p] for p in hist if p in self.cb_model.product_mapping]
            if hist_idx:
                u_profile = self.cb_model.tfidf_matrix[hist_idx].mean(axis=0)
                cb_raw = self.cb_model.cosine_similarity(u_profile, self.cb_model.tfidf_matrix).flatten()
            else:
                cb_raw = np.zeros(n_items)
            user_cb_scores[u] = self._normalize(cb_raw, self.cb_min, self.cb_max)

        # Re-attach cosine_similarity bound to class since we used it directly above
        from sklearn.metrics.pairwise import cosine_similarity
        self.cb_model.cosine_similarity = cosine_similarity
            
        logger.info(f"{'Alpha':<10} | {'Prec@10':<10} | {'Rec@10':<10} | {'MAP@10':<10} | {'NDCG@10':<10}")
        logger.info("-" * 65)
        
        results = []
        for alpha in alphas:
            predictions = []
            for u in valid_val_users:
                cf_s = user_cf_scores[u]
                cb_s = user_cb_scores[u]
                
                # Hybrid Formula
                hybrid_s = (alpha * cf_s) + ((1.0 - alpha) * cb_s)
                
                # Exclude train items
                train_idx = [self.cf_model.item_mapping[p] for p in train_histories[u] if p in self.cf_model.item_mapping]
                hybrid_s[train_idx] = -np.inf
                
                top_idx = np.argsort(hybrid_s)[::-1][:10]
                predictions.append(item_ids_array[top_idx].tolist())
                
            metrics = evaluator.evaluate(actuals, predictions)
            logger.info(f"{alpha:<10.1f} | {metrics['Precision@10']:<10.4f} | {metrics['Recall@10']:<10.4f} | {metrics['MAP@10']:<10.4f} | {metrics['NDCG@10']:<10.4f}")
            
            if metrics[primary_metric] > best_metric_val:
                best_metric_val = metrics[primary_metric]
                self.best_alpha = alpha
                
        logger.info("-" * 65)
        logger.info(f"Optimal Alpha found: {self.best_alpha} (Based on {primary_metric})")
        return self

    def save(self):
        logger.info(f"Saving Hybrid artifacts to {self.artifacts_dir}...")
        params = {
            "best_alpha": float(self.best_alpha),
            "cf_min": float(self.cf_min),
            "cf_max": float(self.cf_max),
            "cb_min": float(self.cb_min),
            "cb_max": float(self.cb_max),
            "popular_items": self.popular_items
        }
        with open(os.path.join(self.artifacts_dir, "hybrid_params.json"), "w") as f:
            json.dump(params, f, indent=4)

    def load(self):
        logger.info("Loading Hybrid artifacts...")
        self.cb_model.load()
        self.cf_model.load()
        with open(os.path.join(self.artifacts_dir, "hybrid_params.json"), "r") as f:
            params = json.load(f)
            self.best_alpha = params["best_alpha"]
            self.cf_min = params["cf_min"]
            self.cf_max = params["cf_max"]
            self.cb_min = params["cb_min"]
            self.cb_max = params["cb_max"]
            self.popular_items = params.get("popular_items", [])
            
    def recommend_for_user(self, user_id: str, reviews_df: pd.DataFrame, top_k: int = 10):
        if self.cb_model.vectorizer is None or self.cf_model.model is None:
            self.load()
            
        n_items = len(self.cf_model.item_mapping)
        item_ids_array = np.array([self.cf_model.idx_to_item[i] for i in range(n_items)])
        
        # Cold User Fallback
        if user_id not in self.cf_model.user_mapping:
            logger.info(f"Cold user detected: {user_id}. Falling back to popular items.")
            recs = []
            for pid in self.popular_items[:top_k]:
                title = self.cf_model.meta_df[self.cf_model.meta_df['product_id'] == pid].iloc[0]['title']
                recs.append({"product_id": pid, "title": title, "score": 1.0, "source": "Fallback (Popular)"})
            return recs
            
        # Get historical items to exclude
        u_items = reviews_df[reviews_df['user_id'] == user_id]['parent_asin'].tolist()
        
        # CF Score
        u_idx = self.cf_model.user_mapping[user_id]
        cf_raw = self.cf_model.model.predict(u_idx, np.arange(n_items))
        cf_norm = self._normalize(cf_raw, self.cf_min, self.cf_max)
        
        # CB Score
        cb_norm = np.zeros(n_items)
        hist_idx = [self.cb_model.product_mapping[p] for p in u_items if p in self.cb_model.product_mapping]
        if hist_idx:
            from sklearn.metrics.pairwise import cosine_similarity
            u_profile = self.cb_model.tfidf_matrix[hist_idx].mean(axis=0)
            cb_raw = cosine_similarity(u_profile, self.cb_model.tfidf_matrix).flatten()
            cb_norm = self._normalize(cb_raw, self.cb_min, self.cb_max)
            
        # Combine
        alpha = self.best_alpha
        hybrid_scores = (alpha * cf_norm) + ((1.0 - alpha) * cb_norm)
        
        # Exclude historical
        exclude_idx = [self.cf_model.item_mapping[p] for p in u_items if p in self.cf_model.item_mapping]
        hybrid_scores[exclude_idx] = -np.inf
        
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            pid = item_ids_array[idx]
            title = self.cf_model.meta_df[self.cf_model.meta_df['product_id'] == pid].iloc[0]['title']
            recommendations.append({
                "product_id": pid,
                "title": title,
                "hybrid_score": float(hybrid_scores[idx]),
                "cf_contribution": float(alpha * cf_norm[idx]),
                "cb_contribution": float((1.0 - alpha) * cb_norm[idx]),
                "source": "Hybrid"
            })
            
        return recommendations

if __name__ == "__main__":
    config = load_config()
    reviews_df = pd.read_parquet(os.path.join(config['data']['paths']['processed_dir'], "reviews_processed.parquet"))
    hybrid = HybridRecommender(config)
    hybrid.fit(reviews_df)
    hybrid.save()
