import os
import time
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import get_logger, load_config

logger = get_logger(__name__)

class ContentBasedRecommender:
    def __init__(self, config: dict):
        self.config = config
        self.processed_dir = config['data']['paths']['processed_dir']
        self.artifacts_dir = config['data']['paths']['artifacts_dir']
        self.tfidf_params = config['content_based']['tfidf']
        self.combined_text_col = config['features']['combined_text_col']
        
        # Will be populated during fit() or load()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.product_mapping = None
        self.idx_to_product = None
        self.meta_df = None
        
    def fit(self, meta_df: pd.DataFrame):
        """Fits the TF-IDF vectorizer on the provided metadata."""
        start_time = time.time()
        logger.info("Fitting Content-Based Recommender...")
        
        self.meta_df = meta_df.copy()
        
        # Ensure we have strings and drop nulls in combined text
        self.meta_df[self.combined_text_col] = self.meta_df[self.combined_text_col].fillna("")
        
        logger.info(f"Initializing TfidfVectorizer with params: {self.tfidf_params}")
        # Convert ngram_range from list to tuple
        params = self.tfidf_params.copy()
        params['ngram_range'] = tuple(params['ngram_range'])
        
        self.vectorizer = TfidfVectorizer(**params)
        
        logger.info("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.meta_df[self.combined_text_col])
        
        # Create mappings
        logger.info("Creating product_id -> row_index mapping...")
        self.product_mapping = {prod_id: idx for idx, prod_id in enumerate(self.meta_df['product_id'])}
        self.idx_to_product = {idx: prod_id for prod_id, idx in self.product_mapping.items()}
        
        elapsed = time.time() - start_time
        
        logger.info("--- Content-Based Model Fitting Complete ---")
        logger.info(f"Training Time: {elapsed:.2f} seconds")
        logger.info(f"TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")
        logger.info(f"Vocabulary Size: {len(self.vectorizer.vocabulary_)}")
        return self

    def save(self):
        """Saves the fitted artifacts to disk."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Model must be fitted before saving artifacts.")
            
        logger.info(f"Saving artifacts to {self.artifacts_dir}...")
        vectorizer_path = os.path.join(self.artifacts_dir, "tfidf_vectorizer.pkl")
        matrix_path = os.path.join(self.artifacts_dir, "tfidf_matrix.npz")
        mapping_path = os.path.join(self.artifacts_dir, "cb_product_mapping.pkl")
        
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
            
        sp.save_npz(matrix_path, self.tfidf_matrix)
        
        with open(mapping_path, "wb") as f:
            pickle.dump({
                "product2idx": self.product_mapping,
                "idx2product": self.idx_to_product
            }, f)
            
        logger.info("Artifacts saved successfully.")

    def load(self):
        """Loads trained artifacts from disk for inference."""
        logger.info("Loading Content-Based artifacts for inference...")
        vectorizer_path = os.path.join(self.artifacts_dir, "tfidf_vectorizer.pkl")
        matrix_path = os.path.join(self.artifacts_dir, "tfidf_matrix.npz")
        mapping_path = os.path.join(self.artifacts_dir, "cb_product_mapping.pkl")
        meta_path = os.path.join(self.processed_dir, "meta_processed.parquet")
        
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
            
        self.tfidf_matrix = sp.load_npz(matrix_path)
        
        with open(mapping_path, "rb") as f:
            mappings = pickle.load(f)
            self.product_mapping = mappings['product2idx']
            self.idx_to_product = mappings['idx2product']
            
        self.meta_df = pd.read_parquet(meta_path)
        logger.info("Artifacts loaded successfully.")
        return self
        
    def get_explanation(self, query_vec, target_vec, feature_names, top_n=3):
        """Finds overlapping TF-IDF terms to explain the recommendation."""
        overlap = query_vec.multiply(target_vec)
        if overlap.nnz == 0:
            return []
        
        # Get top overlapping terms
        indices = overlap.nonzero()[1]
        scores = overlap.data
        
        # Sort descending by score
        top_indices = indices[np.argsort(scores)[::-1][:top_n]]
        return [feature_names[idx] for idx in top_indices]

    def recommend_similar_items(self, product_id: str, top_k: int = 10):
        """
        Recommends similar items using exact cosine similarity.
        Returns a list of dicts with product details, similarity score, and explanation.
        """
        if self.tfidf_matrix is None:
            self.load()
            
        if product_id not in self.product_mapping:
            logger.warning(f"Product ID {product_id} not found in training data.")
            return []
            
        query_idx = self.product_mapping[product_id]
        query_vec = self.tfidf_matrix[query_idx]
        
        # Compute exact cosine similarity with all items
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top_k + 1 indices (since the queried item itself will be the most similar)
        top_indices = np.argsort(sim_scores)[::-1][:top_k+1]
        
        feature_names = self.vectorizer.get_feature_names_out()
        recommendations = []
        
        for idx in top_indices:
            # Exclude the queried product
            if idx == query_idx:
                continue
                
            sim_score = sim_scores[idx]
            target_product_id = self.idx_to_product[idx]
            target_vec = self.tfidf_matrix[idx]
            
            # Fetch metadata details
            product_info = self.meta_df[self.meta_df['product_id'] == target_product_id].iloc[0]
            
            explanation_terms = self.get_explanation(query_vec, target_vec, feature_names)
            
            recommendations.append({
                "product_id": target_product_id,
                "title": product_info.get("title", ""),
                "similarity_score": round(float(sim_score), 4),
                "explanation_terms": explanation_terms
            })
            
            if len(recommendations) >= top_k:
                break
                
        return recommendations

    def recommend_for_user(self, user_id: str, reviews_df: pd.DataFrame, top_k: int = 10):
        """
        Recommends items for a user by computing a user profile (average of interacted items' TF-IDF vectors).
        Returns a list of recommended product_ids.
        """
        if self.tfidf_matrix is None:
            self.load()
            
        # Get items user interacted with
        user_items = reviews_df[reviews_df['user_id'] == user_id]['parent_asin'].tolist()
        if not user_items:
            logger.warning(f"User ID {user_id} has no interactions.")
            return []
            
        user_item_indices = [self.product_mapping[pid] for pid in user_items if pid in self.product_mapping]
        if not user_item_indices:
            return []
            
        # Compute user profile vector (mean of their item vectors)
        user_profile = self.tfidf_matrix[user_item_indices].mean(axis=0)
        
        # Compute similarities
        sim_scores = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sim_scores)[::-1]
        
        recommendations = []
        for idx in top_indices:
            target_pid = self.idx_to_product[idx]
            # Exclude items the user has already interacted with
            if target_pid in user_items:
                continue
                
            recommendations.append(target_pid)
            if len(recommendations) >= top_k:
                break
                
        return recommendations

if __name__ == "__main__":
    config = load_config()
    # Load raw metadata directly if you want to test running this module standalone
    meta_df_path = os.path.join(config['data']['paths']['processed_dir'], "meta_processed.parquet")
    meta_df = pd.read_parquet(meta_df_path)
    
    recommender = ContentBasedRecommender(config)
    recommender.fit(meta_df)
    recommender.save()
