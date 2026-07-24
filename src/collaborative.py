import os
import time
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
from src.utils import get_logger, load_config

logger = get_logger(__name__)

class CollaborativeRecommender:
    def __init__(self, config: dict):
        self.config = config
        self.artifacts_dir = config['data']['paths']['artifacts_dir']
        self.processed_dir = config['data']['paths']['processed_dir']
        self.params = config['collaborative']['lightfm']
        
        self.model = None
        self.user_mapping = None
        self.idx_to_user = None
        self.item_mapping = None
        self.idx_to_item = None
        self.meta_df = None
        
    def fit(self, interactions: sp.spmatrix):
        """Fits the LightFM model on interactions and performs evaluation."""
        start_time = time.time()
        logger.info("Initializing Collaborative Filtering (LightFM)...")
        
        # Load explicit mappings
        user_map_path = os.path.join(self.artifacts_dir, "user_mapping.pkl")
        item_map_path = os.path.join(self.artifacts_dir, "item_mapping.pkl")
        with open(user_map_path, "rb") as f:
            u_map = pickle.load(f)
            self.user_mapping = u_map['user2idx']
            self.idx_to_user = u_map['idx2user']
            
        with open(item_map_path, "rb") as f:
            i_map = pickle.load(f)
            self.item_mapping = i_map['item2idx']
            self.idx_to_item = i_map['idx2product']
            
        logger.info(f"Interaction Matrix Shape: {interactions.shape}")
        
        # Train / Test split for evaluation
        logger.info("Splitting dataset into 80/20 train/test...")
        train_mat, test_mat = random_train_test_split(interactions, test_percentage=0.2)
        
        self.model = LightFM(
            no_components=self.params['no_components'],
            loss=self.params['loss'],
            learning_rate=self.params['learning_rate']
        )
        
        logger.info(f"Training LightFM for {self.params['epochs']} epochs...")
        self.model.fit(train_mat, epochs=self.params['epochs'], num_threads=4)
        
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds.")
        
        # Evaluation
        logger.info("Evaluating Collaborative Model...")
        train_precision = precision_at_k(self.model, train_mat, k=10).mean()
        test_precision = precision_at_k(self.model, test_mat, train_interactions=train_mat, k=10).mean()
        
        train_recall = recall_at_k(self.model, train_mat, k=10).mean()
        test_recall = recall_at_k(self.model, test_mat, train_interactions=train_mat, k=10).mean()
        
        train_auc = auc_score(self.model, train_mat).mean()
        test_auc = auc_score(self.model, test_mat, train_interactions=train_mat).mean()
        
        logger.info("--- Evaluation Metrics ---")
        logger.info(f"Precision@10: Train={train_precision:.4f} | Test={test_precision:.4f}")
        logger.info(f"Recall@10: Train={train_recall:.4f} | Test={test_recall:.4f}")
        logger.info(f"AUC: Train={train_auc:.4f} | Test={test_auc:.4f}")
        
        return self

    def save(self):
        """Saves LightFM model and embeddings."""
        if self.model is None:
            raise ValueError("Model must be fitted before saving.")
            
        logger.info(f"Saving artifacts to {self.artifacts_dir}...")
        model_path = os.path.join(self.artifacts_dir, "lightfm_model.pkl")
        user_emb_path = os.path.join(self.artifacts_dir, "user_embeddings.npy")
        item_emb_path = os.path.join(self.artifacts_dir, "item_embeddings.npy")
        
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            
        np.save(user_emb_path, self.model.user_embeddings)
        np.save(item_emb_path, self.model.item_embeddings)
        
        logger.info("Artifacts saved successfully.")

    def load(self):
        """Loads trained artifacts for inference."""
        logger.info("Loading Collaborative Filtering artifacts...")
        model_path = os.path.join(self.artifacts_dir, "lightfm_model.pkl")
        user_map_path = os.path.join(self.artifacts_dir, "user_mapping.pkl")
        item_map_path = os.path.join(self.artifacts_dir, "item_mapping.pkl")
        meta_path = os.path.join(self.processed_dir, "meta_processed.parquet")
        
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
        with open(user_map_path, "rb") as f:
            u_map = pickle.load(f)
            self.user_mapping = u_map['user2idx']
            self.idx_to_user = u_map['idx2user']
            
        with open(item_map_path, "rb") as f:
            i_map = pickle.load(f)
            self.item_mapping = i_map['item2idx']
            self.idx_to_item = i_map['idx2product']
            
        self.meta_df = pd.read_parquet(meta_path)
        logger.info("Artifacts loaded successfully.")
        return self

    def recommend_for_user(self, user_id: str, top_k: int = 10, exclude_known: bool = True, interactions=None):
        """Generates recommendations for a user."""
        if self.model is None:
            self.load()
            
        if user_id not in self.user_mapping:
            logger.warning(f"User ID {user_id} not found in mapping.")
            return []
            
        user_idx = self.user_mapping[user_id]
        n_items = len(self.item_mapping)
        item_indices = np.arange(n_items)
        
        # Predict scores
        scores = self.model.predict(user_idx, item_indices)
        
        if exclude_known and interactions is not None:
            known_item_indices = interactions[user_idx].indices
            scores[known_item_indices] = -np.inf
            
        top_item_indices = np.argsort(-scores)[:top_k]
        
        recommendations = []
        for idx in top_item_indices:
            target_product_id = self.idx_to_item[idx]
            product_info = self.meta_df[self.meta_df['product_id'] == target_product_id].iloc[0]
            recommendations.append({
                "product_id": target_product_id,
                "title": product_info.get("title", ""),
                "score": float(scores[idx])
            })
            
        return recommendations

    def similar_items(self, product_id: str, top_k: int = 10):
        """Finds similar items based on LightFM item embeddings."""
        if self.model is None:
            self.load()
            
        if product_id not in self.item_mapping:
            return []
            
        item_idx = self.item_mapping[product_id]
        item_embeddings = self.model.item_embeddings
        
        # Compute cosine similarity
        target_emb = item_embeddings[item_idx]
        norms = np.linalg.norm(item_embeddings, axis=1)
        sim_scores = np.dot(item_embeddings, target_emb) / (norms * norms[item_idx] + 1e-9)
        
        top_indices = np.argsort(-sim_scores)[:top_k+1]
        
        recommendations = []
        for idx in top_indices:
            if idx == item_idx:
                continue
                
            target_product_id = self.idx_to_item[idx]
            product_info = self.meta_df[self.meta_df['product_id'] == target_product_id].iloc[0]
            recommendations.append({
                "product_id": target_product_id,
                "title": product_info.get("title", ""),
                "similarity_score": float(sim_scores[idx])
            })
            if len(recommendations) >= top_k:
                break
        return recommendations

    def similar_users(self, user_id: str, top_k: int = 10):
        """Finds similar users based on LightFM user embeddings."""
        if self.model is None:
            self.load()
            
        if user_id not in self.user_mapping:
            return []
            
        user_idx = self.user_mapping[user_id]
        user_embeddings = self.model.user_embeddings
        
        target_emb = user_embeddings[user_idx]
        norms = np.linalg.norm(user_embeddings, axis=1)
        sim_scores = np.dot(user_embeddings, target_emb) / (norms * norms[user_idx] + 1e-9)
        
        top_indices = np.argsort(-sim_scores)[:top_k+1]
        
        similar_users_list = []
        for idx in top_indices:
            if idx == user_idx:
                continue
                
            similar_users_list.append({
                "user_id": self.idx_to_user[idx],
                "similarity_score": float(sim_scores[idx])
            })
            if len(similar_users_list) >= top_k:
                break
        return similar_users_list

if __name__ == "__main__":
    config = load_config()
    matrix_path = os.path.join(config['data']['paths']['artifacts_dir'], "interaction_matrix.npz")
    interactions = sp.load_npz(matrix_path)
    
    recommender = CollaborativeRecommender(config)
    recommender.fit(interactions)
    recommender.save()
