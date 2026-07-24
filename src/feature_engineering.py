import json
import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.processed_dir = config['data']['paths']['processed_dir']
        self.artifacts_dir = config['data']['paths']['artifacts_dir']
        self.combined_text_col = config['features']['combined_text_col']
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def _safe_join(self, x):
        """Safely joins list/array or returns string."""
        if isinstance(x, (list, np.ndarray)):
            return " ".join([str(i) for i in x])
        return str(x) if pd.notnull(x) else ""

    def process(self, reviews_df: pd.DataFrame, meta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Starting feature engineering...")
        
        # Handle textual features
        text_cols = ['title', 'brand', 'categories', 'description']
        for col in text_cols:
            if col not in meta_df.columns:
                meta_df[col] = ""
            else:
                meta_df[col] = meta_df[col].apply(self._safe_join)
                
        logger.info("Combining text features...")
        meta_df[self.combined_text_col] = meta_df.apply(
            lambda row: " ".join([row[col] for col in text_cols if row[col]]), 
            axis=1
        )
        
        # 1. Explicit ID Mappings
        logger.info("Creating explicit mappings for users and items...")
        unique_users = reviews_df['user_id'].unique()
        unique_items = reviews_df['parent_asin'].unique()
        
        user2idx = {u: i for i, u in enumerate(unique_users)}
        idx2user = {i: u for i, u in enumerate(unique_users)}
        
        item2idx = {item: i for i, item in enumerate(unique_items)}
        idx2item = {i: item for i, item in enumerate(unique_items)}
        
        with open(os.path.join(self.artifacts_dir, "user_mapping.pkl"), "wb") as f:
            pickle.dump({'user2idx': user2idx, 'idx2user': idx2user}, f)
            
        with open(os.path.join(self.artifacts_dir, "item_mapping.pkl"), "wb") as f:
            pickle.dump({'item2idx': item2idx, 'idx2item': idx2item}, f)
            
        # Map IDs to contiguous encoded IDs
        reviews_df['user_id_encoded'] = reviews_df['user_id'].map(user2idx)
        reviews_df['item_id_encoded'] = reviews_df['parent_asin'].map(item2idx)
        
        meta_df = meta_df[meta_df['parent_asin'].isin(item2idx.keys())].copy()
        meta_df['item_id_encoded'] = meta_df['parent_asin'].map(item2idx)
        
        # 2. Sparse CSR Interaction Matrix
        logger.info("Creating CSR sparse interaction matrix...")
        data_col = 'rating_normalized' if 'rating_normalized' in reviews_df.columns else 'rating'
        interaction_matrix = sp.csr_matrix(
            (reviews_df[data_col], (reviews_df['user_id_encoded'], reviews_df['item_id_encoded'])),
            shape=(len(unique_users), len(unique_items))
        )
        sp.save_npz(os.path.join(self.artifacts_dir, "interaction_matrix.npz"), interaction_matrix)
        
        # 3. Expanded Dataset Statistics
        logger.info("Calculating extended dataset statistics...")
        num_users = len(unique_users)
        num_items = len(unique_items)
        num_interactions = len(reviews_df)
        sparsity = 1.0 - (num_interactions / (num_users * num_items)) if num_users * num_items > 0 else 0
        
        stats = {
            "total_interactions": int(num_interactions),
            "unique_users": int(num_users),
            "unique_products": int(num_items),
            "average_interactions_per_user": float(num_interactions / num_users) if num_users > 0 else 0,
            "average_interactions_per_item": float(num_interactions / num_items) if num_items > 0 else 0,
            "average_rating": float(reviews_df['rating'].mean()),
            "dataset_sparsity": float(sparsity),
            "dataset_density": float(1.0 - sparsity)
        }
        
        with open(os.path.join(self.artifacts_dir, "dataset_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)
            
        # 4. Save metadata correctly
        meta_df = meta_df.rename(columns={'parent_asin': 'product_id'})
        
        processed_meta_cols = ['product_id', 'item_id_encoded', 'title', 'brand', 'categories', 'description', self.combined_text_col]
        processed_meta_cols = [c for c in processed_meta_cols if c in meta_df.columns]
        
        reviews_path = os.path.join(self.processed_dir, "reviews_processed.parquet")
        meta_path = os.path.join(self.processed_dir, "meta_processed.parquet")
        
        reviews_df.to_parquet(reviews_path, index=False)
        meta_df[processed_meta_cols].to_parquet(meta_path, index=False)
        
        logger.info(f"Saved processed datasets to {self.processed_dir}")
        return reviews_df, meta_df
