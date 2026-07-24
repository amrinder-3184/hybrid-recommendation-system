import os
import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.interim_dir = config['data']['paths']['interim_dir']
        self.min_user_reviews = config['preprocessing']['min_user_reviews']
        self.min_item_reviews = config['preprocessing']['min_item_reviews']
        self.normalize_ratings = config['preprocessing']['normalize_ratings']
        os.makedirs(self.interim_dir, exist_ok=True)
        
    def process(self, reviews_df: pd.DataFrame, meta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Starting preprocessing pipeline...")
        
        # 1. Missing Values Handling
        reviews_df = reviews_df.dropna(subset=['user_id', 'parent_asin', 'rating'])
        meta_df = meta_df.dropna(subset=['parent_asin', 'title'])
        
        # 2. Duplicate Removal
        initial_reviews = len(reviews_df)
        reviews_df = reviews_df.drop_duplicates(subset=['user_id', 'parent_asin'])
        logger.info(f"Removed {initial_reviews - len(reviews_df)} duplicate reviews.")
        
        # 3. Filtering Users and Items (k-core filtering)
        logger.info(f"Filtering users with < {self.min_user_reviews} and items with < {self.min_item_reviews} reviews.")
        
        while True:
            start_len = len(reviews_df)
            
            # Filter users
            user_counts = reviews_df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_reviews].index
            reviews_df = reviews_df[reviews_df['user_id'].isin(valid_users)]
            
            # Filter items
            item_counts = reviews_df['parent_asin'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_reviews].index
            reviews_df = reviews_df[reviews_df['parent_asin'].isin(valid_items)]
            
            if len(reviews_df) == start_len:
                break
                
        logger.info(f"After filtering, Reviews Shape: {reviews_df.shape}")
        
        # Filter metadata to only include items that survived the filtering
        meta_df = meta_df[meta_df['parent_asin'].isin(reviews_df['parent_asin'].unique())]
        
        # 4. Rating Normalization (Min-Max Scaling to 0-1)
        if self.normalize_ratings:
            min_rating = reviews_df['rating'].min()
            max_rating = reviews_df['rating'].max()
            if max_rating > min_rating:
                reviews_df['rating_normalized'] = (reviews_df['rating'] - min_rating) / (max_rating - min_rating)
            else:
                reviews_df['rating_normalized'] = 1.0
            logger.info("Ratings normalized.")
            
        # Save intermediate results
        reviews_path = os.path.join(self.interim_dir, "reviews_interim.parquet")
        meta_path = os.path.join(self.interim_dir, "meta_interim.parquet")
        
        reviews_df.to_parquet(reviews_path, index=False)
        meta_df.to_parquet(meta_path, index=False)
        
        logger.info(f"Saved interim datasets to {self.interim_dir}")
        return reviews_df, meta_df
