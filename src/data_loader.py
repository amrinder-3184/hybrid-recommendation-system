import os
import pandas as pd
from datasets import load_dataset
from src.utils import get_logger

logger = get_logger(__name__)

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = config['data']['paths']['raw_dir']
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the review and metadata datasets from Hugging Face and saves them to raw_dir.
        If already downloaded, loads from local disk.
        """
        review_path = os.path.join(self.raw_dir, "reviews.parquet")
        meta_path = os.path.join(self.raw_dir, "meta.parquet")
        
        if os.path.exists(review_path) and os.path.exists(meta_path):
            logger.info("Loading datasets from local raw directory.")
            reviews_df = pd.read_parquet(review_path)
            meta_df = pd.read_parquet(meta_path)
            return reviews_df, meta_df
            
        logger.info(f"Downloading dataset from Hugging Face: {self.config['data']['dataset_name']}")
        try:
            # Load Reviews
            reviews_dataset = load_dataset(
                self.config['data']['dataset_name'], 
                self.config['data']['dataset_config'], 
                split="full", 
                trust_remote_code=True
            )
            reviews_df = reviews_dataset.to_pandas()
            
            # Load Metadata
            meta_dataset = load_dataset(
                self.config['data']['dataset_name'], 
                self.config['data']['dataset_meta_config'], 
                split="full", 
                trust_remote_code=True
            )
            meta_df = meta_dataset.to_pandas()
            
            logger.info(f"Saving raw datasets to {self.raw_dir}")
            reviews_df.to_parquet(review_path, index=False)
            meta_df.to_parquet(meta_path, index=False)
            
            return reviews_df, meta_df
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
