import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

class DataValidator:
    def __init__(self):
        self.required_review_cols = {'user_id', 'parent_asin', 'rating', 'timestamp'}
        self.required_meta_cols = {'parent_asin', 'title'}
        
    def validate(self, reviews_df: pd.DataFrame, meta_df: pd.DataFrame) -> bool:
        """
        Validates the schema of the incoming raw datasets.
        """
        logger.info("Validating raw datasets...")
        
        review_cols = set(reviews_df.columns)
        meta_cols = set(meta_df.columns)
        
        # Check if required columns are present. In some datasets user_id is 'user_id', item is 'parent_asin' or 'item_id'.
        # We will flexibly check, but enforce our core requirements.
        missing_review_cols = self.required_review_cols - review_cols
        missing_meta_cols = self.required_meta_cols - meta_cols
        
        if missing_review_cols:
            logger.error(f"Missing required review columns: {missing_review_cols}")
            raise ValueError(f"Missing required review columns: {missing_review_cols}")
            
        if missing_meta_cols:
            logger.error(f"Missing required metadata columns: {missing_meta_cols}")
            raise ValueError(f"Missing required metadata columns: {missing_meta_cols}")
            
        logger.info("Validation successful. All required columns are present.")
        logger.info(f"Reviews Shape: {reviews_df.shape}")
        logger.info(f"Metadata Shape: {meta_df.shape}")
        
        return True
