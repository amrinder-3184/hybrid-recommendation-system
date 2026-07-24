from src.utils import load_config, get_logger
from src.data_loader import DataLoader
from src.validation import DataValidator
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

logger = get_logger(__name__)

class PipelineRunner:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.loader = DataLoader(self.config)
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(self.config)
        self.engineer = FeatureEngineer(self.config)
        
    def run(self):
        logger.info("--- Starting Phase 1 Pipeline ---")
        
        # 1. Load Data
        reviews_df, meta_df = self.loader.load_data()
        
        # 2. Validate Data
        self.validator.validate(reviews_df, meta_df)
        
        # 3. Preprocess Data
        reviews_df, meta_df = self.preprocessor.process(reviews_df, meta_df)
        
        # 4. Feature Engineering
        reviews_df, meta_df = self.engineer.process(reviews_df, meta_df)
        
        logger.info("--- Phase 1 Pipeline Completed Successfully ---")

if __name__ == "__main__":
    pipeline = PipelineRunner()
    pipeline.run()
