import pytest
import pandas as pd
from src.preprocessing import DataPreprocessor

@pytest.fixture
def config():
    return {
        'data': {'paths': {'interim_dir': 'data/interim'}},
        'preprocessing': {
            'min_user_reviews': 2,
            'min_item_reviews': 2,
            'normalize_ratings': True
        }
    }

@pytest.fixture
def raw_data():
    reviews_df = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u4'],
        'parent_asin': ['i1', 'i2', 'i1', 'i2', 'i3', 'i1'],
        'rating': [5, 4, 3, 2, 5, 1],
        'timestamp': [1, 2, 3, 4, 5, 6]
    })
    
    meta_df = pd.DataFrame({
        'parent_asin': ['i1', 'i2', 'i3'],
        'title': ['Item 1', 'Item 2', 'Item 3']
    })
    
    return reviews_df, meta_df

def test_preprocessing(config, raw_data, tmp_path):
    # Override interim dir to tmp_path for testing
    config['data']['paths']['interim_dir'] = str(tmp_path)
    
    preprocessor = DataPreprocessor(config)
    reviews_df, meta_df = raw_data
    
    # Add duplicate and missing value to test
    reviews_df.loc[len(reviews_df)] = ['u1', 'i1', 5, 7] # duplicate user-item
    reviews_df.loc[len(reviews_df)] = [None, 'i1', 5, 8] # missing user_id
    
    proc_reviews, proc_meta = preprocessor.process(reviews_df, meta_df)
    
    # 1. Missing value should be removed
    assert proc_reviews['user_id'].isnull().sum() == 0
    
    # 2. Duplicate should be removed (we added 2 bad rows, leaving 6 valid unique interactions)
    # Then filtering applies: min_user_reviews=2, min_item_reviews=2
    # Users: u1(2), u2(2), u3(1), u4(1) => u3 and u4 removed
    # Items left: i1, i2 from u1, u2. 
    # Let's check remaining
    assert set(proc_reviews['user_id']) == {'u1', 'u2'}
    
    # 3. Ratings normalization
    assert 'rating_normalized' in proc_reviews.columns
    assert proc_reviews['rating_normalized'].max() <= 1.0
    assert proc_reviews['rating_normalized'].min() >= 0.0
