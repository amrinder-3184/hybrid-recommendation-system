import os
import pytest
import pandas as pd
import numpy as np
import scipy.sparse as sp
from src.content_based import ContentBasedRecommender

@pytest.fixture
def config(tmp_path):
    return {
        'data': {
            'paths': {
                'processed_dir': str(tmp_path),
                'artifacts_dir': str(tmp_path)
            }
        },
        'features': {
            'combined_text_col': 'combined_text'
        },
        'content_based': {
            'tfidf': {
                'max_features': 100,
                'min_df': 1,
                'max_df': 1.0,
                'ngram_range': [1, 1],
                'stop_words': 'english'
            }
        }
    }

@pytest.fixture
def setup_data(config):
    # Create dummy metadata
    meta_df = pd.DataFrame({
        'product_id': ['p1', 'p2', 'p3', 'p4'],
        'title': ['Game A', 'Game B', 'Game C', 'Movie D'],
        'combined_text': [
            'action adventure shooter',
            'action adventure puzzle',
            'sports racing',
            'action adventure movie'
        ]
    })
    
    meta_path = os.path.join(config['data']['paths']['processed_dir'], "meta_processed.parquet")
    meta_df.to_parquet(meta_path, index=False)
    
    return meta_df

def test_content_based_training(config, setup_data):
    recommender = ContentBasedRecommender(config)
    
    # Test fit and save
    recommender.fit(setup_data)
    recommender.save()
    
    # Assert artifacts were created
    assert os.path.exists(os.path.join(config['data']['paths']['artifacts_dir'], "tfidf_vectorizer.pkl"))
    assert os.path.exists(os.path.join(config['data']['paths']['artifacts_dir'], "tfidf_matrix.npz"))
    assert os.path.exists(os.path.join(config['data']['paths']['artifacts_dir'], "cb_product_mapping.pkl"))
    
    # Assert matrix shape
    assert recommender.tfidf_matrix.shape[0] == 4
    assert recommender.tfidf_matrix.shape[1] > 0

def test_recommend_similar_items(config, setup_data):
    recommender = ContentBasedRecommender(config)
    recommender.fit(setup_data)
    recommender.save() 
    
    # Reload for inference to test load()
    inf_recommender = ContentBasedRecommender(config)
    inf_recommender.load()
    
    # Test inference
    recs = inf_recommender.recommend_similar_items('p1', top_k=2)
    
    # p1 should recommend p2 and p4 (most similar based on text)
    assert len(recs) == 2
    
    # Check that queried item is excluded
    product_ids = [r['product_id'] for r in recs]
    assert 'p1' not in product_ids
    
    # Check explanation logic is populated
    assert 'explanation_terms' in recs[0]
    assert isinstance(recs[0]['explanation_terms'], list)
    
    # Check similarity score is float
    assert isinstance(recs[0]['similarity_score'], float)

def test_recommend_unknown_item(config, setup_data):
    recommender = ContentBasedRecommender(config)
    recommender.fit(setup_data)
    
    recs = recommender.recommend_similar_items('unknown_product', top_k=2)
    assert len(recs) == 0
