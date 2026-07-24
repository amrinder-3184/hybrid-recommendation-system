import os
import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.collaborative import CollaborativeRecommender
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender


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
            'tfidf': {'max_features': 100, 'min_df': 1, 'max_df': 1.0, 'ngram_range': [1, 1], 'stop_words': 'english'}
        },
        'collaborative': {
            'lightfm': {'no_components': 2, 'loss': 'warp', 'learning_rate': 0.05, 'epochs': 1}
        },
        'hybrid': {
            'validation_subset_size': 2,
            'alpha_grid': [0.5],
            'primary_metric': 'NDCG@10'
        }
    }

@pytest.fixture
def setup_models(config):
    # Setup data
    meta_df = pd.DataFrame({
        'product_id': ['p1', 'p2', 'p3'],
        'title': ['T1', 'T2', 'T3'],
        'combined_text': ['a b c', 'd e f', 'a b f']
    })
    meta_df.to_parquet(os.path.join(config['data']['paths']['processed_dir'], "meta_processed.parquet"), index=False)
    
    reviews_df = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u2'],
        'parent_asin': ['p1', 'p2', 'p1', 'p2', 'p3'],
        'rating': [5, 4, 3, 2, 1],
        'rating_normalized': [1, 0.8, 0.6, 0.4, 0.2]
    })
    reviews_df.to_parquet(os.path.join(config['data']['paths']['processed_dir'], "reviews_processed.parquet"), index=False)
    
    # Train CB
    cb = ContentBasedRecommender(config)
    cb.fit(meta_df)
    cb.save()
    
    # Train CF
    interactions = sp.csr_matrix(np.array([[1.0, 1.0, 0], [1.0, 1.0, 1.0]]))
    cf = CollaborativeRecommender(config)
    
    # mock mapping so CF matches our users
    user2idx = {'u1': 0, 'u2': 1}
    idx2user = {0: 'u1', 1: 'u2'}
    item2idx = {'p1': 0, 'p2': 1, 'p3': 2}
    idx2item = {0: 'p1', 1: 'p2', 2: 'p3'}
    
    art_dir = config['data']['paths']['artifacts_dir']
    with open(os.path.join(art_dir, "user_mapping.pkl"), "wb") as f:
        pickle.dump({'user2idx': user2idx, 'idx2user': idx2user}, f)
        
    with open(os.path.join(art_dir, "item_mapping.pkl"), "wb") as f:
        pickle.dump({'item2idx': item2idx, 'idx2product': idx2item}, f)
        
    cf.fit(interactions)
    cf.save()
    
    return reviews_df

def test_hybrid_fit_and_inference(config, setup_models):
    reviews_df = setup_models
    hybrid = HybridRecommender(config)
    hybrid.fit(reviews_df)
    hybrid.save()
    
    # Check artifacts
    assert os.path.exists(os.path.join(config['data']['paths']['artifacts_dir'], "hybrid_params.json"))
    
    # Inference known user
    hybrid.load()
    recs = hybrid.recommend_for_user('u1', reviews_df, top_k=2)
    assert len(recs) == 1 # u1 interacted with p1, p2. Only p3 left
    assert recs[0]['product_id'] == 'p3'
    assert 'cf_contribution' in recs[0]
    assert 'cb_contribution' in recs[0]
    
def test_hybrid_cold_user(config, setup_models):
    reviews_df = setup_models
    hybrid = HybridRecommender(config)
    hybrid.fit(reviews_df) # to populate popular_items
    hybrid.load()
    
    recs = hybrid.recommend_for_user('unknown_user', reviews_df, top_k=2)
    assert len(recs) > 0
    assert recs[0]['source'] == 'Fallback (Popular)'
