import os
import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.collaborative import CollaborativeRecommender


@pytest.fixture
def config(tmp_path):
    return {
        'data': {
            'paths': {
                'processed_dir': str(tmp_path),
                'artifacts_dir': str(tmp_path)
            }
        },
        'collaborative': {
            'lightfm': {
                'no_components': 2,
                'loss': 'warp',
                'learning_rate': 0.05,
                'epochs': 1
            }
        }
    }

@pytest.fixture
def setup_data(config):
    # Mock Mappings
    user2idx = {'u1': 0, 'u2': 1}
    idx2user = {0: 'u1', 1: 'u2'}
    item2idx = {'p1': 0, 'p2': 1, 'p3': 2}
    idx2item = {0: 'p1', 1: 'p2', 2: 'p3'}
    
    art_dir = config['data']['paths']['artifacts_dir']
    with open(os.path.join(art_dir, "user_mapping.pkl"), "wb") as f:
        pickle.dump({'user2idx': user2idx, 'idx2user': idx2user}, f)
        
    with open(os.path.join(art_dir, "item_mapping.pkl"), "wb") as f:
        pickle.dump({'item2idx': item2idx, 'idx2product': idx2item}, f)
        
    # Mock Meta
    meta_df = pd.DataFrame({
        'product_id': ['p1', 'p2', 'p3'],
        'title': ['T1', 'T2', 'T3']
    })
    meta_df.to_parquet(os.path.join(config['data']['paths']['processed_dir'], "meta_processed.parquet"))
    
    # Mock Interactions
    interactions = sp.csr_matrix(np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0]
    ]))
    
    return interactions

def test_collaborative_training(config, setup_data):
    recommender = CollaborativeRecommender(config)
    recommender.fit(setup_data)
    recommender.save()
    
    art_dir = config['data']['paths']['artifacts_dir']
    assert os.path.exists(os.path.join(art_dir, "lightfm_model.pkl"))
    assert os.path.exists(os.path.join(art_dir, "user_embeddings.npy"))
    assert os.path.exists(os.path.join(art_dir, "item_embeddings.npy"))

def test_collaborative_inference(config, setup_data):
    recommender = CollaborativeRecommender(config)
    recommender.fit(setup_data)
    recommender.save()
    
    recommender.load()
    recs = recommender.recommend_for_user('u1', top_k=2, interactions=setup_data)
    
    assert len(recs) > 0
    # u1 interacted with p1, p2, so p3 should be recommended if exclude_known is true
    rec_ids = [r['product_id'] for r in recs]
    assert 'p3' in rec_ids

def test_similar_users_and_items(config, setup_data):
    recommender = CollaborativeRecommender(config)
    recommender.fit(setup_data)
    recommender.load()
    
    sim_items = recommender.similar_items('p1')
    assert len(sim_items) > 0
    assert sim_items[0]['product_id'] != 'p1'
    
    sim_users = recommender.similar_users('u1')
    assert len(sim_users) > 0
    assert sim_users[0]['user_id'] != 'u1'
