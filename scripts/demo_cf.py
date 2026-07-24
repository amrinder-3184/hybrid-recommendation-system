import argparse
import scipy.sparse as sp
import os
from src.collaborative import CollaborativeRecommender
from src.utils import load_config

def demo_cli():
    print("Initializing Collaborative Filtering Demo...")
    config = load_config()
    
    matrix_path = os.path.join(config['data']['paths']['artifacts_dir'], "interaction_matrix.npz")
    try:
        interactions = sp.load_npz(matrix_path)
    except:
        interactions = None
        print("Warning: No interaction matrix found. Predictions will not exclude known items.")
        
    recommender = CollaborativeRecommender(config)
    try:
        recommender.load()
    except Exception as e:
        print("Failed to load artifacts. Please train the Collaborative model first.")
        return

    parser = argparse.ArgumentParser(description="Collaborative Filtering CLI Demo")
    parser.add_argument("--user_id", type=str, help="User ID to query", nargs='?')
    args = parser.parse_args()

    user_id = args.user_id
    if not user_id:
        user_id = input("\nEnter a User ID to get recommendations: ").strip()

    print(f"\nFetching Collaborative recommendations for user: {user_id}...\n")
    recs = recommender.recommend_for_user(user_id, top_k=5, exclude_known=True, interactions=interactions)
    
    if not recs:
        print(f"No recommendations found. Make sure user '{user_id}' exists in the training set.")
        return

    print("--- Top 5 Recommended Products (CF) ---")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['title']} (ID: {r['product_id']})")
        print(f"   Model Score: {r['score']}\n")

if __name__ == "__main__":
    demo_cli()
