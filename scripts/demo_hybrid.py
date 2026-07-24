import argparse
import os

import pandas as pd

from src.hybrid import HybridRecommender
from src.utils import load_config


def demo_cli():
    print("Initializing Hybrid Recommendation Demo...")
    config = load_config()
    
    reviews_path = os.path.join(config['data']['paths']['processed_dir'], "reviews_processed.parquet")
    try:
        reviews_df = pd.read_parquet(reviews_path)
    except Exception:
        print("Could not load reviews_processed.parquet. Please run Phase 1.")
        return

    hybrid = HybridRecommender(config)
    try:
        hybrid.load()
    except Exception:
        print("Failed to load artifacts. Please train CB, CF, and Hybrid models first.")
        return

    parser = argparse.ArgumentParser(description="Hybrid Recommendation CLI Demo")
    parser.add_argument("--user_id", type=str, help="User ID to query", nargs='?')
    args = parser.parse_args()

    user_id = args.user_id
    if not user_id:
        user_id = input("\nEnter a User ID (or random string for cold-start): ").strip()

    print(f"\nFetching Hybrid recommendations for user: {user_id}...\n")
    print(f"Optimal Alpha (CF Weight): {hybrid.best_alpha}\n")
    
    recs = hybrid.recommend_for_user(user_id, reviews_df, top_k=5)
    
    if not recs:
        print("No recommendations returned.")
        return

    print("--- Top 5 Recommended Products (Hybrid) ---")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['title']} (ID: {r['product_id']})")
        if r.get('source') == 'Fallback (Popular)':
            print(f"   Source: {r['source']} | Score: {r['score']}\n")
        else:
            print(f"   Final Hybrid Score: {r['hybrid_score']:.4f}")
            print(f"   [ CF Contribution: {r['cf_contribution']:.4f} | CB Contribution: {r['cb_contribution']:.4f} ]\n")

if __name__ == "__main__":
    demo_cli()
