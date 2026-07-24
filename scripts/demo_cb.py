import argparse

from src.content_based import ContentBasedRecommender
from src.utils import load_config


def demo_cli():
    print("Initializing Content-Based Recommender Demo...")
    config = load_config()
    recommender = ContentBasedRecommender(config)
    try:
        recommender.load()
    except Exception:
        print("Failed to load artifacts. Please run Phase 1 and Phase 2 training first.")
        return

    parser = argparse.ArgumentParser(description="Content-Based Recommendation CLI Demo")
    parser.add_argument("--product_id", type=str, help="Product ID to query", nargs='?')
    args = parser.parse_args()

    product_id = args.product_id
    if not product_id:
        product_id = input("\nEnter a Product ID to get similar recommendations: ").strip()

    print(f"\nFetching recommendations for: {product_id}...\n")
    recs = recommender.recommend_similar_items(product_id, top_k=5)
    
    if not recs:
        print(f"No recommendations found. Make sure '{product_id}' exists in the training set.")
        return

    print("--- Top 5 Similar Products ---")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['title']} (ID: {r['product_id']})")
        print(f"   Similarity Score: {r['similarity_score']}")
        print(f"   Why? Overlapping terms: {r['explanation_terms']}\n")

if __name__ == "__main__":
    demo_cli()
