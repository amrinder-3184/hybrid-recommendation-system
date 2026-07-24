import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=5)
        return res.json(), res.headers.get("X-Process-Time")
    except Exception as e:
        return {"status": "error", "details": str(e)}, None

def get_user_recommendations(user_id, top_k=10):
    res = requests.get(f"{API_URL}/recommend/user/{user_id}?top_k={top_k}", timeout=10)
    res.raise_for_status()
    return res.json(), res.headers.get("X-Process-Time")

def get_item_cb_similar(product_id, top_k=10):
    res = requests.get(f"{API_URL}/recommend/item/{product_id}?top_k={top_k}", timeout=10)
    res.raise_for_status()
    return res.json(), res.headers.get("X-Process-Time")

def get_item_cf_similar(product_id, top_k=10):
    res = requests.get(f"{API_URL}/similar/items/{product_id}?top_k={top_k}", timeout=10)
    res.raise_for_status()
    return res.json(), res.headers.get("X-Process-Time")

def get_popular(top_k=10):
    res = requests.get(f"{API_URL}/popular?top_k={top_k}", timeout=10)
    res.raise_for_status()
    return res.json(), res.headers.get("X-Process-Time")
