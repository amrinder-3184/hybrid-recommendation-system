import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies.model_dep import get_inference_service

class MockInferenceService:
    initialized = True
    def recommend_for_user(self, user_id, top_k=10):
        return [{"product_id": "p1", "title": "Game 1", "score": 0.9, "source": "Hybrid"}]
        
    def recommend_for_item(self, product_id, top_k=10):
        return [{"product_id": "p2", "title": "Game 2", "score": 0.8, "source": "Content-Based"}]

    def similar_items_cf(self, product_id, top_k=10):
        return [{"product_id": "p3", "title": "Game 3", "score": 0.7, "source": "Collaborative"}]
        
    def similar_users(self, user_id, top_k=10):
        return [{"user_id": "u2", "similarity_score": 0.8}]
        
    def popular_items(self, top_k=10):
        return [{"product_id": "p4", "title": "Popular Game", "score": 1.0, "source": "Fallback (Popular)"}]

def override_get_inference_service():
    return MockInferenceService()

app.dependency_overrides[get_inference_service] = override_get_inference_service
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_recommend_user():
    response = client.get("/recommend/user/u1?top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["query_id"] == "u1"
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["product_id"] == "p1"

def test_recommend_item():
    response = client.get("/recommend/item/p1")
    assert response.status_code == 200
    
def test_similar_items():
    response = client.get("/similar/items/p1")
    assert response.status_code == 200
    
def test_similar_users():
    response = client.get("/similar/users/u1")
    assert response.status_code == 200
    
def test_popular():
    response = client.get("/popular")
    assert response.status_code == 200
