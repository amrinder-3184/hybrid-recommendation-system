from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies.model_dep import get_inference_service
from api.schemas.responses import RecommendationResponse, SimilarUsersResponse
from api.services.inference_service import InferenceService

router = APIRouter(prefix="", tags=["recommendations"])

@router.get("/recommend/user/{user_id}", response_model=RecommendationResponse)
def get_user_recommendations(user_id: str, top_k: int = Query(10, ge=1, le=50), service: InferenceService = Depends(get_inference_service)):
    try:
        recs = service.recommend_for_user(user_id, top_k)
        return RecommendationResponse(query_id=user_id, query_type="user", recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend/item/{product_id}", response_model=RecommendationResponse)
def get_item_recommendations(product_id: str, top_k: int = Query(10, ge=1, le=50), service: InferenceService = Depends(get_inference_service)):
    try:
        recs = service.recommend_for_item(product_id, top_k)
        return RecommendationResponse(query_id=product_id, query_type="item_cb", recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar/items/{product_id}", response_model=RecommendationResponse)
def get_similar_items_cf(product_id: str, top_k: int = Query(10, ge=1, le=50), service: InferenceService = Depends(get_inference_service)):
    try:
        recs = service.similar_items_cf(product_id, top_k)
        return RecommendationResponse(query_id=product_id, query_type="item_cf", recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar/users/{user_id}", response_model=SimilarUsersResponse)
def get_similar_users(user_id: str, top_k: int = Query(10, ge=1, le=50), service: InferenceService = Depends(get_inference_service)):
    try:
        users = service.similar_users(user_id, top_k)
        return SimilarUsersResponse(user_id=user_id, similar_users=users)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/popular", response_model=RecommendationResponse)
def get_popular(top_k: int = Query(10, ge=1, le=50), service: InferenceService = Depends(get_inference_service)):
    try:
        recs = service.popular_items(top_k)
        return RecommendationResponse(query_id="popular", query_type="global", recommendations=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
