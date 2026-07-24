from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendationItem(BaseModel):
    product_id: str
    title: str
    score: float
    cf_contribution: Optional[float] = None
    cb_contribution: Optional[float] = None
    explanation_terms: Optional[List[str]] = None
    source: str

class RecommendationResponse(BaseModel):
    query_id: str
    query_type: str
    recommendations: List[RecommendationItem]

class SimilarUserItem(BaseModel):
    user_id: str
    similarity_score: float

class SimilarUsersResponse(BaseModel):
    user_id: str
    similar_users: List[SimilarUserItem]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
