
from pydantic import BaseModel


class RecommendationItem(BaseModel):
    product_id: str
    title: str
    score: float
    cf_contribution: float | None = None
    cb_contribution: float | None = None
    explanation_terms: list[str] | None = None
    source: str

class RecommendationResponse(BaseModel):
    query_id: str
    query_type: str
    recommendations: list[RecommendationItem]

class SimilarUserItem(BaseModel):
    user_id: str
    similarity_score: float

class SimilarUsersResponse(BaseModel):
    user_id: str
    similar_users: list[SimilarUserItem]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
