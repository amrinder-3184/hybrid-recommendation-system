from fastapi import APIRouter, Depends
from api.dependencies.model_dep import get_inference_service
from api.services.inference_service import InferenceService
from api.schemas.responses import HealthResponse

router = APIRouter(tags=["system"])

@router.get("/health", response_model=HealthResponse)
def health_check():
    from api.services.inference_service import inference_service
    return HealthResponse(status="ok", models_loaded=inference_service.initialized)

@router.post("/reload-models")
def reload_models(service: InferenceService = Depends(get_inference_service)):
    service.reload()
    return {"message": "Models reloaded successfully"}
