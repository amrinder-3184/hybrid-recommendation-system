from api.services.inference_service import InferenceService, inference_service


def get_inference_service() -> InferenceService:
    if not inference_service.initialized:
        inference_service.initialize()
    return inference_service
