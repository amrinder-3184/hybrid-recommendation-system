from api.services.inference_service import inference_service, InferenceService

def get_inference_service() -> InferenceService:
    if not inference_service.initialized:
        inference_service.initialize()
    return inference_service
