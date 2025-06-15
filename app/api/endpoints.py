__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"


from fastapi import APIRouter, BackgroundTasks
from app.api.schemas import IntentRequest, IntentResponse
from core.training import train_model

router = APIRouter()
# Import the preloaded model


@router.post("/predict_intent/")
async def predict_intent(request: IntentRequest):
    from app.main import intent_classifier

    predicted_intent = intent_classifier.predict(
        request.text, request.customer_domain
    )
    return IntentResponse(intent=predicted_intent)


# This is the endpoint for training the model in the background
@router.post("/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model)
    return {"message": "Model training has started in the background."}
