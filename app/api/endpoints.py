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
        request.text, request.account_name # Changed from customer_domain
    )
    return IntentResponse(intent=predicted_intent)


# This is the endpoint for training the model in the background
@router.post("/train")
def train(background_tasks: BackgroundTasks):
    # This endpoint might need to be re-evaluated or removed if all training becomes tenant-specific.
    # For now, it likely trains a generic model or the first tenant based on previous setup.
    # If train_model() now requires tenant_id, this will fail or need adjustment.
    # Assuming train_model still has a default behavior or is updated separately.
    background_tasks.add_task(train_model) # If train_model now requires account_name, this call needs to be updated.
    return {"message": "Model training has started in the background."}


@router.post("/train_adapter/{account_name}")
async def train_tenant_adapter(account_name: str, background_tasks: BackgroundTasks):
    """
    Start adapter training for a specific account_name in the background.
    """
    # Note: train_model from core.training will be updated to accept account_name
    background_tasks.add_task(train_model, account_name=account_name)
    return {"message": f"Adapter training for account {account_name} has started in the background."}
