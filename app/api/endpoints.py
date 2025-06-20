__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

from app.api.schemas import IntentRequest, IntentResponse
from core.training import train_model
from fastapi import APIRouter, BackgroundTasks, HTTPException

# Create an API router to group related endpoints.
router = APIRouter()


@router.post("/predict_intent/", response_model=IntentResponse)
async def predict_intent(request: IntentRequest):
    """
    Predicts the intent of a given text for a specific account.

    This endpoint uses the singleton `intent_classifier` instance to
    perform inference. It handles errors gracefully, such as when a model
    for the requested account is not found.

    Args:
        request: An `IntentRequest` object containing the user's text and
                 the account name.

    Returns:
        An `IntentResponse` object with the predicted intent.
    """
    # Import the classifier instance from the main application module.
    # This avoids circular dependencies and ensures the singleton pattern.
    from app.main import intent_classifier

    try:
        predicted_intent = intent_classifier.predict(
            request.text, request.account_name
        )
        return IntentResponse(intent=predicted_intent)
    except FileNotFoundError as e:
        # If the model/adapter for the account doesn't exist, return a
        # 404 error.
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # For any other unexpected errors, return a generic 500 error.
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )


@router.post("/train_adapter/{account_name}")
async def train_tenant_adapter(
    account_name: str, background_tasks: BackgroundTasks
):
    """
    Triggers a background task to train a new adapter for an account.

    This endpoint immediately returns a confirmation message while the
    `train_model` function runs in the background. This is ideal for
    long-running processes like model training, as it doesn't block the
    API server.

    Args:
        account_name: The unique identifier for the account.
        background_tasks: FastAPI's dependency for running background jobs.
    """
    # Schedule the `train_model` function to run after the response is sent.
    background_tasks.add_task(train_model, account_name=account_name)
    return {
        "message": f"Adapter training for {account_name} has started."
    }
