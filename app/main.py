__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

from fastapi import FastAPI
from app.api.endpoints import router as api_router
from core.inference import IntentClassifier


app = FastAPI(
    title="NLP Service",
    version="1.0.0",
    description="Intent classification using a shared RoBERTa model",
)


# Initialize the IntentClassifier once at startup
intent_classifier = IntentClassifier()


# Include your API router
app.include_router(api_router, prefix="/api")
