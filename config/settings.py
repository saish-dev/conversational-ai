__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class Settings:
    # Model path where the unified RoBERTa model will be saved
    MODEL_PATH = os.getenv("MODEL_PATH", "app/models/roberta")

    # Combined dataset path
    DATASET_PATH = os.getenv("DATASET_PATH", "datasets/combined_dataset.csv")

    # Log path for training logs
    LOGGING_PATH = os.getenv("LOGGING_PATH", "models/logs")
    LABEL_ENCODER_PATH = os.getenv(
        "LABEL_ENCODER_PATH", "models/roberta/label_encoder.pkl"
    )
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

    class Config:
        env_file = ".env"


settings = Settings()
