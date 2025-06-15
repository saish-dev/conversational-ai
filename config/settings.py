__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class Settings:
    # Model path where tenant-specific models (adapters) will be saved under subdirectories
    MODEL_PATH = os.getenv("MODEL_PATH", "app/models/roberta")

    # Name of the base transformer model
    BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "roberta-base")

    # Default adapter configuration string
    ADAPTER_CONFIG_STRING = os.getenv("ADAPTER_CONFIG_STRING", "pfeiffer")

    # Original combined dataset path. Now primarily serves as a base path for locating the 'dataset' directory
    # where tenant-specific datasets (e.g., 'dataset/tenantA_dataset.csv') are stored.
    # The combined file itself is not directly used by training/inference after tenant-specific split.
    DATASET_PATH = os.getenv("DATASET_PATH", "datasets/combined_dataset.csv")

    # Log path for training logs (base directory)
    LOGGING_PATH = os.getenv("LOGGING_PATH", "models/logs")

    # Path for the old global label encoder (potentially obsolete, as encoders are now per-tenant)
    LABEL_ENCODER_PATH = os.getenv(
        "LABEL_ENCODER_PATH", "models/roberta/label_encoder.pkl"
    )
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

    class Config:
        env_file = ".env"


settings = Settings()
