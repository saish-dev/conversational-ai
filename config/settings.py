__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import os
from dotenv import load_dotenv

# Load environment variables from a .env file at the project root.
# This makes it easy to configure the application without hardcoding
# values.
load_dotenv()


class Settings:
    """
    Centralized configuration settings for the application.

    This class reads configuration from environment variables, providing
    default values for a standard development setup.
    """

    # --- Model and Adapter Configuration ---

    # Base path where all models, adapters, and tenant-specific artifacts
    # are stored. Each tenant will have a subdirectory here.
    MODEL_PATH = os.getenv("MODEL_PATH", "app/models/roberta")

    # The name of the foundational Hugging Face transformer model.
    # All adapters are built on top of this model.
    BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "roberta-base")

    # The configuration for the adapter architecture (e.g., "pfeiffer",
    # "houlsby"). This defines the structure and complexity of the
    # adapter layers.
    ADAPTER_CONFIG_STRING = os.getenv("ADAPTER_CONFIG_STRING", "pfeiffer")

    # --- Data and Logging Configuration ---

    # The directory where tenant-specific datasets are stored. The format
    # is expected to be {account_name}_dataset.csv.
    DATASET_DIR = os.getenv("DATASET_DIR", "dataset")

    # Path for storing training logs.
    LOGGING_PATH = os.getenv("LOGGING_PATH", "models/logs")

    # --- Azure Configuration ---

    # The environment mode (e.g., "development", "production").
    # This can be used to enable or disable certain features, like
    # loading data from Azure.
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Connection string for Azure Blob Storage, used for loading datasets
    # in a production environment.
    AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

    class Config:
        # Specifies the .env file to be used by Pydantic's BaseSettings.
        env_file = ".env"


# Create a singleton instance of the settings to be used throughout the
# application.
settings = Settings()
