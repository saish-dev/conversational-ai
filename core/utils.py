__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
from config.settings import settings


def load_local_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from a local CSV file.

    Args:
        path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def get_dataset_from_azure(
    blob_connection_string: str, container_name: str, blob_name: str
) -> pd.DataFrame:
    """
    Retrieve a dataset from an Azure Blob storage.

    Args:
        blob_connection_string (str): Connection string to the Azure Blob
        storage.
        container_name (str): Name of the container in the Blob storage.
        blob_name (str): Name of the blob to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    # Initialize the BlobServiceClient using the provided connection string
    blob_service_client = BlobServiceClient.from_connection_string(
        blob_connection_string
    )

    # Get a BlobClient for the specific container and blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name
    )

    # Download the blob's content to memory and read it as a CSV into a df
    blob_data = blob_client.download_blob()
    return pd.read_csv(blob_data.content_as_text())


def get_combined_dataset():
    """
    Retrieve the combined dataset based on the environment.

    In production/UAT environments, the dataset is retrieved from an Azure Blob
    storage. In other environments, the dataset is loaded from a local file
    path.

    Returns:
        pd.DataFrame: DataFrame containing the combined dataset.
    """
    # Check if the dataset is from remote or local path
    azure_connection_string = settings.AZURE_CONNECTION_STRING
    container_name = settings.AZURE_CONTAINER_NAME
    dataset_path = settings.DATASET_PATH
    if settings.ENVIRONMENT in ["production", "uat"]:
        # Retrieve the dataset from the Azure Blob storage
        return get_dataset_from_azure(
            azure_connection_string, container_name, dataset_path
        )

    else:
        # Load the dataset from a local file path
        return load_local_dataset(dataset_path)
