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


def get_tenant_dataset_path(account_name: str) -> str:
    """
    Constructs and returns the path to an account's dataset file.

    It assumes datasets are stored locally in the directory defined by
    `settings.DATASET_DIR`, with a filename convention of
    `{account_name}_dataset.csv`.

    Args:
        account_name (str): The unique identifier for the account/tenant.

    Returns:
        The full path to the account's dataset file.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    # In a production environment, this function could be extended to
    # fetch data from a cloud source like Azure Blob Storage.
    # if settings.ENVIRONMENT == "production":
    #     # Logic to download from Azure and return a local path
    #     pass

    dataset_filename = f"{account_name}_dataset.csv"
    dataset_path = os.path.join(settings.DATASET_DIR, dataset_filename)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found for account '{account_name}' at "
            f"{dataset_path}"
        )

    return dataset_path
