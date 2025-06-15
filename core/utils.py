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


def get_tenant_dataset_path(account_name: str) -> str: # Changed tenant_id to account_name
    """
    Constructs and returns the path to the account's specific dataset CSV file.
    It also checks if the file exists.

    Args:
        account_name (str): The identifier for the account/tenant. # Changed tenant_id to account_name

    Returns:
        str: The path to the account's dataset file. # Changed tenant to account

    Raises:
        FileNotFoundError: If the account-specific dataset file does not exist. # Changed tenant to account
    """
    # Construct the path to the account's specific dataset file
    # Assumes account datasets are stored in the 'dataset/' directory named as '{account_name}_dataset.csv'
    # settings.DATASET_PATH currently points to "datasets/combined_dataset.csv"
    # We'll use the directory part of DATASET_PATH or assume "dataset/" if it's not structured as a dir.

    base_dataset_dir = os.path.dirname(settings.DATASET_PATH)
    if not base_dataset_dir: # If DATASET_PATH is just a filename like "combined_dataset.csv"
        base_dataset_dir = "dataset" # Default to "dataset/"

    tenant_dataset_filename = f"{account_name}_dataset.csv" # Changed tenant_id to account_name
    tenant_dataset_file_path = os.path.join(base_dataset_dir, tenant_dataset_filename)

    if not os.path.exists(tenant_dataset_file_path):
        raise FileNotFoundError(
            f"Dataset file not found for account '{account_name}' at {tenant_dataset_file_path}" # Changed tenant_id to account_name
        )

    # For now, this function will focus on local file paths.
    # Azure logic from the old get_combined_dataset might be integrated here or in training.py later if needed.
    # if settings.ENVIRONMENT in ["production", "uat"]:
    #     # This part would need to be adapted to fetch tenant-specific files from Azure
    #     # For example, blob_name might become f"{tenant_id}_dataset.csv"
    #     # return get_dataset_from_azure(
    #     # azure_connection_string, container_name, tenant_dataset_filename
    #     # )
    #     pass # Placeholder for Azure logic

    return tenant_dataset_file_path
