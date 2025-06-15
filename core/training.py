__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import torch
import joblib
import pandas as pd
import os # Added for os.makedirs
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.adapters import RobertaModelWithHeads # Added for adapter support
from config.settings import settings
from core.utils import get_tenant_dataset_path, load_local_dataset # Updated import


class IntentDataset(Dataset):
    """
    A custom Dataset class for handling text and label data for
    intent classification. (Removed domain from description)

    Attributes:
        encodings (dict): Tokenized inputs including input IDs and attention
        masks.
        labels (List[int]): List of intent labels.
    """

    def __init__(self, texts, labels, tokenizer): # Removed domains
        """
        Initialize the IntentDataset with text, labels, and tokenizer.

        Args:
            texts (List[str]): The input texts (keywords).
            labels (List[int]): The intent labels for each text.
            tokenizer: The tokenizer used to encode the text data.
        """
        # Tokenize the texts (keywords) directly
        self.encodings = tokenizer(
            texts, # Changed from concatenating domain and text
            truncation=True,
            padding=True,
            max_length=128,
        )
        self.labels = labels
        # self.domains = domains # Removed

    def __getitem__(self, idx):
        """
        Retrieve a single item from the dataset at the given index.

        Args:
            idx (int): Index of the item to be retrieved.

        Returns:
            dict: A dictionary containing input IDs, attention mask, and label.
            (Removed domain from returned dict)
        """
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx]
            ),
            "labels": torch.tensor(self.labels[idx]),
            # "domain": self.domains[idx],  # Removed domain information
        }

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)


def train_model(account_name: str):
    """
    Train the intent classification model for a specific account_name using adapters.

    This function loads the dataset for the given account_name,
    preprocesses the data, loads a base RoBERTa model, adds and trains an
    adapter for the account_name, and saves the trained adapter and label encoder.

    Args:
        account_name (str): The identifier for the account/tenant.

    Returns:
        None
    """
    # Determine the device to be used (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load account-specific dataset
    try:
        tenant_dataset_file_path = get_tenant_dataset_path(account_name) # Changed tenant_id to account_name
        # Assuming load_local_dataset is appropriate for loading the tenant-specific file
        # If Azure loading is needed for tenant-specific files, this part needs adjustment
        df = load_local_dataset(tenant_dataset_file_path)
    except FileNotFoundError as e:
        print(e)
        print(f"Skipping training for account {account_name} due to missing dataset.") # Changed tenant_id to account_name
        return

    if df.empty:
        print(f"No data found for account_name: {account_name} in {tenant_dataset_file_path}. Skipping training.") # Changed tenant_id to account_name
        return

    # The DataFrame should already be account-specific.
    # It should contain 'keywords' and 'intent_name' columns.
    texts = df["keywords"].tolist() # Changed from df["text"]
    intents = df["intent_name"].tolist() # Changed from df["intent"]
    # domains = df["domain"].tolist() # Removed, as domain info is not used in tokenizer

    # Encode the intent labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)
    num_labels = len(set(labels))

    # Initialize the tokenizer and dataset
    tokenizer = RobertaTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
    intent_dataset = IntentDataset(texts, labels, tokenizer) # Removed domains argument

    # Load the base RoBERTa model with a prediction head
    model = RobertaModelWithHeads.from_pretrained(
        settings.BASE_MODEL_NAME, num_labels=num_labels
    )

    # Add a new adapter for the account
    model.add_adapter(account_name, config=settings.ADAPTER_CONFIG_STRING) # Changed tenant_id to account_name
    # Activate the adapter for training
    model.train_adapter(account_name) # Changed tenant_id to account_name

    # Move the model to the appropriate device (GPU/CPU)
    model.to(device)

    # Define account-specific paths
    account_model_path = os.path.join(settings.MODEL_PATH, account_name) # Changed tenant_id to account_name
    adapter_path = os.path.join(account_model_path, "adapter")
    label_encoder_path = os.path.join(account_model_path, "label_encoder.pkl")

    # Create directories if they don't exist
    os.makedirs(adapter_path, exist_ok=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=adapter_path, # Output directory for adapter checkpoints
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        logging_dir=os.path.join(settings.MODEL_PATH, "logs", account_name), # Account-specific logs, changed tenant_id
        logging_steps=10,
        save_total_limit=1,
    )

    # Initialize Trainer and start training
    trainer = Trainer(model=model, args=training_args, train_dataset=intent_dataset)
    trainer.train()

    # Save the trained adapter and label encoder
    model.save_adapter(adapter_path, account_name) # Changed tenant_id to account_name
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Adapter for account {account_name} saved to {adapter_path}") # Changed tenant_id to account_name
    print(f"Label encoder for account {account_name} saved to {label_encoder_path}") # Changed tenant_id to account_name
