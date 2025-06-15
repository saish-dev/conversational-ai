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
    A custom Dataset class for handling text, domain, and label data for
    intent classification.

    Attributes:
        encodings (dict): Tokenized inputs including input IDs and attention
        masks.
        labels (List[int]): List of intent labels.
        domains (List[str]): List of domains corresponding to each text.
    """

    def __init__(self, texts, labels, domains, tokenizer):
        """
        Initialize the IntentDataset with text, domain, labels, and tokenizer.

        Args:
            texts (List[str]): The input texts.
            labels (List[int]): The intent labels for each text.
            domains (List[str]): The domain information for each text.
            tokenizer: The tokenizer used to encode the text data.
        """
        # Concatenate domain and text for each entry
        self.encodings = tokenizer(
            [f"{domain} {text}" for domain, text in zip(domains, texts)],
            truncation=True,
            padding=True,
            max_length=128,
        )
        self.labels = labels
        self.domains = domains

    def __getitem__(self, idx):
        """
        Retrieve a single item from the dataset at the given index.

        Args:
            idx (int): Index of the item to be retrieved.

        Returns:
            dict: A dictionary containing input IDs, attention mask, label,
            and domain.
        """
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx]
            ),
            "labels": torch.tensor(self.labels[idx]),
            "domain": self.domains[idx],  # Still storing domain information
        }

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)


def train_model(tenant_id: str):
    """
    Train the intent classification model for a specific tenant using adapters.

    This function loads the dataset, filters it for the given tenant_id,
    preprocesses the data, loads a base RoBERTa model, adds and trains an
    adapter for the tenant, and saves the trained adapter and label encoder.

    Args:
        tenant_id (str): The identifier for the tenant.

    Returns:
        None
    """
    # Determine the device to be used (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tenant-specific dataset
    try:
        tenant_dataset_file_path = get_tenant_dataset_path(tenant_id)
        # Assuming load_local_dataset is appropriate for loading the tenant-specific file
        # If Azure loading is needed for tenant-specific files, this part needs adjustment
        df = load_local_dataset(tenant_dataset_file_path)
    except FileNotFoundError as e:
        print(e)
        print(f"Skipping training for tenant {tenant_id} due to missing dataset.")
        return

    if df.empty:
        print(f"No data found for tenant_id: {tenant_id} in {tenant_dataset_file_path}. Skipping training.")
        return

    # The DataFrame should already be tenant-specific, but ensure 'domain' column exists if used by IntentDataset
    # For IntentDataset: texts, labels, domains. 'domains' list can be df["domain"].tolist() or [tenant_id] * len(texts)
    texts = df["text"].tolist()
    intents = df["intent"].tolist()
    domains = df["domain"].tolist()

    # Encode the intent labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)
    num_labels = len(set(labels))

    # Initialize the tokenizer and dataset
    tokenizer = RobertaTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
    intent_dataset = IntentDataset(texts, labels, domains, tokenizer)

    # Load the base RoBERTa model with a prediction head
    model = RobertaModelWithHeads.from_pretrained(
        settings.BASE_MODEL_NAME, num_labels=num_labels
    )

    # Add a new adapter for the tenant
    model.add_adapter(tenant_id, config=settings.ADAPTER_CONFIG_STRING)
    # Activate the adapter for training
    model.train_adapter(tenant_id)

    # Move the model to the appropriate device (GPU/CPU)
    model.to(device)

    # Define tenant-specific paths
    tenant_model_path = os.path.join(settings.MODEL_PATH, tenant_id)
    adapter_path = os.path.join(tenant_model_path, "adapter")
    label_encoder_path = os.path.join(tenant_model_path, "label_encoder.pkl")

    # Create directories if they don't exist
    os.makedirs(adapter_path, exist_ok=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=adapter_path, # Output directory for adapter checkpoints
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        logging_dir=os.path.join(settings.MODEL_PATH, "logs", tenant_id), # Tenant-specific logs
        logging_steps=10,
        save_total_limit=1,
    )

    # Initialize Trainer and start training
    trainer = Trainer(model=model, args=training_args, train_dataset=intent_dataset)
    trainer.train()

    # Save the trained adapter and label encoder
    model.save_adapter(adapter_path, tenant_id)
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Adapter for tenant {tenant_id} saved to {adapter_path}")
    print(f"Label encoder for tenant {tenant_id} saved to {label_encoder_path}")
