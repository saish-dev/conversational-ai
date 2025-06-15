__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import torch
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from config.settings import settings
from core.utils import get_combined_dataset


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


def train_model():
    """
    Train the intent classification model using the combined dataset.

    This function loads the dataset, preprocesses the data using a tokenizer
    and label encoder, initializes the model with the appropriate number of
    labels, and trains the model using specified training arguments.
    The trained model and the label encoder are saved to disk.

    Returns:
        None
    """
    # Determine the device to be used (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess the dataset
    dataset = get_combined_dataset()
    df = pd.read_csv(dataset)
    texts = df["text"].tolist()
    intents = df["intent"].tolist()
    domains = df["domain"].tolist()  # New column: domain

    # Encode the intent labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)

    # Initialize the tokenizer and dataset
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataset = IntentDataset(texts, labels, domains, tokenizer)

    # Load the model with the appropriate number of labels
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=len(set(labels))
    )

    # Move the model to the appropriate device (GPU/CPU)
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=settings.MODEL_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        logging_dir="models/logs",
        logging_steps=10,
        save_total_limit=1,
    )

    # Initialize Trainer and start training
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    # Save the trained model and label encoder
    model.save_pretrained(settings.MODEL_PATH)
    joblib.dump(label_encoder, settings.LABEL_ENCODER_PATH)
