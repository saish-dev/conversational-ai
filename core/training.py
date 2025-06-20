__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import os
import joblib
import pandas as pd
import torch
from config.settings import settings
from core.utils import get_tenant_dataset_path, load_local_dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.trainer import Trainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments
from transformers.adapters.models.roberta import RobertaAdapterModel


class IntentDataset(Dataset):
    """
    Custom PyTorch Dataset for intent classification.

    This class prepares the text and labels for the model by tokenizing
    the input texts and structuring them for consumption by the Trainer.
    """

    def __init__(self, texts, labels, tokenizer):
        """
        Initializes the dataset.

        Args:
            texts (list[str]): A list of raw text inputs (utterances).
            labels (list[int]): A list of corresponding integer-encoded labels.
            tokenizer: The Hugging Face tokenizer to process the text.
        """
        # Tokenize all texts at once for efficiency.
        # `truncation=True` ensures that inputs longer than the model's
        # maximum length are cut down. `padding=True` adds padding to
        # shorter inputs to create uniform-length batches.
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves a single tokenized item and its label from the dataset.

        This is required by PyTorch's Dataset class.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary containing the item's `input_ids`,
                  `attention_mask`, and `labels` as PyTorch tensors.
        """
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        This is required by PyTorch's Dataset class.
        """
        return len(self.labels)


def train_model(account_name: str):
    """
    Trains a new adapter for a given account.

    This function orchestrates the entire training pipeline:
    1. Loads the account-specific dataset.
    2. Prepares the data (text cleaning, label encoding).
    3. Initializes the RoBERTa model and tokenizer.
    4. Adds and configures a new adapter and classification head for the
       account.
    5. Sets up and runs the Hugging Face Trainer.
    6. Saves the trained adapter and label encoder to disk.

    Args:
        account_name (str): The unique identifier for the account/tenant.
    """
    # Use GPU if available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset specific to the account.
    try:
        dataset_path = get_tenant_dataset_path(account_name)
        df = load_local_dataset(dataset_path)
    except FileNotFoundError as e:
        print(e)
        print(f"Skipping training for {account_name}: dataset not found.")
        return

    if df.empty:
        print(f"Skipping training for {account_name}: dataset is empty.")
        return

    # Extract texts and intents from the DataFrame.
    texts = df["keywords"].tolist()
    # Strip whitespace from intents to prevent labeling errors.
    intents = [str(intent).strip() for intent in df["intent_name"].tolist()]

    if not intents:
        print(f"Skipping training for {account_name}: no intents found.")
        return

    # Convert string labels (e.g., "greet") to integer indices (e.g., 0).
    # The label encoder mapping is saved for later use during inference.
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)
    num_labels = len(label_encoder.classes_)  # type: ignore

    # Initialize the tokenizer and create the PyTorch dataset.
    tokenizer = RobertaTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
    intent_dataset = IntentDataset(texts, list(labels), tokenizer)  # type: ignore

    # Load the pre-trained RoBERTa model with adapter support.
    model = RobertaAdapterModel.from_pretrained(
        settings.BASE_MODEL_NAME, num_labels=num_labels
    )
    # The from_pretrained method may return a tuple; we only need the model.
    if isinstance(model, tuple):
        model = model[0]

    # This is the core of the multi-tenant strategy:
    # 1. Add a new adapter layer named after the account.
    # 2. Add a new classification head for this adapter.
    # During training, only the adapter and head weights are updated,
    # keeping the base model frozen.
    model.add_adapter(account_name, config=settings.ADAPTER_CONFIG_STRING)
    model.add_classification_head(account_name, num_labels=num_labels)

    # Activate the new adapter and head for the upcoming training session.
    model.train_adapter([account_name])
    model.set_active_adapters([account_name])
    model.active_head = account_name
    model.to(device)

    # Define paths for saving the trained adapter and artifacts.
    account_model_path = os.path.join(settings.MODEL_PATH, account_name)
    adapter_path = os.path.join(account_model_path, "adapter")
    label_encoder_path = os.path.join(
        account_model_path, "label_encoder.pkl"
    )
    os.makedirs(adapter_path, exist_ok=True)

    # Configure the training process.
    # These arguments control aspects like epochs, batch size, and logging.
    training_args = TrainingArguments(
        output_dir=adapter_path,
        num_train_epochs=40,
        per_device_train_batch_size=8,
        save_strategy=IntervalStrategy.STEPS,
        logging_dir=os.path.join(settings.MODEL_PATH, "logs", account_name),
        logging_steps=10,
        save_total_limit=1,  # Only keep the best checkpoint.
    )

    # For debugging: confirm which adapter is active before training.
    print(f"Training adapter: {account_name}")
    print(f"Active adapters: {model.active_adapters}")
    print(f"Active head: {model.active_head}")

    # Initialize the Trainer, which handles the training loop.
    trainer = Trainer(
        model=model, args=training_args, train_dataset=intent_dataset
    )
    trainer.train()  # type: ignore

    # After training, save the adapter and the label encoder for inference.
    model.save_adapter(adapter_path, account_name)
    joblib.dump(label_encoder, label_encoder_path)

    print(f"Adapter for {account_name} saved to {adapter_path}")
    print(f"Label encoder for {account_name} saved to {label_encoder_path}")
