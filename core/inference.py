__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import os
import torch
import joblib
from transformers import RobertaTokenizer
from transformers.adapters import RobertaModelWithHeads # Added for adapter support
from config.settings import settings


class IntentClassifier:
    def __init__(self):
        """
        Initialize the IntentClassifier with tokenizer, base model,
        and caches for label encoders and active adapter.
        """
        self.base_model_path = settings.MODEL_PATH # Assuming MODEL_PATH points to dir with tenant adapters
        self.tokenizer = RobertaTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
        self.model = RobertaModelWithHeads.from_pretrained(settings.BASE_MODEL_NAME)
        self.model.eval() # Set model to evaluation mode
        self.loaded_label_encoders = {}
        self.active_adapter_name = None # Renamed from active_tenant_adapter

        # Determine the device to be used (GPU or CPU) and move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"IntentClassifier using device: {self.device}")

    def load_adapter_and_encoder(self, account_name: str): # Changed tenant_id to account_name
        """
        Load the adapter and label encoder for the given account_name.

        Args:
            account_name (str): The identifier for the account/tenant.

        Raises:
            FileNotFoundError: If the adapter or label encoder for the account is not found.
        """
        adapter_path = os.path.join(self.base_model_path, account_name, "adapter") # Changed tenant_id to account_name
        label_encoder_path = os.path.join(self.base_model_path, account_name, "label_encoder.pkl") # Changed tenant_id to account_name

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"❌ Adapter not found for account {account_name} at {adapter_path}" # Changed tenant_id to account_name
            )
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(
                f"❌ Label encoder not found for account {account_name} at {label_encoder_path}" # Changed tenant_id to account_name
            )

        # Load adapter if not already loaded or if a different one was active
        # Note: Adapters are uniquely named by account_name when loaded.
        # We only need to load it once. `load_adapter` handles if it's already loaded under that name.
        self.model.load_adapter(adapter_path, load_as=account_name, set_active=False) # Load but don't set active yet, changed tenant_id

        if account_name not in self.loaded_label_encoders: # Changed tenant_id to account_name
            self.loaded_label_encoders[account_name] = joblib.load(label_encoder_path) # Changed tenant_id to account_name

        # self.active_adapter_name = account_name # Will be set in predict when set_active_adapters is called

    def predict(self, text: str, account_name: str) -> str: # Changed domain to account_name
        """
        Predict the intent label of the given text based on the account_name.

        Args:
            text (str): The input text.
            account_name (str): The account identifier associated with the input.

        Returns:
            str: Predicted intent.
        """
        # account_name is now directly used

        # Load adapter and label encoder if necessary
        if account_name != self.active_adapter_name or account_name not in self.loaded_label_encoders: # Renamed active_tenant_adapter
            self.load_adapter_and_encoder(account_name)

        # Set the active adapter for this prediction
        self.model.set_active_adapters(account_name)
        self.active_adapter_name = account_name # Renamed active_tenant_adapter

        label_encoder = self.loaded_label_encoders[account_name]

        # Preprocess input - tokenizer now takes only the text (keywords)
        # domain_text = f"{account_name} {text}" # This concatenation is removed
        inputs = self.tokenizer(
            text, # Changed from domain_text
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()

        predicted_intent = label_encoder.inverse_transform(
            [predicted_class_idx]
        )[0]
        return predicted_intent
