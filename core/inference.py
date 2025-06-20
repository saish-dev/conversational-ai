__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import os
import torch
import joblib
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.adapters.models.roberta import RobertaAdapterModel
from config.settings import settings


class IntentClassifier:
    """
    Handles intent prediction using a multi-tenant adapter model.

    This class encapsulates the logic for loading the base model,
    dynamically loading tenant-specific adapters, and performing
    predictions. It caches loaded adapters and label encoders to optimize
    performance for repeated requests for the same tenant.
    """

    def __init__(self):
        """
        Initializes the classifier.

        This loads the base RoBERTa model and tokenizer and prepares caches
        for tenant-specific artifacts.
        """
        self.base_model_path = settings.MODEL_PATH
        self.tokenizer = RobertaTokenizer.from_pretrained(
            settings.BASE_MODEL_NAME
        )
        # Load the base model, which will serve as the foundation for all
        # tenant-specific adapters.
        self.model = RobertaAdapterModel.from_pretrained(
            settings.BASE_MODEL_NAME
        )
        if isinstance(self.model, tuple):
            self.model = self.model[0]

        # Set the model to evaluation mode, as this class is only for
        # inference.
        self.model.eval()

        # Caches to store artifacts for already-loaded tenants to avoid
        # redundant disk I/O.
        self.loaded_label_encoders = {}
        self.active_adapter_name = None

        # Set device to GPU if available for faster inference.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        print(f"IntentClassifier using device: {self.device}")

    def load_adapter_and_encoder(self, account_name: str):
        """
        Loads the adapter and label encoder for a specific account.

        This method is called on-demand the first time a prediction is
        requested for a new account.

        Args:
            account_name (str): The identifier for the account/tenant.

        Raises:
            FileNotFoundError: If the required adapter or label encoder
                               for the account does not exist.
        """
        adapter_path = os.path.join(
            self.base_model_path, account_name, "adapter"
        )
        label_encoder_path = os.path.join(
            self.base_model_path, account_name, "label_encoder.pkl"
        )

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found for account {account_name}"
            )
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(
                f"Label encoder not found for account {account_name}"
            )

        # Load the pre-trained adapter weights into the base model.
        # `load_as` gives the adapter a unique name for future activation.
        self.model.load_adapter(  # type: ignore
            adapter_path, load_as=account_name, set_active=False
        )

        # Load and cache the label encoder if it's not already in memory.
        if account_name not in self.loaded_label_encoders:
            self.loaded_label_encoders[account_name] = joblib.load(
                label_encoder_path
            )

    def predict(self, text: str, account_name: str) -> str:
        """
        Predicts the intent for a given text and account.

        Args:
            text (str): The user's input utterance.
            account_name (str): The account identifier.

        Returns:
            str: The predicted intent name (e.g., "greet").
        """
        # Lazy-load the adapter and encoder only when needed.
        # This is efficient if an account's model is requested for the
        # first time or if switching between different accounts.
        if (
            account_name != self.active_adapter_name
            or account_name not in self.loaded_label_encoders
        ):
            self.load_adapter_and_encoder(account_name)

        # Set the active adapter for this specific prediction request.
        model = (
            self.model[0] if isinstance(self.model, tuple) else self.model
        )
        model.set_active_adapters([account_name])
        self.active_adapter_name = account_name

        # Retrieve the correct label encoder for this account.
        label_encoder = self.loaded_label_encoders[account_name]

        # Tokenize the input text and move it to the active device.
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Perform inference. `torch.no_grad()` disables gradient
        # calculations, which is crucial for speeding up predictions.
        with torch.no_grad():
            outputs = model(**inputs)
            # The model outputs raw scores (logits); we get the prediction
            # by finding the index of the highest score.
            predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()

        # Convert the predicted index back to its string label.
        predicted_intent = label_encoder.inverse_transform(
            [predicted_class_idx]
        )[0]
        return predicted_intent


# A singleton instance of the classifier to be used by the API and tests.
# This ensures the model is loaded only once when the application starts.
intent_classifier_instance = IntentClassifier()


def predict_intent_for_test(text: str, account_name: str) -> str:
    """
    A simple wrapper for testing the prediction logic.
    """
    return intent_classifier_instance.predict(text, account_name)
