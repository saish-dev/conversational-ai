__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

import os
import torch
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from config.settings import settings


class IntentClassifier:
    def __init__(self):
        """
        Initialize the IntentClassifier with model path, tokenizer,
        label encoder, and load the model.
        """
        self.model_path = settings.MODEL_PATH
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = self.load_model()
        self.label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)

    def load_model(self):
        """
        Load the trained intent classification model.

        Returns:
            torch.nn.Module: Loaded intent classification model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"❌ Model directory not found at {self.model_path}"
            )

        model = RobertaForSequenceClassification.from_pretrained(
            self.model_path
        )
        model.eval()
        return model

    def predict(self, text: str, domain: str) -> str:
        """
        Predict the intent label of the given text based on the domain.

        Args:
            text (str): The input text.
            domain (str): The domain associated with the input.

        Returns:
            str: Predicted intent.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Preprocess input
        domain_text = f"{domain} {text}"
        inputs = self.tokenizer(
            domain_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()

        predicted_intent = self.label_encoder.inverse_transform(
            [predicted_class_idx]
        )[0]
        return predicted_intent
