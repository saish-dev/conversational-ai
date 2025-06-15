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
        self.active_tenant_adapter = None

        # Determine the device to be used (GPU or CPU) and move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"IntentClassifier using device: {self.device}")

    def load_adapter_and_encoder(self, tenant_id: str):
        """
        Load the adapter and label encoder for the given tenant_id.

        Args:
            tenant_id (str): The identifier for the tenant.

        Raises:
            FileNotFoundError: If the adapter or label encoder for the tenant is not found.
        """
        adapter_path = os.path.join(self.base_model_path, tenant_id, "adapter")
        label_encoder_path = os.path.join(self.base_model_path, tenant_id, "label_encoder.pkl")

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"❌ Adapter not found for tenant {tenant_id} at {adapter_path}"
            )
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(
                f"❌ Label encoder not found for tenant {tenant_id} at {label_encoder_path}"
            )

        # Load adapter if not already loaded or if a different one was active
        # Note: Adapters are uniquely named by tenant_id when loaded.
        # We only need to load it once. `load_adapter` handles if it's already loaded under that name.
        self.model.load_adapter(adapter_path, load_as=tenant_id, set_active=False) # Load but don't set active yet

        if tenant_id not in self.loaded_label_encoders:
            self.loaded_label_encoders[tenant_id] = joblib.load(label_encoder_path)

        # self.active_tenant_adapter = tenant_id # Will be set in predict when set_active_adapters is called

    def predict(self, text: str, domain: str) -> str:
        """
        Predict the intent label of the given text based on the tenant_id (domain).

        Args:
            text (str): The input text.
            domain (str): The tenant identifier (domain) associated with the input.

        Returns:
            str: Predicted intent.
        """
        tenant_id = domain # Rename for clarity within this method

        # Load adapter and label encoder if necessary
        if tenant_id != self.active_tenant_adapter or tenant_id not in self.loaded_label_encoders:
            self.load_adapter_and_encoder(tenant_id)

        # Set the active adapter for this prediction
        self.model.set_active_adapters(tenant_id)
        self.active_tenant_adapter = tenant_id

        label_encoder = self.loaded_label_encoders[tenant_id]

        # Preprocess input - consistency with training: f"{domain} {text}"
        domain_text = f"{tenant_id} {text}"
        inputs = self.tokenizer(
            domain_text,
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
