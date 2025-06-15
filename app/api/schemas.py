__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

from pydantic import BaseModel


class IntentRequest(BaseModel):
    text: str
    customer_domain: str


class IntentResponse(BaseModel):
    intent: str
