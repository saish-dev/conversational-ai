__author__ = "Trellissoft"
__copyright__ = "Copyright 2025, Trellissoft"

from pydantic import BaseModel


class IntentRequest(BaseModel):
    text: str
    account_name: str


class IntentResponse(BaseModel):
    intent: str
