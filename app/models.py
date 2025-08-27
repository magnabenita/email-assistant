#app/models.py
from pydantic import BaseModel
from typing import Optional

class Email(BaseModel):
    id: str
    subject: Optional[str]
    body: str

class RecommendationRequest(BaseModel):
    email_id: str  # existing email in DB
    top_k: int = 5
