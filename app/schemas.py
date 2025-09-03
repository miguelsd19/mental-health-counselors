from pydantic import BaseModel, Field
from typing import List, Optional


class IngestItem(BaseModel):
    question: str
    answer: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None


class AskRequest(BaseModel):
    query: str = Field(..., description="Free-text del terapeuta")
    top_k: Optional[int] = None


class Retrieved(BaseModel):
    id: int
    score: float
    question: str
    answer: str
    source: Optional[str] = None
    tags: Optional[list] = None


class AskResponse(BaseModel):
    suggestion: str
    retrieved: List[Retrieved]