from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's natural-language question.")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The final answer text shown to the user.")
    sources: List[str] = Field(
        default_factory=list,
        description="Chunk/document IDs used to ground the answer. Empty for general_question answers.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the answer, between 0 and 1."
    )
