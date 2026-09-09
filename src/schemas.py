from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InjectionAssessment(BaseModel):
    is_injection: bool = Field(description="True only if the input attempts to override, reveal, or bypass system/policy instructions.")
    risk: Literal["low", "medium", "high"] = "low"
    reason: str = ""


class GroundednessAssessment(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="How fully the answer is supported by the supplied context.")
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""


class AnswerQualityAssessment(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class SourceCitation(BaseModel):
    id: str
    source: str
    page: int | None = None
    section: str | None = None
    excerpt: str = ""


class AskResult(BaseModel):
    answer: str
    blocked: bool = False
    block_reason: str | None = None
    pii_redacted: bool = False
    injection_risk: str = "low"
    groundedness_score: float | None = None
    grounded: bool | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)
