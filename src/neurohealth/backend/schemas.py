from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from neurohealth.phase1.types import AgeGroup, UrgencyLevel


SeverityLevel = Literal["mild", "moderate", "severe"]
SearchMethod = Literal["bm25", "keyword"]
FeedbackRating = Literal["helpful", "not_helpful", "unsafe"]


class IntakeProfile(BaseModel):
    age_group: AgeGroup
    sex_at_birth: str | None = None
    location: str | None = None
    symptom_category: str
    duration: str
    severity: SeverityLevel
    conditions: str | None = None
    medications: str | None = None

    @field_validator(
        "sex_at_birth",
        "location",
        "symptom_category",
        "duration",
        "conditions",
        "medications",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class SessionCreateRequest(BaseModel):
    intake: IntakeProfile | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    created_at_utc: str


class TriageDecision(BaseModel):
    urgency: UrgencyLevel
    matched_rule_id: str | None = None
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    symptom_cluster: str
    recommended_care_level: str
    specialty: str
    missing_fields: list[str] = Field(default_factory=list)


class NearbyProvider(BaseModel):
    name: str
    distance_km: float
    estimated_wait_minutes: int
    care_level: str
    specialty: str
    location_hint: str
    maps_url: str | None = None


class KnowledgeSearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    source_id: str
    score: float
    snippet: str
    citation_id: str
    title: str | None = None
    citation_url: str | None = None


class SessionStateResponse(BaseModel):
    session_id: str
    created_at_utc: str
    updated_at_utc: str
    intake: IntakeProfile | None = None
    latest_triage: TriageDecision | None = None
    chat: list[ChatMessage] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at_utc: str
    updated_at_utc: str
    intake: IntakeProfile | None = None


class IntakeUpdateResponse(BaseModel):
    session_id: str
    intake: IntakeProfile
    updated_at_utc: str


class MessageCreateRequest(BaseModel):
    text: str = Field(min_length=2, max_length=4000)
    include_providers: bool = True
    message_kind: Literal["chat", "intake_bootstrap"] = "chat"

    @field_validator("text", mode="before")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return str(value).strip()


class ChatReplyResponse(BaseModel):
    session_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage
    triage: TriageDecision
    intake: IntakeProfile | None = None
    nearby_providers: list[NearbyProvider] = Field(default_factory=list)
    citations: list[KnowledgeSearchResult] = Field(default_factory=list)
    response_mode: Literal["llm_rag", "rules_fallback", "provider_lookup", "location_request"] = "rules_fallback"
    dataset_files_used: list[str] = Field(default_factory=list)


class NearbyProvidersRequest(BaseModel):
    location: str
    urgency: UrgencyLevel
    recommended_care_level: str | None = None
    specialty: str | None = None

    @field_validator("location", mode="before")
    @classmethod
    def _strip_location(cls, value: str) -> str:
        return str(value).strip()


class NearbyProvidersResponse(BaseModel):
    providers: list[NearbyProvider]


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    method: SearchMethod = "bm25"
    age_group: AgeGroup | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        return str(value).strip()


class KnowledgeSearchResponse(BaseModel):
    query: str
    method: SearchMethod
    results: list[KnowledgeSearchResult] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    session_id: str
    rating: FeedbackRating
    comment: str | None = None

    @field_validator("session_id", mode="before")
    @classmethod
    def _strip_session(cls, value: str) -> str:
        return str(value).strip()

    @field_validator("comment", mode="before")
    @classmethod
    def _strip_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class FeedbackResponse(BaseModel):
    accepted: bool
    feedback_id: str


class DatasetStats(BaseModel):
    triage_rules: int
    routing_rules: int
    knowledge_documents: int
    knowledge_chunks: int
    conversation_samples: int


class ConfigResponse(BaseModel):
    age_groups: list[str]
    severity_levels: list[str]
    urgency_levels: list[str]
    symptom_clusters: list[str]
    dataset_stats: DatasetStats
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_enabled: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"]
    timestamp_utc: str
    sessions_active: int
    feedback_items: int
    dataset_stats: DatasetStats
    llm_enabled: bool = False
    llm_model: str | None = None
    llm_last_error: str | None = None
