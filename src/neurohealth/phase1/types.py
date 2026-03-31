from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class AllowedUsage(str, Enum):
    REDISTRIBUTABLE = "redistributable"
    REDISTRIBUTABLE_WITH_ATTRIBUTION = "redistributable_with_attribution"
    QUARANTINE = "quarantine"


class UrgencyLevel(str, Enum):
    SELF_CARE = "self_care"
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class AgeGroup(str, Enum):
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    OLDER_ADULT = "older_adult"


class IntentClass(str, Enum):
    SYMPTOM_CHECK = "symptom_check"
    MEDICATION_QUESTION = "medication_question"
    PREVENTIVE_CARE = "preventive_care"
    CHRONIC_FOLLOWUP = "chronic_followup"
    APPOINTMENT_NAVIGATION = "appointment_navigation"


class SourceRecord(BaseModel):
    source_id: str
    name: str
    url: str
    owner: str
    license_type: str
    allowed_usage: AllowedUsage
    attribution_required: bool
    refresh_cadence: str
    ingestion_method: str
    license_evidence_url: str
    status: str
    notes: str | None = None


class KnowledgeDoc(BaseModel):
    doc_id: str
    source_id: str
    title: str
    raw_text: str
    publication_date: str | None = None
    license_type: str
    citation_url: str
    retrieved_at_utc: str
    language: str = "en"
    audience_age_groups: list[AgeGroup] = Field(default_factory=list)
    content_hash: str


class KnowledgeChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_id: str
    chunk_index: int
    chunk_text: str
    medical_topics: list[str] = Field(default_factory=list)
    age_applicability: list[AgeGroup] = Field(default_factory=list)
    citation_id: str


class TriageRule(BaseModel):
    rule_id: str
    symptom_pattern: list[str]
    age_group: AgeGroup
    urgency_level: UrgencyLevel
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class RoutingRule(BaseModel):
    routing_id: str
    urgency_level: UrgencyLevel
    symptom_cluster: str
    recommended_care_level: str
    specialty: str


class DialogueTurn(BaseModel):
    role: str
    text: str


class DialogueSample(BaseModel):
    sample_id: str
    turns: list[DialogueTurn]
    intent_class: IntentClass
    urgency_label: UrgencyLevel
    safe_response_label: str
    age_group: AgeGroup
    routing_specialty: str
    evidence_ids: list[str] = Field(default_factory=list)
    provenance: dict[str, Any]


class ValidationIssue(BaseModel):
    gate: str
    severity: str
    issue_type: str
    message: str
    record_id: str | None = None


class ValidationReport(BaseModel):
    generated_at_utc: str
    pass_fail: dict[str, bool]
    metrics: dict[str, Any]
    issues: list[ValidationIssue]
    human_review_queue_size: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
