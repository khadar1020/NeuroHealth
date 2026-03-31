from __future__ import annotations

from neurohealth.phase1.types import (
    AgeGroup,
    DialogueSample,
    DialogueTurn,
    IntentClass,
    SourceRecord,
    TriageRule,
    UrgencyLevel,
)
from neurohealth.phase1.validation import validate_all


def _sample_dialogue(sample_id: str, text: str, urgency: UrgencyLevel, age_group: AgeGroup) -> DialogueSample:
    return DialogueSample(
        sample_id=sample_id,
        turns=[
            DialogueTurn(role="user", text=text),
            DialogueTurn(role="assistant", text="Seek emergency care immediately if symptoms worsen."),
        ],
        intent_class=IntentClass.SYMPTOM_CHECK,
        urgency_label=urgency,
        safe_response_label="policy_safe_template",
        age_group=age_group,
        routing_specialty="emergency",
        evidence_ids=["triage_policy_table_v1"],
        provenance={"source_id": "synthetic_dialogues", "license": "INTERNAL-OPEN"},
    )


def test_validation_detects_red_flag_not_escalated():
    sources = {
        "synthetic_dialogues": SourceRecord(
            source_id="synthetic_dialogues",
            name="Synthetic",
            url="local://synthetic",
            owner="NeuroHealth",
            license_type="INTERNAL-OPEN",
            allowed_usage="redistributable",
            attribution_required=False,
            refresh_cadence="monthly",
            ingestion_method="generator",
            license_evidence_url="local://",
            status="approved",
            notes="",
        )
    }

    triage_rules = [
        TriageRule(
            rule_id="triage_001",
            symptom_pattern=["chest pain", "shortness of breath"],
            age_group=AgeGroup.ADULT,
            urgency_level=UrgencyLevel.EMERGENCY,
            rationale="Emergency signs",
            evidence_ids=["cdc"],
        )
    ]

    d = _sample_dialogue(
        "s1",
        "I have chest pain and shortness of breath",
        UrgencyLevel.ROUTINE,
        AgeGroup.ADULT,
    )

    report, queue = validate_all(
        source_lookup=sources,
        knowledge_docs=[
            {
                "doc_id": "d1",
                "source_id": "synthetic_dialogues",
                "title": "x",
                "raw_text": "Emergency guidance text.",
                "citation_url": "https://example.com",
            }
        ],
        knowledge_chunks=[{"chunk_id": "c1", "chunk_text": "x"}],
        triage_rules=triage_rules,
        dialogues=[d],
        candidate_status={"admitted": [], "quarantined": []},
        dedup_stats={"dedup_removed_docs": 0, "dedup_removed_dialogues": 0},
    )

    assert report.pass_fail["gate3_safety"] is False
    assert any(i.issue_type == "red_flag_not_escalated" for i in report.issues)
    assert len(queue) >= 1
