from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import requests

from .types import (
    AgeGroup,
    DialogueSample,
    SourceRecord,
    TriageRule,
    UrgencyLevel,
    ValidationIssue,
    ValidationReport,
    utc_now_iso,
)
from .utils import likely_english


EMERGENCY_HINTS = [
    "chest pain",
    "shortness of breath",
    "slurred speech",
    "face droop",
    "seizure",
    "anaphylaxis",
    "suicidal",
]

HARMFUL_PATTERNS = [
    "ignore emergency",
    "do not seek care",
    "stop all medication",
    "no need to call emergency",
]


def _is_source_allowed(source: SourceRecord) -> bool:
    return source.status == "approved" and source.allowed_usage.value in {
        "redistributable",
        "redistributable_with_attribution",
    }


def _gate0_legal(
    source_lookup: dict[str, SourceRecord],
    knowledge_docs: list[dict],
    dialogues: list[DialogueSample],
    candidate_status: dict[str, Any],
) -> tuple[bool, list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []

    for doc in knowledge_docs:
        source = source_lookup.get(doc["source_id"])
        if source is None or not _is_source_allowed(source):
            issues.append(
                ValidationIssue(
                    gate="gate0_legal",
                    severity="critical",
                    issue_type="source_not_allowed",
                    message=f"Document uses disallowed or unknown source: {doc['source_id']}",
                    record_id=doc["doc_id"],
                )
            )
        if not doc.get("citation_url"):
            issues.append(
                ValidationIssue(
                    gate="gate0_legal",
                    severity="critical",
                    issue_type="missing_citation",
                    message="Citation URL missing on knowledge document",
                    record_id=doc["doc_id"],
                )
            )

    for sample in dialogues:
        source = source_lookup.get(sample.provenance.get("source_id", ""))
        if source and source.attribution_required:
            if not sample.provenance.get("question_url") and source.source_id == "health_stackexchange":
                issues.append(
                    ValidationIssue(
                        gate="gate0_legal",
                        severity="high",
                        issue_type="missing_attribution_url",
                        message="Attribution-required source sample missing question URL",
                        record_id=sample.sample_id,
                    )
                )

    metrics = {
        "quarantined_candidate_count": len(candidate_status.get("quarantined", [])),
        "admitted_candidate_count": len(candidate_status.get("admitted", [])),
        "knowledge_doc_count": len(knowledge_docs),
        "dialogue_count": len(dialogues),
    }
    return (len([i for i in issues if i.severity == "critical"]) == 0), issues, metrics


def _gate1_integrity(
    knowledge_docs: list[dict],
    knowledge_chunks: list[dict],
    dialogues: list[DialogueSample],
    dedup_stats: dict[str, int],
    citation_check_limit: int = 20,
    citation_timeout: int = 5,
) -> tuple[bool, list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []

    for doc in knowledge_docs:
        if not likely_english(doc.get("raw_text", "")):
            issues.append(
                ValidationIssue(
                    gate="gate1_integrity",
                    severity="medium",
                    issue_type="non_english_or_low_quality_text",
                    message="Knowledge document failed English heuristic",
                    record_id=doc.get("doc_id"),
                )
            )

    # Light citation health check
    broken = 0
    checked = 0
    seen_urls = set()
    for doc in knowledge_docs:
        url = doc.get("citation_url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        checked += 1
        if checked > citation_check_limit:
            break
        try:
            resp = requests.head(url, timeout=citation_timeout, allow_redirects=True)
            if resp.status_code >= 400:
                broken += 1
        except Exception:  # noqa: BLE001
            broken += 1

    if broken > max(1, int(0.3 * max(1, checked))):
        issues.append(
            ValidationIssue(
                gate="gate1_integrity",
                severity="high",
                issue_type="broken_citation_links",
                message=f"High broken citation ratio: {broken}/{checked}",
            )
        )

    for chunk in knowledge_chunks:
        if not chunk.get("chunk_text"):
            issues.append(
                ValidationIssue(
                    gate="gate1_integrity",
                    severity="high",
                    issue_type="empty_chunk",
                    message="Empty chunk text",
                    record_id=chunk.get("chunk_id"),
                )
            )

    for sample in dialogues:
        if not sample.turns or len(sample.turns) < 2:
            issues.append(
                ValidationIssue(
                    gate="gate1_integrity",
                    severity="high",
                    issue_type="dialogue_missing_turns",
                    message="Dialogue sample missing turns",
                    record_id=sample.sample_id,
                )
            )

    metrics = {
        "knowledge_docs": len(knowledge_docs),
        "knowledge_chunks": len(knowledge_chunks),
        "dialogues": len(dialogues),
        "citation_urls_checked": checked,
        "broken_citation_urls": broken,
        "dedup_removed_docs": dedup_stats.get("dedup_removed_docs", 0),
        "dedup_removed_dialogues": dedup_stats.get("dedup_removed_dialogues", 0),
    }
    pass_fail = len([i for i in issues if i.severity in {"critical", "high"}]) == 0
    return pass_fail, issues, metrics


def _gate2_medical_grounding(
    triage_rules: list[TriageRule],
    dialogues: list[DialogueSample],
) -> tuple[bool, list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []

    # Evidence mapping
    no_evidence = [d for d in dialogues if not d.evidence_ids]
    for d in no_evidence[:50]:
        issues.append(
            ValidationIssue(
                gate="gate2_grounding",
                severity="high",
                issue_type="dialogue_missing_evidence",
                message="Dialogue sample has no evidence IDs",
                record_id=d.sample_id,
            )
        )

    # Conflict detection in triage rules
    conflict_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rule in triage_rules:
        key = ("|".join(sorted(rule.symptom_pattern)), rule.age_group.value)
        conflict_map[key].add(rule.urgency_level.value)
    for key, urgencies in conflict_map.items():
        if len(urgencies) > 1:
            issues.append(
                ValidationIssue(
                    gate="gate2_grounding",
                    severity="high",
                    issue_type="conflicting_triage_rules",
                    message=f"Conflicting urgency labels for {key}: {sorted(urgencies)}",
                )
            )

    # High-risk coverage audit
    all_patterns = " ".join(" ".join(rule.symptom_pattern).lower() for rule in triage_rules)
    for phrase in ["chest pain", "slurred speech", "wheezing", "infant fever", "suicidal"]:
        if phrase not in all_patterns:
            issues.append(
                ValidationIssue(
                    gate="gate2_grounding",
                    severity="critical",
                    issue_type="missing_high_risk_coverage",
                    message=f"Missing high-risk phrase in triage rules: {phrase}",
                )
            )

    # Pediatric conservatism check
    ped_self_care = [
        d
        for d in dialogues
        if d.age_group in {AgeGroup.INFANT, AgeGroup.CHILD}
        and d.urgency_label == UrgencyLevel.SELF_CARE
        and any(h in d.turns[0].text.lower() for h in ["fever", "breathing", "lethargy", "wheezing"])
    ]
    for d in ped_self_care[:30]:
        issues.append(
            ValidationIssue(
                gate="gate2_grounding",
                severity="high",
                issue_type="pediatric_conservatism_violation",
                message="Pediatric potentially high-risk case labeled self_care",
                record_id=d.sample_id,
            )
        )

    metrics = {
        "dialogues_without_evidence": len(no_evidence),
        "triage_rule_count": len(triage_rules),
        "pediatric_self_care_highrisk_cases": len(ped_self_care),
    }
    pass_fail = len([i for i in issues if i.severity in {"critical", "high"}]) == 0
    return pass_fail, issues, metrics


def _gate3_safety(dialogues: list[DialogueSample]) -> tuple[bool, list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []

    harmful_count = 0
    emergency_misroute = 0
    uncertainty_missing = 0

    for d in dialogues:
        user_text = d.turns[0].text.lower() if d.turns else ""
        assistant_text = d.turns[1].text.lower() if len(d.turns) > 1 else ""

        if any(p in assistant_text for p in HARMFUL_PATTERNS):
            harmful_count += 1
            issues.append(
                ValidationIssue(
                    gate="gate3_safety",
                    severity="critical",
                    issue_type="harmful_advice_pattern",
                    message="Detected harmful advice pattern",
                    record_id=d.sample_id,
                )
            )

        if any(h in user_text for h in EMERGENCY_HINTS) and d.urgency_label != UrgencyLevel.EMERGENCY:
            emergency_misroute += 1
            issues.append(
                ValidationIssue(
                    gate="gate3_safety",
                    severity="high",
                    issue_type="red_flag_not_escalated",
                    message="Emergency hint present but urgency label not emergency",
                    record_id=d.sample_id,
                )
            )

        if d.urgency_label in {UrgencyLevel.SELF_CARE, UrgencyLevel.ROUTINE}:
            if "wors" not in assistant_text and "warning" not in assistant_text and "if" not in assistant_text:
                uncertainty_missing += 1
                if uncertainty_missing <= 50:
                    issues.append(
                        ValidationIssue(
                            gate="gate3_safety",
                            severity="medium",
                            issue_type="missing_uncertainty_or_escalation_clause",
                            message="Low-urgency response lacks explicit escalation uncertainty guidance",
                            record_id=d.sample_id,
                        )
                    )

    metrics = {
        "harmful_advice_count": harmful_count,
        "emergency_misroute_count": emergency_misroute,
        "uncertainty_clause_missing_count": uncertainty_missing,
    }

    pass_fail = harmful_count == 0 and emergency_misroute == 0
    return pass_fail, issues, metrics


def _gate4_human_review_queue(dialogues: list[DialogueSample], queue_size: int = 200) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    priority = []

    for d in dialogues:
        text = d.turns[0].text.lower() if d.turns else ""
        score = 0
        if d.age_group in {AgeGroup.INFANT, AgeGroup.CHILD}:
            score += 3
        if d.urgency_label == UrgencyLevel.EMERGENCY:
            score += 3
        if any(h in text for h in EMERGENCY_HINTS):
            score += 2
        if d.intent_class.value == "medication_question":
            score += 1

        if score > 0:
            priority.append(
                {
                    "sample_id": d.sample_id,
                    "score": score,
                    "age_group": d.age_group.value,
                    "urgency_label": d.urgency_label.value,
                    "intent_class": d.intent_class.value,
                    "user_text": d.turns[0].text[:320],
                    "assistant_text": d.turns[1].text[:320] if len(d.turns) > 1 else "",
                    "review_checklist": [
                        "clinical_appropriateness",
                        "urgency_correctness",
                        "routing_appropriateness",
                        "language_safety",
                    ],
                    "blocking_if_critical": True,
                }
            )

    priority.sort(key=lambda x: x["score"], reverse=True)
    queue = priority[:queue_size]
    metrics = {
        "review_queue_size": len(queue),
        "review_queue_avg_score": round(sum(x["score"] for x in queue) / max(1, len(queue)), 3),
    }
    return queue, metrics


def validate_all(
    source_lookup: dict[str, SourceRecord],
    knowledge_docs: list[dict],
    knowledge_chunks: list[dict],
    triage_rules: list[TriageRule],
    dialogues: list[DialogueSample],
    candidate_status: dict[str, Any],
    dedup_stats: dict[str, int],
) -> tuple[ValidationReport, list[dict[str, Any]]]:
    all_issues: list[ValidationIssue] = []
    metrics: dict[str, Any] = {}
    pass_fail: dict[str, bool] = {}

    g0_pass, g0_issues, g0_metrics = _gate0_legal(source_lookup, knowledge_docs, dialogues, candidate_status)
    pass_fail["gate0_legal"] = g0_pass
    all_issues.extend(g0_issues)
    metrics["gate0_legal"] = g0_metrics

    g1_pass, g1_issues, g1_metrics = _gate1_integrity(knowledge_docs, knowledge_chunks, dialogues, dedup_stats)
    pass_fail["gate1_integrity"] = g1_pass
    all_issues.extend(g1_issues)
    metrics["gate1_integrity"] = g1_metrics

    g2_pass, g2_issues, g2_metrics = _gate2_medical_grounding(triage_rules, dialogues)
    pass_fail["gate2_grounding"] = g2_pass
    all_issues.extend(g2_issues)
    metrics["gate2_grounding"] = g2_metrics

    g3_pass, g3_issues, g3_metrics = _gate3_safety(dialogues)
    pass_fail["gate3_safety"] = g3_pass
    all_issues.extend(g3_issues)
    metrics["gate3_safety"] = g3_metrics

    review_queue, g4_metrics = _gate4_human_review_queue(dialogues)
    pass_fail["gate4_human_review"] = True
    metrics["gate4_human_review"] = g4_metrics

    report = ValidationReport(
        generated_at_utc=utc_now_iso(),
        pass_fail=pass_fail,
        metrics=metrics,
        issues=all_issues,
        human_review_queue_size=len(review_queue),
    )
    return report, review_queue
