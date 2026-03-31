from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_knowledge_docs
from .ingestion.gov_pages import ingest_gov_pages
from .ingestion.health_stackexchange import fetch_health_stackexchange_dialogues
from .ingestion.medquad_public_qa import ingest_medquad_public_qa
from .ingestion.medlineplus_connect import ingest_medlineplus_connect
from .ingestion.open_medqa_candidates import evaluate_open_medqa_candidates
from .ingestion.synthetic_dialogues import generate_synthetic_dialogues
from .normalize import dedupe_dialogues, dedupe_knowledge_docs, intent_distribution, urgency_distribution
from .ontology import build_symptom_condition_graph, load_routing_rules, load_triage_rules
from .source_registry import approved_sources, load_source_registry, source_lookup
from .types import SourceRecord
from .utils import ensure_dir, read_json, write_json, write_jsonl
from .validation import validate_all


@dataclass
class Step1Config:
    project_root: str
    max_medlineplus_codes: int = 100
    stackexchange_pages: int = 10
    stackexchange_page_size: int = 100
    synthetic_dialogue_count: int = 50000


def _paths(project_root: str) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "root": root,
        "config": root / "configs",
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "release_kb": root / "outputs" / "phase1" / "kb_release_v1",
        "release_val": root / "outputs" / "phase1" / "validation_release_v1",
    }


def _build_license_manifest(sources: list[SourceRecord], candidate_status: dict) -> dict:
    return {
        "approved_sources": [
            {
                "source_id": s.source_id,
                "name": s.name,
                "owner": s.owner,
                "license_type": s.license_type,
                "allowed_usage": s.allowed_usage.value,
                "attribution_required": s.attribution_required,
                "license_evidence_url": s.license_evidence_url,
                "refresh_cadence": s.refresh_cadence,
            }
            for s in sources
        ],
        "quarantined_candidates": candidate_status.get("quarantined", []),
        "admitted_candidates": candidate_status.get("admitted", []),
    }


def _issue_counts_by_gate(issues: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for issue in issues:
        gate = issue.get("gate", "unknown")
        sev = issue.get("severity", "unknown")
        if gate not in out:
            out[gate] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        if sev in out[gate]:
            out[gate][sev] += 1
    return out


def run_phase1_pipeline(cfg: Step1Config) -> dict:
    p = _paths(cfg.project_root)
    for key in ["raw", "processed", "release_kb", "release_val"]:
        ensure_dir(p[key])

    source_registry_path = p["config"] / "source_registry.json"
    icd_path = p["config"] / "icd10_seed_codes.json"
    gov_seed_path = p["config"] / "gov_seed_urls.json"
    triage_path = p["config"] / "triage_seed_rules.json"
    routing_path = p["config"] / "routing_map.json"

    source_records = load_source_registry(str(source_registry_path))
    source_map = source_lookup(source_records)
    approved = approved_sources(source_records)

    icd_codes = read_json(icd_path)
    gov_seeds = read_json(gov_seed_path)
    triage_rules = load_triage_rules(read_json(triage_path))
    routing_rules = load_routing_rules(read_json(routing_path))

    candidate_status = evaluate_open_medqa_candidates()

    # Ingestion
    med_docs = ingest_medlineplus_connect(
        source=source_map["medlineplus_connect"],
        icd_codes=icd_codes,
        max_codes=cfg.max_medlineplus_codes,
    )
    gov_docs = ingest_gov_pages(source_lookup=source_map, seed_urls=gov_seeds)
    medquad_docs: list = []
    medquad_dialogues: list = []
    if "medquad_public_qa" in source_map:
        medquad_docs, medquad_dialogues = ingest_medquad_public_qa(
            source=source_map["medquad_public_qa"],
            raw_dir=p["raw"],
        )
    knowledge_docs = med_docs + gov_docs + medquad_docs

    hse_dialogues = fetch_health_stackexchange_dialogues(
        source=source_map["health_stackexchange"],
        pages=cfg.stackexchange_pages,
        page_size=cfg.stackexchange_page_size,
    )
    synthetic_dialogues = generate_synthetic_dialogues(total_samples=cfg.synthetic_dialogue_count)
    dialogues = hse_dialogues + medquad_dialogues + synthetic_dialogues

    # Normalize and deduplicate
    dedup_docs, removed_docs = dedupe_knowledge_docs(knowledge_docs)
    dedup_dialogues, removed_dialogues = dedupe_dialogues(dialogues)

    chunks = chunk_knowledge_docs(dedup_docs)
    graph = build_symptom_condition_graph(triage_rules, chunks)

    dedup_stats = {
        "dedup_removed_docs": removed_docs,
        "dedup_removed_dialogues": removed_dialogues,
    }

    docs_dict = [doc.model_dump() for doc in dedup_docs]
    chunks_dict = [chunk.model_dump() for chunk in chunks]
    dialogues_dict = [sample.model_dump() for sample in dedup_dialogues]
    triage_dict = [rule.model_dump() for rule in triage_rules]
    routing_dict = [rule.model_dump() for rule in routing_rules]

    report, review_queue = validate_all(
        source_lookup=source_map,
        knowledge_docs=docs_dict,
        knowledge_chunks=chunks_dict,
        triage_rules=triage_rules,
        dialogues=dedup_dialogues,
        candidate_status=candidate_status,
        dedup_stats=dedup_stats,
    )
    report_dict = report.model_dump()

    license_manifest = _build_license_manifest(approved, candidate_status)

    summary = {
        "knowledge_documents": len(docs_dict),
        "knowledge_chunks": len(chunks_dict),
        "conversation_corpus": len(dialogues_dict),
        "triage_rules": len(triage_dict),
        "routing_rules": len(routing_dict),
        "intent_distribution": intent_distribution(dedup_dialogues),
        "urgency_distribution": urgency_distribution(dedup_dialogues),
        "dedup_stats": dedup_stats,
        "validation_pass_fail": report_dict["pass_fail"],
        "issue_counts_by_gate": _issue_counts_by_gate([i for i in report_dict["issues"]]),
    }

    # Raw artifacts
    write_json(p["raw"] / "source_registry.snapshot.json", [s.model_dump() for s in source_records])
    write_json(p["raw"] / "open_medqa_candidate_status.json", candidate_status)

    # Processed artifacts
    write_jsonl(p["processed"] / "knowledge_documents.jsonl", docs_dict)
    write_jsonl(p["processed"] / "knowledge_chunks.jsonl", chunks_dict)
    write_jsonl(p["processed"] / "conversation_corpus.jsonl", dialogues_dict)
    write_json(p["processed"] / "symptom_condition_graph.json", graph)
    write_json(p["processed"] / "triage_policy_table.json", triage_dict)
    write_json(p["processed"] / "routing_map.json", routing_dict)

    # KB release package
    write_json(p["release_kb"] / "source_registry.json", [s.model_dump() for s in source_records])
    write_json(p["release_kb"] / "license_manifest.json", license_manifest)
    write_jsonl(p["release_kb"] / "knowledge_documents.jsonl", docs_dict)
    write_jsonl(p["release_kb"] / "knowledge_chunks.jsonl", chunks_dict)
    write_json(p["release_kb"] / "symptom_condition_graph.json", graph)
    write_json(p["release_kb"] / "triage_policy_table.json", triage_dict)
    write_json(p["release_kb"] / "routing_map.json", routing_dict)
    write_jsonl(p["release_kb"] / "conversation_corpus.jsonl", dialogues_dict)
    write_json(p["release_kb"] / "dataset_summary.json", summary)

    # Validation release package
    issues = report_dict["issues"]
    write_json(p["release_val"] / "validation_report.json", report_dict)
    write_jsonl(p["release_val"] / "human_review_queue.jsonl", review_queue)
    write_json(
        p["release_val"] / "legal_report.json",
        {
            "gate": "gate0_legal",
            "pass": report_dict["pass_fail"].get("gate0_legal", False),
            "issues": [i for i in issues if i.get("gate") == "gate0_legal"],
            "license_manifest_ref": "../kb_release_v1/license_manifest.json",
        },
    )
    write_json(
        p["release_val"] / "quality_report.json",
        {
            "gate": "gate1_integrity",
            "pass": report_dict["pass_fail"].get("gate1_integrity", False),
            "issues": [i for i in issues if i.get("gate") == "gate1_integrity"],
            "metrics": report_dict["metrics"].get("gate1_integrity", {}),
        },
    )
    write_json(
        p["release_val"] / "safety_report.json",
        {
            "gate": "gate3_safety",
            "pass": report_dict["pass_fail"].get("gate3_safety", False),
            "issues": [i for i in issues if i.get("gate") == "gate3_safety"],
            "metrics": report_dict["metrics"].get("gate3_safety", {}),
        },
    )

    review_template = """# Human Review Notes (Gate 4)\n\nUse this checklist for sampled records from human_review_queue.jsonl:\n\n- [ ] Clinical appropriateness\n- [ ] Urgency correctness\n- [ ] Routing appropriateness\n- [ ] Language safety\n- [ ] Critical finding? (if yes, block release)\n\n## Reviewer Log\n\n| sample_id | reviewer | finding | severity | action_required |\n|---|---|---|---|---|\n| | | | | |\n"""
    (p["release_val"] / "review_notes_template.md").write_text(review_template, encoding="utf-8")

    return {
        "paths": {k: str(v) for k, v in p.items()},
        "summary": summary,
    }


# Backward-compatible alias while the repo transitions from old naming.
run_step1_pipeline = run_phase1_pipeline
