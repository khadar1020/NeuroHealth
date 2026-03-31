from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import requests

from .health_stackexchange import _infer_age_group, _infer_intent, _infer_urgency
from .synthetic_dialogues import _routing_specialty
from ..types import AgeGroup, DialogueSample, DialogueTurn, KnowledgeDoc, SourceRecord, utc_now_iso
from ..utils import ensure_dir, normalize_whitespace, slugify, stable_hash

LOGGER = logging.getLogger(__name__)

ARCHIVE_URL = "https://codeload.github.com/abachaa/MedQuAD/zip/refs/heads/master"
EXCLUDED_SUBSETS = {
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
}


def _download_archive(cache_path: Path, timeout: int = 60) -> Path:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    ensure_dir(cache_path.parent)
    resp = requests.get(ARCHIVE_URL, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return cache_path


def _itertext(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return normalize_whitespace(" ".join(text for text in elem.itertext()))


def _safe_age_tags(text: str) -> list[AgeGroup]:
    age = _infer_age_group(text)
    if age == AgeGroup.ADULT:
        return [
            AgeGroup.INFANT,
            AgeGroup.CHILD,
            AgeGroup.ADOLESCENT,
            AgeGroup.ADULT,
            AgeGroup.OLDER_ADULT,
        ]
    return [age]


def _parse_medquad_document(
    xml_bytes: bytes,
    source: SourceRecord,
    subset_name: str,
) -> tuple[KnowledgeDoc | None, list[DialogueSample]]:
    root = ET.fromstring(xml_bytes)
    doc_id = root.attrib.get("id", "")
    citation_url = root.attrib.get("url") or source.url
    source_label = root.attrib.get("source", subset_name)
    focus = _itertext(root.find("Focus")) or f"{source_label} health topic"

    knowledge_sections: list[str] = []
    dialogues: list[DialogueSample] = []

    for pair in root.findall(".//QAPair"):
        question_el = pair.find("Question")
        answer_el = pair.find("Answer")
        question = _itertext(question_el)
        answer = _itertext(answer_el)
        if len(question) < 15 or len(answer) < 30:
            continue

        qid = (question_el.attrib.get("qid") if question_el is not None else None) or f"{doc_id}-{pair.attrib.get('pid', '0')}"
        qtype = (question_el.attrib.get("qtype") if question_el is not None else None) or "information"
        combined = normalize_whitespace(f"{focus}. {question} {answer}")

        dialogues.append(
            DialogueSample(
                sample_id=f"medquad-{doc_id}-{slugify(qid)}",
                turns=[
                    DialogueTurn(role="user", text=question),
                    DialogueTurn(role="assistant", text=answer),
                ],
                intent_class=_infer_intent(combined, [qtype, source_label.lower(), focus.lower()]),
                urgency_label=_infer_urgency(combined),
                safe_response_label="attributed_for_review",
                age_group=_infer_age_group(combined),
                routing_specialty=_routing_specialty(_infer_intent(combined, [qtype, focus.lower()]), _infer_urgency(combined), combined),
                evidence_ids=[f"medquad-doc-{doc_id}"],
                provenance={
                    "source_id": source.source_id,
                    "document_id": doc_id,
                    "document_source": source_label,
                    "document_url": citation_url,
                    "focus": focus,
                    "qtype": qtype,
                    "license": source.license_type,
                    "attribution_required": True,
                },
            )
        )
        knowledge_sections.append(f"Question: {question} Answer: {answer}")

    if not knowledge_sections:
        return None, dialogues

    body = normalize_whitespace(f"Topic: {focus}. Source: {source_label}. " + " ".join(knowledge_sections))
    knowledge_doc = KnowledgeDoc(
        doc_id=f"medquad-doc-{doc_id}-{slugify(focus)}",
        source_id=source.source_id,
        title=focus,
        raw_text=body,
        publication_date=None,
        license_type=source.license_type,
        citation_url=citation_url,
        retrieved_at_utc=utc_now_iso(),
        language="en",
        audience_age_groups=_safe_age_tags(body),
        content_hash=stable_hash(body),
    )
    return knowledge_doc, dialogues


def ingest_medquad_public_qa(
    source: SourceRecord,
    raw_dir: str | Path,
    timeout: int = 60,
) -> tuple[list[KnowledgeDoc], list[DialogueSample]]:
    cache_path = Path(raw_dir) / "medquad_master.zip"
    try:
        archive_path = _download_archive(cache_path, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to download MedQuAD archive: %s", exc)
        return [], []

    docs: list[KnowledgeDoc] = []
    dialogues: list[DialogueSample] = []

    try:
        with zipfile.ZipFile(archive_path) as zf:
            members = [name for name in zf.namelist() if name.endswith(".xml")]
            for member in members:
                parts = member.split("/")
                if len(parts) < 3:
                    continue
                subset_name = parts[1]
                if subset_name in EXCLUDED_SUBSETS:
                    continue

                try:
                    knowledge_doc, qa_dialogues = _parse_medquad_document(zf.read(member), source, subset_name)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to parse MedQuAD member %s: %s", member, exc)
                    continue

                if knowledge_doc is not None:
                    docs.append(knowledge_doc)
                dialogues.extend(qa_dialogues)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read MedQuAD archive: %s", exc)
        return [], []

    return docs, dialogues
