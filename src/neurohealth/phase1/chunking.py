from __future__ import annotations

import re

from .types import AgeGroup, KnowledgeChunk, KnowledgeDoc


TOPIC_KEYWORDS = {
    "cardiology": ["chest pain", "heart", "cardiac", "angina", "myocardial"],
    "neurology": ["stroke", "seizure", "dizziness", "weakness"],
    "respiratory": ["cough", "breathing", "asthma", "wheezing", "pneumonia"],
    "infectious_disease": ["fever", "infection", "viral", "covid", "flu"],
    "mental_health": ["anxiety", "depression", "suicid", "mental health"],
    "pediatrics": ["infant", "child", "newborn", "pediatric", "toddler"],
    "endocrinology": ["diabetes", "blood sugar", "insulin"],
    "dermatology": ["rash", "skin", "itch", "hives"],
}


def _extract_topics(text: str) -> list[str]:
    low = text.lower()
    topics: list[str] = []
    for topic, keys in TOPIC_KEYWORDS.items():
        if any(k in low for k in keys):
            topics.append(topic)
    return topics or ["general_health"]


def _extract_age_tags(text: str) -> list[AgeGroup]:
    low = text.lower()
    tags: list[AgeGroup] = []
    if "infant" in low or "newborn" in low:
        tags.append(AgeGroup.INFANT)
    if "child" in low or "toddler" in low or "pediatric" in low:
        tags.append(AgeGroup.CHILD)
    if "teen" in low or "adolescent" in low:
        tags.append(AgeGroup.ADOLESCENT)
    if "older adult" in low or "elderly" in low:
        tags.append(AgeGroup.OLDER_ADULT)
    if not tags:
        tags = [AgeGroup.ADULT]
    return tags


def chunk_knowledge_docs(
    docs: list[KnowledgeDoc],
    target_chars: int = 850,
    overlap_sentences: int = 1,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []

    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.raw_text)
        sentences = [s.strip() for s in sentences if s and s.strip()]

        current: list[str] = []
        current_len = 0
        chunk_index = 0

        def flush_chunk():
            nonlocal current, current_len, chunk_index
            if not current:
                return
            text = " ".join(current).strip()
            if len(text) < 50:
                current, current_len = [], 0
                return
            chunk_id = f"{doc.doc_id}-chunk-{chunk_index:04d}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_id=doc.source_id,
                    chunk_index=chunk_index,
                    chunk_text=text,
                    medical_topics=_extract_topics(text),
                    age_applicability=_extract_age_tags(text),
                    citation_id=f"{doc.source_id}:{doc.doc_id}",
                )
            )
            chunk_index += 1
            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                current_len = sum(len(x) for x in current)
            else:
                current, current_len = [], 0

        for sentence in sentences:
            if current_len + len(sentence) > target_chars and current:
                flush_chunk()
            current.append(sentence)
            current_len += len(sentence)

        flush_chunk()

    return chunks
