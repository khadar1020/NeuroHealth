from __future__ import annotations

from collections import defaultdict

from .types import DialogueSample, KnowledgeDoc


def dedupe_knowledge_docs(docs: list[KnowledgeDoc]) -> tuple[list[KnowledgeDoc], int]:
    seen_hashes: set[str] = set()
    deduped: list[KnowledgeDoc] = []
    removed = 0
    for doc in docs:
        if doc.content_hash in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(doc.content_hash)
        deduped.append(doc)
    return deduped, removed


def dedupe_dialogues(samples: list[DialogueSample]) -> tuple[list[DialogueSample], int]:
    seen: set[str] = set()
    out: list[DialogueSample] = []
    removed = 0
    for sample in samples:
        key = "|".join(turn.text.strip().lower() for turn in sample.turns)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(sample)
    return out, removed


def intent_distribution(samples: list[DialogueSample]) -> dict[str, int]:
    dist: dict[str, int] = defaultdict(int)
    for sample in samples:
        dist[sample.intent_class.value] += 1
    return dict(dist)


def urgency_distribution(samples: list[DialogueSample]) -> dict[str, int]:
    dist: dict[str, int] = defaultdict(int)
    for sample in samples:
        dist[sample.urgency_label.value] += 1
    return dict(dist)
