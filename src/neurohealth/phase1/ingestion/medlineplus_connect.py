from __future__ import annotations

import logging
from typing import Any

import requests

from ..types import AgeGroup, KnowledgeDoc, SourceRecord, utc_now_iso
from ..utils import normalize_whitespace, slugify, stable_hash

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://connect.medlineplus.gov/service"


def _extract_text(field: Any) -> str:
    if isinstance(field, dict):
        if "_value" in field:
            return str(field.get("_value", ""))
        return normalize_whitespace(" ".join(str(v) for v in field.values()))
    return normalize_whitespace(str(field or ""))


def _extract_links(entry: dict[str, Any]) -> list[str]:
    links = entry.get("link", [])
    if isinstance(links, dict):
        links = [links]
    out = []
    for link in links:
        if not isinstance(link, dict):
            continue
        href = link.get("href") or link.get("_href")
        if href:
            out.append(str(href))
    return out


def ingest_medlineplus_connect(
    source: SourceRecord,
    icd_codes: list[dict[str, str]],
    max_codes: int = 50,
    timeout: int = 20,
) -> list[KnowledgeDoc]:
    docs: list[KnowledgeDoc] = []
    for item in icd_codes[:max_codes]:
        code = item.get("code", "").strip()
        label = item.get("label", "").strip()
        if not code:
            continue

        params = {
            "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.90",
            "mainSearchCriteria.v.c": code,
            "knowledgeResponseType": "application/json",
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("MedlinePlus request failed for %s: %s", code, exc)
            continue

        entries = ((payload.get("feed") or {}).get("entry")) or []
        if isinstance(entries, dict):
            entries = [entries]

        for idx, entry in enumerate(entries):
            title = _extract_text(entry.get("title")) or f"MedlinePlus Topic {code}"
            summary = _extract_text(entry.get("summary"))
            links = _extract_links(entry)
            citation_url = links[0] if links else "https://medlineplus.gov/"
            body = normalize_whitespace(f"ICD-10: {code} ({label}). {summary}")
            if len(body) < 80:
                continue

            doc_id = f"medlineplus-{code.lower()}-{idx}-{slugify(title)}"
            docs.append(
                KnowledgeDoc(
                    doc_id=doc_id,
                    source_id=source.source_id,
                    title=title,
                    raw_text=body,
                    publication_date=None,
                    license_type=source.license_type,
                    citation_url=citation_url,
                    retrieved_at_utc=utc_now_iso(),
                    language="en",
                    audience_age_groups=[
                        AgeGroup.INFANT,
                        AgeGroup.CHILD,
                        AgeGroup.ADOLESCENT,
                        AgeGroup.ADULT,
                        AgeGroup.OLDER_ADULT,
                    ],
                    content_hash=stable_hash(body),
                )
            )
    return docs
