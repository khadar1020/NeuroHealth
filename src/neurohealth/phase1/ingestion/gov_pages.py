from __future__ import annotations

import logging
from typing import Any

import requests
from bs4 import BeautifulSoup

from ..types import AgeGroup, KnowledgeDoc, SourceRecord, utc_now_iso
from ..utils import normalize_whitespace, slugify, stable_hash

LOGGER = logging.getLogger(__name__)


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for selector in ["script", "style", "noscript", "nav", "footer", "header"]:
        for tag in soup.select(selector):
            tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return ""

    texts = [normalize_whitespace(t.get_text(" ", strip=True)) for t in main.find_all(["h1", "h2", "h3", "p", "li"])]
    texts = [t for t in texts if t]
    return normalize_whitespace(" ".join(texts))


def ingest_gov_pages(
    source_lookup: dict[str, SourceRecord],
    seed_urls: list[dict[str, str]],
    timeout: int = 20,
) -> list[KnowledgeDoc]:
    docs: list[KnowledgeDoc] = []

    for item in seed_urls:
        source_id = item.get("source_id", "")
        source = source_lookup.get(source_id)
        if source is None:
            continue
        url = item.get("url", "")
        topic = item.get("topic", "General health")
        if not url:
            continue

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "NeuroHealth-Step1/1.0"},
                timeout=timeout,
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to fetch gov page %s: %s", url, exc)
            continue

        text = _extract_text_from_html(html)
        if len(text) < 120:
            continue

        title = topic
        doc_id = f"{source_id}-{slugify(topic)}-{slugify(url)}"
        docs.append(
            KnowledgeDoc(
                doc_id=doc_id,
                source_id=source_id,
                title=title,
                raw_text=text,
                publication_date=None,
                license_type=source.license_type,
                citation_url=url,
                retrieved_at_utc=utc_now_iso(),
                language="en",
                audience_age_groups=[
                    AgeGroup.INFANT,
                    AgeGroup.CHILD,
                    AgeGroup.ADOLESCENT,
                    AgeGroup.ADULT,
                    AgeGroup.OLDER_ADULT,
                ],
                content_hash=stable_hash(text),
            )
        )

    return docs
