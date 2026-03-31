from __future__ import annotations

from collections import defaultdict
import html
import logging
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from ..types import AgeGroup, DialogueSample, DialogueTurn, IntentClass, SourceRecord, UrgencyLevel
from ..utils import normalize_whitespace, slugify

LOGGER = logging.getLogger(__name__)
API_BASE = "https://api.stackexchange.com/2.3"

EMERGENCY_HINTS = {
    "chest pain",
    "shortness of breath",
    "stroke",
    "unconscious",
    "anaphylaxis",
    "seizure",
    "suicidal",
}


def _html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw or "", "html.parser")
    text = soup.get_text(" ", strip=True)
    return normalize_whitespace(html.unescape(text))


def _infer_intent(text: str, tags: list[str]) -> IntentClass:
    low = text.lower()
    if "medication" in low or "dose" in low or "drug" in low:
        return IntentClass.MEDICATION_QUESTION
    if "prevent" in low or "vaccine" in low or "screen" in low:
        return IntentClass.PREVENTIVE_CARE
    if "follow up" in low or "chronic" in low or "diabetes" in low:
        return IntentClass.CHRONIC_FOLLOWUP
    if "appointment" in low or "doctor" in low or "specialist" in low:
        return IntentClass.APPOINTMENT_NAVIGATION
    if "symptom" in low or tags:
        return IntentClass.SYMPTOM_CHECK
    return IntentClass.SYMPTOM_CHECK


def _infer_urgency(text: str) -> UrgencyLevel:
    low = text.lower()
    if any(key in low for key in EMERGENCY_HINTS):
        return UrgencyLevel.EMERGENCY
    if "worse" in low or "severe" in low or "high fever" in low:
        return UrgencyLevel.URGENT
    if "persistent" in low:
        return UrgencyLevel.ROUTINE
    return UrgencyLevel.ROUTINE


def _infer_age_group(text: str) -> AgeGroup:
    low = text.lower()
    if "infant" in low or "newborn" in low:
        return AgeGroup.INFANT
    if "toddler" in low or "child" in low:
        return AgeGroup.CHILD
    if "teen" in low or "adolescent" in low:
        return AgeGroup.ADOLESCENT
    if "elderly" in low or "older" in low:
        return AgeGroup.OLDER_ADULT
    return AgeGroup.ADULT


def _pick_best_answer(question: dict[str, Any], answers: list[dict[str, Any]]) -> tuple[str, str | None]:
    accepted_id = str(question.get("accepted_answer_id") or "")
    accepted_answer = None
    if accepted_id:
        accepted_answer = next((ans for ans in answers if str(ans.get("answer_id")) == accepted_id), None)

    ranked_answers = []
    if accepted_answer is not None:
        ranked_answers.append(accepted_answer)

    remaining = [ans for ans in answers if accepted_answer is None or ans is not accepted_answer]
    remaining.sort(key=lambda ans: int(ans.get("score", 0)), reverse=True)
    ranked_answers.extend(remaining)

    for ans in ranked_answers:
        answer_text = _html_to_text(ans.get("body", ""))
        if len(answer_text) >= 30:
            owner = ((ans.get("owner") or {}).get("display_name"))
            return answer_text, owner

    return "", None


def fetch_health_stackexchange_dialogues(
    source: SourceRecord,
    pages: int = 5,
    page_size: int = 100,
    sleep_sec: float = 0.4,
    timeout: int = 20,
) -> list[DialogueSample]:
    samples: list[DialogueSample] = []
    seen_ids: set[str] = set()

    for page in range(1, pages + 1):
        q_params = {
            "order": "desc",
            "sort": "votes",
            "site": "health",
            "filter": "withbody",
            "page": page,
            "pagesize": page_size,
        }
        try:
            q_resp = requests.get(f"{API_BASE}/questions", params=q_params, timeout=timeout)
            q_resp.raise_for_status()
            q_payload = q_resp.json()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("StackExchange questions fetch failed on page %s: %s", page, exc)
            continue

        questions = q_payload.get("items", [])
        if not questions:
            break

        question_ids = [str(q.get("question_id")) for q in questions if q.get("question_id")]
        answers_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)

        if question_ids:
            ids_joined = ";".join(question_ids[:100])
            answer_page = 1
            while True:
                a_params = {
                    "order": "desc",
                    "sort": "votes",
                    "site": "health",
                    "filter": "withbody",
                    "page": answer_page,
                    "pagesize": 100,
                }
                try:
                    a_resp = requests.get(f"{API_BASE}/questions/{ids_joined}/answers", params=a_params, timeout=timeout)
                    a_resp.raise_for_status()
                    a_payload = a_resp.json()
                    for ans in a_payload.get("items", []):
                        qid = str(ans.get("question_id") or "")
                        if qid:
                            answers_by_question[qid].append(ans)
                    if not a_payload.get("has_more"):
                        break
                    answer_page += 1
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("StackExchange answers fetch failed on page %s: %s", page, exc)
                    break

        for q in questions:
            qid = str(q.get("question_id") or "")
            if not qid or qid in seen_ids:
                continue
            seen_ids.add(qid)

            q_title = _html_to_text(q.get("title", ""))
            q_body = _html_to_text(q.get("body", ""))
            question_text = normalize_whitespace(f"{q_title}. {q_body}")
            if len(question_text) < 50:
                continue

            answer_text, answer_owner = _pick_best_answer(q, answers_by_question.get(qid, []))
            if len(answer_text) < 30:
                continue

            tags = [str(t) for t in (q.get("tags") or [])]
            combined = f"{question_text} {answer_text}"
            intent = _infer_intent(combined, tags)
            urgency = _infer_urgency(combined)
            age_group = _infer_age_group(combined)

            question_url = q.get("link") or f"https://health.stackexchange.com/questions/{qid}"
            author = ((q.get("owner") or {}).get("display_name"))

            sample_id = f"hse-{qid}-{slugify(q_title)}"
            samples.append(
                DialogueSample(
                    sample_id=sample_id,
                    turns=[
                        DialogueTurn(role="user", text=question_text),
                        DialogueTurn(role="assistant", text=answer_text),
                    ],
                    intent_class=intent,
                    urgency_label=urgency,
                    safe_response_label="attributed_for_review",
                    age_group=age_group,
                    routing_specialty="primary_care",
                    evidence_ids=[f"hse-question-{qid}"],
                    provenance={
                        "source_id": source.source_id,
                        "question_id": qid,
                        "question_url": question_url,
                        "author": author,
                        "answer_author": answer_owner,
                        "license": source.license_type,
                        "attribution_required": True,
                        "tags": tags,
                    },
                )
            )

        backoff = q_payload.get("backoff")
        if isinstance(backoff, int) and backoff > 0:
            time.sleep(backoff)
        else:
            time.sleep(sleep_sec)

    return samples
