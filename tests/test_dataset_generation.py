from __future__ import annotations

from neurohealth.phase1.ingestion.health_stackexchange import _pick_best_answer
from neurohealth.phase1.ingestion.medquad_public_qa import _parse_medquad_document
from neurohealth.phase1.ingestion.synthetic_dialogues import (
    _routing_specialty,
    generate_synthetic_dialogues,
)
from neurohealth.phase1.types import IntentClass, SourceRecord, UrgencyLevel


def test_routing_specialty_does_not_confuse_asking_with_skin_terms():
    text = (
        "I am asking for me: moderate fatigue with congestion for two days while at work. "
        "I have no major medical history. What should I do next?"
    )
    assert _routing_specialty(IntentClass.SYMPTOM_CHECK, UrgencyLevel.ROUTINE, text) == "primary_care"


def test_pick_best_answer_falls_back_to_highest_scored_answer():
    question = {"question_id": 42}
    answers = [
        {
            "answer_id": 100,
            "question_id": 42,
            "score": 2,
            "body": "<p>This is short but still long enough to count as an answer for testing.</p>",
            "owner": {"display_name": "Lower Score"},
        },
        {
            "answer_id": 101,
            "question_id": 42,
            "score": 8,
            "body": "<p>This answer should win because it has the highest score and enough content to be selected.</p>",
            "owner": {"display_name": "Higher Score"},
        },
    ]

    answer_text, answer_owner = _pick_best_answer(question, answers)

    assert "highest score" in answer_text
    assert answer_owner == "Higher Score"


def test_synthetic_generator_produces_more_diverse_routing_labels():
    samples = generate_synthetic_dialogues(total_samples=200, seed=7)
    labels = {sample.routing_specialty for sample in samples}

    assert len(labels) >= 6
    assert "dermatology" in labels
    assert "primary_care" in labels


def test_parse_medquad_document_produces_doc_and_dialogues():
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<Document id="0000001" source="CDC" url="https://example.gov/topic">
  <Focus>Asthma</Focus>
  <QAPairs>
    <QAPair pid="1">
      <Question qid="0000001-1" qtype="information">What is asthma?</Question>
      <Answer>Asthma is a chronic disease that affects the airways and can make breathing harder.</Answer>
    </QAPair>
    <QAPair pid="2">
      <Question qid="0000001-2" qtype="prevention">How can asthma attacks be prevented?</Question>
      <Answer>People can reduce risk by following their treatment plan and avoiding known triggers.</Answer>
    </QAPair>
  </QAPairs>
</Document>
"""
    source = SourceRecord(
        source_id="medquad_public_qa",
        name="MedQuAD Public QA",
        url="https://github.com/abachaa/MedQuAD",
        owner="Asma Ben Abacha et al.",
        license_type="CC-BY-4.0",
        allowed_usage="redistributable_with_attribution",
        attribution_required=True,
        refresh_cadence="quarterly",
        ingestion_method="github_archive",
        license_evidence_url="https://raw.githubusercontent.com/abachaa/MedQuAD/master/readme.txt",
        status="approved",
        notes="",
    )

    knowledge_doc, dialogues = _parse_medquad_document(xml, source, "9_CDC_QA")

    assert knowledge_doc is not None
    assert knowledge_doc.title == "Asthma"
    assert len(dialogues) == 2
    assert dialogues[0].provenance["source_id"] == "medquad_public_qa"
