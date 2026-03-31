from __future__ import annotations

from neurohealth.backend.schemas import ChatMessage, IntakeProfile, NearbyProvider
from neurohealth.backend.service import BackendService
from neurohealth.phase1.types import AgeGroup, UrgencyLevel


def _make_service(monkeypatch) -> BackendService:
    monkeypatch.setattr(BackendService, "_load_processed_artifacts", lambda self: None)
    service = BackendService()
    service.llm_enabled = False
    return service


def test_generate_nearby_providers_prefers_live_results(monkeypatch):
    service = _make_service(monkeypatch)
    live_provider = NearbyProvider(
        name="Apollo Hospital",
        distance_km=2.4,
        estimated_wait_minutes=18,
        care_level="urgent_care",
        specialty="general_medicine",
        location_hint="Jubilee Hills, Hyderabad",
    )
    monkeypatch.setattr(
        service,
        "_fetch_live_nearby_providers",
        lambda location, urgency, recommended_care_level=None, specialty=None: [live_provider],
    )

    response = service.generate_nearby_providers(
        location="Hyderabad 500081",
        urgency=UrgencyLevel.URGENT,
        recommended_care_level="urgent_care",
        specialty="general_medicine",
    )

    assert response.providers == [live_provider]


def test_generate_nearby_providers_falls_back_when_live_lookup_empty(monkeypatch):
    service = _make_service(monkeypatch)
    monkeypatch.setattr(
        service,
        "_fetch_live_nearby_providers",
        lambda location, urgency, recommended_care_level=None, specialty=None: [],
    )

    response = service.generate_nearby_providers(
        location="Hyderabad 500081",
        urgency=UrgencyLevel.URGENT,
        recommended_care_level="urgent_care",
        specialty="general_medicine",
    )

    assert len(response.providers) == 3
    assert response.providers[0].name.startswith("Rapid Urgent Care Clinic")
    assert response.providers[0].location_hint == "Hyderabad 500081"


def test_location_reply_updates_intake_and_returns_provider_lookup(monkeypatch):
    service = _make_service(monkeypatch)
    live_provider = NearbyProvider(
        name="Apollo Hospital",
        distance_km=2.4,
        estimated_wait_minutes=18,
        care_level="urgent_care",
        specialty="general_medicine",
        location_hint="Jubilee Hills, Hyderabad",
        maps_url="https://www.google.com/maps/search/?api=1&query=Apollo+Hospital+Jubilee+Hills+Hyderabad",
    )
    monkeypatch.setattr(
        service,
        "_fetch_live_nearby_providers",
        lambda location, urgency, recommended_care_level=None, specialty=None: [live_provider],
    )

    created = service.create_session(
        IntakeProfile(
            age_group=AgeGroup.ADULT,
            symptom_category="chest_pain_breathing",
            duration="few_hours",
            severity="moderate",
            conditions="hypertension",
        )
    )
    state = service.sessions[created.session_id]
    state.chat.append(
        ChatMessage(
            role="assistant",
            text="Please share your city or postal code so I can suggest nearby hospitals or clinics.",
            created_at_utc="2026-03-26T10:00:00Z",
        )
    )

    reply = service.reply_to_message(
        session_id=created.session_id,
        text="I am in Hyderabad Madhapur",
        include_providers=True,
        message_kind="chat",
    )

    assert reply is not None
    assert reply.response_mode == "provider_lookup"
    assert reply.intake is not None
    assert reply.intake.location == "Hyderabad Madhapur"
    assert reply.nearby_providers[0].maps_url is not None
    assert "Open in Google Maps" in reply.assistant_message.text
    assert reply.citations == []
    assert reply.dataset_files_used == []


def test_provider_query_does_not_get_saved_as_location(monkeypatch):
    service = _make_service(monkeypatch)
    created = service.create_session(
        IntakeProfile(
            age_group=AgeGroup.ADULT,
            symptom_category="respiratory_distress_moderate",
            duration="few_hours",
            severity="moderate",
        )
    )

    reply = service.reply_to_message(
        session_id=created.session_id,
        text="can you give me near by hospitals for this",
        include_providers=True,
        message_kind="chat",
    )

    assert reply is not None
    assert reply.response_mode == "location_request"
    assert reply.intake is not None
    assert reply.intake.location is None
    assert "city or postal code" in reply.assistant_message.text.lower()


def test_invalid_saved_location_is_cleared_before_provider_lookup(monkeypatch):
    service = _make_service(monkeypatch)
    created = service.create_session(
        IntakeProfile(
            age_group=AgeGroup.ADULT,
            location="by hospitals for this",
            symptom_category="respiratory_distress_moderate",
            duration="few_hours",
            severity="severe",
        )
    )

    reply = service.reply_to_message(
        session_id=created.session_id,
        text="can you show nearby hospitals",
        include_providers=True,
        message_kind="chat",
    )

    assert reply is not None
    assert reply.response_mode == "location_request"
    assert reply.intake is not None
    assert reply.intake.location is None


def test_llm_intent_router_is_used_before_fallback(monkeypatch):
    service = _make_service(monkeypatch)
    service.llm_enabled = True
    service.llm_provider = "ollama"
    monkeypatch.setattr(
        service,
        "_call_ollama_chat",
        lambda messages, temperature=0.1, num_predict=220: "provider_lookup",
    )
    monkeypatch.setattr(service, "_is_provider_location_query", lambda text: False)

    intent = service._classify_request_intent(
        user_text="please open maps for the closest clinic",
        history=[],
        intake=None,
        message_kind="chat",
    )

    assert intent == "provider_lookup"


def test_dynamic_missing_fields_use_message_context_without_intake(monkeypatch):
    service = _make_service(monkeypatch)
    created = service.create_session()

    reply = service.reply_to_message(
        session_id=created.session_id,
        text="My 8 year old child has fever since this morning and it is severe.",
        include_providers=True,
        message_kind="chat",
    )

    assert reply is not None
    assert reply.triage.missing_fields == []
    assert reply.triage.symptom_cluster == "pediatric_fever"


def test_location_reply_without_saved_intake_returns_provider_lookup(monkeypatch):
    service = _make_service(monkeypatch)
    live_provider = NearbyProvider(
        name="Apollo Hospital",
        distance_km=2.4,
        estimated_wait_minutes=18,
        care_level="urgent_care",
        specialty="general_medicine",
        location_hint="Jubilee Hills, Hyderabad",
        maps_url="https://www.google.com/maps/search/?api=1&query=Apollo+Hospital+Jubilee+Hills+Hyderabad",
    )
    monkeypatch.setattr(
        service,
        "_fetch_live_nearby_providers",
        lambda location, urgency, recommended_care_level=None, specialty=None: [live_provider],
    )

    created = service.create_session()
    first_reply = service.reply_to_message(
        session_id=created.session_id,
        text="I have severe wheezing since this morning. What should I do?",
        include_providers=True,
        message_kind="chat",
    )
    assert first_reply is not None

    location_prompt = service.reply_to_message(
        session_id=created.session_id,
        text="Can you show nearby hospitals?",
        include_providers=True,
        message_kind="chat",
    )
    assert location_prompt is not None
    assert location_prompt.response_mode == "location_request"

    provider_reply = service.reply_to_message(
        session_id=created.session_id,
        text="Hyderabad Madhapur",
        include_providers=True,
        message_kind="chat",
    )

    assert provider_reply is not None
    assert provider_reply.response_mode == "provider_lookup"
    assert provider_reply.intake is None
    assert provider_reply.nearby_providers[0].maps_url is not None
    assert "Open in Google Maps" in provider_reply.assistant_message.text


def test_fallback_clinical_reply_uses_remedy_steps_not_evidence_basis(monkeypatch):
    service = _make_service(monkeypatch)
    created = service.create_session()

    reply = service.reply_to_message(
        session_id=created.session_id,
        text="I have severe wheezing since this morning. What should I do?",
        include_providers=True,
        message_kind="chat",
    )

    assert reply is not None
    assert "Remedy steps" in reply.assistant_message.text or "Remedy Steps" in reply.assistant_message.text
    assert "Evidence Basis" not in reply.assistant_message.text
    assert "Evidence reference" not in reply.assistant_message.text
