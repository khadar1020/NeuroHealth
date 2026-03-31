from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from neurohealth.phase1.types import AgeGroup, UrgencyLevel

from .schemas import (
    ChatReplyResponse,
    ConfigResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IntakeProfile,
    IntakeUpdateResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    MessageCreateRequest,
    NearbyProvidersRequest,
    NearbyProvidersResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStateResponse,
)
from .service import BackendService


@lru_cache(maxsize=1)
def get_backend_service() -> BackendService:
    return BackendService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm up data and indexes on startup.
    get_backend_service()
    yield


app = FastAPI(
    title="NeuroHealth Backend API",
    description=(
        "API for intake capture, conversational triage, care routing, "
        "knowledge retrieval, and nearby provider suggestions."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "name": "NeuroHealth Backend API",
        "docs": "/docs",
        "status": "ok",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: BackendService = Depends(get_backend_service)) -> HealthResponse:
    return service.health()


@app.get("/config", response_model=ConfigResponse, tags=["system"])
def config(service: BackendService = Depends(get_backend_service)) -> ConfigResponse:
    return service.config()


@app.get("/triage/rules", response_model=list[dict[str, Any]], tags=["policies"])
def triage_rules(
    age_group: AgeGroup | None = Query(default=None),
    service: BackendService = Depends(get_backend_service),
) -> list[dict[str, Any]]:
    return service.list_triage_rules(age_group=age_group)


@app.get("/routing/rules", response_model=list[dict[str, Any]], tags=["policies"])
def routing_rules(
    urgency: UrgencyLevel | None = Query(default=None),
    service: BackendService = Depends(get_backend_service),
) -> list[dict[str, Any]]:
    return service.list_routing_rules(urgency=urgency)


@app.post("/sessions", response_model=SessionCreateResponse, tags=["sessions"])
def create_session(
    payload: SessionCreateRequest,
    service: BackendService = Depends(get_backend_service),
) -> SessionCreateResponse:
    return service.create_session(intake=payload.intake)


@app.get("/sessions/{session_id}", response_model=SessionStateResponse, tags=["sessions"])
def get_session(
    session_id: str,
    service: BackendService = Depends(get_backend_service),
) -> SessionStateResponse:
    state = service.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@app.patch("/sessions/{session_id}/intake", response_model=IntakeUpdateResponse, tags=["sessions"])
def update_intake(
    session_id: str,
    payload: IntakeProfile,
    service: BackendService = Depends(get_backend_service),
) -> IntakeUpdateResponse:
    updated = service.update_intake(session_id=session_id, intake=payload)
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return updated


@app.post("/sessions/{session_id}/messages", response_model=ChatReplyResponse, tags=["chat"])
def message(
    session_id: str,
    payload: MessageCreateRequest,
    service: BackendService = Depends(get_backend_service),
) -> ChatReplyResponse:
    response = service.reply_to_message(
        session_id=session_id,
        text=payload.text,
        include_providers=payload.include_providers,
        message_kind=payload.message_kind,
    )
    if response is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return response


@app.post("/providers/nearby", response_model=NearbyProvidersResponse, tags=["providers"])
def nearby_providers(
    payload: NearbyProvidersRequest,
    service: BackendService = Depends(get_backend_service),
) -> NearbyProvidersResponse:
    return service.generate_nearby_providers(
        location=payload.location,
        urgency=payload.urgency,
        recommended_care_level=payload.recommended_care_level,
        specialty=payload.specialty,
    )


@app.post("/knowledge/search", response_model=KnowledgeSearchResponse, tags=["knowledge"])
def knowledge_search(
    payload: KnowledgeSearchRequest,
    service: BackendService = Depends(get_backend_service),
) -> KnowledgeSearchResponse:
    return service.search_knowledge(
        query=payload.query,
        top_k=payload.top_k,
        method=payload.method,
        age_group=payload.age_group,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def feedback(
    payload: FeedbackRequest,
    service: BackendService = Depends(get_backend_service),
) -> FeedbackResponse:
    if service.get_session(payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return service.submit_feedback(payload)
