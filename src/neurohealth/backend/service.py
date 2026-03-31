from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from neurohealth.phase1.types import AgeGroup, UrgencyLevel, utc_now_iso

from .schemas import (
    ChatMessage,
    ChatReplyResponse,
    ConfigResponse,
    DatasetStats,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IntakeProfile,
    IntakeUpdateResponse,
    KnowledgeSearchResult,
    KnowledgeSearchResponse,
    NearbyProvider,
    NearbyProvidersResponse,
    SessionCreateResponse,
    SessionStateResponse,
    TriageDecision,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
LOCATION_REPLY_PREFIX_RE = re.compile(
    r"^\s*(i am in|i'm in|im in|located in|my location is|i live in|city is|postal code is|zip code is|zipcode is|at)\s+",
    re.IGNORECASE,
)
HEALTH_QUERY_HINT_RE = re.compile(
    r"(pain|fever|cough|breath|breathing|headache|symptom|vomit|vomiting|nausea|diabetes|blood sugar|wheez|"
    r"dizz|injury|rash|infection|doctor should i see|what should i do|help|treatment|medicine|care)",
    re.IGNORECASE,
)
PROVIDER_LOCATION_QUERY_RE = re.compile(
    r"("
    r"near\s+me|near\s+my|near\s+by|nearby|nearest|closest|google\s+maps?|map\s+link|directions|"
    r"at\s+my\s+location|around\s+me|around\s+here"
    r")",
    re.IGNORECASE,
)
PROVIDER_WORD_RE = re.compile(r"\b(hospital|hospitals|clinic|clinics|doctor|doctors|provider|providers)\b", re.IGNORECASE)
LOCATION_NOISE_RE = re.compile(
    r"(for this|this issue|this problem|my symptoms|current symptoms|nearby hospitals?|nearest hospitals?|"
    r"closest hospitals?|nearby clinics?|nearest clinics?|closest clinics?|my location|current location)",
    re.IGNORECASE,
)
EXPLICIT_AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|year-old|yrs?\s*old|yo|y/o)\b", re.IGNORECASE)
DURATION_HINT_RE = re.compile(
    r"(since (?:this )?(?:morning|afternoon|evening|yesterday|last night)|"
    r"for \d+\s*(?:hour|day|week|month)s?|"
    r"for (?:a|an|one|two|three|four|five|six|seven)\s*(?:hour|day|week|month)s?|"
    r"few hours|few days|one day|two days|week|weeks|month|started (?:today|yesterday))",
    re.IGNORECASE,
)
SEVERITY_HINT_RE = re.compile(
    r"\b(mild|moderate|severe|worse|worsening|persistent|constant|unable to|difficulty breathing|can't|cannot|very bad|serious)\b",
    re.IGNORECASE,
)

DEFAULT_SYMPTOM_CLUSTERS = [
    "anaphylaxis",
    "chest_pain_breathing",
    "common_respiratory",
    "mental_health_crisis",
    "mental_health_non_crisis",
    "metabolic_risk",
    "mild_skin",
    "pediatric_fever",
    "respiratory_distress_moderate",
    "stroke_signs",
]

URGENCY_PRIORITY = {
    UrgencyLevel.EMERGENCY.value: 4,
    UrgencyLevel.URGENT.value: 3,
    UrgencyLevel.ROUTINE.value: 2,
    UrgencyLevel.SELF_CARE.value: 1,
}


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_id: str
    chunk_text: str
    citation_id: str
    age_applicability: set[str]


@dataclass(slots=True)
class GeoPoint:
    lat: float
    lon: float
    label: str


@dataclass(slots=True)
class SessionState:
    session_id: str
    created_at_utc: str
    updated_at_utc: str
    intake: IntakeProfile | None
    chat: list[ChatMessage]
    latest_triage: TriageDecision | None


class BackendService:
    """NeuroHealth backend runtime service.

    This service loads Phase 1 processed artifacts, manages in-memory sessions,
    and executes lightweight triage/routing + retrieval-assisted responses.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or self._resolve_project_root()
        self._load_env_file(self.project_root / ".env")
        self.data_dir = self.project_root / "data" / "processed"
        self.dataset_files_for_chat = [
            "data/processed/triage_policy_table.json",
            "data/processed/routing_map.json",
            "data/processed/knowledge_chunks.jsonl",
            "data/processed/knowledge_documents.jsonl",
        ]

        self.llm_provider = os.getenv("NEUROHEALTH_LLM_PROVIDER", "ollama").strip().lower()
        self.llm_model = os.getenv("NEUROHEALTH_LLM_MODEL", "llama3.2:3b").strip()
        self.llm_enabled = os.getenv("NEUROHEALTH_LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        self.llm_timeout_seconds = float(os.getenv("NEUROHEALTH_LLM_TIMEOUT_SECONDS", "45"))
        self.live_maps_enabled = os.getenv("NEUROHEALTH_LIVE_MAPS_ENABLED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self.maps_provider = os.getenv("NEUROHEALTH_MAPS_PROVIDER", "osm").strip().lower()
        self.nominatim_url = os.getenv(
            "NEUROHEALTH_NOMINATIM_URL",
            "https://nominatim.openstreetmap.org/search",
        ).strip()
        self.overpass_url = os.getenv(
            "NEUROHEALTH_OVERPASS_URL",
            "https://overpass-api.de/api/interpreter",
        ).strip()
        self.maps_request_timeout_seconds = float(os.getenv("NEUROHEALTH_MAPS_TIMEOUT_SECONDS", "12"))
        self.maps_user_agent = os.getenv(
            "NEUROHEALTH_MAPS_USER_AGENT",
            "NeuroHealth/0.1 (clinical-assistant prototype)",
        ).strip()
        self.last_llm_error: str | None = None

        self.triage_rules: list[dict[str, Any]] = []
        self.routing_rules: list[dict[str, Any]] = []
        self.document_index: dict[str, dict[str, str]] = {}
        self.chunks: list[ChunkRecord] = []
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avg_doc_len: float = 1.0
        self.conversation_sample_count = 0

        self.sessions: dict[str, SessionState] = {}
        self.feedback_store: list[dict[str, str]] = []
        self._geocode_cache: dict[str, GeoPoint | None] = {}
        self._provider_lookup_cache: dict[tuple[str, str, str, str], list[NearbyProvider]] = {}

        self._load_processed_artifacts()

    @staticmethod
    def _resolve_project_root() -> Path:
        env_root = os.getenv("NEUROHEALTH_PROJECT_ROOT")
        if env_root:
            return Path(env_root).expanduser().resolve()
        # /.../src/neurohealth/backend/service.py -> project root
        return Path(__file__).resolve().parents[3]

    @staticmethod
    def _load_env_file(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_key = key.strip()
            env_value = value.strip().strip('"').strip("'")
            if env_key and env_key not in os.environ:
                os.environ[env_key] = env_value

    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower().strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _count_jsonl_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else []

    def _load_processed_artifacts(self) -> None:
        triage_path = self.data_dir / "triage_policy_table.json"
        routing_path = self.data_dir / "routing_map.json"
        docs_path = self.data_dir / "knowledge_documents.jsonl"
        chunks_path = self.data_dir / "knowledge_chunks.jsonl"
        conversation_path = self.data_dir / "conversation_corpus.jsonl"

        self.triage_rules = self._load_json(triage_path)
        self.routing_rules = self._load_json(routing_path)
        self.conversation_sample_count = self._count_jsonl_lines(conversation_path)

        self._load_document_index(docs_path)
        self._load_chunk_index(chunks_path)

    def _load_document_index(self, docs_path: Path) -> None:
        if not docs_path.exists():
            return
        with docs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                doc_id = str(payload.get("doc_id", "")).strip()
                if not doc_id:
                    continue
                self.document_index[doc_id] = {
                    "title": str(payload.get("title", "")).strip(),
                    "citation_url": str(payload.get("citation_url", "")).strip(),
                    "source_id": str(payload.get("source_id", "")).strip(),
                }

    def _load_chunk_index(self, chunks_path: Path) -> None:
        if not chunks_path.exists():
            return

        doc_frequency: Counter[str] = Counter()

        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                chunk_text = str(payload.get("chunk_text", "")).strip()
                tokens = self._tokenize(chunk_text)
                term_freq = Counter(tokens)
                chunk_idx = len(self.chunks)

                self.chunks.append(
                    ChunkRecord(
                        chunk_id=str(payload.get("chunk_id", f"chunk-{chunk_idx:06d}")),
                        doc_id=str(payload.get("doc_id", "")),
                        source_id=str(payload.get("source_id", "")),
                        chunk_text=chunk_text,
                        citation_id=str(payload.get("citation_id", "")),
                        age_applicability={
                            str(item).strip()
                            for item in payload.get("age_applicability", [])
                            if str(item).strip()
                        },
                    )
                )
                self.doc_lengths.append(max(1, len(tokens)))

                for token, frequency in term_freq.items():
                    self.postings[token].append((chunk_idx, frequency))
                doc_frequency.update(term_freq.keys())

        total_docs = max(1, len(self.chunks))
        self.avg_doc_len = sum(self.doc_lengths) / total_docs

        for token, df in doc_frequency.items():
            self.idf[token] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

    def dataset_stats(self) -> DatasetStats:
        return DatasetStats(
            triage_rules=len(self.triage_rules),
            routing_rules=len(self.routing_rules),
            knowledge_documents=len(self.document_index),
            knowledge_chunks=len(self.chunks),
            conversation_samples=self.conversation_sample_count,
        )

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            timestamp_utc=utc_now_iso(),
            sessions_active=len(self.sessions),
            feedback_items=len(self.feedback_store),
            dataset_stats=self.dataset_stats(),
            llm_enabled=self.llm_enabled,
            llm_model=self.llm_model if self.llm_enabled else None,
            llm_last_error=self.last_llm_error,
        )

    def config(self) -> ConfigResponse:
        routing_clusters = {str(rule.get("symptom_cluster", "")).strip() for rule in self.routing_rules}
        all_clusters = sorted(cluster for cluster in routing_clusters.union(DEFAULT_SYMPTOM_CLUSTERS) if cluster)
        return ConfigResponse(
            age_groups=[group.value for group in AgeGroup],
            severity_levels=["mild", "moderate", "severe"],
            urgency_levels=[level.value for level in UrgencyLevel],
            symptom_clusters=all_clusters,
            dataset_stats=self.dataset_stats(),
            llm_provider=self.llm_provider if self.llm_enabled else None,
            llm_model=self.llm_model if self.llm_enabled else None,
            llm_enabled=self.llm_enabled,
        )

    def list_triage_rules(self, age_group: AgeGroup | None = None) -> list[dict[str, Any]]:
        if age_group is None:
            return self.triage_rules
        return [rule for rule in self.triage_rules if rule.get("age_group") == age_group.value]

    def list_routing_rules(self, urgency: UrgencyLevel | None = None) -> list[dict[str, Any]]:
        if urgency is None:
            return self.routing_rules
        return [rule for rule in self.routing_rules if rule.get("urgency_level") == urgency.value]

    def create_session(self, intake: IntakeProfile | None = None) -> SessionCreateResponse:
        now = utc_now_iso()
        session_id = uuid4().hex
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            created_at_utc=now,
            updated_at_utc=now,
            intake=intake,
            chat=[],
            latest_triage=None,
        )
        return SessionCreateResponse(
            session_id=session_id,
            created_at_utc=now,
            updated_at_utc=now,
            intake=intake,
        )

    def get_session(self, session_id: str) -> SessionStateResponse | None:
        state = self.sessions.get(session_id)
        if state is None:
            return None
        return self._serialize_session(state)

    def update_intake(self, session_id: str, intake: IntakeProfile) -> IntakeUpdateResponse | None:
        state = self.sessions.get(session_id)
        if state is None:
            return None
        state.intake = intake
        state.updated_at_utc = utc_now_iso()
        return IntakeUpdateResponse(
            session_id=session_id,
            intake=intake,
            updated_at_utc=state.updated_at_utc,
        )

    def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        method: str = "bm25",
        age_group: AgeGroup | None = None,
    ) -> KnowledgeSearchResponse:
        tokens = self._tokenize(query)
        if not tokens:
            return KnowledgeSearchResponse(query=query, method=method, results=[])

        scores: defaultdict[int, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75
        age_value = age_group.value if age_group else None

        for token in tokens:
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf.get(token, 0.0)

            for chunk_idx, tf in postings:
                chunk = self.chunks[chunk_idx]
                if age_value and chunk.age_applicability and age_value not in chunk.age_applicability:
                    continue

                if method == "keyword":
                    scores[chunk_idx] += tf
                    continue

                doc_len = self.doc_lengths[chunk_idx]
                denom = tf + (k1 * (1 - b + b * (doc_len / self.avg_doc_len)))
                score = idf * ((tf * (k1 + 1)) / denom)
                scores[chunk_idx] += score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[KnowledgeSearchResult] = []
        for chunk_idx, score in ranked:
            chunk = self.chunks[chunk_idx]
            doc_meta = self.document_index.get(chunk.doc_id, {})
            snippet = self._build_snippet(chunk.chunk_text, tokens)
            results.append(
                KnowledgeSearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_id=chunk.source_id or doc_meta.get("source_id", ""),
                    score=round(float(score), 4),
                    snippet=snippet,
                    citation_id=chunk.citation_id,
                    title=doc_meta.get("title") or None,
                    citation_url=doc_meta.get("citation_url") or None,
                )
            )

        return KnowledgeSearchResponse(query=query, method=method, results=results)

    def reply_to_message(
        self,
        session_id: str,
        text: str,
        include_providers: bool = True,
        message_kind: str = "chat",
    ) -> ChatReplyResponse | None:
        state = self.sessions.get(session_id)
        if state is None:
            return None

        now = utc_now_iso()
        user_message = ChatMessage(role="user", text=text.strip(), created_at_utc=now)
        state.chat.append(user_message)

        intake = state.intake
        if intake and intake.location:
            sanitized_location = self._normalize_location_candidate(intake.location)
            if sanitized_location != intake.location:
                intake = intake.model_copy(update={"location": sanitized_location or None})
                state.intake = intake
        is_intake_bootstrap = message_kind == "intake_bootstrap"
        request_intent = self._classify_request_intent(
            user_text=text,
            history=state.chat[-8:],
            intake=intake,
            message_kind=message_kind,
        )
        active_location = intake.location if intake and intake.location else None

        if request_intent == "location_answer":
            extracted_location = self._extract_location_from_text(text)
            if extracted_location:
                active_location = extracted_location
                if intake:
                    intake = intake.model_copy(update={"location": extracted_location})
                    state.intake = intake

        if is_intake_bootstrap and intake:
            merged_text = " ".join(
                part
                for part in [
                    intake.symptom_category,
                    intake.duration,
                    intake.severity,
                    intake.conditions if intake.conditions else "",
                ]
                if part
            )
        elif request_intent in {"provider_lookup", "location_answer"} and intake:
            merged_text = " ".join(
                part
                for part in [
                    intake.symptom_category,
                    intake.duration,
                    intake.severity,
                    intake.conditions if intake.conditions else "",
                ]
                if part
            )
        elif request_intent in {"provider_lookup", "location_answer"} and state.latest_triage:
            merged_text = " ".join(
                part
                for part in [
                    state.latest_triage.symptom_cluster.replace("_", " "),
                    state.latest_triage.rationale,
                ]
                if part
            )
        else:
            merged_text = " ".join(
                part for part in [text, intake.symptom_category if intake else "", intake.conditions if intake else ""] if part
            )
        effective_age_group = intake.age_group if intake and intake.age_group else self._infer_age_group_from_text(text)
        matched_rule = self._pick_best_triage_rule(merged_text, effective_age_group)

        if request_intent in {"provider_lookup", "location_answer"} and state.latest_triage and not intake:
            urgency = state.latest_triage.urgency.value
            symptom_cluster = state.latest_triage.symptom_cluster
            recommended_care = state.latest_triage.recommended_care_level
            specialty = state.latest_triage.specialty
            missing_fields = []
            rationale = state.latest_triage.rationale
            evidence_ids = list(state.latest_triage.evidence_ids)
        else:
            urgency = matched_rule.get("urgency_level") if matched_rule else self._map_urgency_from_keywords(merged_text)
            symptom_cluster = self._infer_symptom_cluster(merged_text, intake, urgency, effective_age_group)
            route = self._find_route(urgency, symptom_cluster)
            recommended_care = str(route.get("recommended_care_level", "")) if route else self._default_care(urgency)
            specialty = str(route.get("specialty", "")) if route else self._default_specialty(urgency)
            missing_fields = self._missing_critical_fields(
                intake=intake,
                user_text=text,
                request_intent=request_intent,
                urgency=urgency,
                effective_age_group=effective_age_group,
            )
            rationale = (
                str(matched_rule.get("rationale", "")).strip()
                if matched_rule
                else "Pattern and severity suggest conservative safety-first routing."
            )
            evidence_ids = list(matched_rule.get("evidence_ids", [])) if matched_rule else []
        retrieval = KnowledgeSearchResponse(query=merged_text or text, method="bm25", results=[])
        if request_intent not in {"provider_lookup", "location_answer"}:
            retrieval = self.search_knowledge(
                query=merged_text or text,
                top_k=3,
                method="bm25",
                age_group=effective_age_group,
            )

        providers: list[NearbyProvider] = []
        if include_providers and active_location:
            providers = self.generate_nearby_providers(
                location=active_location,
                urgency=UrgencyLevel(urgency),
                recommended_care_level=recommended_care,
                specialty=specialty,
            ).providers

        triage = TriageDecision(
            urgency=UrgencyLevel(urgency),
            matched_rule_id=str(matched_rule.get("rule_id")) if matched_rule else None,
            rationale=rationale,
            evidence_ids=evidence_ids,
            symptom_cluster=symptom_cluster,
            recommended_care_level=recommended_care,
            specialty=specialty,
            missing_fields=missing_fields,
        )
        assistant_text: str | None = None
        response_mode: str = "rules_fallback"
        dataset_files_used = self.dataset_files_for_chat
        citations = retrieval.results

        if request_intent in {"provider_lookup", "location_answer"}:
            assistant_text = self._build_provider_lookup_reply(
                location=active_location,
                triage=triage,
                providers=providers,
            )
            response_mode = "provider_lookup" if providers else "location_request"
            dataset_files_used = []
            citations = []
        else:
            force_rules_reply = (not is_intake_bootstrap) and request_intent == "smalltalk"
            if not force_rules_reply:
                assistant_text = self._generate_llm_reply(
                    user_text=text,
                    intake=intake,
                    triage=triage,
                    providers=providers,
                    citations=retrieval.results,
                    history=state.chat[-8:],
                    message_kind=message_kind,
                )
                response_mode = "llm_rag" if assistant_text else "rules_fallback"

            if not assistant_text:
                assistant_text = self._build_assistant_reply_text(
                    user_text=text,
                    triage=triage,
                    location=active_location,
                    providers=providers,
                    citations=retrieval.results,
                    message_kind=message_kind,
                )
        assistant_message = ChatMessage(role="assistant", text=assistant_text, created_at_utc=utc_now_iso())

        state.chat.append(assistant_message)
        state.latest_triage = triage
        state.updated_at_utc = assistant_message.created_at_utc

        return ChatReplyResponse(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            triage=triage,
            intake=state.intake,
            nearby_providers=providers,
            citations=citations,
            response_mode=response_mode,
            dataset_files_used=dataset_files_used,
        )

    def generate_nearby_providers(
        self,
        location: str,
        urgency: UrgencyLevel,
        recommended_care_level: str | None = None,
        specialty: str | None = None,
    ) -> NearbyProvidersResponse:
        live_providers = self._fetch_live_nearby_providers(
            location=location,
            urgency=urgency,
            recommended_care_level=recommended_care_level,
            specialty=specialty,
        )
        if live_providers:
            return NearbyProvidersResponse(providers=live_providers)

        if urgency == UrgencyLevel.EMERGENCY:
            names = [
                "City Emergency Hospital",
                "Metro Trauma Center",
                "24x7 Critical Care Institute",
            ]
            waits = [8, 12, 15]
        elif urgency == UrgencyLevel.URGENT:
            names = [
                "Rapid Urgent Care Clinic",
                "Same-Day Care Center",
                "Community Health Urgent Unit",
            ]
            waits = [18, 24, 28]
        else:
            names = [
                "Primary Care Family Clinic",
                "Neighborhood Health Clinic",
                "Multi-Specialty Outpatient Center",
            ]
            waits = [35, 42, 48]

        care_level = recommended_care_level or self._default_care(urgency.value)
        provider_specialty = specialty or self._default_specialty(urgency.value)

        providers = [
            NearbyProvider(
                name=f"{name} - {location}",
                distance_km=round(1.2 + idx * 1.1, 1),
                estimated_wait_minutes=waits[idx],
                care_level=care_level,
                specialty=provider_specialty,
                location_hint=location,
                maps_url=self._build_google_maps_url(f"{name} {location}"),
            )
            for idx, name in enumerate(names)
        ]
        return NearbyProvidersResponse(providers=providers)

    def _fetch_live_nearby_providers(
        self,
        location: str,
        urgency: UrgencyLevel,
        recommended_care_level: str | None = None,
        specialty: str | None = None,
    ) -> list[NearbyProvider]:
        if not self.live_maps_enabled or self.maps_provider != "osm":
            return []

        normalized_location = location.strip()
        if not normalized_location:
            return []

        cache_key = (
            normalized_location.lower(),
            urgency.value,
            (recommended_care_level or "").strip().lower(),
            (specialty or "").strip().lower(),
        )
        if cache_key in self._provider_lookup_cache:
            return self._provider_lookup_cache[cache_key]

        geo_point = self._geocode_location(normalized_location)
        if geo_point is None:
            self._provider_lookup_cache[cache_key] = []
            return []

        providers = self._search_osm_nearby(
            geo_point=geo_point,
            query_location=normalized_location,
            urgency=urgency,
            recommended_care_level=recommended_care_level,
            specialty=specialty,
        )
        self._provider_lookup_cache[cache_key] = providers
        return providers

    def _geocode_location(self, location: str) -> GeoPoint | None:
        cache_key = location.strip().lower()
        if cache_key in self._geocode_cache:
            return self._geocode_cache[cache_key]

        params = urllib.parse.urlencode(
            {
                "q": location,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
            }
        )
        url = f"{self.nominatim_url}?{params}"
        payload = self._request_json(url=url)
        if not isinstance(payload, list) or not payload:
            self._geocode_cache[cache_key] = None
            return None

        top_hit = payload[0]
        try:
            point = GeoPoint(
                lat=float(top_hit["lat"]),
                lon=float(top_hit["lon"]),
                label=str(top_hit.get("display_name", location)).strip() or location,
            )
        except (KeyError, TypeError, ValueError):
            point = None

        self._geocode_cache[cache_key] = point
        return point

    def _search_osm_nearby(
        self,
        geo_point: GeoPoint,
        query_location: str,
        urgency: UrgencyLevel,
        recommended_care_level: str | None = None,
        specialty: str | None = None,
    ) -> list[NearbyProvider]:
        radius_m = self._provider_search_radius_meters(urgency)
        overpass_query = f"""
[out:json][timeout:12];
(
  node(around:{radius_m},{geo_point.lat},{geo_point.lon})["amenity"~"hospital|clinic|doctors"];
  way(around:{radius_m},{geo_point.lat},{geo_point.lon})["amenity"~"hospital|clinic|doctors"];
  relation(around:{radius_m},{geo_point.lat},{geo_point.lon})["amenity"~"hospital|clinic|doctors"];
);
out center tags;
""".strip()

        payload = self._request_json(
            url=self.overpass_url,
            method="POST",
            data=urllib.parse.urlencode({"data": overpass_query}).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
        )
        if not isinstance(payload, dict):
            return []

        elements = payload.get("elements")
        if not isinstance(elements, list):
            return []

        candidate_rows: list[tuple[int, float, NearbyProvider]] = []
        seen: set[tuple[str, str]] = set()
        care_level = recommended_care_level or self._default_care(urgency.value)
        provider_specialty = specialty or self._default_specialty(urgency.value)

        for item in elements:
            if not isinstance(item, dict):
                continue
            tags = item.get("tags")
            if not isinstance(tags, dict):
                continue

            name = str(tags.get("name", "")).strip()
            amenity = str(tags.get("amenity", "")).strip().lower()
            if not name or amenity not in {"hospital", "clinic", "doctors"}:
                continue

            lat, lon = self._extract_provider_coordinates(item)
            if lat is None or lon is None:
                continue

            address = self._format_osm_address(tags) or geo_point.label or query_location
            dedupe_key = (name.lower(), address.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            distance_km = round(self._haversine_km(geo_point.lat, geo_point.lon, lat, lon), 1)
            candidate_rows.append(
                (
                    self._amenity_priority(amenity, urgency),
                    distance_km,
                    NearbyProvider(
                        name=name,
                        distance_km=distance_km,
                        estimated_wait_minutes=self._estimate_provider_wait_minutes(
                            urgency=urgency,
                            amenity=amenity,
                            distance_km=distance_km,
                        ),
                        care_level=care_level,
                        specialty=provider_specialty,
                        location_hint=address,
                        maps_url=self._build_google_maps_url(f"{name} {address or query_location}"),
                    ),
                )
            )

        candidate_rows.sort(key=lambda item: (item[0], item[1]))
        return [provider for _, _, provider in candidate_rows[:3]]

    def _request_json(
        self,
        url: str,
        method: str = "GET",
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        request_headers = {
            "Accept": "application/json",
            "User-Agent": self.maps_user_agent,
        }
        if headers:
            request_headers.update(headers)

        request = urllib.request.Request(
            url=url,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.maps_request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _provider_search_radius_meters(urgency: UrgencyLevel) -> int:
        if urgency == UrgencyLevel.EMERGENCY:
            return 15000
        if urgency == UrgencyLevel.URGENT:
            return 10000
        return 8000

    @staticmethod
    def _extract_provider_coordinates(item: dict[str, Any]) -> tuple[float | None, float | None]:
        try:
            if "lat" in item and "lon" in item:
                return float(item["lat"]), float(item["lon"])
            center = item.get("center") or {}
            if "lat" in center and "lon" in center:
                return float(center["lat"]), float(center["lon"])
        except (TypeError, ValueError):
            return None, None
        return None, None

    @staticmethod
    def _format_osm_address(tags: dict[str, Any]) -> str:
        parts = [
            str(tags.get("addr:housenumber", "")).strip(),
            str(tags.get("addr:street", "")).strip(),
            str(tags.get("addr:city", "")).strip(),
            str(tags.get("addr:postcode", "")).strip(),
        ]
        address = ", ".join(part for part in parts if part)
        return address or str(tags.get("addr:full", "")).strip()

    @staticmethod
    def _amenity_priority(amenity: str, urgency: UrgencyLevel) -> int:
        if urgency == UrgencyLevel.EMERGENCY:
            ranking = {"hospital": 0, "clinic": 1, "doctors": 2}
        elif urgency == UrgencyLevel.URGENT:
            ranking = {"clinic": 0, "hospital": 1, "doctors": 2}
        else:
            ranking = {"doctors": 0, "clinic": 1, "hospital": 2}
        return ranking.get(amenity, 9)

    @staticmethod
    def _estimate_provider_wait_minutes(urgency: UrgencyLevel, amenity: str, distance_km: float) -> int:
        base = {
            UrgencyLevel.EMERGENCY: 12,
            UrgencyLevel.URGENT: 24,
            UrgencyLevel.ROUTINE: 36,
            UrgencyLevel.SELF_CARE: 42,
        }[urgency]
        amenity_adjustment = {"hospital": 0, "clinic": 4, "doctors": 8}.get(amenity, 5)
        distance_adjustment = min(int(distance_km * 2), 10)
        return base + amenity_adjustment + distance_adjustment

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        earth_radius_km = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return earth_radius_km * c

    @staticmethod
    def _build_google_maps_url(query: str) -> str:
        encoded = urllib.parse.quote_plus(query.strip())
        return f"https://www.google.com/maps/search/?api=1&query={encoded}"

    def submit_feedback(self, payload: FeedbackRequest) -> FeedbackResponse:
        feedback_id = f"fb_{uuid4().hex[:12]}"
        self.feedback_store.append(
            {
                "feedback_id": feedback_id,
                "session_id": payload.session_id,
                "rating": payload.rating,
                "comment": payload.comment or "",
                "created_at_utc": utc_now_iso(),
            }
        )
        return FeedbackResponse(accepted=True, feedback_id=feedback_id)

    def _serialize_session(self, state: SessionState) -> SessionStateResponse:
        return SessionStateResponse(
            session_id=state.session_id,
            created_at_utc=state.created_at_utc,
            updated_at_utc=state.updated_at_utc,
            intake=state.intake,
            latest_triage=state.latest_triage,
            chat=state.chat,
        )

    def _pick_best_triage_rule(
        self,
        text: str,
        age_group: AgeGroup | None,
    ) -> dict[str, Any] | None:
        normalized_text = self._normalize(text)
        best_rule: dict[str, Any] | None = None

        for rule in self.triage_rules:
            rule_age = str(rule.get("age_group", "")).strip()
            if age_group and rule_age and rule_age != age_group.value:
                continue

            symptom_pattern = [self._normalize(str(token)) for token in rule.get("symptom_pattern", [])]
            if not symptom_pattern:
                continue
            if not all(pattern in normalized_text for pattern in symptom_pattern):
                continue

            if best_rule is None:
                best_rule = rule
                continue

            current_priority = URGENCY_PRIORITY.get(str(rule.get("urgency_level", "")), 0)
            best_priority = URGENCY_PRIORITY.get(str(best_rule.get("urgency_level", "")), 0)
            if current_priority > best_priority:
                best_rule = rule

        return best_rule

    def _map_urgency_from_keywords(self, text: str) -> str:
        normalized = self._normalize(text)
        if re.search(
            r"(chest pain|shortness of breath|suicidal|seizure|unconscious|facial droop|slurred speech)",
            normalized,
        ):
            return UrgencyLevel.EMERGENCY.value
        if re.search(
            r"(wheezing|high fever|persistent high blood sugar|severe|worse|difficulty breathing)",
            normalized,
        ):
            return UrgencyLevel.URGENT.value
        if re.search(r"(itchy rash|mild|follow up|routine|occasional)", normalized):
            return UrgencyLevel.SELF_CARE.value
        return UrgencyLevel.ROUTINE.value

    def _infer_symptom_cluster(
        self,
        text: str,
        intake: IntakeProfile | None,
        urgency: str,
        effective_age_group: AgeGroup | None = None,
    ) -> str:
        normalized = self._normalize(f"{text} {intake.symptom_category if intake else ''}")
        age_group = intake.age_group if intake and intake.age_group else effective_age_group

        if re.search(r"(chest pain|shortness of breath|breathing)", normalized):
            return "chest_pain_breathing"
        if re.search(r"(facial droop|slurred speech|arm weakness|stroke)", normalized):
            return "stroke_signs"
        if re.search(r"(anaphylaxis|throat swelling|hives with breathing)", normalized):
            return "anaphylaxis"
        if re.search(r"(suicidal|self harm|mental health crisis)", normalized):
            return "mental_health_crisis"
        if re.search(r"(wheezing|difficulty breathing|respiratory distress)", normalized):
            return "respiratory_distress_moderate"
        if (
            re.search(r"(fever|child fever|infant fever)", normalized)
            and age_group in {AgeGroup.CHILD, AgeGroup.INFANT}
        ):
            return "pediatric_fever"
        if re.search(r"(blood sugar|polyuria|polydipsia|diabetes)", normalized):
            return "metabolic_risk"
        if re.search(r"(rash|skin|itchy)", normalized):
            return "mild_skin"
        if re.search(r"(anxiety|depression|mental health)", normalized):
            return "mental_health_non_crisis"
        if urgency == UrgencyLevel.EMERGENCY.value:
            return "chest_pain_breathing"
        if urgency == UrgencyLevel.URGENT.value:
            return "respiratory_distress_moderate"
        if urgency == UrgencyLevel.SELF_CARE.value:
            return "mild_skin"
        return "common_respiratory"

    def _find_route(self, urgency: str, symptom_cluster: str) -> dict[str, Any] | None:
        for rule in self.routing_rules:
            if rule.get("urgency_level") == urgency and rule.get("symptom_cluster") == symptom_cluster:
                return rule
        for rule in self.routing_rules:
            if rule.get("urgency_level") == urgency:
                return rule
        return None

    @staticmethod
    def _default_care(urgency: str) -> str:
        if urgency == UrgencyLevel.EMERGENCY.value:
            return "emergency_department"
        if urgency == UrgencyLevel.URGENT.value:
            return "urgent_care"
        if urgency == UrgencyLevel.SELF_CARE.value:
            return "self_care"
        return "primary_care_visit"

    @staticmethod
    def _default_specialty(urgency: str) -> str:
        if urgency == UrgencyLevel.EMERGENCY.value:
            return "emergency"
        if urgency == UrgencyLevel.URGENT.value:
            return "general_medicine"
        return "primary_care"

    def _missing_critical_fields(
        self,
        intake: IntakeProfile | None,
        user_text: str,
        request_intent: str,
        urgency: str,
        effective_age_group: AgeGroup | None,
    ) -> list[str]:
        if request_intent != "clinical_guidance" or self._is_smalltalk_or_ack(user_text):
            return []
        if urgency == UrgencyLevel.EMERGENCY.value:
            return []
        missing = []
        if effective_age_group is None:
            missing.append("patient age group")
        if not (intake and intake.duration) and not self._has_duration_hint(user_text):
            missing.append("symptom duration")
        severity_known = (
            bool(intake and intake.severity)
            or self._has_severity_hint(user_text)
            or urgency in {UrgencyLevel.URGENT.value, UrgencyLevel.EMERGENCY.value}
        )
        if not severity_known:
            missing.append("severity")
        return missing

    @staticmethod
    def _has_duration_hint(text: str) -> bool:
        return bool(DURATION_HINT_RE.search(text or ""))

    @staticmethod
    def _has_severity_hint(text: str) -> bool:
        return bool(SEVERITY_HINT_RE.search(text or ""))

    @staticmethod
    def _infer_age_group_from_text(text: str) -> AgeGroup | None:
        raw_text = (text or "").strip()
        normalized = raw_text.lower()
        match = EXPLICIT_AGE_RE.search(raw_text)
        if match:
            years = int(match.group(1))
            if years <= 1:
                return AgeGroup.INFANT
            if years <= 12:
                return AgeGroup.CHILD
            if years <= 17:
                return AgeGroup.ADOLESCENT
            if years >= 65:
                return AgeGroup.OLDER_ADULT
            return AgeGroup.ADULT
        if re.search(r"\b(newborn|infant|baby|month-old)\b", normalized):
            return AgeGroup.INFANT
        if re.search(r"\b(toddler|child|children|kid|kids|son|daughter|school-age)\b", normalized):
            return AgeGroup.CHILD
        if re.search(r"\b(teen|teenager|adolescent)\b", normalized):
            return AgeGroup.ADOLESCENT
        if re.search(r"\b(elderly|senior|older adult)\b", normalized):
            return AgeGroup.OLDER_ADULT
        if re.search(r"\badult\b", normalized):
            return AgeGroup.ADULT
        return None

    @staticmethod
    def _is_smalltalk_or_ack(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        return normalized in {
            "hi",
            "hello",
            "hey",
            "ok",
            "okay",
            "thanks",
            "thank you",
            "got it",
            "cool",
            "fine",
        }

    def _is_provider_location_query(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        has_provider_word = bool(PROVIDER_WORD_RE.search(normalized))
        has_location_request = bool(PROVIDER_LOCATION_QUERY_RE.search(normalized))
        has_inline_location = bool(self._extract_inline_location_hint(text))
        is_generic_specialty_request = bool(
            re.search(r"(which doctor should i see|doctor should i see|what doctor should i see)", normalized)
        )
        if is_generic_specialty_request and not has_location_request and not has_inline_location:
            return False
        return has_provider_word and (has_location_request or has_inline_location)

    def _classify_request_intent(
        self,
        user_text: str,
        history: list[ChatMessage],
        intake: IntakeProfile | None,
        message_kind: str = "chat",
    ) -> str:
        if message_kind == "intake_bootstrap":
            return "clinical_guidance"
        if self._is_smalltalk_or_ack(user_text):
            return "smalltalk"
        if self._last_assistant_requested_location(history) and self._looks_like_location_reply(user_text):
            return "location_answer"
        llm_intent = self._classify_request_intent_with_llm(
            user_text=user_text,
            history=history,
            intake=intake,
        )
        if llm_intent:
            return llm_intent
        if self._is_provider_location_query(user_text):
            return "provider_lookup"
        return "clinical_guidance"

    def _classify_request_intent_with_llm(
        self,
        user_text: str,
        history: list[ChatMessage],
        intake: IntakeProfile | None,
    ) -> str | None:
        if not self.llm_enabled or self.llm_provider != "ollama":
            return None

        last_assistant_requested_location = self._last_assistant_requested_location(history)
        saved_location = self._normalize_location_candidate(intake.location) if intake and intake.location else ""
        llm_reply = self._call_ollama_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify the latest NeuroHealth user message. "
                        "Return exactly one label and nothing else: "
                        "clinical_guidance, provider_lookup, location_answer, or smalltalk. "
                        "provider_lookup means the user is asking for nearby hospitals, clinics, doctors, maps, directions, "
                        "or location-based care options. "
                        "location_answer means the user is directly giving a city, area, landmark, or postal code, usually "
                        "after the assistant asked for location. "
                        "clinical_guidance means symptoms, triage, urgency, doctor specialty, or care advice. "
                        "smalltalk means greeting or acknowledgement."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            f"Latest user message: {user_text}",
                            f"Assistant recently asked for location: {'yes' if last_assistant_requested_location else 'no'}",
                            f"Saved intake location is available: {'yes' if saved_location else 'no'}",
                        ]
                    ),
                },
            ],
            temperature=0.0,
            num_predict=12,
        )
        if not llm_reply:
            return None

        match = re.search(
            r"\b(clinical_guidance|provider_lookup|location_answer|smalltalk)\b",
            llm_reply.strip().lower(),
        )
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _last_assistant_requested_location(history: list[ChatMessage]) -> bool:
        for turn in reversed(history):
            if turn.role != "assistant":
                continue
            normalized = turn.text.lower()
            return bool(
                re.search(
                    r"(share|add).*(city|postal code|zip code)|nearby hospitals|nearby clinics|"
                    r"location was not provided|location not provided|suggest nearby hospitals or clinics",
                    normalized,
                )
            )
        return False

    @staticmethod
    def _looks_like_location_reply(text: str) -> bool:
        candidate = (text or "").strip()
        if not candidate:
            return False
        if LOCATION_REPLY_PREFIX_RE.search(candidate):
            return bool(BackendService._normalize_location_candidate(candidate))
        if re.search(r"\b\d{5,6}\b", candidate):
            return bool(BackendService._normalize_location_candidate(candidate))
        return bool(BackendService._normalize_location_candidate(candidate))

    @staticmethod
    def _extract_location_from_text(text: str) -> str:
        candidate = LOCATION_REPLY_PREFIX_RE.sub("", (text or "").strip())
        candidate = re.sub(r"^(city|postal code|zip code|zipcode)\s*[:\-]?\s*", "", candidate, flags=re.IGNORECASE)
        return BackendService._normalize_location_candidate(candidate)

    @staticmethod
    def _extract_inline_location_hint(text: str) -> str:
        raw_text = (text or "").strip()
        for pattern in (
            r"\b(?:in|at)\s+([a-z0-9][a-z0-9,\-.' ]{1,60})$",
            r"\bnear\s+([a-z0-9][a-z0-9,\-.' ]{1,60})$",
        ):
            match = re.search(pattern, raw_text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = BackendService._normalize_location_candidate(match.group(1))
            if candidate:
                return candidate
        return ""

    @staticmethod
    def _normalize_location_candidate(text: str | None) -> str:
        candidate = str(text or "").strip().strip(".,!?")
        if not candidate:
            return ""
        if HEALTH_QUERY_HINT_RE.search(candidate):
            return ""
        if LOCATION_NOISE_RE.search(candidate):
            return ""
        if re.search(r"\b(nearby|nearest|closest|google maps?|map links?|directions)\b", candidate, re.IGNORECASE):
            return ""
        if re.search(r"\b(hospital|clinic|doctor|provider)s?\b.*\b(for this|available)\b", candidate, re.IGNORECASE):
            return ""
        if candidate.lower() in {"my location", "my area", "my place", "current location", "location"}:
            return ""
        if len(candidate.split()) > 8:
            return ""
        if not re.fullmatch(r"[a-zA-Z0-9,\-.' ]+", candidate):
            return ""
        return candidate

    @staticmethod
    def _build_snippet(text: str, query_tokens: list[str], max_chars: int = 260) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return ""

        lower = clean.lower()
        anchor = -1
        for token in query_tokens:
            pos = lower.find(token)
            if pos != -1:
                anchor = pos
                break

        if anchor == -1 or len(clean) <= max_chars:
            return clean[:max_chars]

        start = max(0, anchor - (max_chars // 3))
        end = min(len(clean), start + max_chars)
        snippet = clean[start:end]
        return f"...{snippet}" if start > 0 else snippet

    @staticmethod
    def _format_provider_display(provider: NearbyProvider, include_maps_link: bool = False) -> str:
        label = provider.name
        hint = (provider.location_hint or "").strip()
        if hint and hint.lower() not in label.lower():
            label = f"{label} - {hint}"
        line = f"{label} ({provider.distance_km:.1f} km, ~{provider.estimated_wait_minutes} min wait)"
        if include_maps_link and provider.maps_url:
            line += f" - [Open in Google Maps]({provider.maps_url})"
        return line

    def _build_provider_lookup_reply(
        self,
        location: str | None,
        triage: TriageDecision,
        providers: list[NearbyProvider],
    ) -> str:
        if not location:
            return "Please share your city or postal code so I can suggest nearby hospitals or clinics."

        if not providers:
            return (
                "I could not find nearby hospitals right now. Please try again in a moment or refine the location "
                "with city, area, or postal code."
            )

        lines = [
            "Nearby options based on your location:",
        ]
        for provider in providers[:3]:
            lines.append(f"- {self._format_provider_display(provider, include_maps_link=True)}")
        lines.append(
            "Recommended care path: "
            f"{triage.recommended_care_level.replace('_', ' ')} ({triage.specialty.replace('_', ' ')})"
        )
        return "\n".join(lines)

    @staticmethod
    def _build_remedy_steps(triage: TriageDecision) -> list[str]:
        cluster = triage.symptom_cluster
        urgency = triage.urgency

        if cluster in {"chest_pain_breathing", "respiratory_distress_moderate", "common_respiratory"}:
            if urgency == UrgencyLevel.EMERGENCY:
                return [
                    "Sit upright, loosen tight clothing, and avoid walking or exertion.",
                    "Use your prescribed rescue inhaler only if you already have one and know how to use it.",
                    "Keep a family member or friend with you while arranging urgent help.",
                ]
            if urgency == UrgencyLevel.URGENT:
                return [
                    "Rest in an upright position and avoid heavy activity.",
                    "Sip water or warm fluids if you can tolerate them.",
                    "Use your usual prescribed breathing medicine if it was previously advised.",
                ]
            return [
                "Rest, hydrate, and avoid activities that make breathing worse.",
                "Warm fluids or steam may help if they are usually comfortable for you.",
                "Track whether cough, wheeze, or breathing difficulty is improving.",
            ]

        if cluster == "pediatric_fever":
            return [
                "Offer frequent fluids and keep clothing light and comfortable.",
                "Monitor temperature and activity level over the next few hours.",
                "Use age-appropriate fever medicine only if it has been used safely before and label directions fit the child.",
            ]

        if cluster == "metabolic_risk":
            return [
                "Drink water and avoid sugary drinks for now.",
                "Check blood sugar if you have a monitor available.",
                "Take usual prescribed diabetes medicines exactly as already directed.",
            ]

        if cluster == "mild_skin":
            return [
                "Keep the area clean and dry and avoid scratching.",
                "Pause any new soaps, creams, or cosmetics that may be irritating the skin.",
                "Use a gentle moisturizer or your usual symptom relief only if it has been safe for you before.",
            ]

        if cluster == "mental_health_non_crisis":
            return [
                "Move to a quiet place and slow your breathing for a few minutes.",
                "Reach out to a trusted person and let them know how you are feeling.",
                "Continue any prescribed medicines and daily routines as already directed.",
            ]

        if cluster == "mental_health_crisis":
            return [
                "Stay with a trusted person and avoid being alone.",
                "Move away from anything that could be used for self-harm.",
                "Call local emergency services or a crisis line immediately.",
            ]

        if urgency == UrgencyLevel.EMERGENCY:
            return [
                "Avoid driving yourself if symptoms are severe or worsening.",
                "Rest in the safest comfortable position while help is arranged.",
                "Keep your phone nearby and have another person stay with you if possible.",
            ]
        if urgency == UrgencyLevel.URGENT:
            return [
                "Rest, hydrate, and avoid strenuous activity until you are evaluated.",
                "Continue regular prescribed medicines unless a clinician told you otherwise.",
                "Keep note of symptom changes so you can describe them clearly at the visit.",
            ]
        if urgency == UrgencyLevel.SELF_CARE:
            return [
                "Rest, drink fluids, and monitor whether symptoms are improving.",
                "Use your usual over-the-counter symptom relief only if it is normally safe for you.",
                "Avoid triggers that clearly make the symptoms worse.",
            ]
        return [
            "Rest, hydrate, and reduce strenuous activity for now.",
            "Continue regular prescribed medicines unless you were told otherwise.",
            "Track changes in symptoms so you can share them during follow-up care.",
        ]

    def _postprocess_llm_reply(
        self,
        text: str,
        triage: TriageDecision,
        *,
        allow_structured_sections: bool,
    ) -> str:
        lines = text.strip().splitlines()
        kept: list[str] = []
        skip_evidence_block = False
        heading_re = re.compile(
            r"^\*{0,2}\s*(Urgency Summary|Immediate Next Actions|Remedy Steps|Warning Signs(?: to Escalate Care)?|Recommended Care Path)\s*\*{0,2}:?\s*$",
            re.IGNORECASE,
        )
        evidence_heading_re = re.compile(
            r"^\*{0,2}\s*Evidence (Basis|Reference)\s*\*{0,2}:?\s*$",
            re.IGNORECASE,
        )
        evidence_line_re = re.compile(r"^\s*Evidence (Basis|Reference):", re.IGNORECASE)

        for raw_line in lines:
            stripped = raw_line.strip()
            if evidence_heading_re.match(stripped) or evidence_line_re.match(stripped):
                skip_evidence_block = True
                continue
            if skip_evidence_block:
                if heading_re.match(stripped):
                    skip_evidence_block = False
                elif not stripped:
                    continue
                else:
                    continue
            kept.append(raw_line)

        cleaned = "\n".join(kept).strip()
        remedy_steps = self._build_remedy_steps(triage)
        if remedy_steps and "remedy steps" not in cleaned.lower():
            if allow_structured_sections:
                cleaned = cleaned.rstrip() + "\n\n**Remedy Steps:**\n" + "\n".join(f"- {step}" for step in remedy_steps)
            else:
                cleaned = cleaned.rstrip() + "\n\nRemedy steps: " + "; ".join(remedy_steps[:2]) + "."
        return cleaned.strip()

    def _generate_llm_reply(
        self,
        user_text: str,
        intake: IntakeProfile | None,
        triage: TriageDecision,
        providers: list[NearbyProvider],
        citations: list[KnowledgeSearchResult],
        history: list[ChatMessage],
        message_kind: str = "chat",
    ) -> str | None:
        if not self.llm_enabled:
            return None
        if self.llm_provider != "ollama":
            self.last_llm_error = f"unsupported_provider:{self.llm_provider}"
            return None

        intake_summary = {
            "age_group": intake.age_group.value if intake else "unknown",
            "sex_at_birth": intake.sex_at_birth if intake and intake.sex_at_birth else "not_provided",
            "location": intake.location if intake and intake.location else "not_provided",
            "symptom_category": intake.symptom_category if intake else "unknown",
            "duration": intake.duration if intake else "unknown",
            "severity": intake.severity if intake else "unknown",
            "conditions": intake.conditions if intake and intake.conditions else "not_provided",
            "medications": intake.medications if intake and intake.medications else "not_provided",
        }

        triage_policy_table = [
            {
                "rule_id": rule.get("rule_id"),
                "age_group": rule.get("age_group"),
                "symptom_pattern": rule.get("symptom_pattern"),
                "urgency_level": rule.get("urgency_level"),
                "rationale": rule.get("rationale"),
            }
            for rule in self.triage_rules
        ]
        routing_policy_table = [
            {
                "routing_id": rule.get("routing_id"),
                "urgency_level": rule.get("urgency_level"),
                "symptom_cluster": rule.get("symptom_cluster"),
                "recommended_care_level": rule.get("recommended_care_level"),
                "specialty": rule.get("specialty"),
            }
            for rule in self.routing_rules
        ]

        citation_lines = []
        for item in citations[:3]:
            title = item.title or item.doc_id
            source = item.source_id or "unknown_source"
            line = f"- [{source}] {title}: {item.snippet}"
            citation_lines.append(line[:320])

        provider_lines = [f"- {self._format_provider_display(item)}" for item in providers[:3]]
        history_lines = [f"{turn.role.upper()}: {turn.text}" for turn in history[-6:]]
        is_intake_bootstrap = message_kind == "intake_bootstrap"
        assistant_turn_count = sum(1 for turn in history if turn.role == "assistant")
        is_follow_up = assistant_turn_count >= 1 and not is_intake_bootstrap
        is_smalltalk = (not is_intake_bootstrap) and self._is_smalltalk_or_ack(user_text)
        is_location_query = (not is_intake_bootstrap) and self._is_provider_location_query(user_text)
        has_location = bool(intake and intake.location and intake.location.strip())

        if is_intake_bootstrap:
            response_style_instruction = (
                "This is the initial intake summary turn. Use concise headings: "
                "Urgency Summary, Immediate Next Actions, Remedy Steps, Warning Signs to Escalate Care. "
                "Do not ask for location unless the user explicitly requests nearby providers."
            )
        elif is_smalltalk:
            response_style_instruction = (
                "The user sent a short acknowledgement message. Reply in 1 short sentence, "
                "invite the next symptom question, and do not repeat triage template sections."
            )
        elif is_location_query and not has_location:
            response_style_instruction = (
                "The user asked for location-based doctor or hospital suggestions, but location is missing. "
                "Ask for city or postal code only in 1 short sentence."
            )
        elif is_location_query and provider_lines:
            response_style_instruction = (
                "The user asked for nearby care options. Give a direct answer with up to 3 nearby options "
                "from the provided list, plus one short safety note. Avoid repeating full prior triage sections."
            )
        elif is_follow_up:
            response_style_instruction = (
                "This is a follow-up turn. Answer the latest question directly in 2-4 short lines. "
                "Do not use section headings and do not repeat the full intake template unless urgency changed. "
                "Include brief safe remedy steps when useful."
            )
        else:
            response_style_instruction = (
                "This is the first clinical guidance turn. Use concise headings: "
                "Urgency Summary, Immediate Next Actions, Remedy Steps, Warning Signs."
            )

        system_prompt = (
            "You are NeuroHealth, a safety-first health assistant. "
            "Use the provided triage and routing policy context to give actionable care-navigation guidance. "
            "Do not provide a disease diagnosis, but DO provide triage guidance, next steps, warning signs, and care routing. "
            "Do not list nearby providers unless the user explicitly asks for location-based options. "
            "Never respond with generic refusal text such as 'I cannot provide diagnosis or treatment'. "
            "Do not include sections called Evidence Basis, Evidence Reference, Sources, or Citations in the answer. "
            "Instead include practical remedy steps or supportive-care steps the user can safely take now when appropriate. "
            "Ask concise clarifying questions only when critical information is missing. "
            "For emergency urgency, clearly instruct immediate emergency care. "
            "Keep the response concise and practical for patients."
        )
        user_prompt = "\n".join(
            [
                "Clinical context:",
                f"- Triage urgency: {triage.urgency.value}",
                f"- Recommended care: {triage.recommended_care_level}",
                f"- Specialty: {triage.specialty}",
                f"- Triage rationale: {triage.rationale}",
                f"- Matched triage rule id: {triage.matched_rule_id or 'none'}",
                f"- Missing fields: {', '.join(triage.missing_fields) if triage.missing_fields else 'none'}",
                "Intake profile:",
                json.dumps(intake_summary, ensure_ascii=True),
                "Full triage policy table (all rules):",
                json.dumps(triage_policy_table, ensure_ascii=True),
                "Full routing policy table (all rules):",
                json.dumps(routing_policy_table, ensure_ascii=True),
                "Recent conversation:",
                "\n".join(history_lines) if history_lines else "none",
                f"Latest user message: {user_text}",
                "Retrieved evidence snippets:",
                "\n".join(citation_lines) if citation_lines else "none",
                "Nearby providers:",
                "\n".join(provider_lines) if provider_lines else "none",
                "Response style instruction:",
                response_style_instruction,
            ]
        )

        llm_reply = self._call_ollama_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        if not llm_reply:
            return None

        cleaned = llm_reply.strip()
        if self._looks_like_refusal(cleaned):
            return None
        cleaned = self._postprocess_llm_reply(
            cleaned,
            triage,
            allow_structured_sections=not is_follow_up and not is_smalltalk,
        )
        if triage.urgency == UrgencyLevel.EMERGENCY and "emergency" not in cleaned.lower():
            cleaned += "\n\nSafety note: seek emergency care immediately and do not delay."
        return cleaned

    def _call_ollama_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        num_predict: int = 220,
    ) -> str | None:
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }

        request = urllib.request.Request(
            url=f"{self.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.llm_timeout_seconds) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            content = str((parsed.get("message") or {}).get("content", "")).strip()
            if not content:
                raise ValueError("ollama response did not include assistant content")
            self.last_llm_error = None
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            self.last_llm_error = f"ollama_error:{str(exc)[:280]}"
            return None

    @staticmethod
    def _looks_like_refusal(text: str) -> bool:
        normalized = text.lower()
        refusal_patterns = [
            "i can't provide a response",
            "i cannot provide a response",
            "i can't provide diagnosis",
            "i cannot provide diagnosis",
            "contact a qualified healthcare professional",
            "is there anything else i can help you with",
        ]
        return any(pattern in normalized for pattern in refusal_patterns)

    def _build_assistant_reply_text(
        self,
        user_text: str,
        triage: TriageDecision,
        location: str | None,
        providers: list[NearbyProvider],
        citations: list[KnowledgeSearchResult],
        message_kind: str = "chat",
    ) -> str:
        is_intake_bootstrap = message_kind == "intake_bootstrap"
        is_smalltalk = (not is_intake_bootstrap) and self._is_smalltalk_or_ack(user_text)
        is_location_query = (not is_intake_bootstrap) and self._is_provider_location_query(user_text)

        if is_smalltalk:
            return (
                "I am ready to help. Please share current symptoms, duration, and severity "
                "so I can triage and guide next steps."
            )

        if is_location_query:
            if not location:
                return "Please share your city or postal code so I can suggest nearby hospitals or clinics."
            if providers:
                lines = ["Nearby options based on your location:"]
                for provider in providers[:3]:
                    lines.append(f"- {self._format_provider_display(provider, include_maps_link=True)}")
                lines.append(
                    "Recommended care path: "
                    f"{triage.recommended_care_level.replace('_', ' ')} ({triage.specialty.replace('_', ' ')})"
                )
                return "\n".join(lines)

        remedy_steps = self._build_remedy_steps(triage)

        if is_intake_bootstrap:
            immediate_action_map = {
                UrgencyLevel.EMERGENCY: (
                    "Seek emergency care immediately. Do not delay for more chat."
                ),
                UrgencyLevel.URGENT: (
                    "Arrange urgent same-day medical evaluation at urgent care or emergency services if symptoms worsen."
                ),
                UrgencyLevel.ROUTINE: (
                    "Book a primary care visit soon and continue symptom monitoring."
                ),
                UrgencyLevel.SELF_CARE: (
                    "Follow conservative self-care and monitor for any worsening symptoms."
                ),
            }
            warning_signs_map = {
                UrgencyLevel.EMERGENCY: [
                    "Breathing difficulty or chest pain that worsens",
                    "Confusion, fainting, or reduced responsiveness",
                    "Severe dehydration or persistent vomiting",
                ],
                UrgencyLevel.URGENT: [
                    "Persistent high fever or worsening breathing",
                    "Severe pain, repeated vomiting, or unusual weakness",
                    "Symptoms not improving over the next few hours",
                ],
                UrgencyLevel.ROUTINE: [
                    "New breathing difficulty or chest pain",
                    "High fever lasting beyond expected recovery",
                    "Any rapid worsening of current symptoms",
                ],
                UrgencyLevel.SELF_CARE: [
                    "Symptoms persist longer than expected",
                    "Pain or fever increases despite home care",
                    "Any new red-flag symptoms appear",
                ],
            }

            lines = [
                "**Urgency Summary:**",
                (
                    f"{triage.urgency.value.replace('_', ' ').title()} care level. "
                    f"Recommended path: {triage.recommended_care_level.replace('_', ' ')} "
                    f"({triage.specialty.replace('_', ' ')})."
                ),
                "",
                "**Immediate Next Actions:**",
                immediate_action_map[triage.urgency],
                "",
                "**Remedy Steps:**",
            ]
            for step in remedy_steps:
                lines.append(f"- {step}")
            lines.extend(
                [
                    "",
                    "**Warning Signs to Escalate Care:**",
                ]
            )
            for warning in warning_signs_map[triage.urgency]:
                lines.append(f"- {warning}")
            if triage.missing_fields:
                lines.append(f"Clarifying request: please share {', '.join(triage.missing_fields)}.")
            if location:
                lines.append("Location is saved. Ask for nearby hospital options anytime.")
            else:
                lines.append("Location was not provided. Share city or postal code if you want nearby options.")
            return "\n".join(lines)

        lines = [
            f"Urgency assessment: {triage.urgency.value.upper()}",
            "Recommended care path: "
            f"{triage.recommended_care_level.replace('_', ' ')} ({triage.specialty.replace('_', ' ')})",
            f"Why: {triage.rationale}",
            "Remedy steps: " + "; ".join(remedy_steps[:2]),
        ]
        if triage.missing_fields:
            lines.append(f"Clarifying request: please share {', '.join(triage.missing_fields)}.")
        if location:
            lines.append("Location is saved. Ask for nearby hospital options if needed.")
        else:
            lines.append("Location not provided. Share city or postal code if you want nearby options.")
        if triage.urgency == UrgencyLevel.EMERGENCY:
            lines.append("Safety note: seek emergency care immediately and do not delay for additional chat.")
        return "\n".join(lines)
