# NeuroHealth

NeuroHealth is an AI-powered health assistant project focused on symptom understanding, urgency triage, care navigation, and safe conversational guidance.

## Repository Layout

- `src/neurohealth/phase1/` - Phase 1 data foundation pipeline
- `configs/` - Source registry, seeds, triage rules, and routing definitions
- `data/` - Raw snapshots and processed intermediate artifacts
- `outputs/` - Generated documents, PDFs, and Phase 1 dataset releases
- `docs/phase1/` - Phase 1 documentation
- `scripts/` - Pipeline runners and proposal generators
- `tests/` - Validation and regression tests
- `requirements/` - Environment-specific dependency files

## Key Commands

Install Phase 1 dependencies:

```bash
python3 -m pip install -r requirements/phase1.txt
```

Install backend dependencies:

```bash
python3 -m pip install -r requirements/backend.txt
```

Run the Phase 1 pipeline:

```bash
python3 scripts/run_phase1_pipeline.py --project-root /Users/khadar/Desktop/NeuroHealth
```

Run the backend API:

```bash
python3 scripts/run_backend.py
```

Optional backend configuration can be provided through environment variables or a local `.env` file in the project root. A reference file is available at `.env.example`.

Or with uvicorn directly:

```bash
python3 -m uvicorn neurohealth.backend.app:app --app-dir src --reload --port 8000
```

Backend docs will be available at:

```text
http://127.0.0.1:8000/docs
```

Core backend endpoints:

- `GET /health`
- `GET /config`
- `GET /triage/rules`
- `GET /routing/rules`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `PATCH /sessions/{session_id}/intake`
- `POST /sessions/{session_id}/messages`
- `POST /providers/nearby`
- `POST /knowledge/search`
- `POST /feedback`

Live nearby-provider lookup:

- When the user asks for nearby hospitals or clinics, NeuroHealth first checks the saved intake location.
- If location is available, the backend tries a live OpenStreetMap-based search (Nominatim geocoding + Overpass nearby lookup).
- If live lookup is unavailable, the system falls back to the local template provider suggestions so the chat flow still works.

Generate proposal artifacts:

```bash
python3 scripts/build_phase1_proposal_docx.py
python3 scripts/build_neurohealth_proposal_docx.py
python3 scripts/build_neurohealth_proposal_pdf.py
```
# NeuroHealth
