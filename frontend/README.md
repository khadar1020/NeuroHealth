# NeuroHealth Frontend Demo

This frontend provides:

- A NeuroHealth loading screen with left-to-right title animation
- A structured intake form (age group, location, symptom category, duration, severity, conditions, medications)
- Intake-first flow: intake appears before chat
- A chat interface for free-text symptom questions after intake completion
- Post-intake layout: chat takes the main area and intake moves to a compact right-side card
- Expand/Collapse + Edit controls for the intake card
- "Other" options in dropdowns with user-entered custom text
- Triage + routing logic driven by Phase 1 processed datasets
- Location-aware nearby facility suggestions

## Run locally

From the project root:

```bash
python3 -m http.server 4173 --directory /Users/khadar/Desktop/NeuroHealth
```

Then open:

```text
http://127.0.0.1:4173/frontend/index.html
```

The frontend loads:

- `/data/processed/triage_policy_table.json`
- `/data/processed/routing_map.json`

If these files are unavailable, the app falls back to heuristic triage logic.
