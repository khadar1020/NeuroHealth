from __future__ import annotations

import random
import re

from ..types import AgeGroup, DialogueSample, DialogueTurn, IntentClass, UrgencyLevel


SYMPTOM_BANK = {
    UrgencyLevel.EMERGENCY: [
        "severe chest pain with shortness of breath",
        "heavy chest pressure spreading to the left arm",
        "face drooping with slurred speech",
        "sudden confusion with one-sided weakness",
        "throat swelling and trouble breathing",
        "unconscious episode with seizure activity",
        "fainting with ongoing palpitations",
        "suicidal thoughts with a plan to self-harm",
        "severe trouble breathing at rest",
        "coughing up blood with chest tightness",
    ],
    UrgencyLevel.URGENT: [
        "high fever for two days with worsening cough",
        "persistent vomiting and dehydration signs",
        "wheezing and breathing difficulty during the night",
        "very high blood sugar with frequent urination",
        "severe abdominal pain with repeated nausea",
        "painful urination with fever and back pain",
        "rapidly spreading painful rash",
        "migraine with vomiting and light sensitivity",
        "swollen painful calf after recent travel",
        "infant fever with poor feeding",
    ],
    UrgencyLevel.ROUTINE: [
        "dry cough and sore throat for three days",
        "fatigue with mild headache and congestion",
        "persistent heartburn after meals",
        "mild knee pain after activity",
        "ongoing anxiety symptoms affecting sleep",
        "recurrent urinary burning without fever",
        "intermittent palpitations without fainting",
        "itchy eczema flare on the hands",
        "constipation with abdominal bloating",
        "chronic low back pain after standing",
    ],
    UrgencyLevel.SELF_CARE: [
        "mild itchy rash without breathing issues",
        "occasional tension headache after long screen use",
        "minor runny nose and mild congestion",
        "small bruise with no swelling",
        "mild seasonal allergy symptoms",
        "brief heartburn after spicy food",
        "light muscle soreness after exercise",
        "mild dry skin flare in winter",
        "minor nausea after a heavy meal",
        "temporary difficulty sleeping after travel",
    ],
}

FOLLOWUPS = [
    "What should I do next?",
    "Do I need to see a doctor today?",
    "Is home care enough for now?",
    "Which specialist should I visit?",
    "What warning signs should I watch for?",
    "Should I monitor this for another day or two?",
    "Is this something that can wait for a clinic appointment?",
]

PREVENTIVE_PROMPTS = [
    "I also want prevention tips.",
    "How can I reduce this risk in future?",
    "Any lifestyle advice for this condition?",
    "What can I do to stop this from happening again?",
]

MEDICATION_PROMPTS = [
    "Could this be related to my current medication and should I adjust the dose?",
    "I recently started a new medicine. Could it be causing this?",
    "Should I ask a doctor before taking my next dose?",
]

CHRONIC_FOLLOWUP_PROMPTS = [
    "This has happened before and I want follow-up guidance.",
    "I have dealt with this on and off and need a longer-term plan.",
    "How should I track this before my next follow-up visit?",
]

APPOINTMENT_NAVIGATION_PROMPTS = [
    "Which type of clinician should I book with first?",
    "Would this be better for urgent care, primary care, or a specialist?",
    "Who is the right doctor for this situation?",
]

DURATIONS = [
    "for a few hours",
    "since this morning",
    "for one day",
    "for two days",
    "for three days",
    "for about a week",
    "on and off for two weeks",
]

SEVERITIES = [
    "mild",
    "moderate",
    "severe",
    "intermittent",
    "persistent",
    "worsening",
]

COMORBIDITIES = [
    "I have asthma",
    "I have type 2 diabetes",
    "I have high blood pressure",
    "I have eczema",
    "I have migraines",
    "I have GERD",
    "I have no major medical history",
    "I have seasonal allergies",
    "I am currently pregnant",
    "I take thyroid medication",
    "I had a recent viral infection",
    "I have chronic kidney disease",
    "I have anxiety",
]

CONTEXTS = [
    "after exercise",
    "after a heavy meal",
    "during the night",
    "after missing sleep",
    "while at work",
    "while traveling",
    "after taking a new supplement",
    "without any clear trigger",
    "after a stressful week",
    "after being around someone sick",
]

INTRO_TEMPLATES = [
    "I need advice {age_phrase}: {severity} {symptom} {duration} {context}. {history}. {followup}",
    "Can you help me figure out what to do {age_phrase}? It is {severity} {symptom} {duration} {context}. {history}. {followup}",
    "I am asking {age_phrase} because there is {severity} {symptom} {duration} {context}. {history}. {followup}",
    "I want guidance {age_phrase}: there has been {severity} {symptom} {duration} {context}. {history}. {followup}",
]


def _contains_phrase(text: str, phrase: str) -> bool:
    if " " in phrase or "-" in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _mentions_any(text: str, phrases: list[str]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _routing_specialty(intent: IntentClass, urgency: UrgencyLevel, text: str) -> str:
    lower = text.lower()
    if urgency == UrgencyLevel.EMERGENCY:
        return "emergency"
    if _mentions_any(lower, ["infant", "child", "toddler", "pediatric", "teenager"]):
        return "pediatrics"
    if _mentions_any(lower, ["suicidal", "anxiety", "panic", "depress", "mood"]):
        return "psychiatry"
    if _mentions_any(lower, ["blood sugar", "diabetes", "thyroid", "polyuria", "polydipsia"]):
        return "endocrinology"
    if _mentions_any(lower, ["pregnant", "pregnancy", "prenatal", "menstrual"]):
        return "obstetrics_gynecology"
    if _mentions_any(lower, ["rash", "eczema", "hives", "itchy", "skin flare"]):
        return "dermatology"
    if _mentions_any(lower, ["wheezing", "asthma", "shortness of breath", "cough", "breathing"]):
        return "pulmonology"
    if _mentions_any(lower, ["headache", "migraine", "seizure", "dizziness", "numbness", "weakness"]):
        return "neurology"
    if _mentions_any(lower, ["knee", "back pain", "joint", "ankle", "shoulder", "muscle soreness", "calf"]):
        return "orthopedics"
    if _mentions_any(lower, ["urination", "urinary", "dysuria", "uti", "flank pain"]):
        return "urology"
    if _mentions_any(lower, ["heartburn", "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation"]):
        return "gastroenterology"
    if _mentions_any(lower, ["palpitations", "blood pressure", "chest pain", "heart attack"]):
        return "cardiology"
    if intent == IntentClass.APPOINTMENT_NAVIGATION:
        return "primary_care"
    return "primary_care"


def _assistant_response(urgency: UrgencyLevel, age_group: AgeGroup) -> str:
    if urgency == UrgencyLevel.EMERGENCY:
        return (
            "This may be a medical emergency. Seek emergency care immediately or call local emergency services. "
            "Do not delay if symptoms are severe or worsening."
        )
    if urgency == UrgencyLevel.URGENT:
        if age_group in {AgeGroup.INFANT, AgeGroup.CHILD}:
            return (
                "Because this concerns a minor and symptoms may worsen, seek same-day urgent medical evaluation. "
                "Go to urgent care or pediatric emergency services if breathing or alertness changes."
            )
        return (
            "You should seek same-day urgent evaluation. Monitor red flags such as worsening breathing difficulty, "
            "persistent high fever, or dehydration."
        )
    if urgency == UrgencyLevel.ROUTINE:
        return (
            "Arrange a routine clinic visit and monitor symptoms. If warning signs appear or symptoms worsen, "
            "escalate to urgent care."
        )
    return (
        "Self-care may be reasonable now with hydration, rest, and symptom monitoring. "
        "If red-flag symptoms appear, seek professional care."
    )


def _user_prompt(
    rnd: random.Random,
    age_phrase: str,
    severity: str,
    symptom: str,
    duration: str,
    context: str,
    history: str,
    intent: IntentClass,
) -> str:
    text = rnd.choice(INTRO_TEMPLATES).format(
        age_phrase=age_phrase,
        severity=severity,
        symptom=symptom,
        duration=duration,
        context=context,
        history=history,
        followup=rnd.choice(FOLLOWUPS),
    )
    if intent == IntentClass.PREVENTIVE_CARE:
        text += f" {rnd.choice(PREVENTIVE_PROMPTS)}"
    elif intent == IntentClass.MEDICATION_QUESTION:
        text += f" {rnd.choice(MEDICATION_PROMPTS)}"
    elif intent == IntentClass.CHRONIC_FOLLOWUP:
        text += f" {rnd.choice(CHRONIC_FOLLOWUP_PROMPTS)}"
    elif intent == IntentClass.APPOINTMENT_NAVIGATION:
        text += f" {rnd.choice(APPOINTMENT_NAVIGATION_PROMPTS)}"
    return text


def generate_synthetic_dialogues(total_samples: int = 50000, seed: int = 42) -> list[DialogueSample]:
    rnd = random.Random(seed)
    age_groups = [
        AgeGroup.INFANT,
        AgeGroup.CHILD,
        AgeGroup.ADOLESCENT,
        AgeGroup.ADULT,
        AgeGroup.OLDER_ADULT,
    ]
    intents = [
        IntentClass.SYMPTOM_CHECK,
        IntentClass.MEDICATION_QUESTION,
        IntentClass.PREVENTIVE_CARE,
        IntentClass.CHRONIC_FOLLOWUP,
        IntentClass.APPOINTMENT_NAVIGATION,
    ]

    urgency_weights = [
        (UrgencyLevel.SELF_CARE, 0.18),
        (UrgencyLevel.ROUTINE, 0.44),
        (UrgencyLevel.URGENT, 0.25),
        (UrgencyLevel.EMERGENCY, 0.13),
    ]

    samples: list[DialogueSample] = []
    for i in range(total_samples):
        urgency = rnd.choices(
            [x[0] for x in urgency_weights],
            weights=[x[1] for x in urgency_weights],
            k=1,
        )[0]

        age_group = rnd.choice(age_groups)
        intent = rnd.choice(intents)
        symptom = rnd.choice(SYMPTOM_BANK[urgency])

        # Conservative pediatric escalation: no self-care labels for infant/child in v1.
        if age_group in {AgeGroup.INFANT, AgeGroup.CHILD} and urgency == UrgencyLevel.SELF_CARE:
            urgency = UrgencyLevel.ROUTINE

        age_phrase = {
            AgeGroup.INFANT: "for my infant",
            AgeGroup.CHILD: "for my child",
            AgeGroup.ADOLESCENT: "for a teenager",
            AgeGroup.ADULT: "for me",
            AgeGroup.OLDER_ADULT: "for an older adult",
        }[age_group]

        severity = rnd.choice(SEVERITIES)
        duration = rnd.choice(DURATIONS)
        context = rnd.choice(CONTEXTS)
        history = rnd.choice(COMORBIDITIES)
        user_text = _user_prompt(rnd, age_phrase, severity, symptom, duration, context, history, intent)
        assistant_text = _assistant_response(urgency, age_group)

        sample_id = f"syn-{i:06d}"
        samples.append(
            DialogueSample(
                sample_id=sample_id,
                turns=[
                    DialogueTurn(role="user", text=user_text),
                    DialogueTurn(role="assistant", text=assistant_text),
                ],
                intent_class=intent,
                urgency_label=urgency,
                safe_response_label="policy_safe_template",
                age_group=age_group,
                routing_specialty=_routing_specialty(intent, urgency, user_text),
                evidence_ids=["triage_policy_table_v1"],
                provenance={
                    "source_id": "synthetic_dialogues",
                    "generator": "template_v2",
                    "seed": seed,
                    "license": "INTERNAL-OPEN",
                },
            )
        )

    return samples
