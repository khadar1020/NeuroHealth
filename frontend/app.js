const apiBaseUrl = window.NEUROHEALTH_API_BASE || "http://127.0.0.1:8000";
const triagePath = "/data/processed/triage_policy_table.json";
const routingPath = "/data/processed/routing_map.json";
const backendDatasetFiles = [
  "data/processed/triage_policy_table.json",
  "data/processed/routing_map.json",
  "data/processed/knowledge_chunks.jsonl",
  "data/processed/knowledge_documents.jsonl",
];
const fallbackDatasetFiles = [triagePath, routingPath];

const state = {
  triageRules: [],
  routingRules: [],
  intake: null,
  chat: [],
  cardExpanded: false,
  sessionId: null,
  backendOnline: false,
  backendConfig: null,
  isProcessing: false,
  chatProgressTimer: null,
  hasShownInitialGuidance: false,
};

const urgencyPriority = {
  emergency: 4,
  urgent: 3,
  routine: 2,
  self_care: 1,
};

const knownSexValues = ["female", "male", "intersex"];
const knownSymptomValues = [
  "chest_pain_breathing",
  "stroke_signs",
  "respiratory_distress_moderate",
  "pediatric_fever",
  "metabolic_risk",
  "mild_skin",
  "mental_health_non_crisis",
  "mental_health_crisis",
  "general",
];

const els = {
  loadingView: document.getElementById("loadingView"),
  appView: document.getElementById("appView"),
  intakeStage: document.getElementById("intakeStage"),
  intakeForm: document.getElementById("intakeForm"),
  sexAtBirthSelect: document.getElementById("sexAtBirthSelect"),
  sexOtherWrap: document.getElementById("sexOtherWrap"),
  symptomCategorySelect: document.getElementById("symptomCategorySelect"),
  symptomOtherWrap: document.getElementById("symptomOtherWrap"),
  chatStage: document.getElementById("chatStage"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  chatLog: document.getElementById("chatLog"),
  quickPrompts: document.getElementById("quickPrompts"),
  messageTemplate: document.getElementById("messageTemplate"),
  urgencyBadge: document.getElementById("urgencyBadge"),
  chatSendBtn: document.getElementById("chatSendBtn"),
  chatProgress: document.getElementById("chatProgress"),
  chatProgressText: document.getElementById("chatProgressText"),
  intakeMiniCard: document.getElementById("intakeMiniCard"),
  intakeMiniCompact: document.getElementById("intakeMiniCompact"),
  intakeMiniDetails: document.getElementById("intakeMiniDetails"),
  toggleIntakeBtn: document.getElementById("toggleIntakeBtn"),
  editIntakeBtn: document.getElementById("editIntakeBtn"),
};

const defaultChatInputPlaceholder = els.chatInput?.getAttribute("placeholder") || "Describe symptoms, duration, and any warning signs...";
const defaultSendButtonHtml = els.chatSendBtn?.innerHTML || "";
const defaultSendButtonAriaLabel = els.chatSendBtn?.getAttribute("aria-label") || "Send message";
const loadingSendButtonHtml = '<span class="send-button-spinner" aria-hidden="true"></span><span class="sr-only">Preparing response</span>';
const durationHintRe = /(since (?:this )?(?:morning|afternoon|evening|yesterday|last night)|for \d+\s*(?:hour|day|week|month)s?|for (?:a|an|one|two|three|four|five|six|seven)\s*(?:hour|day|week|month)s?|few hours|few days|one day|two days|week|weeks|month|started (?:today|yesterday))/i;
const severityHintRe = /\b(mild|moderate|severe|worse|worsening|persistent|constant|unable to|difficulty breathing|can't|cannot|very bad|serious)\b/i;
const locationReplyPrefixRe = /^\s*(i am in|i'm in|im in|located in|my location is|i live in|city is|postal code is|zip code is|zipcode is|at)\s+/i;
const healthQueryHintRe = /(pain|fever|cough|breath|breathing|headache|symptom|vomit|vomiting|nausea|diabetes|blood sugar|wheez|dizz|injury|rash|infection|doctor should i see|what should i do|help|treatment|medicine|care)/i;
const providerWordRe = /\b(hospital|hospitals|clinic|clinics|doctor|doctors|provider|providers)\b/i;
const providerLocationHintRe = /(near\s+me|near\s+my|near\s+by|nearby|nearest|closest|google\s+maps?|map\s+link|directions|at\s+my\s+location|around\s+me|around\s+here)/i;
const locationNoiseRe = /(for this|this issue|this problem|my symptoms|current symptoms|nearby hospitals?|nearest hospitals?|closest hospitals?|nearby clinics?|nearest clinics?|closest clinics?|my location|current location)/i;

function normalize(text) {
  return (text || "").toLowerCase().trim();
}

function formatToken(text) {
  if (!text) return "Not provided";
  return String(text).replaceAll("_", " ");
}

function formatAssistantMessageText(reply) {
  return cleanAssistantText(reply?.text || "");
}

function createEmptyIntakeState() {
  return {
    ageGroup: "",
    sexAtBirth: "",
    location: "",
    symptomCategory: "",
    duration: "",
    severity: "",
    conditions: "",
    medications: "",
  };
}

function inferAgeGroupFromText(text) {
  const raw = String(text || "").trim();
  const normalized = normalize(raw);
  const explicitAge = raw.match(/\b(\d{1,3})\s*(?:years?\s*old|year-old|yrs?\s*old|yo|y\/o)\b/i);
  if (explicitAge) {
    const years = Number.parseInt(explicitAge[1], 10);
    if (!Number.isNaN(years)) {
      if (years <= 1) return "infant";
      if (years <= 12) return "child";
      if (years <= 17) return "adolescent";
      if (years >= 65) return "older_adult";
      return "adult";
    }
  }
  if (/\b(newborn|infant|baby|2 month old|6 month old|month-old)\b/.test(normalized)) return "infant";
  if (/\b(toddler|child|children|kid|kids|son|daughter|school-age)\b/.test(normalized)) return "child";
  if (/\b(teen|teenager|adolescent)\b/.test(normalized)) return "adolescent";
  if (/\b(elderly|senior|older adult)\b/.test(normalized)) return "older_adult";
  if (/\badult\b/.test(normalized)) return "adult";
  return "";
}

function hasDurationHint(text) {
  return durationHintRe.test(String(text || ""));
}

function hasSeverityHint(text) {
  return severityHintRe.test(String(text || ""));
}

function isShortAcknowledge(text) {
  return ["hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "got it", "cool", "fine"].includes(normalize(text));
}

function normalizeLocationCandidate(text) {
  const candidate = String(text || "").trim().replace(/^[,.\s]+|[,.!?\s]+$/g, "");
  if (!candidate) return "";
  if (healthQueryHintRe.test(candidate)) return "";
  if (locationNoiseRe.test(candidate)) return "";
  if (/\b(nearby|nearest|closest|google maps?|map links?|directions)\b/i.test(candidate)) return "";
  if (/\b(hospital|clinic|doctor|provider)s?\b.*\b(for this|available)\b/i.test(candidate)) return "";
  if (/^(my location|my area|my place|current location|location)$/i.test(candidate)) return "";
  if (!/^[a-zA-Z0-9,\-.' ]+$/.test(candidate)) return "";
  if (candidate.split(/\s+/).length > 8) return "";
  return candidate;
}

function extractInlineLocationHint(text) {
  const raw = String(text || "").trim();
  const patterns = [
    /\b(?:in|at)\s+([a-z0-9][a-z0-9,\-.' ]{1,60})$/i,
    /\bnear\s+([a-z0-9][a-z0-9,\-.' ]{1,60})$/i,
  ];
  for (const pattern of patterns) {
    const match = raw.match(pattern);
    if (!match) continue;
    const candidate = normalizeLocationCandidate(match[1]);
    if (candidate) return candidate;
  }
  return "";
}

function isProviderLocationRequest(text) {
  const normalized = normalize(text);
  const hasProviderWord = providerWordRe.test(normalized);
  const hasLocationHint = providerLocationHintRe.test(normalized);
  const hasInlineLocation = Boolean(extractInlineLocationHint(text));
  const genericSpecialtyRequest = /(which doctor should i see|doctor should i see|what doctor should i see)/i.test(normalized);
  if (genericSpecialtyRequest && !hasLocationHint && !hasInlineLocation) return false;
  return hasProviderWord && (hasLocationHint || hasInlineLocation);
}

function assistantAskedForLocation() {
  for (let index = state.chat.length - 1; index >= 0; index -= 1) {
    const turn = state.chat[index];
    if (turn.role !== "assistant") continue;
    return /(share|add).*(city|postal code|zip code)|nearby hospitals|nearby clinics|location not provided|location was not provided|suggest nearby hospitals or clinics/.test(normalize(turn.text));
  }
  return false;
}

function looksLikeLocationReply(text) {
  const candidate = (text || "").trim();
  if (!candidate) return false;
  if (locationReplyPrefixRe.test(candidate)) return Boolean(normalizeLocationCandidate(extractLocationFromText(candidate)));
  if (/\b\d{5,6}\b/.test(candidate)) return Boolean(normalizeLocationCandidate(candidate));
  return Boolean(normalizeLocationCandidate(candidate));
}

function extractLocationFromText(text) {
  return normalizeLocationCandidate((text || "")
    .replace(locationReplyPrefixRe, "")
    .replace(/^(city|postal code|zip code|zipcode)\s*[:\-]?\s*/i, "")
    .trim()
    .replace(/^[,.\s]+|[,.\s]+$/g, ""));
}

function captureLocationFromChat(text) {
  const shouldUsePromptContext = assistantAskedForLocation() && looksLikeLocationReply(text);
  const nextLocation = shouldUsePromptContext ? extractLocationFromText(text) : "";
  if (!nextLocation) return false;
  if (!state.intake) {
    state.intake = createEmptyIntakeState();
  }
  state.intake = { ...state.intake, location: nextLocation };
  renderIntakeCard(state.intake);
  return true;
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function cleanAssistantText(text) {
  return String(text || "")
    .replace(/This assistant provides guidance support and is not a medical diagnosis\.?/gi, "")
    .replace(/^\s*\*{0,2}Evidence (Basis|Reference)\*{0,2}:?\s*$/gim, "")
    .replace(/^\s*Evidence (Basis|Reference):.*$/gim, "")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]+\n/g, "\n")
    .trim();
}

function formatAssistantInline(text) {
  const escaped = escapeHtml(text);
  const withLinks = escaped.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return withLinks.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function isPlainSectionHeading(line) {
  return /^(Urgency Summary|Immediate Next Actions|Remedy Steps|Warning Signs(?: to Escalate Care)?|Recommended Care Path)\:?$/i.test(
    line.trim(),
  );
}

function renderAssistantMessage(text) {
  const cleaned = cleanAssistantText(text);
  if (!cleaned) return "";

  const lines = cleaned.split(/\n+/).map((line) => line.trim()).filter(Boolean);
  return lines.map((line) => {
    const headingMatch = line.match(/^\*\*(.+?)\*\*:?\s*$/);
    if (headingMatch || isPlainSectionHeading(line)) {
      const headingText = headingMatch ? headingMatch[1].replace(/:\s*$/, "") : line.replace(/:\s*$/, "");
      return `<div class="message-section-title">${formatAssistantInline(headingText)}</div>`;
    }
    if (/^[-*]\s+/.test(line)) {
      return `<div class="message-list-item">${formatAssistantInline(line.replace(/^[-*]\s+/, ""))}</div>`;
    }
    return `<p class="message-line">${formatAssistantInline(line)}</p>`;
  }).join("");
}

function addMessage(role, text) {
  const template = els.messageTemplate.content.cloneNode(true);
  const article = template.querySelector(".message");
  const roleNode = template.querySelector(".message-role");
  const bodyNode = template.querySelector(".message-body");

  article.classList.add(role);
  roleNode.textContent = role === "assistant" ? "NeuroHealth Assistant" : "You";
  if (role === "assistant") {
    bodyNode.innerHTML = renderAssistantMessage(text);
  } else {
    bodyNode.textContent = text;
  }
  els.chatLog.appendChild(template);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
  state.chat.push({ role, text });
}

function setChatProgress(text, visible) {
  if (!els.chatProgress || !els.chatProgressText) return;
  els.chatProgressText.textContent = text || "";
  els.chatProgress.classList.toggle("hidden", !visible);
}

function setChatInteractionLock(locked) {
  if (els.chatInput) {
    els.chatInput.disabled = locked;
    els.chatInput.placeholder = locked ? "Please wait while NeuroHealth is preparing your response..." : defaultChatInputPlaceholder;
  }
  if (els.chatSendBtn) {
    els.chatSendBtn.disabled = locked;
    els.chatSendBtn.classList.toggle("is-loading", locked);
    els.chatSendBtn.innerHTML = locked ? loadingSendButtonHtml : defaultSendButtonHtml;
    els.chatSendBtn.setAttribute("aria-label", locked ? "Preparing response" : defaultSendButtonAriaLabel);
    els.chatSendBtn.setAttribute("title", locked ? "Preparing response" : defaultSendButtonAriaLabel);
  }
  if (els.quickPrompts) {
    els.quickPrompts.querySelectorAll(".chip").forEach((chip) => {
      chip.disabled = locked;
    });
  }
}

function startProgressStages(stages) {
  if (state.chatProgressTimer) {
    window.clearInterval(state.chatProgressTimer);
    state.chatProgressTimer = null;
  }

  if (!Array.isArray(stages) || stages.length === 0) {
    setChatProgress("LLM is preparing your response...", true);
    return;
  }

  let index = 0;
  setChatProgress(stages[index], true);
  state.chatProgressTimer = window.setInterval(() => {
    if (index >= stages.length - 1) return;
    index += 1;
    setChatProgress(stages[index], true);
  }, 1250);
}

function stopProgressStages() {
  if (state.chatProgressTimer) {
    window.clearInterval(state.chatProgressTimer);
    state.chatProgressTimer = null;
  }
  setChatProgress("", false);
}

async function withAssistantProcessing(stages, task) {
  if (state.isProcessing) return null;
  state.isProcessing = true;
  setChatInteractionLock(true);
  startProgressStages(stages);

  try {
    return await task();
  } finally {
    stopProgressStages();
    setChatInteractionLock(false);
    state.isProcessing = false;
    if (els.chatInput) els.chatInput.focus();
  }
}

function intakeGuidanceStages() {
  if (state.backendOnline) {
    return [
      "LLM is analyzing intake details...",
      "NeuroHealth is selecting the right clinical workflow...",
      "LLM is preparing initial clinical guidance...",
    ];
  }
  return [
    "Applying local clinical safety rules...",
    "Preparing initial guidance from intake...",
  ];
}

function messageGuidanceStages(text = "") {
  const providerFlow = isProviderLocationRequest(text) || (assistantAskedForLocation() && looksLikeLocationReply(text));
  if (providerFlow) {
    return [
      "LLM is checking whether this needs location-based care...",
      "Checking saved location context...",
      "Searching nearby hospitals and Google Maps links...",
    ];
  }
  if (state.backendOnline) {
    return [
      "LLM is understanding your message...",
      "NeuroHealth is choosing the right response path...",
      "LLM is preparing your response...",
    ];
  }
  return [
    "Applying local triage rules...",
    "Preparing fallback response...",
  ];
}

function setUrgencyBadge(urgency) {
  els.urgencyBadge.className = `urgency-badge ${urgency || "neutral"}`;
  if (!urgency) {
    els.urgencyBadge.textContent = "No triage yet";
    return;
  }
  els.urgencyBadge.textContent = `Urgency: ${urgency.replace("_", " ")}`;
}

function showLoadingView() {
  els.loadingView.classList.remove("hidden");
  els.appView.classList.add("hidden");
}

function showIntakeView() {
  els.loadingView.classList.add("hidden");
  els.appView.classList.remove("hidden");
  els.intakeStage.classList.remove("hidden");
  els.chatStage.classList.add("hidden");
}

function showChatView() {
  els.loadingView.classList.add("hidden");
  els.appView.classList.remove("hidden");
  els.intakeStage.classList.add("hidden");
  els.chatStage.classList.remove("hidden");
}

function setOtherFieldVisibility(selectEl, wrapEl, inputName) {
  const input = els.intakeForm.elements.namedItem(inputName);
  const isOther = selectEl.value === "other";
  wrapEl.classList.toggle("hidden", !isOther);
  if (input) {
    input.required = isOther;
    if (!isOther) input.value = "";
  }
}

function renderSummaryRows(rows, container) {
  container.innerHTML = rows
    .map(
      ([key, value]) => `
      <div class="summary-row">
        <p class="summary-key">${key}</p>
        <p class="summary-value">${value}</p>
      </div>
    `
    )
    .join("");
}

function renderIntakeCard(intake) {
  const current = intake || createEmptyIntakeState();
  const compactRows = [
    ["Age group", formatToken(current.ageGroup)],
    ["Severity", formatToken(current.severity)],
    ["Location", current.location || "Not provided"],
  ];
  const detailRows = [
    ["Symptom category", formatToken(current.symptomCategory)],
    ["Duration", formatToken(current.duration)],
    ["Sex at birth", formatToken(current.sexAtBirth)],
    ["Conditions", current.conditions || "Not provided"],
    ["Medications", current.medications || "Not provided"],
  ];
  renderSummaryRows(compactRows, els.intakeMiniCompact);
  renderSummaryRows(detailRows, els.intakeMiniDetails);
}

function setIntakeCardExpanded(expanded) {
  state.cardExpanded = expanded;
  els.intakeMiniCard.classList.toggle("expanded", expanded);
  els.intakeMiniCard.classList.toggle("collapsed", !expanded);
  els.intakeMiniDetails.classList.toggle("hidden", !expanded);
  els.toggleIntakeBtn.textContent = expanded ? "Collapse" : "Expand";
}

function prefillIntakeForm(intake) {
  if (!intake) return;
  const sexKnown = knownSexValues.includes(intake.sexAtBirth);
  els.sexAtBirthSelect.value = sexKnown ? intake.sexAtBirth : intake.sexAtBirth ? "other" : "";
  const sexOtherInput = els.intakeForm.elements.namedItem("sexAtBirthOther");
  if (sexOtherInput) sexOtherInput.value = sexKnown ? "" : intake.sexAtBirth || "";
  setOtherFieldVisibility(els.sexAtBirthSelect, els.sexOtherWrap, "sexAtBirthOther");

  const symptomKnown = knownSymptomValues.includes(intake.symptomCategory);
  els.symptomCategorySelect.value = symptomKnown ? intake.symptomCategory : "other";
  const symptomOtherInput = els.intakeForm.elements.namedItem("symptomCategoryOther");
  if (symptomOtherInput) symptomOtherInput.value = symptomKnown ? "" : intake.symptomCategory || "";
  setOtherFieldVisibility(els.symptomCategorySelect, els.symptomOtherWrap, "symptomCategoryOther");

  const directFields = ["ageGroup", "location", "duration", "severity", "conditions", "medications"];
  for (const fieldName of directFields) {
    const field = els.intakeForm.elements.namedItem(fieldName);
    if (field) field.value = intake[fieldName] || "";
  }
}

function clearIntakeForm() {
  els.intakeForm.reset();
  setOtherFieldVisibility(els.sexAtBirthSelect, els.sexOtherWrap, "sexAtBirthOther");
  setOtherFieldVisibility(els.symptomCategorySelect, els.symptomOtherWrap, "symptomCategoryOther");
}

function toBackendIntake(intake) {
  return {
    age_group: intake.ageGroup,
    sex_at_birth: intake.sexAtBirth || null,
    location: intake.location || null,
    symptom_category: intake.symptomCategory,
    duration: intake.duration,
    severity: intake.severity,
    conditions: intake.conditions || null,
    medications: intake.medications || null,
  };
}

function fromBackendIntake(intake) {
  if (!intake) return null;
  return {
    ageGroup: intake.age_group || "",
    sexAtBirth: intake.sex_at_birth || "",
    location: intake.location || "",
    symptomCategory: intake.symptom_category || "",
    duration: intake.duration || "",
    severity: intake.severity || "",
    conditions: intake.conditions || "",
    medications: intake.medications || "",
  };
}

async function apiRequest(path, options = {}) {
  const method = options.method || "GET";
  const payload = options.payload ?? null;
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    body: payload ? JSON.stringify(payload) : undefined,
  });

  if (!response.ok) {
    const body = await response.text();
    const error = new Error(`API ${method} ${path} failed with status ${response.status}`);
    error.status = response.status;
    error.body = body;
    throw error;
  }

  return response.json();
}

async function checkBackendStatus() {
  try {
    const [health, config] = await Promise.all([apiRequest("/health"), apiRequest("/config")]);
    state.backendOnline = true;
    state.backendConfig = config;
    console.info("NeuroHealth backend online:", health, config);
    return true;
  } catch (error) {
    state.backendOnline = false;
    state.backendConfig = null;
    console.warn("Backend unavailable. Falling back to local heuristic mode.", error);
    return false;
  }
}

async function saveIntakeToBackend(intake, retryOnNotFound = true) {
  if (!state.backendOnline) return false;

  const payload = intake ? toBackendIntake(intake) : null;
  try {
    if (!state.sessionId) {
      const created = await apiRequest("/sessions", { method: "POST", payload: payload ? { intake: payload } : {} });
      state.sessionId = created.session_id;
      return true;
    }
    if (!payload) return true;
    await apiRequest(`/sessions/${state.sessionId}/intake`, { method: "PATCH", payload });
    return true;
  } catch (error) {
    if (retryOnNotFound && error.status === 404) {
      state.sessionId = null;
      return saveIntakeToBackend(intake, false);
    }
    state.backendOnline = false;
    console.warn("Failed to persist intake to backend, switching to local fallback.", error);
    return false;
  }
}

async function sendMessageToBackend(text, messageKind = "chat") {
  if (!state.backendOnline || !state.sessionId) return null;

  try {
    return await apiRequest(`/sessions/${state.sessionId}/messages`, {
      method: "POST",
      payload: {
        text,
        include_providers: true,
        message_kind: messageKind,
      },
    });
  } catch (error) {
    if (error.status === 404) {
      state.sessionId = null;
      const restored = await saveIntakeToBackend(state.intake, false);
      if (restored) {
        return sendMessageToBackend(text, messageKind);
      }
    }
    state.backendOnline = false;
    console.warn("Backend message call failed, using local fallback.", error);
    return null;
  }
}

function mapUrgencyFromKeywords(text) {
  const t = normalize(text);
  if (/(chest pain|shortness of breath|suicidal|seizure|unconscious|facial droop|slurred speech)/.test(t)) return "emergency";
  if (/(wheezing|high fever|persistent high blood sugar|severe|worse|difficulty breathing)/.test(t)) return "urgent";
  if (/(itchy rash|mild|follow up|routine|occasional)/.test(t)) return "self_care";
  return "routine";
}

function pickBestTriageRule(fullText, ageGroup) {
  const text = normalize(fullText);
  let bestRule = null;
  for (const rule of state.triageRules) {
    const ageMatch = !rule.age_group || !ageGroup || rule.age_group === ageGroup;
    if (!ageMatch) continue;
    const hasAll = (rule.symptom_pattern || []).every((symptom) => text.includes(normalize(symptom)));
    if (!hasAll) continue;
    if (!bestRule || urgencyPriority[rule.urgency_level] > urgencyPriority[bestRule.urgency_level]) {
      bestRule = rule;
    }
  }
  return bestRule;
}

function inferSymptomCluster(text, intake, urgency, effectiveAgeGroup = "") {
  const t = normalize(`${text} ${intake?.symptomCategory || ""}`);
  const ageGroup = intake?.ageGroup || effectiveAgeGroup;
  if (/(chest pain|shortness of breath|breathing)/.test(t)) return "chest_pain_breathing";
  if (/(facial droop|slurred speech|arm weakness|stroke)/.test(t)) return "stroke_signs";
  if (/(anaphylaxis|throat swelling|hives with breathing)/.test(t)) return "anaphylaxis";
  if (/(suicidal|self harm|mental health crisis)/.test(t)) return "mental_health_crisis";
  if (/(wheezing|difficulty breathing|respiratory distress)/.test(t)) return "respiratory_distress_moderate";
  if (/(fever|child fever|infant fever)/.test(t) && (ageGroup === "child" || ageGroup === "infant")) return "pediatric_fever";
  if (/(blood sugar|polyuria|polydipsia|diabetes)/.test(t)) return "metabolic_risk";
  if (/(rash|skin|itchy)/.test(t)) return "mild_skin";
  if (/(anxiety|depression|mental health)/.test(t)) return "mental_health_non_crisis";
  if (urgency === "emergency") return "chest_pain_breathing";
  if (urgency === "urgent") return "respiratory_distress_moderate";
  if (urgency === "self_care") return "mild_skin";
  return "common_respiratory";
}

function findRoute(urgency, symptomCluster) {
  const exactRoute = state.routingRules.find((item) => item.urgency_level === urgency && item.symptom_cluster === symptomCluster);
  if (exactRoute) return exactRoute;
  return state.routingRules.find((item) => item.urgency_level === urgency) || null;
}

function missingCriticalFields(intake, userText, urgency, options = {}) {
  if (options?.isProviderLookup || isShortAcknowledge(userText)) return [];
  if (urgency === "emergency") return [];
  const missing = [];
  const effectiveAgeGroup = intake?.ageGroup || options?.effectiveAgeGroup || inferAgeGroupFromText(userText);
  const hasDuration = Boolean(intake?.duration) || hasDurationHint(userText);
  const hasSeverity = Boolean(intake?.severity) || hasSeverityHint(userText) || urgency === "urgent";
  if (!effectiveAgeGroup) missing.push("patient age group");
  if (!hasDuration) missing.push("symptom duration");
  if (!hasSeverity) missing.push("severity");
  return missing;
}

function nearbyFacilities(location, route, urgency) {
  const loc = location || "your area";
  const emergencyNames = ["City Emergency Hospital", "Metro Trauma Center", "24x7 Critical Care Institute"];
  const urgentNames = ["Rapid Urgent Care Clinic", "Same-Day Care Center", "Community Health Urgent Unit"];
  const routineNames = ["Primary Care Family Clinic", "Neighborhood Health Clinic", "Multi-Specialty Outpatient Center"];
  const names = urgency === "emergency" ? emergencyNames : urgency === "urgent" ? urgentNames : routineNames;
  const care = route?.recommended_care_level || "clinic";
  return names.map((name, idx) => ({
    name: `${name} - ${loc}`,
    distance: `${(1.3 + idx * 1.1).toFixed(1)} km`,
    care,
    mapsUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${name} ${loc}`)}`,
  }));
}

function buildRemedySteps(symptomCluster, urgency) {
  if (["chest_pain_breathing", "respiratory_distress_moderate", "common_respiratory"].includes(symptomCluster)) {
    if (urgency === "emergency") {
      return [
        "Sit upright, loosen tight clothing, and avoid exertion.",
        "Use a prescribed rescue inhaler only if it was already advised for you.",
        "Keep another person with you while urgent help is arranged.",
      ];
    }
    if (urgency === "urgent") {
      return [
        "Rest upright and avoid heavy activity.",
        "Sip water or warm fluids if you can tolerate them.",
        "Use your usual prescribed breathing medicine if it was previously advised.",
      ];
    }
    return [
      "Rest, hydrate, and avoid activities that worsen breathing.",
      "Warm fluids or steam may help if they are usually comfortable for you.",
      "Track whether cough, wheeze, or breathing trouble is improving.",
    ];
  }

  if (symptomCluster === "pediatric_fever") {
    return [
      "Offer frequent fluids and keep clothing light and comfortable.",
      "Monitor temperature and activity level over the next few hours.",
      "Use age-appropriate fever medicine only if it has been used safely before.",
    ];
  }

  if (symptomCluster === "metabolic_risk") {
    return [
      "Drink water and avoid sugary drinks for now.",
      "Check blood sugar if you have a monitor available.",
      "Take regular diabetes medicines exactly as already prescribed.",
    ];
  }

  if (symptomCluster === "mild_skin") {
    return [
      "Keep the area clean and dry and avoid scratching.",
      "Pause any new soaps, creams, or cosmetics that may be irritating the skin.",
      "Use gentle moisturizer or usual symptom relief only if it is normally safe for you.",
    ];
  }

  if (urgency === "emergency") {
    return [
      "Rest in the safest comfortable position while help is arranged.",
      "Avoid driving yourself if symptoms are severe or worsening.",
      "Keep your phone nearby and have another person stay with you if possible.",
    ];
  }

  if (urgency === "urgent") {
    return [
      "Rest, hydrate, and avoid strenuous activity until you are evaluated.",
      "Continue regular prescribed medicines unless a clinician told you otherwise.",
      "Keep note of symptom changes so you can describe them clearly at the visit.",
    ];
  }

  return [
    "Rest, hydrate, and reduce strenuous activity for now.",
    "Continue regular prescribed medicines unless you were told otherwise.",
    "Track changes in symptoms so you can share them during follow-up care.",
  ];
}

function buildAssistantReplyFallback(userText, intake, messageKind = "chat", options = {}) {
  const isBootstrap = messageKind === "intake_bootstrap";
  const mergedText = `${userText} ${intake?.symptomCategory || ""} ${intake?.conditions || ""}`;
  const effectiveAgeGroup = intake?.ageGroup || inferAgeGroupFromText(mergedText) || inferAgeGroupFromText(userText);
  const matchedRule = pickBestTriageRule(mergedText, effectiveAgeGroup || null);
  const urgency = matchedRule?.urgency_level || mapUrgencyFromKeywords(mergedText);
  const cluster = inferSymptomCluster(mergedText, intake, urgency, effectiveAgeGroup);
  const route = findRoute(urgency, cluster);
  const hospitals = nearbyFacilities(intake?.location, route, urgency);
  const missing = missingCriticalFields(intake, userText, urgency, {
    isProviderLookup: options?.forceProviderLookup || isProviderLocationRequest(userText),
    effectiveAgeGroup,
  });
  const locationAsked = !isBootstrap && (options?.forceProviderLookup || isProviderLocationRequest(userText));
  const remedySteps = buildRemedySteps(cluster, urgency);

  if (!isBootstrap && isShortAcknowledge(userText)) {
    return {
      text: "I am ready to help. Please share current symptoms, duration, and severity to continue triage.",
      urgency: "routine",
      datasetFiles: fallbackDatasetFiles,
      responseMode: "rules_fallback",
      citations: [],
    };
  }

  if (locationAsked && !intake?.location) {
    return {
      text: "Please share your city or postal code so I can suggest nearby hospitals and clinics.",
      urgency,
      datasetFiles: fallbackDatasetFiles,
      responseMode: "rules_fallback",
      citations: [],
    };
  }

  if (locationAsked && intake?.location) {
    const lines = ["Nearby options based on your location:"];
    for (const place of hospitals.slice(0, 3)) {
      lines.push(`- ${place.name} (${place.distance}) - [Open in Google Maps](${place.mapsUrl})`);
    }
    if (route) {
      lines.push(`Recommended care path: ${route.recommended_care_level.replaceAll("_", " ")} (${route.specialty.replaceAll("_", " ")})`);
    }
    return {
      text: lines.join("\n"),
      urgency,
      datasetFiles: fallbackDatasetFiles,
      responseMode: "provider_lookup",
      citations: [],
    };
  }

  if (isBootstrap) {
    const lines = [];
    lines.push("**Urgency Summary:**");
    if (route) {
      lines.push(
        `${urgency.replaceAll("_", " ")} care level. Recommended path: ${route.recommended_care_level.replaceAll("_", " ")} (${route.specialty.replaceAll("_", " ")}).`,
      );
    } else {
      lines.push(`${urgency.replaceAll("_", " ")} care level based on current intake details.`);
    }
    lines.push("");
    lines.push("**Immediate Next Actions:**");
    if (urgency === "emergency") {
      lines.push("Seek emergency care immediately and do not delay for additional chat.");
    } else if (urgency === "urgent") {
      lines.push("Arrange same-day urgent clinical evaluation and continue monitoring symptoms.");
    } else {
      lines.push("Book a primary care follow-up and continue symptom monitoring.");
    }
    lines.push("");
    lines.push("**Remedy Steps:**");
    for (const step of remedySteps) {
      lines.push(`- ${step}`);
    }
    lines.push("");
    lines.push("**Warning Signs to Escalate Care:**");
    lines.push("- New breathing difficulty, chest pain, or confusion");
    lines.push("- High fever that persists or worsens");
    lines.push("- Any rapid worsening in current symptoms");
    if (missing.length > 0) {
      lines.push(`Clarifying question: Please share ${missing.join(", ")} to refine recommendation.`);
    }
    if (intake?.location) {
      lines.push("Location is saved. Ask for nearby hospital options anytime.");
    } else {
      lines.push("Location was not provided. Share city or postal code only if you want nearby options.");
    }
    return {
      text: lines.join("\n"),
      urgency,
      datasetFiles: fallbackDatasetFiles,
      responseMode: "rules_fallback",
      citations: [],
    };
  }

  const lines = [];
  lines.push(`Urgency assessment: ${urgency.toUpperCase()}`);
  if (route) {
    lines.push(`Recommended care path: ${route.recommended_care_level.replaceAll("_", " ")} (${route.specialty.replaceAll("_", " ")})`);
  } else {
    lines.push("Recommended care path: routine primary care follow-up");
  }
  lines.push(matchedRule?.rationale ? `Why: ${matchedRule.rationale}` : "Why: pattern and severity suggest conservative safety-first routing.");
  lines.push(`Remedy steps: ${remedySteps.slice(0, 2).join("; ")}`);
  if (missing.length > 0) {
    lines.push(`Clarifying question: Please share ${missing.join(", ")} to refine recommendation.`);
  }
  if (intake?.location) {
    lines.push("Location is saved. Ask for nearby hospital options if needed.");
  } else {
    lines.push("Location not provided. Share city or postal code if you want nearby options.");
  }
  if (urgency === "emergency") {
    lines.push("Safety note: seek emergency care immediately and do not delay for additional chat.");
  }
  return {
    text: lines.join("\n"),
    urgency,
    datasetFiles: fallbackDatasetFiles,
    responseMode: "rules_fallback",
    citations: [],
  };
}

function buildIntakeOnlyPrompt(intake) {
  return [
    "Provide initial guidance using only the patient intake context.",
    `Age group: ${intake.ageGroup || "unknown"}.`,
    `Primary symptom category: ${intake.symptomCategory || "unknown"}.`,
    `Duration: ${intake.duration || "unknown"}.`,
    `Severity: ${intake.severity || "unknown"}.`,
    `Location: ${intake.location || "not provided"}.`,
    `Conditions: ${intake.conditions || "not provided"}.`,
    `Medications: ${intake.medications || "not provided"}.`,
    "Return urgency, care path, warning signs, and next immediate step in clear language.",
  ].join(" ");
}

async function resolveAssistantResponse(messageText, options = {}) {
  const messageKind = options?.messageKind || "chat";
  if (state.backendOnline && !state.sessionId) {
    await saveIntakeToBackend(state.intake);
  }

  const backendReply = await sendMessageToBackend(messageText, messageKind);
  if (backendReply?.assistant_message?.text) {
    const refreshedIntake = fromBackendIntake(backendReply.intake);
    if (refreshedIntake) {
      state.intake = refreshedIntake;
      renderIntakeCard(state.intake);
    }
    return {
      text: backendReply.assistant_message.text,
      urgency: backendReply.triage?.urgency || null,
      datasetFiles: backendReply.dataset_files_used || backendDatasetFiles,
      responseMode: backendReply.response_mode || "llm_rag",
      citations: backendReply.citations || [],
    };
  }

  return buildAssistantReplyFallback(messageText, state.intake, messageKind, options);
}

async function loadDatasets() {
  try {
    const [triageRes, routingRes] = await Promise.all([fetch(triagePath), fetch(routingPath)]);
    if (!triageRes.ok || !routingRes.ok) throw new Error("Unable to load dataset files");
    state.triageRules = await triageRes.json();
    state.routingRules = await routingRes.json();
  } catch (error) {
    console.warn("Dataset fetch failed in frontend fallback mode:", error);
  }
}

function collectIntakeFromForm(formData) {
  const sexAtBirthRaw = formData.get("sexAtBirth");
  const sexAtBirth = sexAtBirthRaw === "other" ? (formData.get("sexAtBirthOther") || "other") : sexAtBirthRaw;
  const symptomRaw = formData.get("symptomCategory");
  const symptomCategory = symptomRaw === "other" ? (formData.get("symptomCategoryOther") || "other") : symptomRaw;

  return {
    ageGroup: formData.get("ageGroup"),
    sexAtBirth,
    location: formData.get("location"),
    symptomCategory,
    duration: formData.get("duration"),
    severity: formData.get("severity"),
    conditions: formData.get("conditions"),
    medications: formData.get("medications"),
  };
}

function bindEvents() {
  els.sexAtBirthSelect.addEventListener("change", () => {
    setOtherFieldVisibility(els.sexAtBirthSelect, els.sexOtherWrap, "sexAtBirthOther");
  });

  els.symptomCategorySelect.addEventListener("change", () => {
    setOtherFieldVisibility(els.symptomCategorySelect, els.symptomOtherWrap, "symptomCategoryOther");
  });

  const skipIntakeBtn = document.getElementById("skipIntakeBtn");
  if (skipIntakeBtn) {
    skipIntakeBtn.addEventListener("click", async () => {
      state.intake = null;
      renderIntakeCard(null);
      setIntakeCardExpanded(false);
      showChatView();
      setUrgencyBadge(null);

      const backendReady = await saveIntakeToBackend(null);
      if (!state.hasShownInitialGuidance) {
        addMessage(
          "assistant",
          backendReady
            ? "Patient intake skipped for now. Ask your health question in chat, and I will request only the missing details needed for safe guidance."
            : "Patient intake skipped for now. Backend is offline, so I will request only the missing details needed for safe guidance.",
        );
        state.hasShownInitialGuidance = true;
      }
      els.chatInput.focus();
    });
  }

  els.chatInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    if (event.shiftKey) return;
    event.preventDefault();
    if (state.isProcessing) return;
    els.chatForm.requestSubmit();
  });

  els.intakeForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(els.intakeForm);
    state.intake = collectIntakeFromForm(formData);

    const backendSaved = await saveIntakeToBackend(state.intake);
    renderIntakeCard(state.intake);
    setIntakeCardExpanded(false);
    showChatView();

    const intakePrompt = buildIntakeOnlyPrompt(state.intake);
    const intakeReply = await withAssistantProcessing(intakeGuidanceStages(), async () => {
      return resolveAssistantResponse(intakePrompt, { messageKind: "intake_bootstrap" });
    });

    if (intakeReply?.text) {
      addMessage("assistant", formatAssistantMessageText(intakeReply));
      setUrgencyBadge(intakeReply.urgency || null);
    } else {
      const fallbackMsg = backendSaved
        ? "Intake saved. You can ask your health question now."
        : "Intake saved. Backend is offline, so local triage fallback is active for now.";
      addMessage("assistant", fallbackMsg);
    }

    if (!state.hasShownInitialGuidance) {
      addMessage("assistant", "You can now ask follow-up questions in chat.");
      state.hasShownInitialGuidance = true;
    }
  });

  els.toggleIntakeBtn.addEventListener("click", () => {
    setIntakeCardExpanded(!state.cardExpanded);
  });

  els.editIntakeBtn.addEventListener("click", () => {
    if (state.intake) {
      prefillIntakeForm(state.intake);
    } else {
      clearIntakeForm();
    }
    showIntakeView();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  els.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.isProcessing) return;

    const text = els.chatInput.value.trim();
    if (!text) return;

    const providerFlow = isProviderLocationRequest(text) || (assistantAskedForLocation() && looksLikeLocationReply(text));
    const locationCaptured = !state.backendOnline && captureLocationFromChat(text);
    if (locationCaptured) {
      await saveIntakeToBackend(state.intake);
    }

    els.chatInput.value = "";
    addMessage("user", text);

    const reply = await withAssistantProcessing(messageGuidanceStages(text), async () => {
      return resolveAssistantResponse(text, {
        messageKind: "chat",
        forceProviderLookup: providerFlow,
      });
    });

    if (!reply?.text) {
      addMessage("assistant", "I could not process that request. Please try again.");
      return;
    }
    addMessage("assistant", formatAssistantMessageText(reply));
    setUrgencyBadge(reply.urgency || null);
  });

  els.quickPrompts.addEventListener("click", (event) => {
    if (state.isProcessing) return;
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (!target.classList.contains("chip")) return;
    const value = target.dataset.text || "";
    els.chatInput.value = value;
    els.chatInput.focus();
  });
}

async function init() {
  bindEvents();
  setUrgencyBadge(null);
  setChatProgress("", false);
  setChatInteractionLock(false);
  renderIntakeCard(null);

  const params = new URLSearchParams(window.location.search);
  const stage = params.get("stage");

  const dataLoadTask = loadDatasets();
  const backendStatusTask = checkBackendStatus();

  if (stage === "loading") {
    showLoadingView();
    await Promise.all([dataLoadTask, backendStatusTask]);
    return;
  }

  if (stage === "intake") {
    showIntakeView();
    await Promise.all([dataLoadTask, backendStatusTask]);
    return;
  }

  if (stage === "chat") {
    state.intake = {
      ageGroup: "adult",
      sexAtBirth: "female",
      location: "Hyderabad 500081",
      symptomCategory: "chest_pain_breathing",
      duration: "since_morning",
      severity: "moderate",
      conditions: "hypertension",
      medications: "amlodipine",
    };
    renderIntakeCard(state.intake);
    setIntakeCardExpanded(false);
    showChatView();

    await Promise.all([dataLoadTask, backendStatusTask]);
    const saved = await saveIntakeToBackend(state.intake);
    const message = saved
      ? `Demo chat state loaded. Ask a question to test LLM + RAG (${state.backendConfig?.llm_model || "backend"}).`
      : "Demo chat state loaded. Backend unavailable, running local fallback triage.";
    addMessage("assistant", message);
    return;
  }

  showLoadingView();
  await Promise.all([dataLoadTask, backendStatusTask]);
  setTimeout(() => {
    showIntakeView();
  }, 1600);
}

init();
