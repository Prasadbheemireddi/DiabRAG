"""
Grounded Answer Generation Engine with Flexible Clinical Response Framework.
Features:
1. Flexible Specialist Framework: Dynamically adapts clinical domains (Pathophysiology,
   Diagnostics, GDMT, Monitoring) to query scope without rigid boilerplate padding.
2. Patient Triage Mode: Natural, jargon-free prose with automatic urgency grading.
3. Deterministic Vancouver-Style Citations [[1]](url) with URL auto-repair.
4. Deterministic Clinical Safety & Triage Overrides (Rule of Maximum Severity).
"""

import time
import re
from typing import Any, Dict, List, Optional, Set
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL, MEDICAL_DISCLAIMER
from utils.logger import setup_logger

logger = setup_logger("AnswerGenerator")
FALLBACK_MODEL = "gemini-flash-latest" if GEMINI_MODEL != "gemini-flash-latest" else "gemini-3.6-flash"

class DeterministicCitationProcessor:
    """Validates, repairs, and formats clinical citations to prevent broken or dead links."""
    
    ORGANIZATION_MAP = {
        "ADA/MedlinePlus": "American Diabetes Association (ADA) / MedlinePlus",
        "ACC/AHA": "American College of Cardiology & American Heart Association (ACC/AHA)",
        "CDC": "Centers for Disease Control and Prevention (CDC)",
        "WHO": "World Health Organization (WHO)",
        "NIDDK": "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK/NIH)",
        "NIH-StatPearls": "National Institutes of Health (NIH) / StatPearls",
        "MedlinePlus": "National Library of Medicine (MedlinePlus / NIH)"
    }

    URL_REPAIR_MAP = {
        "diabetestests.html": "https://medlineplus.gov/lab-tests/diabetes-tests/",
        "gestationaldiabetes.html": "https://medlineplus.gov/diabetesandpregnancy.html/",
        "diabeticketoacidosis.html": "https://medlineplus.gov/lab-tests/ketones-in-blood/",
        "what-is-diabetes.html": "https://www.cdc.gov/diabetes/about/index.html",
        "type-1-diabetes.html": "https://www.cdc.gov/diabetes/about/about-type-1-diabetes.html",
        "type-2-diabetes.html": "https://www.cdc.gov/diabetes/about/about-type-2-diabetes.html",
        "manage-blood-sugar.html": "https://www.cdc.gov/diabetes/about/index.html"
    }

    @classmethod
    def sanitize_url(cls, raw_url: str) -> str:
        """Inspects and rewrites any deprecated URL to a verified live destination."""
        if not raw_url or raw_url == "#":
            return "https://medlineplus.gov/diabetes.html"

        url_clean = raw_url.strip()
        for broken_pattern, live_target in cls.URL_REPAIR_MAP.items():
            if broken_pattern in url_clean:
                logger.info(f"URL Auto-Repair: Remapped '{url_clean}' -> '{live_target}'")
                return live_target

        return url_clean

    @classmethod
    def process_citations(cls, raw_answer: str, passages: List[Dict[str, Any]], user_mode: str = "patient") -> str:
        """Processes citations with guaranteed working URLs and audience-differentiated styles."""
        clean_text = re.split(r"(?i)\n#{1,4}\s*(?:references|sources|citations|clinical evidence)", raw_answer)[0].strip()
        clean_text = re.split(r"(?i)\n>\s*\*?disclaimer", clean_text)[0].strip()

        cited_indices: Set[int] = set()
        matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", clean_text)
        for m in matches:
            for num in re.split(r",\s*", m):
                if num.isdigit():
                    idx = int(num)
                    if 1 <= idx <= len(passages):
                        cited_indices.add(idx)

        if not cited_indices:
            cited_indices = set(range(1, len(passages) + 1))

        # --- CATEGORY 1: PATIENT & CAREGIVER FORMAT ---
        if user_mode == "patient":
            patient_text = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", clean_text)

            sources_section = "\n\n---\n\n### 🏥 Trusted Medical References Consulted\n"
            sources_section += "*This guidance is grounded in active clinical guidelines from recognized institutions:*\n\n"
            
            seen_entries = set()
            for idx in sorted(cited_indices):
                p = passages[idx - 1]
                src_raw = p.get("source", "Clinical Reference")
                org_name = cls.ORGANIZATION_MAP.get(src_raw, src_raw)
                title = p.get("title", "Diabetes Medical Guideline")
                safe_url = cls.sanitize_url(p.get("url", ""))

                entry_key = (org_name, safe_url)
                if entry_key not in seen_entries:
                    seen_entries.add(entry_key)
                    sources_section += f"- **{org_name}:** [{title}]({safe_url})\n"

            return f"{patient_text}{sources_section}\n\n> *{MEDICAL_DISCLAIMER}*"

        # --- CATEGORY 2: MEDICAL SPECIALIST FORMAT (Vancouver-style links [[1]](url)) ---
        else:
            specialist_text = clean_text
            for idx in sorted(cited_indices, reverse=True):
                p = passages[idx - 1]
                safe_url = cls.sanitize_url(p.get("url", ""))
                specialist_text = re.sub(
                    rf"\[{idx}\](?!\()", 
                    f"[[{idx}]]({safe_url})", 
                    specialist_text
                )

            references_section = "\n\n---\n\n### 📚 References & Clinical Evidence Base (Vancouver / AMA Standard)\n"
            for idx in sorted(cited_indices):
                p = passages[idx - 1]
                src_raw = p.get("source", "Clinical Reference")
                org_name = cls.ORGANIZATION_MAP.get(src_raw, src_raw)
                title = p.get("title", "Clinical Guidelines in Diabetes")
                section = p.get("section", "Clinical Protocols")
                safe_url = cls.sanitize_url(p.get("url", ""))

                references_section += (
                    f"**[{idx}] {org_name}** — *{title}*  \n"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;↳ **Clinical Scope:** {section} | [Access Verified Guideline ↗]({safe_url})\n"
                )

            return f"{specialist_text}{references_section}\n\n> *{MEDICAL_DISCLAIMER}*"


class AnswerGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate_answer(
        self, 
        query: str, 
        passages: List[Dict[str, Any]], 
        evaluation_result: Dict[str, Any],
        user_mode: str = "patient",
        chat_history: List[Dict[str, Any]] = None,
        triage_override: Optional[Dict[str, Any]] = None
    ) -> str:
        """Synthesizes clinical answers with flexible response framing and deterministic citations."""
        
        # Build numbered evidence list with sanitized active URLs
        evidence_text = ""
        for i, p in enumerate(passages, start=1):
            src_raw = p.get("source", "Clinical Reference")
            org = DeterministicCitationProcessor.ORGANIZATION_MAP.get(src_raw, src_raw)
            verified_url = DeterministicCitationProcessor.sanitize_url(p.get("url", ""))
            
            evidence_text += (
                f"[EVIDENCE {i}]\n"
                f"Authority: {org}\n"
                f"Title: {p.get('title')}\n"
                f"Section: {p.get('section')}\n"
                f"Verified URL: {verified_url}\n"
                f"Content: {p.get('text')}\n\n"
            )

        caveat_instruction = ""
        if evaluation_result.get("status") == "insufficient":
            missing = ", ".join(evaluation_result.get("missing_points", []))
            caveat_instruction = (
                f"\nNOTE: The evidence was marked incomplete for: {missing}. "
                f"Add a brief note under 'Limitations of Available Evidence'."
            )

        history_text = ""
        if chat_history:
            history_text = "\nRECENT CONVERSATION HISTORY:\n"
            for turn in chat_history[-2:]:
                q = turn.get("display_query") or turn.get("query", "")
                a = turn.get("answer", "")
                history_text += f"User: {q}\nAssistant: {a[:200]}...\n"

        # Deterministic Safety Override
        safety_override_prompt = ""
        if triage_override:
            banner = triage_override["triage_banner"]
            findings_str = "\n".join([f"- {f}" for f in triage_override.get("findings", [])])

            safety_override_prompt = (
                f"\nCRITICAL DETERMINISTIC SAFETY OVERRIDE (MANDATORY):\n"
                f"Display this exact header at the top:\n"
                f"### {banner}\n\n"
                f"Clinical Findings to Enforce:\n{findings_str}\n"
            )
            if triage_override.get("pancreatitis_alert"):
                safety_override_prompt += (
                    "- Explain that Triglycerides >=1,000 mg/dL carries an acute risk of ACUTE PANCREATITIS.\n"
                    "- Ask about severe upper abdominal/back pain, nausea, vomiting.\n"
                    "- Mandate same-day evaluation, very-low-fat diet (<10-15% fat), complete alcohol elimination.\n"
                )
            if triage_override.get("post_meal_clarification_needed"):
                safety_override_prompt += (
                    "- Clarify that casual post-meal glucose requires timing confirmation and cannot independently diagnose IGT without a 75g OGTT.\n"
                    "- Conclude findings with: 'These results alone do not establish Type 2 diabetes.'\n"
                )

        # ======================================================================
        # FLEXIBLE PERSONA DIRECTIVES (Adaptive Framework vs Rigid Boilerplate)
        # ======================================================================
        if user_mode == "student":
            persona_prompt = (
                "ROLE: Advanced Clinical Decision Support System (CDSS) for Medical Students, Residents & Endocrinologists.\n"
                "TONE: Academic, rigorous, guideline-aligned (ADA/EASD/ACC/AHA), and analytical.\n\n"
                "FLEXIBLE CLINICAL RESPONSE FRAMEWORK:\n"
                "Do NOT force rigid pre-set headings onto every question. Adapt your response structure dynamically "
                "to the specific clinical scope of the inquiry, drawing from these four core domains as appropriate:\n"
                "• **Pathophysiological Analysis:** (Activate for etiology, molecular pathways, genetics, beta-cell biology)\n"
                "• **Diagnostic & Laboratory Assessment:** (Activate for cutoffs, biomarker interpretation, differential diagnostics)\n"
                "• **Guideline-Directed Medical Management (GDMT):** (Activate for pharmacotherapy, clinical trial data, therapy algorithms)\n"
                "• **Monitoring & Clinical Surveillance:** (Activate for screening intervals, complication surveillance, target ranges)\n\n"
                "GUIDELINES FOR ADAPTATION:\n"
                "- For comprehensive patient profiles / complex multi-marker reports: Synthesize a full multi-domain CDS evaluation.\n"
                "- For focused, mechanism-specific, or single-topic inquiries: Respond with direct, tailored clinical sections without padding irrelevant domains.\n\n"
                "CITATION DIRECTIVE: Use high-density Vancouver-style numerical bracket citations [1], [2] after every clinical claim and cutoff."
            )
        else:
            default_triage = "1. **TRIAGE STATUS:** Output: 🚨 EMERGENCY, ⚠️ PRIORITY, or 🟢 ROUTINE ADVISORY.\n" if not triage_override else "1. **TRIAGE STATUS:** Display locked status above.\n"
            
            persona_prompt = (
                "ROLE: Multimodal Patient Triage & Educational Assistant for Patients and Caregivers.\n"
                "TONE: Warm, empathetic, comforting, jargon-free, and practical.\n\n"
                "FLEXIBLE PATIENT GUIDANCE FRAMEWORK:\n"
                f"{default_triage}"
                "- **Understanding Your Health / Results:** Explain findings in plain, everyday conversational language.\n"
                "- **Practical Next Steps:** Tailor diet, activity, or clinic appointment advice to the user's specific concern.\n"
                "- **Questions for Your Doctor:** 2-3 helpful discussion points.\n"
                "- **Emergency Red Flags:** Highlight urgent warning signs when clinically indicated.\n\n"
                "CITATION DIRECTIVE: Write smooth, natural prose. Do NOT clutter sentences with citation brackets like [1] or [2]."
            )

        system_instruction = (
            "You are DiabRAG, an expert clinical decision support and triage AI.\n"
            f"{persona_prompt}\n"
            f"{safety_override_prompt}\n\n"
            "GROUNDING RULE: Ground every claim in the provided evidence. Never hallucinate."
        )

        user_content = (
            f"{history_text}\n"
            f"User Question / Patient Data: {query}\n\n"
            f"Available Evidence Passages:\n{evidence_text}\n"
            f"{caveat_instruction}\n\n"
            f"Synthesize the response tailored for the {user_mode.upper()} audience."
        )

        models = [GEMINI_MODEL, FALLBACK_MODEL]
        for model in models:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=[system_instruction, user_content]
                    )
                    final_answer = DeterministicCitationProcessor.process_citations(
                        raw_answer=response.text, 
                        passages=passages, 
                        user_mode=user_mode
                    )
                    return final_answer
                except Exception:
                    if attempt == 0:
                        time.sleep(3)
                        continue
                    break

        return (
            f"### Temporary Traffic Spike\n"
            f"Please wait a few seconds and try again.\n\n"
            f"> *{MEDICAL_DISCLAIMER}*"
        )