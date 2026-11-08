"""
Deterministic Clinical Triage & Biomarker Safety Engine.
Enforces ADA, ACC/AHA, and Endocrine Society guideline guardrails.
"""

from typing import Any, Dict, List, Optional
import re
from utils.logger import setup_logger

logger = setup_logger("ClinicalTriageEngine")

class ClinicalTriageEngine:
    @staticmethod
    def extract_biomarkers(text: str) -> Dict[str, float]:
        """Extracts numerical biomarkers including fasting vs postprandial glucose."""
        biomarkers = {}
        t_clean = text.replace(",", "").lower()

        # 1. Post-Meal / Postprandial Glucose (mg/dL)
        pp_match = re.search(
            r"(?:post[- ]?(?:meal|prandial)(?:\s+(?:plasma\s+)?glucose|\s+blood\s+sugar)?|ppbs|ppbg|2[- ]?hour\s+(?:post[- ]?meal|ogtt))[\s:]+([0-9]+(?:\.[0-9]+)?)",
            t_clean
        )
        if pp_match:
            biomarkers["post_meal_glucose"] = float(pp_match.group(1))

        # 2. Fasting Glucose / Blood Sugar (mg/dL)
        fpg_match = re.search(
            r"(?:fasting\s+(?:plasma\s+)?glucose|fasting\s+blood\s+sugar|fpg|fbs)[\s:]+([0-9]+(?:\.[0-9]+)?)",
            t_clean
        )
        if fpg_match:
            biomarkers["fasting_glucose"] = float(fpg_match.group(1))
        elif "glucose" in t_clean and "post_meal_glucose" not in biomarkers:
            g_match = re.search(r"(?:glucose|blood\s+sugar)[\s:]+([0-9]+(?:\.[0-9]+)?)", t_clean)
            if g_match:
                biomarkers["fasting_glucose"] = float(g_match.group(1))

        # 3. HbA1c (%)
        a1c_match = re.search(r"(?:hba1c|a1c|glycated\s+hemoglobin)[\s:]+([0-9]+(?:\.[0-9]+)?)", t_clean)
        if a1c_match:
            biomarkers["hba1c"] = float(a1c_match.group(1))

        # 4. Triglycerides (mg/dL)
        tg_match = re.search(r"triglycerides?[\s:]+([0-9]+(?:\.[0-9]+)?)", t_clean)
        if tg_match:
            biomarkers["triglycerides"] = float(tg_match.group(1))

        # 5. Cholesterol & Ketones
        tc_match = re.search(r"total\s+cholesterol[\s:]+([0-9]+(?:\.[0-9]+)?)", t_clean)
        if tc_match:
            biomarkers["total_cholesterol"] = float(tc_match.group(1))
        hdl_match = re.search(r"hdl(?:\s+cholesterol)?[\s:]+([0-9]+(?:\.[0-9]+)?)", t_clean)
        if hdl_match:
            biomarkers["hdl"] = float(hdl_match.group(1))

        if any(k in t_clean for k in ["ketones: positive", "ketones positive", "ketones: moderate", "ketones: large", "urine ketones: positive"]):
            biomarkers["ketones_positive"] = 1.0

        return biomarkers

    @classmethod
    def evaluate_triage(cls, text: str) -> Optional[Dict[str, Any]]:
        """Deterministically assigns clinical urgency tiers based on multi-biomarker hierarchy."""
        biomarkers = cls.extract_biomarkers(text)
        if not biomarkers:
            return None

        t_clean = text.lower()
        findings: List[str] = []
        urgency_level = "ROUTINE"
        triage_banner = "🟢 TRIAGE STATUS: ROUTINE HEALTH ADVISORY (Within Normal Limits)"
        pancreatitis_alert = False
        dka_alert = False
        hypo_alert = False
        post_meal_clarification_needed = False

        # --- 1. SEVERE LIPID GUARDRAIL (Pancreatitis Risk) ---
        tg = biomarkers.get("triglycerides")
        if tg is not None:
            if tg >= 1000.0:
                urgency_level = "EMERGENCY_REVIEW"
                pancreatitis_alert = True
                triage_banner = "🚨 TRIAGE: URGENT SAME-DAY MEDICAL REVIEW (Pancreatitis Risk Alert)"
                findings.append(
                    f"Triglycerides of {tg:,.0f} mg/dL is SEVERELY ELEVATED (≥1,000 mg/dL threshold), "
                    "which carries a substantial risk of acute pancreatitis requiring immediate clinician assessment."
                )
            elif tg >= 500.0:
                urgency_level = "PRIORITY"
                triage_banner = "⚠️ TRIAGE STATUS: PRIORITY CLINICAL REVIEW (Severe Hypertriglyceridemia)"
                findings.append(f"Triglycerides of {tg:,.0f} mg/dL is severely elevated (≥500 mg/dL).")

        # --- 2. GLYCEMIC GUARDRAILS: FASTING GLUCOSE & KETONES ---
        fpg = biomarkers.get("fasting_glucose")
        has_ketones = biomarkers.get("ketones_positive", 0.0) == 1.0

        if fpg is not None:
            if fpg < 54.0:
                urgency_level = "EMERGENCY"
                hypo_alert = True
                triage_banner = "🚨 TRIAGE STATUS: CRITICAL EMERGENCY (Severe Hypoglycemia <54 mg/dL)"
                findings.append(f"Fasting Glucose: {fpg} mg/dL is dangerously low (<54 mg/dL).")
            elif fpg >= 250.0 and has_ketones:
                urgency_level = "EMERGENCY"
                dka_alert = True
                triage_banner = "🚨 TRIAGE STATUS: CRITICAL EMERGENCY (Suspected Diabetic Ketoacidosis)"
                findings.append(f"Fasting Glucose: {fpg} mg/dL with Positive Ketones indicates acute DKA risk.")
            elif fpg >= 126.0:
                if urgency_level not in ["EMERGENCY_REVIEW", "EMERGENCY"]:
                    urgency_level = "PRIORITY"
                    triage_banner = "⚠️ TRIAGE STATUS: PRIORITY OUTPATIENT EVALUATION (Diabetes Diagnostic Threshold Met)"
                findings.append(f"Fasting Glucose: {fpg} mg/dL meets diabetes threshold (≥126 mg/dL) requiring confirmatory testing.")
            elif 100.0 <= fpg <= 125.0:
                if urgency_level == "ROUTINE":
                    urgency_level = "NON_EMERGENCY_FOLLOWUP"
                    triage_banner = "🟡 TRIAGE: NON-EMERGENCY MEDICAL FOLLOW-UP (Prediabetes Range)"
                findings.append(
                    f"Fasting Blood Glucose: {fpg:.0f} mg/dL falls within the prediabetes range (100–125 mg/dL), "
                    "also termed Impaired Fasting Glucose (IFG)."
                )

        # --- 3. POST-MEAL GLUCOSE INTERPRETATION ---
        ppg = biomarkers.get("post_meal_glucose")
        if ppg is not None:
            if ppg >= 200.0 and any(s in t_clean for s in ["thirsty", "polyuria", "weight loss", "fatigue"]):
                if urgency_level not in ["EMERGENCY_REVIEW", "EMERGENCY"]:
                    urgency_level = "PRIORITY"
                    triage_banner = "⚠️ TRIAGE STATUS: PRIORITY OUTPATIENT EVALUATION (Hyperglycemia with Symptoms)"
                findings.append(f"Post-meal glucose of {ppg} mg/dL with symptoms suggests potential diabetes requiring confirmation.")
            elif 140.0 <= ppg <= 199.0:
                post_meal_clarification_needed = True
                if urgency_level == "ROUTINE":
                    urgency_level = "NON_EMERGENCY_FOLLOWUP"
                    triage_banner = "🟡 TRIAGE: NON-EMERGENCY MEDICAL FOLLOW-UP"
                findings.append(
                    f"Post-meal glucose of {ppg:.0f} mg/dL requires clarification regarding the exact time elapsed after eating "
                    "and test methodology. An isolated casual post-meal reading cannot independently establish Impaired Glucose Tolerance (IGT) "
                    "or Type 2 diabetes without a standardized 2-hour 75g OGTT."
                )

        # --- 4. HBA1C STRATIFICATION ---
        a1c = biomarkers.get("hba1c")
        if a1c is not None:
            if a1c >= 6.5:
                if urgency_level not in ["EMERGENCY_REVIEW", "EMERGENCY"]:
                    urgency_level = "PRIORITY"
                    triage_banner = "⚠️ TRIAGE STATUS: PRIORITY OUTPATIENT EVALUATION (HbA1c ≥6.5%)"
                findings.append(f"HbA1c: {a1c}% meets diabetes criteria (≥6.5%) requiring clinical confirmation.")
            elif 5.7 <= a1c <= 6.4:
                if urgency_level == "ROUTINE":
                    urgency_level = "NON_EMERGENCY_FOLLOWUP"
                    triage_banner = "🟡 TRIAGE: NON-EMERGENCY MEDICAL FOLLOW-UP (Prediabetes Range)"
                findings.append(f"HbA1c: {a1c}% indicates prediabetes (5.7%–6.4%).")

        return {
            "urgency_level": urgency_level,
            "triage_banner": triage_banner,
            "pancreatitis_alert": pancreatitis_alert,
            "dka_alert": dka_alert,
            "hypo_alert": hypo_alert,
            "post_meal_clarification_needed": post_meal_clarification_needed,
            "biomarkers": biomarkers,
            "findings": findings
        }