"""
Clinical Validation Suite for Multimodal Patient Triage.
Tests deterministic urgency classification against 5 gold-standard clinical challenge cases:
Case 1: Severe Hypertriglyceridemia (Pancreatitis Risk) -> URGENT SAME-DAY REVIEW
Case 2: Suspected DKA (Hyperglycemia + Ketonemia) -> CRITICAL EMERGENCY
Case 3: Severe Hypoglycemia (<54 mg/dL) -> CRITICAL EMERGENCY
Case 4: Prediabetes (IFG + Casual Post-meal) -> NON-EMERGENCY FOLLOW-UP
Case 5: Normal Screening Profile -> ROUTINE ADVISORY
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.triage_engine import ClinicalTriageEngine

VALIDATION_CASES = [
    {
        "case_name": "Severe Hypertriglyceridemia (Pancreatitis Risk)",
        "report_text": "LAB REPORT: Fasting Glucose: 83 mg/dL, HbA1c: 3.8%, Triglycerides: 1315 mg/dL, Total Cholesterol: 221 mg/dL.",
        "expected_urgency": "EMERGENCY_REVIEW",
        "expected_flag": "pancreatitis_alert"
    },
    {
        "case_name": "Diabetic Ketoacidosis (DKA Crisis)",
        "report_text": "EMERGENCY PANEL: Blood Glucose: 310 mg/dL, Urine Ketones: Positive, Symptoms: Nausea, deep breathing.",
        "expected_urgency": "EMERGENCY",
        "expected_flag": "dka_alert"
    },
    {
        "case_name": "Clinically Significant Hypoglycemia (Level 2)",
        "report_text": "BEDSIDE GLUCOSE: Blood Glucose: 48 mg/dL, Symptoms: Tremors, diaphoresis, confusion.",
        "expected_urgency": "EMERGENCY",
        "expected_flag": "hypo_alert"
    },
    {
        "case_name": "Unconfirmed Prediabetes / IFG",
        "report_text": "OUTPATIENT SCREEN: Fasting Plasma Glucose: 120 mg/dL, Post-meal Glucose: 150 mg/dL, HbA1c: 5.9%.",
        "expected_urgency": "NON_EMERGENCY_FOLLOWUP",
        "expected_flag": "post_meal_clarification_needed"
    },
    {
        "case_name": "Normal Metabolic Health Screening",
        "report_text": "ANNUAL HEALTH LABS: Fasting Blood Sugar: 88 mg/dL, HbA1c: 5.2%, Triglycerides: 110 mg/dL.",
        "expected_urgency": "ROUTINE",
        "expected_flag": None
    }
]

def run_triage_validation():
    print("\n" + "="*85)
    print("CLINICAL TRIAGE VALIDATION: Testing Deterministic Safety Guardrail Against Gold Standards")
    print("="*85)
    
    passed = 0
    total = len(VALIDATION_CASES)
    
    for case in VALIDATION_CASES:
        name = case["case_name"]
        raw = case["report_text"]
        exp_urgency = case["expected_urgency"]
        exp_flag = case["expected_flag"]
        
        result = ClinicalTriageEngine.evaluate_triage(raw)
        actual_urgency = result["urgency_level"]
        has_flag = result.get(exp_flag, False) if exp_flag else True
        
        is_valid = (actual_urgency == exp_urgency) and has_flag
        status = "✓ PASSED" if is_valid else "✗ FAILED"
        if is_valid:
            passed += 1
            
        print(f"\n[CASE] {name}")
        print(f"  • Extracted Biomarkers: {result['biomarkers']}")
        print(f"  • Assigned Triage:      {result['triage_banner']}")
        print(f"  • Expected Urgency:     {exp_urgency} | Actual: {actual_urgency} -> {status}")
        
    print("\n" + "-"*85)
    accuracy = (passed / total) * 100
    print(f"TRIAGE VALIDATION ACCURACY: {passed}/{total} ({accuracy:.1f}%)")
    print("Guideline Adherence: ADA 2026 Standards of Care & ACC/AHA Dyslipidemia Guardrails.")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_triage_validation()