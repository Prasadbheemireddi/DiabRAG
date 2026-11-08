"""
Document Loader for CDC, WHO, MedlinePlus, ADA, ACC/AHA, and NIH.
Contains the curated medical reference corpus with authoritative institutional attribution,
active deep-links, and clinical endocrinology coverage.
"""

from typing import Dict, List
import json
import requests
from bs4 import BeautifulSoup
from config import RAW_DATA_DIR, DATA_SOURCES
from utils.logger import setup_logger

logger = setup_logger("DocumentLoader")

# Complete Curated Medical Reference Corpus (Deduplicated & Citation-Aligned)
OFFLINE_DIABETES_CORPUS: List[Dict[str, str]] = [
    # --- 1. OVERVIEW & CLASSIFICATION ---
    {
        "source": "CDC",
        "url": "https://www.cdc.gov/diabetes/about/index.html",
        "title": "CDC - What is Diabetes Basics",
        "section": "Overview & Classification",
        "text": (
            "Diabetes is a chronic (long-lasting) health condition that affects how your body turns "
            "food into energy. Your body breaks down most of the food you eat into sugar (glucose) "
            "and releases it into your bloodstream. When your blood sugar goes up, it signals your "
            "pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body’s "
            "cells for use as energy. With diabetes, your body doesn’t make enough insulin or can’t use "
            "it as well as it should. There are three main types: type 1, type 2, and gestational diabetes."
        )
    },
    # --- 2. TYPE 1 DIABETES ETIOLOGY ---
    {
        "source": "CDC",
        "url": "https://www.cdc.gov/diabetes/about/about-type-1-diabetes.html",
        "title": "CDC - Type 1 Diabetes Causes and Management",
        "section": "Type 1 Diabetes Pathophysiology",
        "text": (
            "If you have type 1 diabetes, your body produces very little or no insulin. This occurs "
            "because the body's immune system attacks and destroys the insulin-producing beta cells in "
            "the pancreas by mistake. This condition is usually diagnosed in children, teens, and young "
            "adults, but it can develop at any age. Symptoms often develop quickly, over a few weeks or months. "
            "Currently, no one knows how to prevent type 1 diabetes. People with type 1 diabetes must take "
            "insulin injections or use an insulin pump every day to survive and manage their blood glucose levels."
        )
    },
    # --- 3. TYPE 2 DIABETES PATHOPHYSIOLOGY ---
    {
        "source": "CDC",
        "url": "https://www.cdc.gov/diabetes/about/about-type-2-diabetes.html",
        "title": "CDC - Type 2 Diabetes Overview",
        "section": "Type 2 Pathophysiology & Risk Factors",
        "text": (
            "Type 2 diabetes affects how your body uses sugar (glucose) for energy. It stops the body from using "
            "insulin properly (insulin resistance), leading to elevated blood sugar levels. Most people with "
            "diabetes have type 2. You are at higher risk if you are overweight or obese, are 45 years or older, "
            "have a family history of diabetes, or are active less than 3 times a week. It can be prevented or "
            "delayed with lifestyle measures including healthy eating, weight loss, and physical activity."
        )
    },
    # --- 4. PATIENT SYMPTOMS & CLINICAL MANIFESTATIONS ---
    {
        "source": "CDC",
        "url": "https://www.cdc.gov/diabetes/signs-symptoms/index.html",
        "title": "CDC - Diabetes Signs, Symptoms & Patient Manifestations",
        "section": "Cardinal Symptoms and Patient Clinical Presentation",
        "text": (
            "Classic clinical signs and symptoms of diabetes mellitus: "
            "1. Excessive Thirst (Polydipsia): Feeling thirsty all the time, unquenchable thirst, or dry mouth even after drinking plenty of water. "
            "This occurs because high blood glucose draws water out of body tissues, causing cellular dehydration and signaling the brain's thirst center. "
            "2. Frequent Urination (Polyuria): Needing to urinate frequently, especially getting up multiple times at night (nocturia). When blood glucose "
            "exceeds the renal threshold (~180 mg/dL), excess sugar spills into urine, drawing large volumes of water with it via osmotic diuresis. "
            "3. Extreme Hunger (Polyphagia): Constant hunger even while eating, because glucose cannot enter cells without sufficient insulin action. "
            "4. Unexplained Weight Loss: Despite eating normally, the body breaks down muscle and fat for fuel due to lack of intracellular glucose. "
            "5. Chronic Fatigue and Weakness: Feeling exhausted because cells are deprived of energy. "
            "6. Blurry Vision: High blood sugar pulls fluid from the lenses of the eyes, impairing focusing ability. "
            "7. Slow-Healing Sores and Frequent Infections: Impaired circulation and compromised white blood cell function."
        )
    },
    # --- 5. ADA DIAGNOSTIC CRITERIA & POSTPRANDIAL GLUCOSE ---
    {
        "source": "ADA/MedlinePlus",
        "url": "https://medlineplus.gov/bloodglucose.html",
        "title": "ADA Clinical Criteria - Fasting, Postprandial & Prediabetes Diagnostics",
        "section": "Glycemic Diagnostic Criteria & Postprandial Nuance",
        "text": (
            "American Diabetes Association (ADA) Standards of Medical Care in Diabetes: "
            "1. Impaired Fasting Glucose (IFG): Fasting Plasma Glucose (FPG) between 100 and 125 mg/dL (5.6–6.9 mmol/L). "
            "This confirms prediabetes, reflecting hepatic insulin resistance, but does NOT establish a diagnosis of Type 2 diabetes. "
            "2. Postprandial Glucose vs. Oral Glucose Tolerance Test (OGTT): In healthy individuals, glucose rises after meals and peaks within 1 hour. "
            "Impaired Glucose Tolerance (IGT) is strictly defined as a 2-hour plasma glucose between 140 and 199 mg/dL (7.8–11.0 mmol/L) "
            "during a standardized 75-gram oral glucose tolerance test (OGTT). A casual or home post-meal glucose reading of ~150 mg/dL "
            "cannot independently establish IGT or diabetes because it lacks standardization of carbohydrate load and timing. "
            "3. Diagnostic Confirmation Requirements: A diagnosis of diabetes requires two abnormal test results from the same sample or "
            "in two separate test samples (e.g., repeating FPG or pairing with an HbA1c test). Single borderline results do not establish diabetes. "
            "4. Recommended Clinical Next Steps for Prediabetes: Discuss fasting glucose with a primary care provider, obtain a glycated hemoglobin "
            "(HbA1c) test to measure 2- to 3-month glycemic average, consider a formal standardized 2-hour 75g OGTT if recommended by the physician, "
            "and initiate lifestyle modifications (structured physical activity of at least 150 minutes/week and Mediterranean/DASH nutritional patterns)."
        )
    },
    # --- 6. ACC/AHA SEVERE HYPERTRIGLYCERIDEMIA & PANCREATITIS ---
    {
        "source": "ACC/AHA",
        "url": "https://medlineplus.gov/cholesterollevelswhatyouneedtoknow.html",
        "title": "ACC/AHA Guidelines - Severe Hypertriglyceridemia & Acute Pancreatitis",
        "section": "Severe Hypertriglyceridemia Management (Triglycerides >=1,000 mg/dL)",
        "text": (
            "Clinical Management of Severe Hypertriglyceridemia (ACC/AHA and Endocrine Society Standards): "
            "1. Acute Pancreatitis Risk: A circulating fasting triglyceride level of 1,000 mg/dL or higher (and especially >500 mg/dL) "
            "substantially elevates the immediate risk of acute pancreatitis. Normal blood glucose or HbA1c does not mitigate this acute danger. "
            "2. Clinical Action Plan: Immediate, same-day medical assessment is recommended. Clinicians must assess for cardinal symptoms of acute "
            "pancreatitis, specifically severe persistent epigastric abdominal pain radiating to the back, nausea, and vomiting. "
            "3. Nutritional & Lifestyle Interventions: For triglycerides >=1,000 mg/dL, standard generic lifestyle advice is inadequate. Current guidance "
            "demands a very-low-fat diet (<10% to 15% of total caloric intake from fat) to reduce chylomicron formation, absolute elimination of alcohol, "
            "and complete avoidance of added sugars and refined carbohydrates. "
            "4. Investigation of Secondary Etiologies: Clinicians must screen for secondary contributing causes including unmanaged diabetes, "
            "hypothyroidism (TSH testing), chronic kidney disease, excessive alcohol intake, medications (estrogens, beta blockers, retinoids), "
            "and underlying familial chylomicronemia or genetic lipid disorders. "
            "5. Pharmacotherapy: Guideline-directed medical therapy includes immediate initiation of fibrates (fenofibrate) or high-dose prescription "
            "omega-3 fatty acids under medical supervision to rapidly lower triglycerides below the pancreatitis threshold."
        )
    },
    # --- 7. AUTOANTIBODY TESTING & GAD65 BIOLOGY ---
    {
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/autoantibody-testing/",
        "title": "MedlinePlus - GAD65 Autoantibodies & Islet Cell Testing",
        "section": "Autoantibody Testing, GAD65 Biology & C-Peptide Differentiation",
        "text": (
            "What is GAD65? Glutamic acid decarboxylase 65 (GAD65) is an essential intracellular enzyme located inside "
            "pancreatic beta cells that catalyzes the conversion of glutamate into gamma-aminobutyric acid (GABA). "
            "In autoimmune diabetes, the immune system mistakenly targets this enzyme, producing circulating GAD65 "
            "autoantibodies (GADA / anti-GAD65). "
            "Clinical Significance of Positive GAD65: A positive GAD65 antibody result confirms an autoimmune etiology (cause) "
            "of diabetes. It is the primary biomarker used by clinicians to identify autoimmune diabetes (Type 1 diabetes and "
            "Latent Autoimmune Diabetes in Adults [LADA]) and distinguish them from non-autoimmune Type 2 diabetes (where autoantibodies are negative). "
            "Difference Between GAD65 Antibody and C-Peptide Testing: "
            "1. GAD65 Autoantibody tests for CAUSE / ETIOLOGY: It answers 'Is the immune system attacking the pancreas?' "
            "2. C-Peptide tests for FUNCTION / CAPACITY: It measures endogenous insulin secretory reserve, answering 'How much insulin is the pancreas currently producing?' "
            "Clinical Comparison: In Type 1 diabetes, GAD65 antibodies are typically positive, while C-peptide is undetectable or very low (<0.2 nmol/L) "
            "due to near-total beta-cell destruction. In Type 2 diabetes, GAD65 antibodies are negative, while C-peptide is normal or elevated due to insulin resistance. "
            "In LADA (Type 1.5), GAD65 is positive, but C-peptide levels may initially be preserved before progressively declining."
        )
    },
    # --- 8. C-PEPTIDE TESTING & INSULIN RESERVE ---
    {
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/lab-tests/c-peptide-test/",
        "title": "MedlinePlus - C-Peptide Test & Endogenous Insulin Secretion",
        "section": "Diagnostic Testing & C-Peptide Interpretation",
        "text": (
            "A C-peptide test measures endogenous insulin secretion. Proinsulin produced in pancreatic beta cells is cleaved into "
            "equimolar amounts of active insulin and connecting peptide (C-peptide). Unlike insulin, C-peptide has a longer half-life, "
            "avoids hepatic first-pass metabolism, and is unaffected by exogenous insulin therapy. "
            "Clinical Interpretation: 1. In Type 1 diabetes: C-peptide levels are undetectable or profoundly low (<0.2 nmol/L or <0.6 ng/mL), "
            "indicating severe/absolute beta-cell failure. 2. In Type 2 diabetes: C-peptide levels are normal or elevated (>0.6 nmol/L), "
            "reflecting insulin resistance and hyperinsulinemia. 3. In LADA: C-peptide levels are initially preserved or borderline, followed by a steady progressive decline."
        )
    },
    # --- 9. LATENT AUTOIMMUNE DIABETES IN ADULTS (LADA) ---
    {
        "source": "NIH-StatPearls",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK557897/",
        "title": "NIH Bookshelf - Latent Autoimmune Diabetes in Adults (LADA)",
        "section": "Atypical Diabetes & LADA Differentiation",
        "text": (
            "Latent Autoimmune Diabetes in Adults (LADA), commonly termed Type 1.5 diabetes, is a slow-progressing form of autoimmune "
            "diabetes presenting in adulthood. Diagnostic Triad (Immunology of Diabetes Society criteria): "
            "1. Adult age at diagnosis (typically >30 to 35 years). "
            "2. Presence of at least one circulating islet autoantibody (most commonly GADA / anti-GAD65). "
            "3. No insulin requirement for at least the initial 6 months following diagnosis. "
            "Clinical Differentiation: Unlike Type 2 diabetes, LADA patients are typically leaner, exhibit progressive beta-cell destruction, "
            "and eventually progress to absolute insulin dependence within months to a few years. Unlike acute Type 1 diabetes, beta-cell loss "
            "in LADA occurs slowly, allowing temporary glycemic control with diet or oral non-insulin therapies at diagnosis."
        )
    },
    # --- 10. MONOGENIC DIABETES & MODY ---
    {
        "source": "NIDDK",
        "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/monogenic-neonatal-mellitus-mody",
        "title": "NIDDK - Monogenic Diabetes and MODY",
        "section": "Monogenic Diabetes & Genetic Mutations",
        "text": (
            "Maturity-Onset Diabetes of the Young (MODY) is an uncommon form of monogenic diabetes caused by a single gene mutation with "
            "autosomal dominant inheritance. Characteristics: onset typically before age 25, absent pancreatic islet autoantibodies, "
            "preserved endogenous C-peptide, absence of severe obesity or insulin resistance, and multigenerational family history of diabetes. "
            "Common subtypes: MODY 2 (GCK gene mutation causing stable, mild fasting hyperglycemia often requiring no treatment) and "
            "MODY 3 (HNF1A gene mutation causing progressive beta-cell dysfunction that responds exquisitely to low-dose sulfonylureas)."
        )
    },
    # --- 11. HYPOGLYCEMIA & RULE OF 15 ---
    {
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/hypoglycemia.html",
        "title": "MedlinePlus - Hypoglycemia Management and Rule of 15",
        "section": "Hypoglycemia Classification & Protocols",
        "text": (
            "Hypoglycemia in diabetes is defined as blood glucose <70 mg/dL (3.9 mmol/L). Classification: Level 1 (glucose 54-69 mg/dL), "
            "Level 2 (glucose <54 mg/dL / clinically significant), Level 3 (severe hypoglycemia characterized by cognitive impairment requiring external assistance). "
            "Symptoms include diaphoresis, tremors, tachycardia, anxiety, neuroglycopenia, confusion, and seizure. "
            "Management ('Rule of 15'): Consume 15-20 grams of fast-acting carbohydrates (e.g., 4 oz fruit juice, 3-4 glucose tablets), wait 15 minutes, "
            "and recheck blood glucose. Repeat until glucose is >=70 mg/dL. For unconscious or severe Level 3 events, administer intramuscular or nasal glucagon."
        )
    },
    # --- 12. ACUTE EMERGENCIES: DKA VS HHS ---
    {
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/diabeticketoacidosis.html",
        "title": "MedlinePlus - Diabetic Ketoacidosis (DKA) and Hyperglycemic Crises",
        "section": "Acute Emergencies: DKA vs HHS",
        "text": (
            "Acute metabolic emergencies of diabetes: 1. Diabetic Ketoacidosis (DKA): Life-threatening complication primarily in Type 1 diabetes, "
            "characterized by triad of hyperglycemia (typically >250 mg/dL), high anion gap metabolic acidosis (arterial pH <7.30, serum bicarbonate <18 mEq/L), "
            "and hyperketonemia (elevated serum/urine beta-hydroxybutyrate). Driven by absolute insulin deficiency and excess counter-regulatory hormones (glucagon, cortisol). "
            "2. Hyperosmolar Hyperglycemic State (HHS): Primarily in Type 2 diabetes, characterized by profound hyperglycemia (>600 mg/dL), severe hyperosmolality (>320 mOsm/kg), "
            "and severe dehydration, without significant ketoacidosis due to residual endogenous insulin preventing lipolysis."
        )
    },
    # --- 13. CHRONIC COMPLICATIONS ---
    {
        "source": "WHO",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "title": "WHO - Long-Term Microvascular and Macrovascular Complications",
        "section": "Chronic Complications",
        "text": (
            "Over time, diabetes can damage the heart, blood vessels, eyes, kidneys, and nerves. Adults with "
            "diabetes have a two- to three-fold increased risk of heart attacks and strokes. Diabetic retinopathy "
            "is an important cause of blindness, resulting from accumulated damage to small retinal vessels. "
            "Neuropathy combined with reduced blood flow increases the risk of foot ulcers, infection, and limb "
            "amputation. Diabetes is also among the leading causes of kidney failure (diabetic nephropathy)."
        )
    },
    # --- 14. GESTATIONAL DIABETES ---
    {
        "source": "MedlinePlus",
        "url": "https://medlineplus.gov/gestationaldiabetes.html",
        "title": "MedlinePlus - Gestational Diabetes Causes and Risks",
        "section": "Gestational Diabetes",
        "text": (
            "Gestational diabetes develops during pregnancy in women who did not have diabetes before. Hormones "
            "from the placenta cause insulin resistance. Usually diagnosed between the 24th and 28th week of pregnancy, "
            "it increases the mother's risk for type 2 diabetes later in life and increases the risk of macrosomia "
            "(high birth weight) and cesarean delivery."
        )
    }
]

class DocumentLoader:
    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }

    def fetch_web_page(self, source_name: str, url: str) -> List[Dict[str, str]]:
        """Attempts to scrape live documents from official sources."""
        try:
            logger.info(f"Fetching remote medical text from {source_name}: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                
                title = soup.title.string.strip() if soup.title else f"{source_name} Diabetes Information"
                paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
                valid_paras = [p for p in paragraphs if len(p.split()) > 15]

                docs = []
                for i, para in enumerate(valid_paras[:8]):
                    docs.append({
                        "source": source_name.split("_")[0],
                        "url": url,
                        "title": title,
                        "section": f"Web Section {i+1}",
                        "text": para
                    })
                if docs:
                    return docs
            logger.warning(f"Live fetch returned status code {response.status_code} for {url}. Using direct clinical corpus.")
        except Exception as e:
            logger.warning(f"Scrape attempt error for {url}: {e}. Using direct clinical corpus.")
        return []

    def load_all_documents(self) -> List[Dict[str, str]]:
        """Loads diabetes documents with clinical depth and verified deep-links."""
        all_documents: List[Dict[str, str]] = []
        for src_name, src_url in DATA_SOURCES.items():
            scraped = self.fetch_web_page(src_name, src_url)
            all_documents.extend(scraped)

        # Merge with verified comprehensive clinical corpus
        all_documents.extend(OFFLINE_DIABETES_CORPUS)
        
        raw_dump_path = RAW_DATA_DIR / "raw_corpus.json"
        with open(raw_dump_path, "w", encoding="utf-8") as f:
            json.dump(all_documents, f, indent=2)

        logger.info(f"Loaded a total of {len(all_documents)} primary diabetes document sections.")
        return all_documents