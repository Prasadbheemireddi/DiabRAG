"""
Global Configuration for Diabetes Adaptive RAG.
Reads environment variables from .env with fallback defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# Root directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths from .env
CHROMA_PERSIST_DIR_STR = os.getenv("CHROMA_PERSIST_DIR", "data/vector_db")
BM25_INDEX_DIR_STR = os.getenv("BM25_INDEX_DIR", "data/processed")

VECTOR_DB_DIR = BASE_DIR / CHROMA_PERSIST_DIR_STR
PROCESSED_DATA_DIR = BASE_DIR / BM25_INDEX_DIR_STR
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Ensure all target directories exist
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Google Gemini API & Model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Embedding Model
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", 
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Logging Level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Chunking Parameters
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# In config.py:

# Retrieval & Fusion Hyperparameters (Raised to provide multi-angle context)
TOP_K_BASIC = 4
TOP_K_HYBRID = 7
TOP_K_MULTI_QUERY = 7
RRF_K_CONSTANT = 60

# Fully Verified, Active Medical Source Endpoints
DATA_SOURCES = {
    "CDC_Overview": "https://www.cdc.gov/diabetes/about/index.html",
    "CDC_Symptoms": "https://www.cdc.gov/diabetes/signs-symptoms/index.html",
    "CDC_Risk_Factors": "https://www.cdc.gov/diabetes/risk-factors/index.html",
    "MedlinePlus_Tests": "https://medlineplus.gov/lab-tests/diabetes-tests/",
    "MedlinePlus_Glucose": "https://medlineplus.gov/bloodglucose.html",
    "MedlinePlus_Lipids": "https://medlineplus.gov/cholesterollevelswhatyouneedtoknow.html",
    "MedlinePlus_Autoantibodies": "https://medlineplus.gov/lab-tests/autoantibody-testing/",
    "MedlinePlus_CPeptide": "https://medlineplus.gov/lab-tests/c-peptide-test/",
    "MedlinePlus_Pregnancy": "https://medlineplus.gov/diabetesandpregnancy.html",
    "MedlinePlus_Ketones": "https://medlineplus.gov/lab-tests/ketones-in-blood/",
    "NIH_LADA": "https://www.ncbi.nlm.nih.gov/books/NBK557897/",
    "NIDDK_MODY": "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/monogenic-neonatal-mellitus-mody",
    "WHO_Factsheet": "https://www.who.int/news-room/fact-sheets/detail/diabetes"
}

# Standard Clinical Disclaimer
# Formal Clinical Decision Support Disclaimer (FDA/IMDRF Aligned)
MEDICAL_DISCLAIMER = (
    "Clinical Decision Support (CDSS) Notice: DiabRAG is an informational decision-support tool "
    "grounded in published guidelines (ADA, ACC/AHA, WHO, CDC, NIH). It is designed to assist "
    "in clinical education and risk stratification. It does not replace independent clinical judgment, "
    "formal medical diagnosis, or personalized patient treatment plans."
)