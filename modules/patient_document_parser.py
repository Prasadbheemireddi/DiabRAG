"""
Patient Medical Document & Image Parser.
Extracts clinical text from PDFs, text files, and lab report photos/images
using pypdf and Gemini Vision multimodal OCR.
"""

from typing import Dict, Any, Optional
import io
import re
from pypdf import PdfReader
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL
from utils.logger import setup_logger

logger = setup_logger("PatientDocParser")

class PatientDocumentParser:
    @staticmethod
    def process_uploaded_file(uploaded_file) -> Optional[Dict[str, Any]]:
        """
        Parses uploaded file and returns a structured dictionary:
        {
            'text': extracted_text,
            'file_name': name,
            'is_image': bool,
            'file_bytes': bytes,
            'doc_type': 'Medical Photo / Scan' | 'PDF Laboratory Report' | 'Text Document'
        }
        """
        if uploaded_file is None:
            return None

        file_name = uploaded_file.name.lower()
        file_bytes = uploaded_file.getvalue()
        extracted_text = ""
        is_image = False
        doc_type = "Document"

        try:
            # 1. Handle Medical Lab Images (PNG, JPG, JPEG, WEBP) via Gemini Vision
            if any(file_name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                is_image = True
                doc_type = "Medical Photo / Image Scan"
                logger.info(f"Analyzing medical report image with Gemini Vision: {uploaded_file.name}")
                
                client = genai.Client(api_key=GEMINI_API_KEY)
                mime = uploaded_file.type or "image/jpeg"
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=file_bytes, mime_type=mime),
                        (
                            "You are an expert medical lab multimodal analyst. "
                            "Visually inspect this medical report/image and transcribe:\n"
                            "1. Patient Name/ID and Date (if visible).\n"
                            "2. Every laboratory biomarker test name, numerical result, unit, and standard reference interval.\n"
                            "3. Note any visually flagged abnormal results (e.g., printed bold text, [H] for high, [L] for low, red asterisks).\n"
                            "4. Transcribe any doctor's handwritten notes, remarks, or diagnostic impressions.\n"
                            "Format the output as clean, structured clinical text."
                        )
                    ]
                )
                extracted_text = response.text or ""

            # 2. Handle PDF Lab Documents via pypdf
            elif file_name.endswith(".pdf"):
                doc_type = "PDF Laboratory Report"
                pdf_reader = PdfReader(io.BytesIO(file_bytes))
                for page_idx, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        extracted_text += f"\n--- Page {page_idx + 1} ---\n" + text

            # 3. Handle Plain Text
            elif file_name.endswith(".txt"):
                doc_type = "Text Clinical Document"
                extracted_text = file_bytes.decode("utf-8", errors="ignore")

            cleaned = re.sub(r"\s+", " ", extracted_text).strip()
            truncated = cleaned[:6500]
            
            logger.info(f"Multimodal Parser extracted {len(truncated)} chars from {uploaded_file.name} [{doc_type}]")
            
            return {
                "text": truncated,
                "file_name": uploaded_file.name,
                "is_image": is_image,
                "file_bytes": file_bytes,
                "doc_type": doc_type
            }

        except Exception as e:
            logger.error(f"Error in multimodal processing of {uploaded_file.name}: {e}")
            return None