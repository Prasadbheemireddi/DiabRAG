"""
DiabRAG: A Self-Correcting Adaptive RAG Framework for Clinical Diabetes Decision Support & Multimodal Patient Triage.
Features Dashboard Landing Page, Isolated Dual-Portal Chat Histories (Patient vs Medical Student),
In-bar '+' Multimodal Attachments, and Telemetry Audit.
"""

import streamlit as st
from config import MEDICAL_DISCLAIMER
from modules.document_loader import DocumentLoader
from modules.text_cleaner import TextCleaner
from modules.text_chunker import TextChunker
from modules.vector_store import VectorStore
from modules.bm25_retriever import BM25Retriever
from modules.adaptive_rag import DiabetesAdaptiveRAG
from modules.patient_document_parser import PatientDocumentParser

st.set_page_config(
    page_title="DiabRAG - Clinical CDS & Multimodal Patient Triage",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        border-radius: 8px; 
        padding: 8px 12px; 
        border: 1px solid rgba(255, 255, 255, 0.1); 
    }
    .self-correct-badge {
        background-color: rgba(0, 200, 83, 0.15);
        color: #00e676;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(0, 230, 118, 0.3);
        display: inline-block;
        margin-bottom: 8px;
    }
    .multimodal-chip {
        background-color: rgba(33, 150, 243, 0.15);
        color: #64b5f6;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(33, 150, 243, 0.3);
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_system():
    vector_store = VectorStore()
    bm25 = BM25Retriever()

    if vector_store.count() == 0 or not bm25.corpus_path.exists():
        with st.spinner("Initializing Clinical Knowledge Base from CDC, WHO, MedlinePlus & NIH..."):
            loader = DocumentLoader()
            raw_docs = loader.load_all_documents()
            
            cleaner = TextCleaner()
            cleaned_docs = cleaner.clean_documents(raw_docs)
            
            chunker = TextChunker()
            chunks = chunker.chunk_documents(cleaned_docs)
            
            vector_store.add_chunks(chunks)
            bm25.build_index(chunks)

    return DiabetesAdaptiveRAG()

# --- ISOLATED SESSION STATES FOR EACH PORTAL ---
if "patient_chat_history" not in st.session_state:
    st.session_state["patient_chat_history"] = []

if "student_chat_history" not in st.session_state:
    st.session_state["student_chat_history"] = []

if "active_portal" not in st.session_state:
    st.session_state["active_portal"] = None  # None = Show Dashboard Landing Page

rag_engine = initialize_system()

# ==============================================================================
# VIEW 1: DASHBOARD LANDING PAGE
# ==============================================================================
if st.session_state["active_portal"] is None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("🩺 DiabRAG")
    st.markdown("#### A Self-Correcting Adaptive RAG Framework for Clinical Diabetes Decision Support & Multimodal Patient Triage")
    st.markdown(
        "Evidence-grounded medical intelligence featuring dynamic pipeline routing, "
        "autonomous evidence sufficiency evaluation, and multimodal report parsing across official guidelines (CDC, WHO, MedlinePlus, NIH)."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Select Your Operational Portal:")

    col_patient, col_student = st.columns(2, gap="large")

    # --- CARD 1: PATIENT TRIAGE PORTAL ---
    with col_patient:
        with st.container(border=True):
            st.markdown("### 🧑‍🤝‍🧑 Multimodal Patient Triage Portal")
            st.caption("Patient-centric symptom analysis, lab report interpretation, and safety grading")
            st.markdown("""
            * **3-Tier Triage Status:** Automatic urgency assessment (🔴 Emergency / 🟡 Priority / 🟢 Routine).
            * **Multimodal Report Ingestion:** Attach blood test PDFs or photos directly from the chat bar.
            * **Plain-Language Explanations:** Clear definitions of medical terminology and symptoms.
            * **Actionable Next Steps:** Practical dietary guidance and questions to bring to your physician.
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Enter Patient Triage Portal", type="primary", use_container_width=True):
                st.session_state["active_portal"] = "patient"
                st.rerun()

    # --- CARD 2: CLINICAL DECISION SUPPORT PORTAL ---
    with col_student:
        with st.container(border=True):
            st.markdown("### 🎓 Clinical Decision Support (CDS) Portal")
            st.caption("Evidence-based decision support for medical students, residents & endocrinologists")
            st.markdown("""
            * **Guideline-Directed Management (GDMT):** Management algorithms aligned with ADA/EASD standards.
            * **Diagnostic Differentiation:** Distinguish T1D, T2D, LADA, and MODY via autoantibodies & C-peptide.
            * **Molecular Pathophysiology:** Beta-cell destruction, insulin resistance mechanisms, and genetics.
            * **Self-Reflective Retrieval:** Dynamic multi-query formulation and multi-hop fact verification.
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔬 Enter Clinical CDS Portal", type="primary", use_container_width=True):
                st.session_state["active_portal"] = "student"
                st.rerun()

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.caption(
        f"<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>"
        f"{MEDICAL_DISCLAIMER}</div>", 
        unsafe_allow_html=True
    )

# ==============================================================================
# VIEW 2: DEDICATED WORKSPACE (Isolated Chat Histories)
# ==============================================================================
else:
    active_mode = st.session_state["active_portal"]
    is_student = (active_mode == "student")
    
    # Active portal's isolated history key
    history_key = "student_chat_history" if is_student else "patient_chat_history"
    active_chat_history = st.session_state[history_key]

    nav_col1, nav_col2, nav_col3 = st.columns([1.5, 4, 1])
    with nav_col1:
        if st.button("← Switch Portal", use_container_width=True):
            st.session_state["active_portal"] = None
            st.rerun()
    with nav_col2:
        if is_student:
            st.subheader("🎓 Clinical Decision Support (CDS) Workspace")
        else:
            st.subheader("🧑‍🤝‍🧑 Patient Triage & Guidance Workspace")
    with nav_col3:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state[history_key] = []
            st.session_state.pop("preset_patient_data", None)
            st.session_state.pop("preset_query", None)
            st.rerun()

    st.divider()

    # Presets tailored to CDS or Triage
    if is_student:
        st.markdown("**🎓 High-Yield Clinical Decision Scenarios:**")
        cols = st.columns(4)
        with cols[0]:
            if st.button("🔬 T1D, T2D & LADA Differentiation", use_container_width=True):
                st.session_state["preset_query"] = (
                    "How do you clinically differentiate Type 1, Type 2, and LADA using C-peptide levels and islet autoantibodies (GADA, IA-2A)?"
                )
        with cols[1]:
            if st.button("🧪 DKA vs HHS Pathophysiology", use_container_width=True):
                st.session_state["preset_query"] = (
                    "Compare the pathophysiology, arterial pH, anion gap, and hyperosmolality differences between DKA and HHS."
                )
        with cols[2]:
            if st.button("💊 SGLT2i vs GLP-1 RA Indications", use_container_width=True):
                st.session_state["preset_query"] = (
                    "Detail the mechanisms of action, renal/cardiovascular outcomes, and indications of SGLT2 inhibitors versus GLP-1 receptor agonists."
                )
        with cols[3]:
            if st.button("🧬 MODY 2 vs MODY 3 Genetics", use_container_width=True):
                st.session_state["preset_query"] = (
                    "Outline the molecular genetics and therapeutic distinctions between MODY 2 (GCK) and MODY 3 (HNF1A)."
                )
    else:
        st.markdown("**🧑‍🤝‍🧑 Common Patient & Triage Inquiries:**")
        cols = st.columns(4)
        with cols[0]:
            if st.button("💧 Thirsty All the Time?", use_container_width=True):
                st.session_state["preset_query"] = (
                    "I am feeling thirsty all the time, even after drinking plenty of water. Could this be a sign of diabetes?"
                )
        with cols[1]:
            if st.button("🩺 Warning Signs of Type 2", use_container_width=True):
                st.session_state["preset_query"] = (
                    "What are the most common early warning signs and symptoms of Type 2 diabetes that I should watch for?"
                )
        with cols[2]:
            if st.button("📉 Hypoglycemia: 15-15 Rule", use_container_width=True):
                st.session_state["preset_query"] = (
                    "What is hypoglycemia, what are the symptoms of low blood sugar, and how do I manage it using the 15-15 rule?"
                )
        with cols[3]:
            if st.button("📋 Sample: High-Risk T2D Lab", use_container_width=True):
                st.session_state["preset_patient_data"] = (
                    "PATIENT LAB REPORT:\n"
                    "- Fasting Plasma Glucose: 168 mg/dL\n"
                    "- HbA1c: 8.2%\n"
                    "- Urine Ketones: Negative\n"
                    "- C-Peptide: 2.1 ng/mL\n"
                    "- Symptoms: Frequent nighttime urination, persistent fatigue."
                )
                st.session_state["preset_query"] = "Interpret my lab values, determine my triage status, and outline questions for my doctor."

    st.markdown("<br>", unsafe_allow_html=True)

    # Render ONLY this portal's isolated chat history (Chronological)
    for turn in active_chat_history:
        with st.chat_message("user"):
            if turn.get("image_bytes"):
                st.image(turn["image_bytes"], caption=f"📸 Uploaded Medical Lab Photo: {turn.get('doc_name')}", width=320)
            elif turn.get("has_document"):
                st.markdown(f"<div class='multimodal-chip'>📄 {turn.get('doc_type', 'Document')}: {turn.get('doc_name')}</div>", unsafe_allow_html=True)
            
            st.markdown(f"**{turn.get('display_query', turn['query'])}**")

        with st.chat_message("assistant", avatar="🩺"):
            mode_badge = "🎓 *[Clinical Decision Support]*" if turn.get("user_mode") == "student" else "🧑‍🤝‍🧑 *[Patient Triage]*"
            st.caption(mode_badge)
            st.markdown(turn["answer"])

            with st.expander("🔍 DiabRAG Telemetry & Self-Correction Audit", expanded=False):
                if turn.get("has_document"):
                    st.info(f"👁️ **Multimodal Ingestion:** Processed `{turn.get('doc_name')}` ({turn.get('doc_type')}) via Gemini Vision / Medical OCR.")

                if turn.get("retrieval_round_2_triggered"):
                    st.markdown(
                        "<div class='self-correct-badge'>🛠️ SELF-CORRECTION ACTIVATED: "
                        "Initial evidence was insufficient. Automatically reformulated targeted follow-up query and retrieved supplementary evidence.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("✓ **Sufficiency Verified:** Initial evidence passed fact-checking evaluation in Round 1.")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Complexity", turn["complexity_info"].get("complexity", "medium").upper())
                c2.metric("Pipeline", turn["pipeline_selected"].split()[0])
                c3.metric("Rounds", turn["total_retrieval_rounds"])
                c4.metric("Latency", f"{turn['latency_seconds']} s")

                st.markdown(f"**Classification Reason:** {turn['complexity_info'].get('reason')}")
                eval1 = turn["evaluation_round_1"]
                st.markdown(f"**Round 1 Sufficiency:** `{eval1.get('status', '').upper()}` (Confidence: {eval1.get('confidence', 0):.2f})")

                st.markdown("##### Grounded Evidence Passages")
                for idx, p in enumerate(turn["passages"], 1):
                    st.markdown(f"**[{idx}] {p.get('source')} - {p.get('title')}** | *{p.get('retrieval_strategy')}*")
                    st.text(p.get("text")[:240] + "...")
                    st.markdown(f"[Source URL]({p.get('url')})")
                    st.divider()

    # In-bar '+' Chat Input
    chat_submission = st.chat_input(
        f"Ask DiabRAG ({'CDS Mode' if is_student else 'Triage Mode'}), or click '+' to attach lab PDF/Image...",
        accept_file=True,
        file_type=["pdf", "png", "jpg", "jpeg", "txt"]
    )

    default_query = st.session_state.pop("preset_query", "")
    stored_sample_data = st.session_state.pop("preset_patient_data", None)

    user_query = ""
    active_patient_text = None
    attached_file_name = None
    attached_image_bytes = None
    attached_doc_type = None

    if chat_submission:
        user_query = getattr(chat_submission, "text", "") or ""
        uploaded_files = getattr(chat_submission, "files", []) or []

        if uploaded_files:
            primary_file = uploaded_files[0]
            with st.spinner(f"Analyzing {primary_file.name} with Multimodal Vision OCR..."):
                parsed_res = PatientDocumentParser.process_uploaded_file(primary_file)
                if parsed_res:
                    active_patient_text = parsed_res["text"]
                    attached_file_name = parsed_res["file_name"]
                    attached_doc_type = parsed_res["doc_type"]
                    if parsed_res["is_image"]:
                        attached_image_bytes = parsed_res["file_bytes"]

        if not user_query and active_patient_text:
            user_query = (
                "Please analyze this attached medical report: extract all biomarkers, determine clinical triage status, "
                "and outline evidence-based next steps."
            )

    elif default_query:
        user_query = default_query
        active_patient_text = stored_sample_data
        if stored_sample_data:
            attached_file_name = "Sample T2D Lab Test"
            attached_doc_type = "Sample Laboratory Data"

    if user_query:
        if active_patient_text:
            full_prompt = (
                f"PATIENT MEDICAL DOCUMENT / LAB REPORT ({attached_doc_type}):\n"
                f"```\n{active_patient_text}\n```\n\n"
                f"USER INQUIRY: {user_query}"
            )
        else:
            full_prompt = user_query

        with st.spinner(f"Running DiabRAG ({'CDS' if is_student else 'Triage'})..."):
            result = rag_engine.run(
                query=full_prompt, 
                user_mode=active_mode,
                chat_history=active_chat_history  # Pass only this portal's history
            )
            result["display_query"] = user_query
            result["has_document"] = bool(active_patient_text)
            result["doc_name"] = attached_file_name
            result["doc_type"] = attached_doc_type
            result["image_bytes"] = attached_image_bytes
            result["user_mode"] = active_mode
            
            # Append only to this portal's dedicated history
            st.session_state[history_key].append(result)
            st.rerun()

    # Footer
    st.markdown(
        f"<div style='text-align: center; color: #6c757d; font-size: 0.85rem; padding: 20px;'>"
        f"{MEDICAL_DISCLAIMER}"
        f"</div>",
        unsafe_allow_html=True
    )