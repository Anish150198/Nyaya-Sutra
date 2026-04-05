"""
Streamlit entry point for Nyaya-Sahayak.
Referenced by app.yaml for Databricks App deployment.
"""

import sys
from pathlib import Path
import time
import base64
import os
import re
import requests
from fpdf import FPDF

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from app.state_manager import init_session_state, clear_chat
from app.components.chat_view import render_chat_view
from app.components.scheme_wizard import render_scheme_wizard
from app.components.performance_dashboard import render_performance_dashboard

# --- TEAMMATE's ORCHESTRATOR ---
# Assuming orchestrator.handle exists in their setup
try:
    from agents.orchestrator import handle as orchestrator_handle
except ImportError:
    orchestrator_handle = None

# --- YOUR LEGAL ENGINE ---
from legal_engine import intake_agent, retrieval_agent, drafting_agent

# ==========================================
# 1. PAGE CONFIG & CSS
# ==========================================
st.set_page_config(
    page_title="Nyaya-Sahayak | Legal & Welfare AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Your Custom CSS injected alongside teammate's UI
st.markdown("""
<style>
    .stDownloadButton > button {
        background-color: #ed8936;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 12px 20px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }
    .stDownloadButton > button:hover {
        background-color: #dd6b20;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (TRANSLATION & PDF)
# ==========================================
import google.generativeai as genai

def translate_with_gemini(text, target_language, api_key="YOUR_GEMINI_API_KEY"):
    """Hook for Gemini to handle Indic Translation on the fly."""
    if target_language == "English":
        return text
        
    genai.configure(api_key=api_key)
    # Using the flash model per the teammate's implementation
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    system_prompt = f"""You are an expert Indian legal translator. 
    Translate the following formal legal document from English into {target_language}. 
    
    CRITICAL CONSTRAINTS:
    1. Maintain all formatting, line breaks, and bullet points exactly as they are.
    2. Use formal legal terminology appropriate for {target_language}.
    3. Return ONLY the translated text. Do not add any conversational filler or markdown code blocks (like ```)."""
    
    try:
        # Low temperature ensures strict translation without hallucination
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        
        response = model.generate_content(
            f"{system_prompt}\n\nDOCUMENT TO TRANSLATE:\n{text}",
            generation_config=generation_config
        )
        
        translated_text = response.text.strip()
        
        # Failsafe: Clean markdown wrappers if Gemini adds them
        if translated_text.startswith("```"):
            translated_text = "\n".join(translated_text.split("\n")[1:])
        if translated_text.endswith("```"):
            translated_text = "\n".join(translated_text.split("\n")[:-1])
            
        return translated_text.strip()
        
    except Exception as e:
        import streamlit as st
        st.error(f"Gemini Translation Error: {str(e)}")
        return text

def generate_pdf_base64(text, doc_type):
    pdf = FPDF()
    pdf.add_page()
    
    clean_text = text.replace('**', '').replace('#', '')
    is_devanagari = bool(re.search(r'[\u0900-\u097F]', clean_text))
    font_path = "NotoSansDevanagari-Regular.ttf"
    
    if is_devanagari and os.path.exists(font_path):
        pdf.add_font("NotoSansDevanagari", "", font_path, uni=True)
        pdf.set_font("NotoSansDevanagari", size=11)
        pdf.multi_cell(0, 7, clean_text)
    else:
        pdf.set_font("Times", size=11)
        safe_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_text)
    
    safe_filename = doc_type.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
    pdf_file = f"drafted_{safe_filename}.pdf"                    
    pdf.output(pdf_file)
    
    with open(pdf_file, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()
    return pdf_b64

# ==========================================
# 3. SESSION STATE INIT
# ==========================================
# Initialize teammate's state
init_session_state()

# Initialize your specific Document Drafter state
if "drafter_messages" not in st.session_state:
    st.session_state.drafter_messages = []
if "drafter_pdf_content" not in st.session_state:
    st.session_state.drafter_pdf_content = None
if "drafter_document_ready" not in st.session_state:
    st.session_state.drafter_document_ready = False
if "drafter_generated_doc_type" not in st.session_state:
    st.session_state.drafter_generated_doc_type = "Document"
if "drafter_current_language" not in st.session_state:
    st.session_state.drafter_current_language = "English"
if "drafter_content_en" not in st.session_state:
    st.session_state.drafter_content_en = ""

# ==========================================
# 4. HEADER & SIDEBAR
# ==========================================
st.title("⚖️ Nyaya-Sahayak: Nyaya-Sahayak Lakehouse")
st.caption(
    "AI-powered Indian legal information, document drafting, & government welfare scheme navigator  •  "
    "Powered by Databricks & Gemini"
)

with st.sidebar:
    st.subheader("Chat Settings")
    persona = st.selectbox("Persona", ["citizen"], index=0)
    st.session_state.persona = persona

    language = st.selectbox("Language", ["en", "hi"], index=0 if st.session_state.get("language") == "en" else 1)
    st.session_state.language = language

    act_filter = st.selectbox("Act Filter", ["ALL", "BNS", "BNSS", "BSA", "IPC", "CONSTITUTION", "OTHER_ACTS"], index=0)
    st.session_state.act_filter = act_filter

    compare = st.checkbox("Compare Models", value=st.session_state.get("compare_models", False))
    st.session_state.compare_models = compare

    if st.button("Clear General Chat"):
        clear_chat()
        st.rerun()
        
    if st.button("Clear Drafter Chat"):
        st.session_state.drafter_messages = []
        st.session_state.drafter_document_ready = False
        st.session_state.drafter_pdf_content = None
        st.rerun()

# ==========================================
# 5. TABS UI
# ==========================================
# Added your feature as Tab 3
tab_citizen, tab_wizard, tab_drafter = st.tabs([
    "💬 Chat (Citizen)",
    "🏛️ Scheme Wizard",
    "📝 Document Drafter"
])

# --- TAB 1: TEAMMATE'S CHAT ---
with tab_citizen:
    st.session_state.persona = "citizen"
    if orchestrator_handle:
        render_chat_view(orchestrator_handle, tab_suffix="citizen")
    else:
        st.warning("Orchestrator not connected yet.")

# --- TAB 2: TEAMMATE'S WIZARD ---
with tab_wizard:
    if orchestrator_handle:
        render_scheme_wizard(orchestrator_handle)
    else:
        st.warning("Orchestrator not connected yet.")

# --- TAB 3: YOUR DOCUMENT DRAFTER ---
with tab_drafter:
    st.markdown("### Generate Formal Legal Documents (FIRs, Notices, Agreements)")
    
    if st.session_state.drafter_document_ready:
        d_col1, d_col2 = st.columns(2)
    else:
        d_col1, d_col2 = st.columns([0.7, 0.3])

    # Drafter Left Column (Chat)
    with d_col1:
        drafter_chat_container = st.container(height=500)
        
        with drafter_chat_container:
            if not st.session_state.drafter_messages:
                st.chat_message("assistant").markdown("Namaste! Describe your situation, and I will draft the appropriate legal document for you.")
            
            for message in st.session_state.drafter_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Key is critical here to separate it from teammate's chat input
        if prompt := st.chat_input("Detail the incident/facts to draft a document...", key="drafter_input"):
            st.session_state.drafter_messages.append({"role": "user", "content": prompt})
                
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                with st.spinner("Analyzing facts & consulting legal protocols..."):
                    chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.drafter_messages])
                    plan = intake_agent(chat_history_str)
                    is_ready = plan.get("is_ready_to_draft", False)
                
                if not is_ready:
                    agent_response = plan.get("next_clarification_question", "Could you provide more details?")
                    
                    full_response = ""
                    for chunk in agent_response.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    st.session_state.drafter_messages.append({"role": "assistant", "content": full_response})
                    
                else:
                    doc_type = plan.get("document_type", "Document")
                    facts = plan.get("extracted_facts", {})
                    agent_response = f"Thank you. Drafting your **{doc_type}** now..."                
                    
                    full_response = ""
                    for chunk in agent_response.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    with st.spinner(f"Synthesizing {doc_type} under BNS compliance..."):
                        st.session_state.drafter_document_ready = True
                        st.session_state.drafter_generated_doc_type = doc_type

                        law = retrieval_agent(doc_type, facts)
                        drafted_content = drafting_agent(doc_type, facts, law)
                        
                        st.session_state.drafter_content_en = drafted_content
                        st.session_state.drafter_current_language = "English"

                        # Generate initial English PDF
                        st.session_state.drafter_pdf_content = generate_pdf_base64(drafted_content, doc_type)
                        
                        if "Recommended Next Steps" in drafted_content:
                            extracted_steps = drafted_content.split("Recommended Next Steps")[-1].strip()
                            chat_conclusion = f"Your **{doc_type}** is ready to review! 📄\n\n**💡 Recommended Next Steps:**\n{extracted_steps}\n\n*(You can change the document language in the panel on the right).*"
                        else:
                            chat_conclusion = f"Your **{doc_type}** is ready! 📄 Please review the document carefully. *(You can change the document language in the panel on the right).*"

                        st.session_state.drafter_messages.append({"role": "assistant", "content": chat_conclusion})
                        st.rerun()

    # Drafter Right Column (Document)
    with d_col2:
        if st.session_state.drafter_document_ready and st.session_state.drafter_pdf_content:
            st.subheader("📄 Document Preview")
            
            SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi", "Tamil", "Bengali", "Kannada", "Telugu", "Gujarati", "Malayalam", "Punjabi"]
            selected_lang = st.selectbox(
                "Select Document Language:",
                SUPPORTED_LANGUAGES,
                index=SUPPORTED_LANGUAGES.index(st.session_state.drafter_current_language)
            )
            
            if selected_lang != st.session_state.drafter_current_language:
                with st.spinner(f"Translating document to {selected_lang}..."):
                    if selected_lang == "English":
                        translated_text = st.session_state.drafter_content_en
                    else:
                        # Call the new Gemini function directly
                        translated_text = translate_with_gemini(
                            st.session_state.drafter_content_en, 
                            selected_lang,
                            api_key="AIzaSyC7YuNKwfJ1CRmwXfNM7XG4XhqzatW0KgA" # Use your actual API key here
                        )
                    
                    st.session_state.drafter_pdf_content = generate_pdf_base64(translated_text, st.session_state.drafter_generated_doc_type)
                    st.session_state.drafter_current_language = selected_lang
                    st.rerun()
            
            pdf_bytes = base64.b64decode(st.session_state.drafter_pdf_content)
            
            st.download_button(
                label=f"⬇️ Download PDF ({st.session_state.drafter_current_language})",
                data=pdf_bytes,
                file_name=f"generated_{st.session_state.drafter_generated_doc_type.lower().replace(' ', '_')}_{st.session_state.drafter_current_language[:2].lower()}.pdf",
                mime="application/pdf"
            )
            
            pdf_display = f'<iframe src="data:application/pdf;base64,{st.session_state.drafter_pdf_content}" width="100%" height="550" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

# ==========================================
# 6. FOOTER
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Nyaya-Sahayak** v0.2.0\n\n"
    "Built for Databricks Hackathon\n\n"
    "⚠️ Not legal advice. Consult a qualified advocate."
)