# app.py

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

# --- Core Logic Imports ---
from core.parser import parse_resume
from core.embedding import generate_embedding
from core.matcher import calculate_cosine_similarity, categorize_candidate
from core.analyzer import (
    extract_jd_keywords,
    find_top_evidence_sentences,
    analyze_strengths_and_gaps,
    extract_skills_from_resume,
    highlight_keywords
)
from core.privacy import anonymize_text
from core.logger import log_analysis_event
from core.llm import generate_gemini_analysis
from core.embedding import load_embedding_model

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Resume Shortlister",
    page_icon="🤖",
    layout="wide"
)

# --- Caching Functions ---
@st.cache_resource
def get_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    return load_embedding_model()

# --- State Management ---
if 'results' not in st.session_state: st.session_state.results = []
if 'jd_keywords' not in st.session_state: st.session_state.jd_keywords = []

# --- MODIFIED: Robust API Key Check ---
try:
    # This handles missing secrets file completely
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        gemini_api_key_found = gemini_api_key is not None and gemini_api_key != ""
    except (KeyError, st.errors.StreamlitSecretNotFoundError):
        gemini_api_key_found = False
except Exception:
    # Fallback for any other unexpected errors
    gemini_api_key_found = False

# Load model via cache
model = get_embedding_model()

# --- Helper function for styling skill chips ---
def create_skill_chips_html(skills: List[str]) -> str:
    """Generates HTML for displaying skills as styled chips."""
    styles = "display:inline-block; background-color:#2B3037; color:#E0E0E0; padding:5px 10px; margin:3px; border-radius:15px; font-size:0.85em;"
    chips = "".join([f'<div style="{styles}">{skill}</div>' for skill in skills])
    return f'<div style="line-height: 1.6;">{chips}</div>'

# --- Core Processing Functions ---
def process_job_description(jd_text: str) -> Dict[str, Any]:
    """Processes the job description text."""
    sentences = jd_text.split('\n')
    embedding = generate_embedding(sentences, model)
    keywords = extract_jd_keywords(jd_text)
    st.session_state.jd_keywords = keywords
    return {"text": jd_text, "embedding": embedding, "keywords": keywords}

def process_resume(resume_file: Any, jd_data: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Processes a single resume file against the JD."""
    candidate_data = parse_resume(resume_file)
    if candidate_data.get("error"): return candidate_data

    is_anonymized = st.session_state.get('anonymize', False)
    filename_display = candidate_data["filename"]
    text_for_analysis = candidate_data["raw_text"]
    if is_anonymized:
        text_for_analysis = anonymize_text(candidate_data["raw_text"], candidate_data["filename"])
        filename_display = f"Candidate_{hash(candidate_data['filename'])}"
    
    resume_embedding = generate_embedding(candidate_data["sentences"], model)
    if resume_embedding is None: return {"filename": filename_display, "error": "Could not generate resume embedding."}

    similarity_score = calculate_cosine_similarity(resume_embedding, jd_data["embedding"])
    category_info = categorize_candidate(similarity_score, **thresholds)
    
    evidence = find_top_evidence_sentences(candidate_data["sentences"], jd_data["embedding"], model)
    if is_anonymized:
        evidence = [anonymize_text(sentence) for sentence in evidence]
    
    extracted_skills = extract_skills_from_resume(candidate_data["raw_text"])

    use_gemini = st.session_state.get('use_gemini', False) and gemini_api_key_found
    ai_analysis = {}
    if use_gemini:
        ai_analysis = generate_gemini_analysis(jd_data["text"], text_for_analysis)
        if "error" in ai_analysis:
            st.warning(f"Gemini failed for {filename_display}: {ai_analysis['error']}. Falling back.")
            ai_analysis = analyze_strengths_and_gaps(text_for_analysis, jd_data["keywords"])
            ai_analysis["summary"] = "N/A (AI analysis failed)"
    else:
        ai_analysis = analyze_strengths_and_gaps(text_for_analysis, jd_data["keywords"])
        ai_analysis["summary"] = "N/A (Keyword analysis only)"

    return {
        "Filename": filename_display, "Score": round(similarity_score * 100, 2),
        "Category": category_info["category"], "Status": "Pending",
        "summary": ai_analysis.get("summary", "Error."),
        "strengths": ai_analysis.get("strengths", []), "gaps": ai_analysis.get("gaps", []),
        "evidence": evidence, "skills": extracted_skills
    }

def update_candidate_status(filename: str, new_status: str):
    """Updates the status of a candidate in the session state."""
    for result in st.session_state.results:
        if result["Filename"] == filename:
            result["Status"] = new_status
            break

# --- UI Layout ---
with st.sidebar:
    st.header("1. Job Description")
    st.text_area("Paste the Job Description here", height=200, key="jd_text")
    st.header("2. Upload Resumes")
    # --- MODIFIED: Accept .docx files ---
    st.file_uploader(
        "Upload resumes (PDF or DOCX, max 10)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True, 
        key="resume_uploader"
    )
    st.header("3. Controls & Settings")
    st.toggle("Anonymize Candidates", key="anonymize", value=True)

    st.toggle(
        "Enable Gemini AI Analysis", 
        key="use_gemini", 
        value=False, 
        disabled=not gemini_api_key_found,
        help="Requires a Gemini API key in your secrets.toml file."
    )
    if not gemini_api_key_found:
        st.warning("No Gemini API Key found. AI Analysis is disabled. Please add a secrets.toml file.")

    st.subheader("Category Thresholds")
    strong_thresh = st.slider("Strong Match Threshold (%)", 50, 100, 75)
    good_thresh = st.slider("Good Match Threshold (%)", 40, 90, 65)
    avg_thresh = st.slider("Average Match Threshold (%)", 30, 80, 55)
    analyze_button = st.button("Analyze Resumes", type="primary", use_container_width=True)
    if st.session_state.jd_keywords:
        st.subheader("Extracted JD Keywords")
        st.info(", ".join(st.session_state.jd_keywords))

st.title("🤖 AI-Powered Resume Shortlister")
st.markdown("An interactive dashboard to screen, sort, and analyze candidates.")

if analyze_button:
    if not st.session_state.jd_text.strip(): st.error("Please provide a Job Description.")
    elif not st.session_state.resume_uploader: st.error("Please upload at least one resume.")
    elif len(st.session_state.resume_uploader) > 10: st.error("You can upload a maximum of 10 resumes.")
    else:
        with st.spinner("Analyzing resumes... This may take a moment."):
            # --- MODIFIED: Handle potential parsing errors in the main loop ---
            jd_data = process_job_description(st.session_state.jd_text)
            thresholds = {
                "strong_threshold": strong_thresh / 100,
                "good_threshold": good_thresh / 100,
                "average_threshold": avg_thresh / 100
            }
            
            results = []
            for file in st.session_state.resume_uploader:
                result = process_resume(file, jd_data, thresholds)
                if result.get("error"):
                    st.error(f"Error processing {file.name}: {result['error']}")
                else:
                    results.append(result)

            st.session_state.results = sorted(results, key=lambda x: x.get('Score', 0), reverse=True)
            
            if st.session_state.results:
                try:
                    log_analysis_event(st.session_state.jd_text, st.session_state.results, st.session_state.anonymize)
                    st.toast("Analysis complete!")
                except Exception as e:
                    st.error(f"Failed to write to audit log: {e}")
            else:
                st.warning("Analysis finished, but no resumes could be successfully processed.")


if st.session_state.results:
    st.header("Analysis Results")
    all_categories = ["Strong", "Good", "Average", "Weak"]
    selected_categories = st.multiselect("Filter by Category", options=all_categories, default=all_categories)
    filtered_results = [res for res in st.session_state.results if res.get("Category") in selected_categories]
    
    if not filtered_results:
        st.warning("No candidates match the selected filters.")
    else:
        df_display = pd.DataFrame(filtered_results)
        
        # Create tabs for simple vs advanced view
        tab1, tab2 = st.tabs(["Essential View", "Advanced Analysis"])
        
        with tab1:
            st.subheader("Candidate Overview")
            
            # GDPR-compliant table without numerical scores
            # Show categories instead of numerical scores
            compliant_df = df_display.copy()
            compliant_df = compliant_df[["Filename", "Category", "Status"]]
            st.dataframe(compliant_df, use_container_width=True, hide_index=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download All Results", compliant_df.to_csv(index=False).encode('utf-8'), 'filtered_candidates.csv', 'text/csv')
            with col2:
                shortlisted_df = compliant_df[compliant_df['Status'] == 'Shortlisted']
                st.download_button("Download Shortlist Only", shortlisted_df.to_csv(index=False).encode('utf-8'), 'shortlisted_candidates.csv', 'text/csv')
            
            # Simplified candidate details view
            st.subheader("Candidate Details")
            candidate_filenames = [res["Filename"] for res in filtered_results]
            selected_candidate_file = st.selectbox("Select a candidate", options=candidate_filenames)
            
            if selected_candidate_file:
                selected_candidate_data = next((res for res in filtered_results if res["Filename"] == selected_candidate_file), None)
                
                if selected_candidate_data:
                    # Quick decision buttons
                    st.subheader("Decision")
                    action_cols = st.columns(3)
                    action_cols[0].button("✅ Shortlist", on_click=update_candidate_status, args=(selected_candidate_file, "Shortlisted"), use_container_width=True)
                    action_cols[1].button("🤔 Maybe", on_click=update_candidate_status, args=(selected_candidate_file, "Maybe"), use_container_width=True)
                    action_cols[2].button("❌ Reject", on_click=update_candidate_status, args=(selected_candidate_file, "Rejected"), use_container_width=True)
                    
                    # Key skills at a glance
                    st.subheader("Key Skills")
                    skills_html = create_skill_chips_html(selected_candidate_data.get("skills", []))
                    st.markdown(skills_html, unsafe_allow_html=True)
                    
                    # Simple summary - just the AI summary if available
                    if selected_candidate_data['summary'] != "N/A (Keyword analysis only)" and selected_candidate_data['summary'] != "N/A (AI analysis failed)":
                        st.subheader("Summary")
                        st.markdown(f"> {selected_candidate_data['summary']}")
        
        with tab2:
            # Advanced metrics section
            st.subheader("Detailed Metrics")
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Total Processed", len(df_display))
            kpi_cols[1].metric("Average Score", f"{df_display['Score'].mean():.2f}%" if not df_display.empty else "N/A")
            category_counts = df_display['Category'].value_counts()
            kpi_cols[2].metric("Strong Matches", category_counts.get("Strong", 0))
            kpi_cols[3].metric("Good Matches", category_counts.get("Good", 0))
            
            # Advanced table with scores
            st.markdown("⚠️ **Note:** Numerical scores should be used for internal assessment only and not as the sole basis for hiring decisions.")
            st.dataframe(df_display[["Filename", "Score", "Category", "Status"]], use_container_width=True, hide_index=True)
            
            # Advanced download option
            st.download_button("Download Detailed Results (with scores)", df_display.to_csv(index=False).encode('utf-8'), 'detailed_resume_analysis.csv', 'text/csv')
            
            # Advanced candidate analysis
            st.subheader("Advanced Candidate Analysis")
            adv_candidate = st.selectbox("Select candidate for detailed analysis", options=candidate_filenames, key="adv_candidate_select")
            
            if adv_candidate:
                adv_candidate_data = next((res for res in filtered_results if res["Filename"] == adv_candidate), None)
                
                if adv_candidate_data:
                    # Display numerical score in advanced view
                    st.metric("Match Score", f"{adv_candidate_data['Score']}%")
                    
                    st.subheader("AI Analysis")
                    st.markdown(f"> {adv_candidate_data['summary']}")
                    
                    # Strengths and gaps
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Strengths")
                        for strength in adv_candidate_data.get("strengths", []):
                            st.markdown(f"- {strength}")
                    
                    with col2:
                        st.subheader("Potential Gaps")
                        for gap in adv_candidate_data.get("gaps", []):
                            st.markdown(f"- {gap}")
                    
                    st.subheader("Evidence Sentences")
                    for sentence in adv_candidate_data["evidence"]:
                        highlighted_sentence = highlight_keywords(sentence, st.session_state.jd_keywords)
                        st.markdown(f"- *{highlighted_sentence}*", unsafe_allow_html=True)
                    
                    # This section was moved up to the columns above

