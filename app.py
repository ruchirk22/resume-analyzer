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
)
from core.privacy import anonymize_text
from core.logger import log_analysis_event
from core.llm import generate_gemini_analysis
from core.embedding import load_embedding_model # Direct import for caching

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
if 'results' not in st.session_state:
    st.session_state.results = []
if 'jd_keywords' not in st.session_state:
    st.session_state.jd_keywords = []

# Load model via cache
model = get_embedding_model()

# --- Core Processing Functions ---
def process_job_description(jd_text: str) -> Dict[str, Any]:
    """Processes the job description text."""
    sentences = jd_text.split('\n')
    embedding = generate_embedding(sentences, model)
    keywords = extract_jd_keywords(jd_text)
    st.session_state.jd_keywords = keywords # Store keywords for display
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

    use_gemini = st.session_state.get('use_gemini', False)
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
        "Filename": filename_display,
        "Score": round(similarity_score * 100, 2),
        "Category": category_info["category"],
        "Status": "Pending", # --- NEW: Default status ---
        "summary": ai_analysis.get("summary", "Error."),
        "strengths": ai_analysis.get("strengths", []),
        "gaps": ai_analysis.get("gaps", []),
        "evidence": evidence,
    }

def update_candidate_status(filename: str, new_status: str):
    """Helper function to update the status of a candidate in the session state."""
    for result in st.session_state.results:
        if result["Filename"] == filename:
            result["Status"] = new_status
            break

# --- UI Layout ---
with st.sidebar:
    st.header("1. Job Description")
    st.text_area("Paste the Job Description here", height=200, key="jd_text")

    st.header("2. Upload Resumes")
    st.file_uploader("Upload resumes (PDF, max 10)", type="pdf", accept_multiple_files=True, key="resume_uploader")
    
    st.header("3. Controls & Settings")
    st.toggle("Anonymize Candidates", key="anonymize", value=True)
    st.toggle("Enable Gemini AI Analysis", key="use_gemini", value=False)
    
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
            jd_data = process_job_description(st.session_state.jd_text)
            thresholds = {
                "strong_threshold": strong_thresh / 100,
                "good_threshold": good_thresh / 100,
                "average_threshold": avg_thresh / 100
            }
            results = [process_resume(file, jd_data, thresholds) for file in st.session_state.resume_uploader]
            st.session_state.results = sorted(results, key=lambda x: x.get('Score', 0), reverse=True)
            try:
                log_analysis_event(st.session_state.jd_text, st.session_state.results, st.session_state.anonymize)
                st.toast("Analysis complete!")
            except Exception as e:
                st.error(f"Failed to write to audit log: {e}")

if st.session_state.results:
    st.header("Analysis Results")
    
    all_categories = ["Strong", "Good", "Average", "Weak"]
    selected_categories = st.multiselect("Filter by Category", options=all_categories, default=all_categories)
    
    filtered_results = [res for res in st.session_state.results if res.get("Category") in selected_categories]
    
    if not filtered_results:
        st.warning("No candidates match the selected filters.")
    else:
        df_display = pd.DataFrame(filtered_results)

        # --- NEW: KPI DISPLAY ---
        st.subheader("Screening KPIs")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total Processed", len(df_display))
        kpi_cols[1].metric("Average Score", f"{df_display['Score'].mean():.2f}%" if not df_display.empty else "N/A")
        
        # Safely get category counts
        category_counts = df_display['Category'].value_counts()
        kpi_cols[2].metric("Strong Matches", category_counts.get("Strong", 0))
        kpi_cols[3].metric("Good Matches", category_counts.get("Good", 0))

        # --- UPDATED: Results Table with Status ---
        st.dataframe(df_display[["Filename", "Score", "Category", "Status"]], use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
             st.download_button(
                "Download All Filtered Results", 
                df_display.to_csv(index=False).encode('utf-8'), 
                'filtered_resume_analysis.csv', 'text/csv'
            )
        with col2:
            shortlisted_df = df_display[df_display['Status'] == 'Shortlisted']
            st.download_button(
                "Download Shortlist Only", 
                shortlisted_df.to_csv(index=False).encode('utf-8'), 
                'shortlisted_candidates.csv', 'text/csv'
            )
        
        st.header("Detailed Candidate View")
        candidate_filenames = [res["Filename"] for res in filtered_results]
        selected_candidate_file = st.selectbox("Select a candidate to see details", options=candidate_filenames)
        
        if selected_candidate_file:
            selected_candidate_data = next((res for res in filtered_results if res["Filename"] == selected_candidate_file), None)
            
            if selected_candidate_data:
                # --- NEW: ACTION BUTTONS ---
                st.subheader("Actions")
                action_cols = st.columns(3)
                action_cols[0].button("✅ Shortlist", on_click=update_candidate_status, args=(selected_candidate_file, "Shortlisted"), use_container_width=True)
                action_cols[1].button("🤔 Maybe", on_click=update_candidate_status, args=(selected_candidate_file, "Maybe"), use_container_width=True)
                action_cols[2].button("❌ Reject", on_click=update_candidate_status, args=(selected_candidate_file, "Rejected"), use_container_width=True)

                st.subheader("AI Summary")
                st.markdown(f"> {selected_candidate_data['summary']}")

                st.subheader("Top Evidence Sentences")
                for sentence in selected_candidate_data["evidence"]:
                    st.markdown(f"- *{sentence}*")
                
                st.subheader("Detailed Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("✅ **Strengths**")
                    st.info("\n".join(f"- {s}" for s in selected_candidate_data["strengths"]) if selected_candidate_data["strengths"] else "N/A")
                with col2:
                    st.markdown("❌ **Gaps**")
                    st.warning("\n".join(f"- {g}" for g in selected_candidate_data["gaps"]) if selected_candidate_data["gaps"] else "N/A")

