# core/llm.py

import os
import google.generativeai as genai
import json
import re
from typing import Dict, Any
import streamlit as st # Import Streamlit for secrets access

def configure_gemini_api():
    """
    Configures the Gemini API with the key from Streamlit secrets.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        if api_key:
            genai.configure(api_key=api_key)
            return True
        return False
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return False
    except Exception:
        # Catch all for any other unexpected errors
        return False

def generate_gemini_analysis(jd_text: str, resume_text: str) -> Dict[str, Any]:
    """
    Uses the Gemini API to perform an intelligent analysis of the resume against the JD.
    """
    if not configure_gemini_api():
        return {
            "error": "Gemini API key not configured. Please set it in your Streamlit secrets."
        }

    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    prompt = f"""
    You are an expert AI Tech Recruiter. Your task is to analyze a resume against a job description and provide a concise, structured analysis.
    The analysis must be in a valid JSON format.

    **Job Description:**
    ---
    {jd_text}
    ---

    **Candidate's Resume:**
    ---
    {resume_text}
    ---

    Provide a JSON object with the following three keys:
    1. "summary": A 2-3 sentence summary of the candidate's fit for the role.
    2. "strengths": A list of 3-5 key skills, experiences, or qualifications from the JD that are clearly demonstrated in the resume.
    3. "gaps": A list of 2-3 key requirements from the JD that seem to be missing or are not clearly mentioned in the resume.

    CRITICAL: Respond with nothing but the JSON object. Do not include any explanatory text before or after the JSON.
    """

    try:
        response = model.generate_content(prompt)
        # --- MORE ROBUST PARSING LOGIC ---
        # Use regex to find the JSON block, even if there's extra text.
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not match:
            return {"error": "LLM response did not contain a valid JSON object."}
        
        json_response_text = match.group(0)
        
        # Parse the JSON string into a Python dictionary
        analysis_result = json.loads(json_response_text)

        if all(k in analysis_result for k in ["summary", "strengths", "gaps"]):
            return analysis_result
        else:
            return {"error": "LLM JSON response was missing required keys (summary, strengths, gaps)."}

    except json.JSONDecodeError:
        return {"error": "Failed to decode the JSON from the AI model's response."}
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        return {"error": f"An unexpected error occurred with the AI model. Details: {str(e)}"}

