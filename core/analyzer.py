# core/analyzer.py

import yake
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import streamlit as st # Import Streamlit for caching

# --- NEW CACHED FUNCTION ---
@st.cache_resource
def get_keyword_extractor():
    """
    Initializes and returns a YAKE keyword extractor.
    This function is cached to avoid recreating the object on each run.
    """
    return yake.KeywordExtractor(
        n=3,          # Max n-gram size
        dedupLimit=0.9, # Deduplication threshold
        top=20        # Max number of keywords to return
    )

# --- Updated function ---
def extract_jd_keywords(text: str) -> List[str]:
    """
    Extracts relevant keywords from the job description using a cached extractor.
    """
    # Noise words specific to job descriptions to filter out
    noise_words = {
        'ema', 'silicon valley', 'ema unlimited', 'experience', 'responsibilities',
        'requirements', 'preferred', 'qualifications', 'bonus', 'points'
    }
    
    custom_kw_extractor = get_keyword_extractor()
    keywords = custom_kw_extractor.extract_keywords(text)
    
    # Filter out noise words and return a clean list of keyword phrases
    clean_keywords = [kw for kw, score in keywords if kw.lower() not in noise_words]
    return clean_keywords

from .matcher import calculate_cosine_similarity

# --- ADD THIS LIST ---
# List of common business/location/generic words to exclude from gap analysis
NOISE_WORDS = {
    'ema', 'unlimited', 'inc', 'etc', 'silicon valley', 'bangalore', 'san francisco',
    'pune', 'india', 'world', 'llc', 'corp', 'company', 'organization',
    # Add more as you find them
}

def extract_jd_keywords(jd_text: str, num_keywords: int = 20) -> List[str]:
    """
    Extracts the most relevant keywords from a job description using YAKE.

    Args:
        jd_text (str): The raw text of the job description.
        num_keywords (int): The maximum number of keywords to extract.

    Returns:
        List[str]: A list of the top keywords.
    """
    if not jd_text:
        return []
    
    # YAKE parameters can be tuned for better performance
    kw_extractor = yake.KeywordExtractor(
        n=2, # Max n-gram size
        dedupLimit=0.9, # Deduplication threshold
        top=num_keywords,
        features=None
    )
    keywords = kw_extractor.extract_keywords(jd_text)
    # --- MODIFY THIS LINE ---
    # Filter out noise words from the extracted keywords
    extracted_kws = [kw[0].lower() for kw in keywords]
    return [kw for kw in extracted_kws if kw not in NOISE_WORDS]


def find_top_evidence_sentences(
    resume_sentences: List[str], 
    jd_embedding: np.ndarray, 
    model: SentenceTransformer,
    top_k: int = 3
) -> List[str]:
    """
    Finds the most relevant sentences in a resume compared to the JD embedding.

    Args:
        resume_sentences (List[str]): The list of sentences from the parsed resume.
        jd_embedding (np.ndarray): The embedding vector for the entire job description.
        model (SentenceTransformer): The loaded embedding model.
        top_k (int): The number of top evidence sentences to return.

    Returns:
        List[str]: A list of the top K most relevant sentences.
    """
    if not resume_sentences or jd_embedding is None:
        return []

    # Generate embeddings for each sentence in the resume
    sentence_embeddings = model.encode(resume_sentences, show_progress_bar=False)

    # Calculate similarity of each sentence to the JD embedding
    similarities = [
        calculate_cosine_similarity(sent_emb, jd_embedding)
        for sent_emb in sentence_embeddings
    ]

    # Get the indices of the top K most similar sentences
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return the corresponding sentences
    return [resume_sentences[i] for i in top_indices]

def analyze_strengths_and_gaps(resume_text: str, jd_keywords: List[str]) -> Dict[str, List[str]]:
    """
    Identifies which JD keywords are present (strengths) or missing (gaps) in the resume.

    Args:
        resume_text (str): The raw text of the resume.
        jd_keywords (List[str]): The keywords extracted from the job description.

    Returns:
        Dict[str, List[str]]: A dictionary with 'strengths' and 'gaps'.
    """
    resume_text_lower = resume_text.lower()
    strengths = [kw for kw in jd_keywords if kw in resume_text_lower]
    gaps = [kw for kw in jd_keywords if kw not in resume_text_lower]
    
    return {"strengths": strengths, "gaps": gaps}


