# core/analyzer.py

import yake
import numpy as np
import streamlit as st
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

# --- NEW: Curated list of technical skills for more accurate extraction ---
SKILL_KEYWORDS = [
    # Programming Languages
    'python', 'javascript', 'java', 'c#', 'c++', 'typescript', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin',
    # AI/ML
    'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'openai', 'gpt-4', 'gpt-5',
    'gemini', 'llama', 'hugging face', 'langchain', 'ner', 'nlp', 'computer vision', 'rag', 'multi-agent systems', 'prompt engineering',
    'embeddings', 'vector search', 'agentic ai',
    # Web Development
    'react', 'angular', 'vue.js', 'node.js', 'express.js', 'django', 'flask', 'fastapi', 'html', 'css', 'mern stack',
    # Cloud & DevOps
    'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'github actions', 'jenkins', 'terraform', 'ansible',
    'oracle cloud', 'oci',
    # Databases
    'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'cassandra', 'sqlite', 'pl/sql', 'oracle autonomous db',
    # Others
    'api', 'rest', 'graphql', 'microservices', 'data structures', 'algorithms', 'git', 'agile', 'scrum', 'automation'
]

try:
    from core.matcher import calculate_cosine_similarity
except ImportError:
    def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1 is None or vec2 is None: return 0.0
        vec1, vec2 = np.asarray(vec1), np.asarray(vec2)
        if np.all(vec1==0) or np.all(vec2==0): return 0.0
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return np.clip(dot_product / (norm_vec1 * norm_vec2), 0, 1)

@st.cache_resource
def get_keyword_extractor(n_gram_size=3, top_k=20):
    return yake.KeywordExtractor(n=n_gram_size, dedupLimit=0.9, top=top_k)

def extract_jd_keywords(text: str) -> List[str]:
    noise_words = {
        'ema', 'silicon valley', 'ema unlimited', 'experience', 'responsibilities', 'requirements',
        'preferred', 'qualifications', 'bonus', 'points', 'etc', 'inc', 'llc', 'corp', 'company',
        'organization', 'pune', 'india', 'world'
    }
    custom_kw_extractor = get_keyword_extractor()
    keywords = custom_kw_extractor.extract_keywords(text)
    clean_keywords = [kw.lower() for kw, score in keywords if kw.lower() not in noise_words]
    return list(dict.fromkeys(clean_keywords))

def find_top_evidence_sentences(
    resume_sentences: List[str], jd_embedding: np.ndarray, model: SentenceTransformer, top_k: int = 3
) -> List[str]:
    if not resume_sentences or jd_embedding is None: return []
    sentence_embeddings = model.encode(resume_sentences, show_progress_bar=False)
    similarities = util.cos_sim(sentence_embeddings, jd_embedding)
    effective_top_k = min(top_k, len(resume_sentences))
    if effective_top_k == 0: return []
    
    # Use negative sign to sort in descending order directly - safer approach
    flattened_similarities = similarities.flatten()
    top_indices = np.argsort(-flattened_similarities)[:effective_top_k]
    
    return [resume_sentences[i] for i in top_indices]

def analyze_strengths_and_gaps(resume_text: str, jd_keywords: List[str]) -> Dict[str, List[str]]:
    resume_text_lower = resume_text.lower()
    strengths = [kw for kw in jd_keywords if kw in resume_text_lower]
    gaps = [kw for kw in jd_keywords if kw not in resume_text_lower]
    return {"strengths": strengths, "gaps": gaps}

# --- MODIFIED FUNCTION: SKILL EXTRACTION ---
def extract_skills_from_resume(resume_text: str) -> List[str]:
    """
    Extracts skills by checking for the presence of a curated list of keywords.
    """
    resume_text_lower = resume_text.lower()
    found_skills = []
    # Use regex to find whole words to avoid matching substrings (e.g., 'go' in 'google')
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + re.escape(skill) + r'\b', resume_text_lower):
            found_skills.append(skill)
    return list(dict.fromkeys(found_skills)) # Return unique skills

def highlight_keywords(text: str, keywords: List[str]) -> str:
    if not keywords: return text
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    for keyword in sorted_keywords:
        try:
            # Fix: Use a raw string with single backslash for proper group reference
            text = re.sub(f'({re.escape(keyword)})', r'<mark>\1</mark>', text, flags=re.IGNORECASE)
        except re.error:
            continue
    return text

