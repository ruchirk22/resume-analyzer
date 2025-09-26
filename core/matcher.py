# core/matcher.py

import numpy as np
from typing import Dict

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    Returns a float between 0 and 1.
    """
    # Ensure vectors are numpy arrays and not None
    if vec1 is None or vec2 is None:
        return 0.0
    
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    
    # Check for zero vectors
    if np.all(vec1==0) or np.all(vec2==0):
        return 0.0

    # Formula for cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    # Clip the value to be between 0 and 1 for consistency, handling potential float precision issues
    return np.clip(similarity, 0, 1)

# --- MODIFIED FUNCTION ---
def categorize_candidate(
    score: float, 
    strong_threshold: float = 0.75, 
    good_threshold: float = 0.65, 
    average_threshold: float = 0.55
) -> Dict[str, str]:
    """
    Categorizes a candidate based on their similarity score using dynamic thresholds.
    """
    score_percentage = score * 100
    if score >= strong_threshold:
        return {"category": "Strong", "color": "green"}
    elif score >= good_threshold:
        return {"category": "Good", "color": "#FFD700"} # Yellow
    elif score >= average_threshold:
        return {"category": "Average", "color": "orange"}
    else:
        return {"category": "Weak", "color": "red"}

