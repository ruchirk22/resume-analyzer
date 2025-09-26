# tests/test_analyzer.py
import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.analyzer import (
    extract_jd_keywords,
    find_top_evidence_sentences,
    analyze_strengths_and_gaps,
)
from core.embedding import load_embedding_model

def test_keyword_extraction():
    """Tests the YAKE keyword extraction."""
    print("\n--- Testing Keyword Extraction ---")
    jd_text = "We are looking for a Senior Python Developer with experience in machine learning, FastAPI, and AWS. The ideal candidate will lead a team of developers."
    keywords = extract_jd_keywords(jd_text, num_keywords=5)
    
    assert len(keywords) > 0
    assert "python developer" in keywords
    assert "machine learning" in keywords
    print(f"Extracted Keywords: {keywords}")
    print("Keyword extraction test passed!")

def test_evidence_extraction():
    """Tests finding the top evidence sentences."""
    print("\n--- Testing Evidence Extraction ---")
    model = load_embedding_model()
    
    jd_text = "Seeking a data scientist skilled in natural language processing."
    jd_embedding = model.encode([jd_text])[0] # Get single embedding

    resume_sentences = [
        "I have a background in software engineering.",
        "My primary focus is on NLP and text analytics.", # Most relevant
        "I also have experience with cloud platforms.",
        "I enjoy working with language data.", # Second most relevant
        "My degree is in computer science."
    ]

    evidence = find_top_evidence_sentences(resume_sentences, jd_embedding, model, top_n=2)
    
    assert len(evidence) == 2
    assert "My primary focus is on NLP and text analytics." in evidence
    assert "I enjoy working with language data." in evidence
    print(f"Found Evidence: {evidence}")
    print("Evidence extraction test passed!")

def test_strengths_and_gaps():
    """Tests the strengths and gaps analysis."""
    print("\n--- Testing Strengths and Gaps ---")
    jd_keywords = ["python", "machine learning", "fastapi", "aws", "team lead"]
    resume_text = "I am a Python developer with a passion for machine learning. I have used AWS for deploying models."

    analysis = analyze_strengths_and_gaps(resume_text, jd_keywords)

    assert "python" in analysis["strengths"]
    assert "machine learning" in analysis["strengths"]
    assert "aws" in analysis["strengths"]
    assert "fastapi" in analysis["gaps"]
    assert "team lead" in analysis["gaps"]
    
    print(f"Strengths: {analysis['strengths']}")
    print(f"Gaps: {analysis['gaps']}")
    print("Strengths and gaps test passed!")


if __name__ == "__main__":
    test_keyword_extraction()
    test_evidence_extraction()
    test_strengths_and_gaps()
    print("\nAll analyzer tests passed!")

