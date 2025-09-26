# core/parser.py

import pdfplumber
import re
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing extra whitespace, non-ASCII characters,
    and converting to lowercase.

    Args:
        text (str): The raw text extracted from a resume.

    Returns:
        str: The cleaned text.
    """
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    return text

def chunk_text_into_sentences(text: str) -> List[str]:
    """
    Splits the cleaned text into a list of sentences.

    Args:
        text (str): The cleaned text.

    Returns:
        List[str]: A list of sentences.
    """
    return sent_tokenize(text)

def parse_pdf(file_path: Any) -> str:
    """
    Extracts text from a given PDF file.

    Args:
        file_path (Any): A file-like object or path to the PDF file.

    Returns:
        str: The extracted raw text from the PDF.
    """
    raw_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                raw_text += page_text + "\n"
    return raw_text

def parse_resume(uploaded_file: Any) -> Dict[str, Any]:
    """
    Parses an uploaded resume file (PDF or TXT).
    It extracts, cleans, and chunks the text into sentences.

    Args:
        uploaded_file (Any): The uploaded file object from Streamlit.

    Returns:
        Dict[str, Any]: A dictionary containing the filename, raw text,
                        and a list of sentences.
    """
    filename = uploaded_file.name
    raw_text = ""

    # Check file type and parse accordingly
    if filename.lower().endswith('.pdf'):
        raw_text = parse_pdf(uploaded_file)
    elif filename.lower().endswith('.txt'):
        raw_text = uploaded_file.getvalue().decode('utf-8')
    else:
        # Placeholder for other file types like .docx in the future
        # For now, we can raise an error or return an empty structure
        return {
            'filename': filename,
            'raw_text': '',
            'sentences': [],
            'error': 'Unsupported file type'
        }

    cleaned_text = clean_text(raw_text)
    sentences = chunk_text_into_sentences(cleaned_text)

    candidate_data = {
        'filename': filename,
        'raw_text': raw_text,
        'sentences': sentences
    }
    return candidate_data
