# core/parser.py

import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
import re
import io
import docx # Import the new library

# Download the 'punkt' tokenizer data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

def clean_text(text: str) -> str:
    """Cleans raw text by removing extra whitespace and non-ASCII characters."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.lower()

def chunk_text_into_sentences(text: str) -> list[str]:
    """Splits a block of text into a list of sentences."""
    return sent_tokenize(text)

# --- NEW FUNCTION for DOCX parsing ---
def parse_docx(file_stream) -> str:
    """Extracts text from a .docx file stream."""
    try:
        document = docx.Document(file_stream)
        full_text = [para.text for para in document.paragraphs]
        return '\n'.join(full_text)
    except Exception:
        # docx library can be sensitive to file corruption
        return ""

def parse_pdf(file_path_or_stream) -> str:
    """Extracts text from a .pdf file path or stream."""
    try:
        with pdfplumber.open(file_path_or_stream) as pdf:
            full_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
        return '\n'.join(full_text)
    except Exception:
        # pdfplumber can fail on corrupted or complex PDFs
        return ""

# --- MODIFIED: The main parsing function ---
def parse_resume(uploaded_file) -> dict:
    """
    Parses an uploaded resume file (.pdf or .docx), returning its content.
    Automatically detects the file type based on its name.
    """
    filename = uploaded_file.name
    raw_text = ""

    try:
        # Use a file-like object in memory
        file_stream = io.BytesIO(uploaded_file.getvalue())

        if filename.lower().endswith('.pdf'):
            raw_text = parse_pdf(file_stream)
        elif filename.lower().endswith('.docx'):
            raw_text = parse_docx(file_stream)
        else:
            return {"error": "Unsupported file format. Please upload a PDF or DOCX file."}

        if not raw_text.strip():
            return {"error": f"Could not extract text from {filename}. The file might be empty, corrupted, or image-based."}

        cleaned_text = clean_text(raw_text)
        sentences = chunk_text_into_sentences(cleaned_text)

        return {
            "filename": filename,
            "raw_text": cleaned_text,
            "sentences": sentences
        }
    except Exception as e:
        return {"error": f"An unexpected error occurred while processing {filename}: {str(e)}"}

