# core/privacy.py

import re
from typing import Optional

def anonymize_text(text: str, filename: Optional[str] = None) -> str:
    """
    Anonymizes text by removing emails, phone numbers, and potentially the candidate's name.

    Args:
        text: The raw text from the resume.
        filename: The original filename of the resume, used to help find the name.

    Returns:
        Anonymized text.
    """
    # Regex for emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '[EMAIL REDACTED]', text)

    # Regex for phone numbers (handles various formats)
    phone_pattern = r'\(?\b[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b'
    text = re.sub(phone_pattern, '[PHONE REDACTED]', text)

    # Attempt to remove the candidate's name based on the filename
    # This is a simple heuristic and can be improved with NER later
    if filename:
        # Assumes filename is like 'John_Doe_Resume.pdf' or 'ruchir.kulkarni.pdf'
        # Clean the filename from extensions and separators
        name_from_file = re.sub(r'(\.pdf|\.txt|\.docx)', '', filename, flags=re.IGNORECASE)
        name_parts = re.split(r'[_.\- ]', name_from_file)
        
        for part in name_parts:
            if len(part) > 1: # Avoid stripping single letters
                # Case-insensitive search for each part of the name
                name_regex = re.compile(r'\b' + re.escape(part) + r'\b', re.IGNORECASE)
                text = name_regex.sub('[CANDIDATE NAME REDACTED]', text)

    return text
