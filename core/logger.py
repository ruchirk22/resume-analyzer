# core/logger.py

import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any

LOG_FILE_PATH = os.path.join('data', 'outputs', 'audit_log.csv')

def log_analysis_event(
    jd_text: str, 
    results: List[Dict[str, Any]], 
    anonymized: bool
):
    """
    Logs the details of an analysis run to a CSV file.

    Args:
        jd_text: The text of the job description used.
        results: The list of result dictionaries for each candidate.
        anonymized: A boolean indicating if the anonymization feature was used.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        
        num_resumes = len(results)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a summary of top candidates for the log
        top_candidates = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
        top_scores = {
            res['filename']: res['score'] for res in top_candidates
        }

        log_entry = {
            'timestamp': timestamp,
            'num_resumes_processed': num_resumes,
            'anonymization_enabled': anonymized,
            'top_3_candidates_scores': str(top_scores),  # Store dict as string
            'jd_snippet': jd_text[:150].replace('\n', ' ') + '...'
        }

        # Check if log file exists to write header or not
        file_exists = os.path.isfile(LOG_FILE_PATH)
        
        df = pd.DataFrame([log_entry])
        df.to_csv(LOG_FILE_PATH, mode='a', header=not file_exists, index=False)

    except Exception as e:
        # In a production app, you'd use a proper logging library (e.g., logging)
        # For this Streamlit app, printing to console is sufficient for debugging.
        print(f"Error writing to audit log: {e}")
