# tests/test_parser.py
import sys
import os

# This line adds the parent directory (your project root) to the Python path.
# This allows the script to find and import the 'core' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.parser import parse_resume

# Mock an uploaded file object for testing purposes
class MockUploadedFile:
    """
    A mock class to simulate Streamlit's UploadedFile object.
    Pdfplumber expects a file-like object with read(), seek(), and tell() methods.
    """
    def __init__(self, path):
        self.name = os.path.basename(path)
        self._file = open(path, 'rb')

    def getvalue(self):
        # Reset pointer and read from the beginning
        self._file.seek(0)
        return self._file.read()

    def read(self, *args):
        return self._file.read(*args)

    def seek(self, offset, whence=0):
        # --- FIX: Added the seek method ---
        return self._file.seek(offset, whence)

    def tell(self):
        # --- FIX: Added the tell method for completeness ---
        return self._file.tell()

    def close(self):
        self._file.close()


def test_parse_resume():
    """
    Tests the parse_resume function with a sample PDF.
    """
    # Important: The path is now relative to this test script's location
    # It navigates up one directory ('..') from 'tests/' and then into 'data/resumes/'
    resume_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'resumes', 'sample_resume.pdf')

    if not os.path.exists(resume_path):
        print(f"Error: Test file not found at {resume_path}")
        print("Please add a sample PDF resume to data/resumes/sample_resume.pdf")
        return

    mock_file = MockUploadedFile(resume_path)

    try:
        parsed_data = parse_resume(mock_file)

        print(f"--- Parsed Data for {parsed_data.get('filename')} ---")
        if parsed_data.get('error'):
            print(f"Error: {parsed_data['error']}")
        else:
            assert 'filename' in parsed_data
            assert 'raw_text' in parsed_data
            assert 'sentences' in parsed_data
            assert parsed_data['filename'] == 'sample_resume.pdf'
            assert len(parsed_data['raw_text']) > 100 # Check that some text was extracted
            assert len(parsed_data['sentences']) > 5 # Check that sentences were chunked

            print("Successfully parsed.")
            print(f"\nFirst 500 characters of raw text:\n{parsed_data['raw_text'][:500]}...")
            print(f"\nNumber of sentences: {len(parsed_data['sentences'])}")
            print(f"\nFirst 3 sentences:\n{parsed_data['sentences'][:3]}")
            print("\nTest passed!")

    finally:
        # Ensure the underlying file is always closed
        mock_file.close()


if __name__ == "__main__":
    test_parse_resume()

