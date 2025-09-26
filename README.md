# AI Resume Shortlister

![Resume Shortlister Logo](https://img.shields.io/badge/AI-Resume%20Shortlister-blue)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.24+-red.svg)
![Docker](https://img.shields.io/badge/docker-ready-green.svg)

An AI-powered application that automates the resume screening process by matching candidate profiles against job descriptions using natural language processing and machine learning techniques.

## Features

- **Resume Parsing**: Extract text from PDF and DOCX resume files
- **Job Description Analysis**: Extract key requirements and skills from job descriptions
- **Semantic Matching**: Uses advanced NLP to match resumes against job requirements
- **Candidate Scoring**: Automatically ranks candidates based on relevance to the job
- **Evidence Highlighting**: Shows specific sentences that match job requirements
- **Skills Extraction**: Identifies and displays candidate skills as tags
- **Privacy-First Design**: Option to anonymize candidate information
- **AI-Powered Analysis**: Optional integration with Google's Gemini API for in-depth analysis
- **Categorization**: Groups candidates into Strong, Good, Average, or Weak matches
- **Audit Logging**: Records all analysis activities for compliance

## Technologies Used

- **Streamlit**: Interactive web interface
- **SentenceTransformers**: Semantic text embeddings
- **NLTK & scikit-learn**: Natural language processing
- **PDFPlumber & python-docx**: Document parsing
- **Google Gemini API (optional)**: Advanced AI analysis
- **Pandas**: Data handling and manipulation
- **Docker**: Application containerization

## Prerequisites

- Python 3.11 or higher
- Docker (for containerized deployment)
- Google Gemini API key (optional, for enhanced AI analysis)

## Installation & Setup

### Option 1: Local Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ruchirk22/resume-analyzer
   cd resume-shortlister/Code
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up Google Gemini API:
   - Create a `.streamlit/secrets.toml` file with:

     ```toml
     GEMINI_API_KEY = "your_api_key_here"
     ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

### Option 2: Docker Installation

1. Build the Docker image:

   ```bash
   docker build -t resume-shortlister .
   ```

2. Run the container:

   ```bash
   docker run -p 8501:8501 resume-shortlister
   ```

3. (Optional) To use Google Gemini API with Docker:

   ```bash
   docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key_here" resume-shortlister
   ```

## Usage

1. Open the application in your browser at `http://localhost:8501`
2. Paste the job description in the sidebar
3. Upload up to 10 resumes (PDF or DOCX format)
4. Configure settings (anonymization, AI analysis, threshold values)
5. Click "Analyze Resumes" to start the process
6. View results in either Essential or Advanced view
7. Make hiring decisions (Shortlist/Maybe/Reject) for each candidate
8. Download your results as CSV

## Project Structure

```text
Code/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── README.md              # This documentation
├── core/                  # Core application logic
│   ├── analyzer.py        # Resume analysis functions
│   ├── embedding.py       # Text embedding generation
│   ├── llm.py             # Gemini AI integration
│   ├── logger.py          # Audit logging functions
│   ├── matcher.py         # Resume-JD matching logic
│   ├── parser.py          # PDF/DOCX parsing functions
│   └── privacy.py         # Anonymization functions
├── data/                  # Data storage
│   └── outputs/           # Analysis logs and outputs
└── tests/                 # Unit tests
    ├── test_analyzer.py
    ├── test_embedding.py
    ├── test_matcher.py
    └── test_parser.py
```

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/resume-shortlister/issues).

## Privacy & Data Security

- The application does not store uploaded resumes after analysis
- The anonymization feature removes personally identifiable information
- All data processing happens locally on your machine or within your Docker container
- No data is sent to external servers (except for optional Gemini API analysis)
- Audit logging records analysis actions for compliance without storing resume content

## Acknowledgements

- SentenceTransformer models by UKPLab
- Streamlit for the interactive web framework
- Google for the Gemini API
- The open-source community for all the amazing libraries used in this project
- Developed as part of assignment given by Ema.
