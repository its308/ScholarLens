# 📚 ScholarLens: AI Research Assistant

ScholarLens is an intelligent research assistant that helps researchers explore academic papers and get AI-powered insights about scientific topics. This tool combines state-of-the-art AI technologies to make research more efficient and accessible.

##  Features

- **Smart Paper Search**: Find relevant academic papers from ArXiv based on your research topics with real-time status updates
- **Automatic Summarization**: Get concise summaries of complex research papers using BART-large-CNN
- **Direct Paper Access**: View abstracts and download full PDFs with one-click links to ArXiv
- **Keyword Extraction**: Automatically identify key concepts from each paper
- **Dual Query System**:
  - **General Research Questions**: Get comprehensive answers about research topics powered by Google's Gemini AI
  - **Paper-Specific Queries**: Ask detailed questions about individual papers using Retrieval-Augmented Generation
- **Fallback Mechanism**: Seamlessly switches to RAG when Gemini API is unavailable
- **Sample Topics**: Quick-start your research with predefined topic suggestions

## Technology Stack

- **Frontend**: Streamlit for responsive and interactive UI
- **AI Models**:
  - BART-large-CNN for text summarization
  - Flan-T5 for paper-specific question answering
  - SentenceTransformer (all-MiniLM-L6-v2) for semantic embeddings
  - Google Gemini 2.0 Flash for general research questions
- **Vector Search**: FAISS for efficient similarity search
- **Data Source**: ArXiv API for academic paper retrieval
- **Natural Language Processing**: NLTK for keyword extraction

## Prerequisites

- Python 3.9 or higher
- Google API key for Gemini AI
- Internet connection for ArXiv API access

##  Installation

1. Clone this repository:
```bash
git clone https://github.com/its308/ScholarLens.git
cd ScholarLens
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google API key:
   - Get an API key from [Google AI Studio](https://makersuite.google.com/)
   - Create a `.streamlit/secrets.toml` file with:
     ```
     GOOGLE_API_KEY = "your-api-key-here"
     ```
   - Or replace the placeholder API key in app.py

##  Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Enter a research topic in the search box or select a sample topic

3. Browse the retrieved papers in the tabbed interface:
   - View full abstracts and concise summaries side-by-side
   - Access direct links to ArXiv abstract pages and PDF downloads
   - Explore automatically extracted keywords

4. Ask questions:
   - Select "General Query" for broad research questions
   - Select a specific paper to ask questions about that paper
   - View real-time generation status updates

##  Project Structure

- **app.py**: Streamlit web interface with UI components and user interaction
- **main.py**: Core functionality module containing paper retrieval, embedding, and query processing
- **summarizer.py**: Paper summarization component
- **requirements.txt**: Dependencies list for easy installation
- **papers.json**: Storage for paper data with duplicate detection
- **faiss_index.index**: Vector search index file

##  How It Works

1. **Paper Retrieval**:
   - User enters a research topic
   - System queries ArXiv API with search parameters
   - XML response is parsed into structured paper data

2. **Semantic Search**:
   - SentenceTransformer creates embeddings for each paper
   - FAISS index enables efficient similarity search
   - Most relevant papers are ranked and presented

3. **Paper Analysis**:
   - BART-large-CNN generates concise summaries
   - NLTK extracts key terms from abstracts
   - Paper metadata and links are formatted for display

4. **Question Answering**:
   - **General Queries**: Gemini AI processes broad research questions
   - **RAG Fallback**: If Gemini is unavailable, system falls back to RAG
   - **Paper-Specific Queries**: T5 model generates answers using paper context


## Known Limitations

- Currently limited to ArXiv papers and their abstracts
- Summarization quality varies based on paper complexity
- API rate limits may affect performance during heavy usage
- Large language models may occasionally generate inaccurate responses

## License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Built with Streamlit, Hugging Face Transformers, and Google Gemini
- Uses the ArXiv API for academic paper retrieval

---
Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50435234/84d4317c-5a21-450b-ac8b-e2b9ff0af3d7/paste.txt
[2] https://github.com/its308/ScholarLens

