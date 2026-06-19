# Large-Scale Animal Impound Handbook — Streamlit App

A clean, interactive web application for exploring the **Large-Scale Animal Impound Handbook** (an 80-page practical guide for agencies handling large-scale animal impound cases in Colorado).

## Features

- **Introduction** — Full text and formatting from the first 5 pages of the document, including the key decision-making flowchart (Page 5)
- **Section Navigation** — Clickable list of all main sections in the sidebar (no dropdown)
- **Semantic Search** — Powered by local sentence-transformers embeddings + ChromaDB
- **Keyword fallback** search
- **Natural Language Q&A** — Shows relevant passages (Ollama integration for full LLM answers is optional/local-only)
- **Images & Graphs Gallery** — Embedded images + rendered pages from the original PDF
- **Original PDF Download** — Always available as reference
- **Fully self-contained** after initial model download

## Local Run (Quick Start)

### Option 1: Using the provided script (Windows)
```powershell
cd "C:\Users\just4\pdf-to-app"
powershell -ExecutionPolicy Bypass -File .\Start-App.ps1
```

### Option 2: Manual (recommended for all platforms)
```bash
# 1. Clone or download the repo
cd pdf-to-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

**Note:** The first run will download the embedding model (~90 MB) and build the vector index. Subsequent runs are fast.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (public or private).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app** → select your repo and branch.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

The app will be live at a share.streamlit.io URL. No API keys or external services are required.

### Important Notes for Deployment
- The embedding model will be downloaded on first cloud run (this is normal and cached by Streamlit).
- The original `document.pdf` is included for image rendering and download.
- `document_clean.md` is the primary source for all text, search, and Q&A — no heavy PDF text extraction happens at runtime.
- Ollama is **not available** on Streamlit Cloud. The app gracefully falls back to showing the most relevant passages from the handbook.

## Project Structure

```
pdf-to-app/
├── .streamlit/
│   └── config.toml          # App theme & server settings
├── app.py                   # Main Streamlit application
├── document_clean.md        # Clean, structured Markdown (primary data source)
├── document.pdf             # Original PDF (for images + download reference)
├── requirements.txt         # Python dependencies
├── README.md
└── data/                    # Generated at runtime (embeddings cache)
```

## Data Sources

- **Primary content**: `document_clean.md` (extracted and cleaned from the original PDF using high-quality tools).
- The app only accesses the original PDF to:
  - Render specific pages as images (for visuals and the gallery)
  - Allow users to download the complete original document

## Optional: Local LLM (Ollama)

For enhanced Q&A on your local machine:

```bash
# Install Ollama (https://ollama.com)
ollama serve
ollama pull llama3.2   # or phi3, gemma2, etc.
```

The app will automatically detect Ollama when running locally.

## License & Attribution

This tool is for educational and practical use with the handbook. The content of the handbook remains the property of its authors.

---

Built for easy sharing and deployment to Streamlit Community Cloud.
