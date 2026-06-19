"""
Large-Scale Animal Impound Handbook — Local Web App (Markdown-powered)
Uses document_clean.md as the source of truth for navigation, search, summaries, and Q&A.
Original PDF remains available for download/reference.
"""

import streamlit as st
import fitz  # only for optional page image rendering
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import requests
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ===================== CONFIG =====================
CLEAN_MD_PATH = "document_clean.md"
PDF_PATH = "document.pdf"
DATA_DIR = Path("data")
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_md")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# =============================================================================
# OPTIONAL: Ollama configuration (local LLM only)
# On Streamlit Community Cloud this will not be available.
# The app gracefully falls back to showing relevant passages.
# To use locally:
#   1. Install Ollama (https://ollama.com)
#   2. Run `ollama serve`
#   3. Run `ollama pull llama3.2` (or another model)
# No secrets/API keys are required for core functionality.
# =============================================================================
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

# Section to approximate page ranges (1-indexed) from original PDF structure
# This enables automatic association of visuals with sections
SECTION_PAGES = {
    "introduction": list(range(1, 7)),
    "section-1-preparedness-before-a-case-begins": list(range(7, 15)),
    "section-2-case-initiation-deciding-the-course-of-action": list(range(15, 27)),
    "section-3-laws-and-procedures-governing-animal-cruelty-and-impound-cases": list(range(27, 40)),
    "section-4-operational-response-the-impound": list(range(40, 45)),
    "section-5-cost-of-care-and-financial-management": list(range(45, 57)),
    "section-6-communications": list(range(57, 68)),
    "section-7-after-action-coordination-monitoring-case-progress": list(range(68, 73)),
    "glossary": list(range(73, 78)),
    "appendix": list(range(78, 81)),
}

def get_pages_for_section(sec_id: str, sec_title: str = "") -> List[int]:
    """Return list of pages associated with a section. Uses fuzzy matching on id or title."""
    # Exact
    if sec_id in SECTION_PAGES:
        return SECTION_PAGES[sec_id]
    # Try matching by keywords in title or id
    key = (sec_id + " " + sec_title).lower()
    for sid, pages in SECTION_PAGES.items():
        if sid in key or any(kw in key for kw in sid.split("-")[:3]):
            return pages
    # Fallback
    if not key or "intro" in key:
        return list(range(1, 7))
    return [1]

# ===================== HELPERS =====================
def load_markdown() -> str:
    if not os.path.exists(CLEAN_MD_PATH):
        st.error(f"Clean Markdown not found: {CLEAN_MD_PATH}")
        st.stop()
    with open(CLEAN_MD_PATH, "r", encoding="utf-8") as f:
        return f.read()

def get_introduction_content(md_text: str) -> str:
    """Return the exact text and formatting for the Introduction (first ~5 pages / front matter)."""
    # Cut off before the first main content section to stick to the first 5 pages
    idx = md_text.find("# Section 1:")
    if idx != -1:
        content = md_text[:idx].strip()
    else:
        content = md_text[:4000].strip()  # safe fallback
    # Remove any leading artifact characters from earlier cleaning
    content = content.lstrip("—").strip()
    return content

def parse_sections(md_text: str) -> List[Dict]:
    """Parse main sections from markdown headings."""
    lines = md_text.splitlines()
    sections = []
    current = {"id": "front", "title": "Front Matter", "level": 0, "start_line": 0, "content_start": 0}
    content_lines = []

    for i, line in enumerate(lines):
        if line.startswith("# "):
            if current:
                current["content"] = "\n".join(content_lines)
                sections.append(current)
            title = line[2:].strip()
            sid = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            current = {
                "id": sid or f"sec-{len(sections)}",
                "title": title,
                "level": 1,
                "start_line": i,
                "content_start": i + 1,
            }
            content_lines = []
        elif line.startswith("## ") and current:
            # treat level 2 as subsections but keep under current main
            pass
        else:
            content_lines.append(line)

    if current:
        current["content"] = "\n".join(content_lines)
        sections.append(current)

    # Filter to meaningful sections
    return [s for s in sections if len(s.get("content", "").strip()) > 20]

def chunk_markdown(md_text: str, max_chars: int = 800, overlap: int = 150) -> List[Dict]:
    """Chunk the markdown by sections then sliding window."""
    chunks = []
    chunk_id = 0
    sections = parse_sections(md_text)

    for sec in sections:
        text = sec.get("content", "") or sec.get("title", "")
        if not text.strip():
            continue
        # simple paragraph split
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        current_chunk = ""
        for para in paras:
            if len(current_chunk) + len(para) + 1 > max_chars and current_chunk:
                chunks.append({
                    "id": f"sec-{sec['id']}_c{chunk_id}",
                    "section_id": sec["id"],
                    "section_title": sec["title"],
                    "text": current_chunk.strip(),
                })
                chunk_id += 1
                # overlap
                current_chunk = current_chunk[-overlap:] + " " + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        if current_chunk.strip():
            chunks.append({
                "id": f"sec-{sec['id']}_c{chunk_id}",
                "section_id": sec["id"],
                "section_title": sec["title"],
                "text": current_chunk.strip(),
            })
            chunk_id += 1
    return chunks

# ===================== PDF VISUALS (Graphs, Images, Diagrams) =====================
IMAGE_DIR = DATA_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_data(show_spinner=False)
def extract_embedded_images():
    """Extract embedded images from PDF and save to disk. Returns list of metadata."""
    if not os.path.exists(PDF_PATH):
        return []
    doc = fitz.open(PDF_PATH)
    images_meta = []
    for pno in range(len(doc)):
        page = doc[pno]
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            # Skip very small images (icons, logos)
            if len(image_bytes) < 5000:
                continue
            filename = f"pdf_p{pno+1}_img{idx}.{ext}"
            fpath = IMAGE_DIR / filename
            if not fpath.exists():
                with open(fpath, "wb") as f:
                    f.write(image_bytes)
            images_meta.append({
                "page": pno + 1,
                "index": idx,
                "path": str(fpath),
                "size": len(image_bytes),
                "ext": ext,
            })
    doc.close()
    return images_meta

@st.cache_data(show_spinner=False)
def render_pdf_pages(page_nums: List[int], dpi: int = 110) -> List[Tuple[int, bytes]]:
    """Render specific PDF pages as PNG bytes. Returns list of (page, image_bytes)."""
    if not os.path.exists(PDF_PATH):
        return []
    doc = fitz.open(PDF_PATH)
    results = []
    for p in page_nums:
        if 1 <= p <= len(doc):
            pix = doc[p-1].get_pixmap(dpi=dpi)
            results.append((p, pix.tobytes("png")))
    doc.close()
    return results

def simple_summary(text: str, max_sentences: int = 4) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(s.strip() for s in sentences[:max_sentences] if s.strip())[:900]

# ===================== VECTOR STORE =====================
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner="Setting up vector database for Markdown content...")
def get_chroma_client():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client

def get_or_create_collection(client):
    name = "handbook_md_chunks"
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)

@st.cache_resource(show_spinner="Indexing clean Markdown content (first run)...")
def load_indexed_md():
    md_text = load_markdown()
    sections = parse_sections(md_text)
    chunks = chunk_markdown(md_text)

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    embedder = get_embedder()

    existing = collection.count()
    if existing < max(5, len(chunks) - 20):
        try:
            client.delete_collection(collection.name)
        except Exception:
            pass
        collection = client.create_collection(name=collection.name)

        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        ids = [c["id"] for c in chunks]
        metadatas = [{"section_id": c["section_id"], "section_title": c["section_title"]} for c in chunks]

        batch_size = 128
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size].tolist(),
                metadatas=metadatas[i:i+batch_size],
            )
        print(f"Indexed {len(chunks)} chunks from clean Markdown")

    return {
        "md_text": md_text,
        "sections": sections,
        "chunks": chunks,
        "collection": collection,
        "embedder": embedder,
    }

# ===================== SEARCH =====================
def semantic_search(query: str, collection, embedder, top_k: int = 8) -> List[Dict]:
    if not query.strip():
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    for i in range(len(results["ids"][0])):
        dist = results["distances"][0][i]
        score = round(1 / (1 + dist), 3)
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "section_id": results["metadatas"][0][i]["section_id"],
            "section_title": results["metadatas"][0][i]["section_title"],
            "score": score,
        })
    return hits

def keyword_search(query: str, chunks: List[Dict], top_k: int = 8) -> List[Dict]:
    q = query.lower()
    scored = []
    for c in chunks:
        score = c["text"].lower().count(q) * 2 + int(q in c["text"].lower())
        if score > 0:
            scored.append((score, c))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:top_k]]

# ===================== Q&A =====================
def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def query_ollama(question: str, context_chunks: List[str]) -> Optional[str]:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant for the "Large-Scale Animal Impound Handbook".
Answer using ONLY the provided context from the clean handbook.
Be concise, accurate, and cite sections/pages when possible.
If the answer is not supported, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 700}}
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[Ollama error: {str(e)}]"
    return None

# ===================== RENDER =====================
def render_section(sec: Dict, indexed: Dict):
    st.subheader(sec["title"])

    with st.expander("📋 Quick Summary & Key Points", expanded=True):
        summary = simple_summary(sec.get("content", sec.get("title", "")))
        st.write(summary or "No summary available.")
        # very basic key points
        kws = ["must", "should", "critical", "important", "required", "never", "always", "key"]
        kps = [s.strip() for s in re.split(r'[.!?]', sec.get("content", "")) if any(k in s.lower() for k in kws) and 30 < len(s) < 200]
        if kps:
            st.markdown("**Key points (heuristic):**")
            for kp in list(dict.fromkeys(kps))[:5]:
                st.markdown(f"- {kp}")

    content = sec.get("content", "")
    if content:
        st.markdown(content[:8000] if len(content) > 8000 else content)
    else:
        st.text(sec.get("title", ""))

    # === AUTOMATIC VISUALS FROM ORIGINAL PDF ===
    st.markdown("### 📊 Original PDF Visuals (Graphs, Diagrams, Flowcharts)")
    st.caption("Automatically rendered from the source PDF for fidelity. Flowcharts, tables, and diagrams are shown as they appear in the original.")

    pages = get_pages_for_section(sec["id"], sec.get("title", ""))
    page_renders = render_pdf_pages(pages[:2], dpi=105)  # Show up to 2 representative pages

    if page_renders:
        cols = st.columns(min(2, len(page_renders)))
        for idx, (pnum, img_bytes) in enumerate(page_renders):
            with cols[idx % len(cols)]:
                st.image(img_bytes, caption=f"Page {pnum} from original PDF", use_column_width=True)
    else:
        st.info("No page visuals available for this section.")

    # Show any extracted embedded images for these pages
    embedded = extract_embedded_images()
    relevant_imgs = [im for im in embedded if im["page"] in pages]
    if relevant_imgs:
        st.markdown("**Extracted images from these pages:**")
        img_cols = st.columns(min(3, len(relevant_imgs)))
        for i, img in enumerate(relevant_imgs[:6]):
            with img_cols[i % len(img_cols)]:
                st.image(img["path"], caption=f"Page {img['page']} (embedded)", use_column_width=True)

def show_search_results(hits: List[Dict], indexed: Dict, query: str):
    st.markdown(f"### Search results for: **{query}**")
    for hit in hits:
        with st.container(border=True):
            st.markdown(f"**{hit['section_title']}** (score: {hit['score']:.2f})")
            st.text(hit["text"][:500] + ("..." if len(hit["text"]) > 500 else ""))
            if st.button("Jump to this section", key=f"jump_{hit['id']}"):
                st.session_state["current_section"] = hit["section_id"]
                st.session_state["view"] = "browse"
                st.rerun()

def show_qa(indexed: Dict):
    st.subheader("💬 Ask the Handbook")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                st.caption("Sources: " + ", ".join(msg["sources"]))

    q = st.chat_input("Ask a natural language question...")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        hits = semantic_search(q, indexed["collection"], indexed["embedder"], top_k=6)
        ctx = [h["text"] for h in hits]
        srcs = [h["section_title"] for h in hits]

        with st.chat_message("assistant"):
            if ollama_available():
                ans = query_ollama(q, ctx)
                if ans:
                    st.markdown(ans)
                    st.caption("Sources: " + "; ".join(srcs[:3]))
                    st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
                else:
                    st.info("Showing best matching passages (Ollama gave no response):")
                    for h in hits[:3]:
                        st.text(f"[{h['section_title']}] " + h["text"][:350])
            else:
                st.info("Ollama not running — showing top relevant passages from the clean handbook.")
                for h in hits[:4]:
                    with st.container(border=True):
                        st.markdown(f"**{h['section_title']}**")
                        st.text(h["text"][:380])
                st.session_state.messages.append({"role": "assistant", "content": "Retrieved top passages.", "sources": srcs})

# ===================== MAIN =====================
def main():
    st.set_page_config(page_title="Large-Scale Animal Impound Handbook", page_icon="📕", layout="wide")

    # Incorporate PDF color scheme via custom CSS
    st.markdown("""
    <style>
    /* Main theme colors pulled from the PDF's dark teal/navy + cool gray palette */
    .stApp {
        background-color: #F5F8FA;
    }
    
    /* Headers styled to match PDF — use dark teal for body headers, accent green for emphasis */
    h1, h2, h3, h4 {
        color: #033A4B !important;
        font-family: system-ui, -apple-system, sans-serif;
    }
    
    /* Sidebar styling to echo the PDF's professional look */
    [data-testid="stSidebar"] {
        background-color: #E8EEF1;
        border-right: 1px solid #C5D2D8;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #033A4B !important;
    }
    
    /* Main action buttons (outside sidebar or primary) use dark teal */
    .stButton > button[kind="primary"] {
        background-color: #033A4B;
        color: white;
        border: none;
    }

    /* Sidebar buttons match the main header bar color (#033A4B) with white text */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #033A4B;
        color: white;
        text-align: left;
        border: 1px solid #033A4B;
        border-radius: 4px;
        margin-bottom: 2px;
        padding: 6px 10px;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #024252;
        color: white;
        border-color: #024252;
    }
    
    /* Accent for important captions */
    .stCaption {
        color: #4A6F7A;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header bar modeled directly after the PDF cover
    # Uses exact colors from cover: dark #033A4B background + #49BC97 accent title
    # Three-line stacked title to closely match the large cover typography
    st.markdown("""
    <div style="
        background-color: #033A4B; 
        padding: 8px 16px 6px 16px; 
        margin: -1rem -1rem 0.5rem -1rem;
        border-bottom: 4px solid #49BC97;
    ">
        <div style="line-height: 0.92;">
            <div style="
                color: #49BC97; 
                font-size: 1.18rem; 
                font-weight: 800; 
                letter-spacing: 2px; 
                line-height: 1.0;
            ">LARGE-SCALE</div>
            <div style="
                color: #49BC97; 
                font-size: 1.18rem; 
                font-weight: 800; 
                letter-spacing: 2px; 
                line-height: 1.0;
            ">ANIMAL IMPOUND</div>
            <div style="
                color: #49BC97; 
                font-size: 1.18rem; 
                font-weight: 800; 
                letter-spacing: 2px; 
                line-height: 1.0;
                margin-bottom: 1px;
            ">HANDBOOK</div>
        </div>
        <div style="
            color: #A8C4D0; 
            font-size: 0.72rem; 
            font-weight: 500; 
            letter-spacing: 0.8px; 
            margin-top: 1px;
        ">Who Investigates Animal-Related Cases in Colorado?</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Clean Markdown edition • Semantic search + Q&A • Original PDF available for reference")

    indexed = load_indexed_md()

    # Pre-warm visuals cache (embedded images + page renders will be fast after first use)
    _ = extract_embedded_images()

    # State
    if "current_section" not in st.session_state:
        st.session_state.current_section = ""
    if "view" not in st.session_state:
        st.session_state.view = "introduction"

    # Sidebar
    with st.sidebar:
        st.header("📖 Sections")

        if st.button("Introduction", use_container_width=True):
            st.session_state.view = "introduction"
            st.session_state.current_section = ""

        st.divider()

        current_sec = st.session_state.get("current_section", "")

        # Only show main document sections (skip front-matter which is covered by Introduction)
        real_sections = [s for s in indexed["sections"] if s["title"].startswith("Section ")]

        for sec in real_sections:
            is_current = sec["id"] == current_sec
            label = f"▶ {sec['title']}" if is_current else sec["title"]
            if st.button(label, key=f"nav_{sec['id']}", use_container_width=True):
                st.session_state.current_section = sec["id"]
                st.session_state.view = "browse"
                st.rerun()
        if st.button("📊 Tables & Data (from original PDF)", use_container_width=True):
            st.session_state.view = "tables"
        if st.button("💬 Ask Questions", use_container_width=True):
            st.session_state.view = "qa"
        if st.button("🖼️ Images & Graphs Gallery", use_container_width=True):
            st.session_state.view = "images"

        st.divider()
        st.subheader("🔍 Search the clean handbook")
        sq = st.text_input("Semantic search", key="search", placeholder="three critical questions cost recovery")
        if st.button("Search", type="primary", use_container_width=True) and sq.strip():
            st.session_state.last_search = sq
            st.session_state.view = "search"

        st.divider()
        # PDF Download
        st.subheader("📄 Original PDF")
        if os.path.exists(PDF_PATH):
            with open(PDF_PATH, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="⬇️ Download original PDF",
                data=pdf_bytes,
                file_name="Large-Scale_Animal_Impound_Handbook.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption("Use the original for exact tables, flowcharts, and formatting.")
        else:
            st.warning("Original PDF not found in folder.")

        st.caption(f"Sections: {len(indexed['sections'])} | Chunks: {len(indexed['chunks'])}")

    # Main views
    view = st.session_state.view

    if view == "introduction":
        st.header("Introduction")
        intro_text = get_introduction_content(indexed["md_text"])
        st.markdown(intro_text)

        # Include the flowchart visual from page 5 of the original PDF
        st.markdown("---")
        st.subheader("Visual: Animal Case Initiation Decision-Making Framework")
        st.caption("Flowchart from Page 5 of the original document")
        page_renders = render_pdf_pages([5], dpi=150)
        if page_renders:
            st.image(page_renders[0][1], use_column_width=True)
        else:
            st.info("Visual from page 5 could not be rendered.")
    elif view == "search":
        hits = semantic_search(st.session_state.get("last_search", ""), indexed["collection"], indexed["embedder"], top_k=10)
        show_search_results(hits, indexed, st.session_state.get("last_search", ""))
        with st.expander("Keyword matches"):
            for h in keyword_search(st.session_state.get("last_search", ""), indexed["chunks"], 5):
                st.text(f"[{h['section_title']}] {h['text'][:280]}...")
    elif view == "tables":
        st.header("Tables & Important Data")
        st.info("Full detailed tables and flowcharts are best viewed in the original PDF (use the download button in the sidebar).")
        st.markdown("Key data categories covered in the handbook include staffing/resources, agency roles, urgency/removal timelines, cost components, best practices checklists, and after-action frameworks.")
        st.markdown("See the Appendix in the clean Markdown and the downloadable PDF for the complete set of templates and tools.")
    elif view == "qa":
        show_qa(indexed)
    elif view == "images":
        st.header("🖼️ Images, Graphs & Diagrams from the PDF")
        st.caption("Automatically extracted and rendered visuals (embedded images + full page renders for context).")

        embedded = extract_embedded_images()
        if embedded:
            st.subheader("Embedded Images from PDF")
            cols = st.columns(3)
            for i, img in enumerate(embedded):
                with cols[i % 3]:
                    st.image(img["path"], caption=f"Page {img['page']}", use_column_width=True)
        else:
            st.info("No large embedded images detected (most diagrams are drawn directly in the PDF).")

        st.subheader("Key Page Renders (containing graphs/flowcharts)")
        # Show pages that are known to have visuals + a few representative ones
        key_pages = sorted(set([p for pages in SECTION_PAGES.values() for p in pages if p in [5,19,23,35,63,64,69]]))
        if not key_pages:
            key_pages = [5, 19, 35, 64]
        renders = render_pdf_pages(key_pages[:6], dpi=100)
        for pnum, img_bytes in renders:
            st.image(img_bytes, caption=f"Page {pnum} — Visual reference (flowchart / diagram area)", use_column_width=True)

    else:
        # browse
        sec = next((s for s in indexed["sections"] if s["id"] == st.session_state.current_section), indexed["sections"][0])
        render_section(sec, indexed)

    st.divider()
    st.caption("All content derived from the clean Markdown (document_clean.md). Original PDF kept for reference and legal accuracy.")

if __name__ == "__main__":
    main()
