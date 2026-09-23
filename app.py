import io
import re
from collections import Counter

import streamlit as st
from pypdf import PdfReader


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="AI Knowledge Assistant — Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        .demo-banner {
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1e3a8a;
            margin-bottom: 18px;
            font-size: 0.9rem;
        }

        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }

        .source-box {
            background: #f8fafc;
            border-left: 4px solid #64748b;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 4px;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.82rem;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Constants
# ============================================================
FALLBACK = "The requested information is not available in the provided document."


# ============================================================
# Session state
# ============================================================
defaults = {
    "messages": [],
    "chunks": [],
    "documents": [],
    "indexed": False,
    "last_stats": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Utility functions
# ============================================================
def clean_text(text: str) -> str:
    """Normalize PDF-extracted text."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def extract_pdf(uploaded_file):
    """Extract page-level text from a PDF."""
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def chunk_pages(pages, chunk_size=1000, overlap=200):
    """
    Simple character-based chunker that preserves page metadata.
    This is intentionally lightweight for a demo.
    """
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks = []
    step = max(1, chunk_size - overlap)

    for page_data in pages:
        text = page_data["text"]
        page_number = page_data["page"]

        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start += step

    return chunks


def tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def retrieve(query, chunks, top_k=4):
    """
    Dummy semantic-like retrieval using token overlap.
    This intentionally does NOT use Gemini embeddings or FAISS.
    """
    query_terms = tokenize(query)

    if not query_terms:
        return []

    scored = []

    for chunk in chunks:
        chunk_terms = tokenize(chunk["text"])

        overlap = query_terms.intersection(chunk_terms)
        score = len(overlap)

        # Small bonus when an exact query phrase appears.
        if query.lower().strip() in chunk["text"].lower():
            score += 5

        if score > 0:
            scored.append(
                {
                    **chunk,
                    "score": score,
                    "matched_terms": sorted(overlap),
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def generate_demo_answer(question, retrieved_chunks):
    """
    Deterministic demo answer generator.

    This is NOT an LLM. It extracts relevant sentences from retrieved
    chunks so the UI can demonstrate the RAG workflow without an API key.
    """
    if not retrieved_chunks:
        return FALLBACK

    question_terms = tokenize(question)
    candidates = []

    for chunk in retrieved_chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_terms = tokenize(sentence)
            score = len(question_terms.intersection(sentence_terms))

            if score > 0:
                candidates.append((score, sentence, chunk["page"]))

    candidates.sort(key=lambda x: x[0], reverse=True)

    if not candidates:
        # If retrieval found a chunk but sentence-level matching did not,
        # provide a short excerpt instead of pretending an LLM answered.
        excerpt = retrieved_chunks[0]["text"][:500].strip()
        return (
            "I found relevant content in the document. Here is the most "
            "relevant excerpt from the retrieved context:\n\n"
            f"“{excerpt}…”"
        )

    selected = []
    seen = set()

    for _, sentence, _ in candidates:
        normalized = sentence.lower()
        if normalized not in seen:
            selected.append(sentence)
            seen.add(normalized)

        if len(selected) >= 3:
            break

    return "Based on the retrieved document context:\n\n" + " ".join(selected)


def process_documents(files, chunk_size, overlap):
    all_chunks = []
    document_stats = []

    for uploaded_file in files:
        pages = extract_pdf(uploaded_file)
        chunks = chunk_pages(pages, chunk_size, overlap)

        for chunk in chunks:
            chunk["filename"] = uploaded_file.name

        all_chunks.extend(chunks)

        document_stats.append(
            {
                "filename": uploaded_file.name,
                "pages": len(pages),
                "chunks": len(chunks),
            }
        )

    return all_chunks, document_stats


# ============================================================
# Header
# ============================================================
st.title("🤖 AI Knowledge Assistant (RAG)")
st.caption("Document-grounded question answering — demonstration mode")

st.markdown(
    """
    <div class="demo-banner">
        <strong>DEMO MODE:</strong> This version demonstrates
        RAG workflow without requiring a Gemini API key. 
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ System Configuration")

    st.markdown("### 🔑 API Configuration")
    st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Demo mode — API key not required",
        help="This dummy application does not send data to Gemini.",
        disabled=True,
    )
    st.caption("🟢 Demo mode active — no API key required")

    st.markdown("---")

    st.markdown("### 🎛️ Retrieval Parameters")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=2500,
        value=1000,
        step=100,
        help="Approximate number of characters per chunk.",
    )

    overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Number of overlapping characters between chunks.",
    )

    top_k = st.slider(
        "Top-K Chunks",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Number of relevant chunks used for the answer.",
    )

    st.markdown("---")

    st.markdown("### 📄 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files.",
    )

    if uploaded_files:
        if st.button(
            "🚀 Process & Index Documents",
            use_container_width=True,
            type="primary",
        ):
            if overlap >= chunk_size:
                st.warning(
                    "Chunk overlap must be smaller than chunk size. "
                    "A smaller overlap will be used automatically."
                )

            with st.spinner("Extracting, chunking, and indexing documents..."):
                chunks, stats = process_documents(
                    uploaded_files,
                    chunk_size,
                    overlap,
                )

            st.session_state.chunks = chunks
            st.session_state.documents = stats
            st.session_state.indexed = bool(chunks)
            st.session_state.last_stats = {
                "pages": sum(item["pages"] for item in stats),
                "chunks": len(chunks),
                "files": len(stats),
            }

            if chunks:
                st.success("Documents indexed successfully.")
                st.rerun()
            else:
                st.error("No readable text was found in the uploaded PDF(s).")

    if st.session_state.documents:
        st.markdown("#### 📚 Ingested Library")

        for document in st.session_state.documents:
            st.markdown(
                f"""
                <div style="
                    background:#F8FAFC;
                    border:1px solid #E2E8F0;
                    border-radius:8px;
                    padding:9px 11px;
                    margin-bottom:7px;
                ">
                    <div style="font-weight:600;">
                        📄 {document["filename"]}
                    </div>
                    <div class="small-muted">
                        {document["pages"]} pages • {document["chunks"]} chunks
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chunks = []
            st.session_state.documents = []
            st.session_state.indexed = False
            st.session_state.last_stats = {}
            st.rerun()


# ============================================================
# Main dashboard
# ============================================================
if st.session_state.last_stats:
    stats = st.session_state.last_stats

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="metric-card"><strong>{stats["files"]}</strong><br>'
            f'<span class="small-muted">Documents</span></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div class="metric-card"><strong>{stats["pages"]}</strong><br>'
            f'<span class="small-muted">Pages</span></div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f'<div class="metric-card"><strong>{stats["chunks"]}</strong><br>'
            f'<span class="small-muted">Chunks Indexed</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")


# ============================================================
# Chat history
# ============================================================
if not st.session_state.messages:
    st.info(
        "Upload and index a PDF from the sidebar, then ask a question about "
        "the document."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(
                f"📚 Retrieved Sources ({len(message['sources'])})",
                expanded=False,
            ):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>Source {index}</strong><br>
                            📄 <strong>{source["filename"]}</strong>
                            — Page {source["page"]}
                            <br><br>
                            {source["text"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        f"Retrieval score: {source['score']} "
                        f"| Matched terms: "
                        f"{', '.join(source['matched_terms']) or 'none'}"
                    )


# ============================================================
# Chat input
# ============================================================
question = st.chat_input(
    "Ask a question about your uploaded document..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.indexed:
            answer = (
                "Please upload and process a PDF document before asking "
                "questions."
            )
            sources = []
        else:
            sources = retrieve(
                question,
                st.session_state.chunks,
                top_k=top_k,
            )

            answer = generate_demo_answer(question, sources)

        st.markdown(answer)

        if sources:
            with st.expander(
                f"📚 Retrieved Sources ({len(sources)})",
                expanded=False,
            ):
                for index, source in enumerate(sources, start=1):
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>Source {index}</strong><br>
                            📄 <strong>{source["filename"]}</strong>
                            — Page {source["page"]}
                            <br><br>
                            {source["text"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        f"Retrieval score: {source['score']} "
                        f"| Matched terms: "
                        f"{', '.join(source['matched_terms']) or 'none'}"
                    )
        elif st.session_state.indexed:
            st.caption("No relevant document chunks were retrieved.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
