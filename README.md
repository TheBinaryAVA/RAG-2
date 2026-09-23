# Dummy RAG Streamlit Demo

A local demonstration of the AI Knowledge Assistant interface.

## Important

This is a **demo/mock RAG implementation** intended for UI/video demonstration.

It does:
- PDF extraction
- Page metadata preservation
- Text chunking
- Lightweight keyword-overlap retrieval
- Conversation history
- Source snippets and page numbers
- Grounding-style fallback

It does **not** use:
- Gemini
- Gemini embeddings
- FAISS
- LangChain
- An LLM

Do not present this demo as the actual Gemini/FAISS RAG implementation.

## Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```
