import io, re, traceback, streamlit as st
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv # <--- ADD THIS LINE

# Load environment variables from a .env file (for local development)
load_dotenv()

OCR_AVAILABLE = False
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

from groq import Groq

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Smart PDF Q&A (Groq)", page_icon="📚", layout="wide")
st.title("📚 Smart PDF Q&A — Groq")

# -------------------------
# Helpers
# -------------------------
def looks_like_groq_key(api_key: str) -> bool:
    return bool(re.match(r"^gsk_[A-Za-z0-9]{16,}$", api_key or ""))

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text: str, chunk_size=700, chunk_overlap=150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", "\n\n", ". ", "! ", "? ", " ", ""]
    )
    return [c.strip() for c in splitter.split_text(text) if len(c.strip()) > 40]

def semantic_retrieve(chunks: List[str], query: str, k: int = 5):
    embedder = load_embedder()
    chunk_emb = embedder.encode(chunks, show_progress_bar=False)
    q_emb = embedder.encode([query], show_progress_bar=False)
    sims = cosine_similarity(q_emb, chunk_emb)[0]
    top_indices = np.argsort(sims)[::-1][:k]
    return [{"index": int(i), "similarity": float(sims[i]), "text": chunks[i]} for i in top_indices]

def substring_snippets(full_text: str, token: str, radius: int = 350, max_matches: int = 3):
    token = token.strip()
    if not token: return []
    out = []
    for m in re.finditer(re.escape(token), full_text, re.IGNORECASE):
        s = max(0, m.start() - radius)
        e = min(len(full_text), m.end() + radius)
        out.append(full_text[s:e].strip())
        if len(out) >= max_matches: break
    return out

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    except Exception:
        pass
    if len(text.strip()) < 100 and OCR_AVAILABLE:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
            ocr_texts = [pytesseract.image_to_string(img) for img in images]
            text_ocr = "\n".join(ocr_texts)
            if len(text_ocr.strip()) > len(text.strip()):
                return text_ocr
        except Exception:
            pass
    return text

def make_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

def call_groq_chat(client: Groq, model: str, context: str, question: str, temperature=0.2, max_tokens=800):
    system_msg = (
        "Answer ONLY using the provided document context. "
        "If not in the context, say you cannot find it."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "system", "content": f"Document context:\n{context}"},
        {"role": "user", "content": question},
    ]
    comp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens)
    )
    return comp.choices[0].message.content.strip()

# -------------------------
# Sidebar: API key and options
# -------------------------
with st.sidebar:
    st.header("Step 1 — Enter your Groq API key")
    api_key = st.text_input("GROQ API key", type="password", placeholder="gsk_...")
    show_debug = st.checkbox("Show debug info", value=False)
    chunk_size = st.slider("Chunk size", 400, 2000, 700, 50)
    chunk_overlap = st.slider("Chunk overlap", 50, 500, 150, 25)
    top_k = st.slider("Top‑K chunks", 1, 10, 5, 1)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

key_ok = False
client = None
chosen_model = "llama-3.1-8b-instant"

if api_key and looks_like_groq_key(api_key):
    try:
        client = make_groq_client(api_key)
        key_ok = True
    except Exception as e:
        st.sidebar.error(f"API key validation failed: {e}")
else:
    if api_key:
        st.sidebar.error("Invalid Groq API key format")

# -------------------------
# Step 2 — Upload PDF
# -------------------------
uploaded = st.file_uploader("Step 2 — Upload a PDF", type=["pdf"]) if key_ok else None

if uploaded:
    pdf_bytes = uploaded.read()
    with st.spinner("Extracting text..."):
        doc_text = extract_text_from_pdf_bytes(pdf_bytes)

    if not doc_text.strip():
        st.error("Could not extract text. For scanned PDFs, install Tesseract + Poppler.")
        st.stop()

    if show_debug: st.text(doc_text[:1500])
    chunks = chunk_text(doc_text, chunk_size, chunk_overlap)
    st.session_state.update({"doc_text": doc_text, "doc_chunks": chunks, "model": chosen_model})
    st.success(f"Document loaded — {len(chunks)} chunks created.")

# -------------------------
# Chat interface
# -------------------------
if key_ok and st.session_state.get("doc_chunks"):
    if "chat_history" not in st.session_state: st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    user_q = st.chat_input("Ask anything about the PDF...")
    if user_q:
        st.session_state["chat_history"].append({"role": "user", "content": user_q})
        with st.chat_message("user"): st.markdown(user_q)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving context and calling Groq..."):
                    chunks = st.session_state["doc_chunks"]
                    sem = semantic_retrieve(chunks, user_q, k=top_k)
                    top_sim = sem[0]["similarity"] if sem else 0.0

                    if top_sim >= 0.25:
                        context = "\n".join([r["text"] for r in sem[:3]])
                        debug_note = f"Semantic context used (top sim={top_sim:.3f})"
                    else:
                        tokens = re.findall(r"[A-Za-z][A-Za-z0-9'_\\-]{1,}", user_q)
                        guess = tokens[-1] if tokens else None
                        used_sub = False
                        context = ""
                        if guess:
                            subs = substring_snippets(st.session_state["doc_text"], guess, radius=350, max_matches=3)
                            if subs:
                                context = "\n".join(subs)
                                used_sub = True
                        if not context:
                            context = "\n".join([r["text"] for r in sem[:3]])
                        debug_note = f"{'Substring' if used_sub else 'Semantic (low sim)'} context used (top sim={top_sim:.3f})"

                    if len(context) > 6000: context = context[:6000] + "\n[...]"
                    answer = call_groq_chat(client, st.session_state["model"], context, user_q, temperature=temperature, max_tokens=800)
                    if show_debug: answer = f"{answer}\n---\n{debug_note}"

                st.markdown(answer)
                st.session_state["chat_history"].append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Groq API error: {e}")
                if show_debug: st.text(traceback.format_exc())

# -------------------------
# Utilities
# -------------------------
with st.sidebar:
    st.divider()
    if st.button("Clear chat & state"):
        for k in ("doc_text", "doc_chunks", "chat_history", "model"):
            st.session_state.pop(k, None)
        st.rerun()

