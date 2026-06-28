import streamlit as st
import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="DocTalk AI",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 🎨 MODERN SAAS UI THEME ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base (Safe font mapping to prevent icon breaking) ── */
html, body, p, div, h1, h2, h3, h4, h5, h6, label, textarea, input, button {
    font-family: 'Inter', -apple-system, sans-serif !important;
}
* {
    box-sizing: border-box;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sources expander icon rendering patch ── */
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #71717A !important;
}

[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

/* Centered constraints */
.block-container {
    max-width: 1050px !important; /* Increased from 800px */
    padding: 2rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* Push the button in the 3rd column to the absolute right edge */
[data-testid="column"]:nth-of-type(3) {
    display: flex;
    justify-content: flex-end;
}

/* Top Bar */
.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
    margin-bottom: 2rem;
}
.wordmark {
    font-size: 1.25rem;
    font-weight: 700;
    background: linear-gradient(90deg, #111827, #374151);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.doc-badge {
    background: #F3F4F6;
    color: #4B5563;
    padding: 5px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid #E5E7EB;
    width: 140px;
    height: 32px;
    line-height: 20px;
    display: inline-block;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Hero Section (Upload State) */
.hero-container {
    text-align: center;
    padding: 4rem 0 3rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #111827;
    margin-bottom: 1rem;
    line-height: 1.2;
}
.hero-title span {
    background: linear-gradient(135deg, #0EA5E9, #2563EB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #6B7280;
    max-width: 500px;
    margin: 0 auto 3rem;
    line-height: 1.5;
}

/* Uploader Styling */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #E5E7EB !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
[data-testid="stFileUploader"]:hover {
    border-color: #3B82F6 !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* 🚨 FIX FOR THE GLITCHED OVERLAPPING BUTTON TEXT 🚨 */
[data-testid="stFileUploader"] button {
    background: #111827 !important;
    color: transparent !important; /* Hides the native glitched text */
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    position: relative !important;
    border: none !important;
    min-width: 120px !important;
}

/* Project clean custom text exactly in the center */
[data-testid="stFileUploader"] button::after {
    content: "Select PDF";
    color: white !important;
    position: absolute !important;
    left: 50% !important;
    top: 50% !important;
    transform: translate(-50%, -50%) !important;
    pointer-events: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    text-align: center !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
    border-bottom: 1px solid rgba(0,0,0,0.05) !important;
}
[data-testid="stChatMessage"] p {
    color: #374151 !important;
    line-height: 1.6 !important;
    font-size: 0.95rem !important;
}

/* Input Area */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 0.95rem !important;
}

/* Dark Mode Overrides */
@media (prefers-color-scheme: dark) {
    .hero-title { color: #F9FAFB; }
    .wordmark { background: linear-gradient(90deg, #F9FAFB, #D1D5DB); -webkit-background-clip: text; }
    .doc-badge { background: #1F2937; color: #D1D5DB; border-color: #374151; }
    [data-testid="stFileUploader"] { background: #111827 !important; border-color: #374151 !important; }
    [data-testid="stFileUploader"] button { background: #3B82F6 !important; }
    [data-testid="stChatMessage"] p { color: #E5E7EB !important; }
    [data-testid="stChatInput"] { border-color: #374151 !important; background: #1F2937 !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        groq_api_key=get_api_key()
    )

def build_pipeline(uploaded_files):
    file_key = "_".join(f"{f.name}_{f.size}" for f in uploaded_files)
    if st.session_state.get("file_key") == file_key:
        return (
            st.session_state["retriever"],
            st.session_state["chain"],
            st.session_state["doc_names"]
        )

    all_docs, doc_names = [], []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        pages = PyPDFLoader(tmp_path).load()
        name  = uf.name.replace(".pdf", "").replace("_", " ")
        for pg in pages:
            pg.metadata["doc_name"] = name
        all_docs.extend(pages)
        doc_names.append(name)
        os.unlink(tmp_path)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    ).split_documents(all_docs)

    vs        = FAISS.from_documents(chunks, get_embeddings())
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.7}
    )

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an elite document intelligence assistant.
Answer using ONLY information found in the context below.

RULES:
1. Direct answers only. Skip "According to the document".
2. Include exact numbers, dates, and figures.
3. Use clean formatting (bullet points for lists).
4. If the answer isn't in the documents, state: "I couldn't find that in the uploaded documents."
5. Never hallucinate outside knowledge.

Context:
{context}"""),
        ("human", "{question}")
    ])

    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    st.session_state.update({
        "file_key":  file_key,
        "retriever": retriever,
        "chain":     chain,
        "doc_names": doc_names
    })
    return retriever, chain, doc_names

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

def ask(q, retriever, chain):
    docs    = retriever.invoke(q)
    context = format_docs(docs)
    sources = list({
        d.metadata.get("doc_name",
            Path(d.metadata.get("source", "")).stem.replace("_", " "))
        for d in docs
    })
    return chain.invoke({"context": context, "question": q}), sources


# ── Custom Avatars ──
USER_AVATAR = "👤"
BOT_AVATAR = "✨"

# ══════════════════════════════════════════════════════════
# STATE A — No document uploaded
# ══════════════════════════════════════════════════════════
if not st.session_state.get("file_key"):
    
    st.markdown("""
        <div class="top-bar" style="border: none;">
            <div class="wordmark">DocTalk</div>
        </div>
        <div class="hero-container">
            <div class="hero-title">Talk to your <span>documents.</span></div>
            <div class="hero-subtitle">Upload PDFs, research papers, or contracts and extract answers instantly using Llama-3.</div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        with st.spinner("Processing document embeddings..."):
            build_pipeline(uploaded_files)
        st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════
# STATE B — Document loaded, show chat
# ══════════════════════════════════════════════════════════
retriever = st.session_state["retriever"]
chain     = st.session_state["chain"]
doc_names = st.session_state["doc_names"]

# 3-Column Layout: Left (Logo), Center (Badges), Right (Button)
col1, col2, col3 = st.columns([1.2, 4, 1.2])

with col1:
    st.markdown("""
        <div style="height: 100%; display: flex; align-items: center; padding-top: 0.5rem;">
            <span class="wordmark" style="font-size: 1.4rem;">DocTalk</span>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Added title="..." so users can hover to see the full name if it gets cut off
    badges = "".join(f'<div class="doc-badge" title="{n}">📄 {n}</div>' for n in doc_names)
    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; padding-top: 0.3rem;">
            {badges}
        </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button("Start New Session", use_container_width=True):
        for k in ["messages","file_key","retriever","chain","doc_names"]:
            st.session_state.pop(k, None)
        st.rerun()

st.markdown("<hr style='margin: 0.5rem 0 2rem; border-color: rgba(0,0,0,0.08);'>", unsafe_allow_html=True)

# Welcome message
if "messages" not in st.session_state:
    names = " & ".join(doc_names[:2])
    extra = f" + {len(doc_names)-2} more" if len(doc_names) > 2 else ""
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"**{names}{extra}** successfully indexed. What would you like to know?",
        "sources": []
    }]

# Chat history rendering
for msg in st.session_state.messages:
    avatar = BOT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

# Input Handling
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role":"user", "content":prompt, "sources":[]})
    
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Analyzing text..."):
            answer, sources = ask(prompt, retriever, chain)
        st.markdown(answer)
        if sources:
            with st.expander("View Sources"):
                for s in sources:
                    st.markdown(f"- `{s}`")

    st.session_state.messages.append({
        "role":"assistant", "content":answer, "sources":sources
    })
