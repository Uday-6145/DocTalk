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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocTalk — Ask Your Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.01em;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] div { color: #94A3B8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #F1F5F9 !important; }
[data-testid="stSidebar"] hr { border-color: #1E293B !important; }

/* Upload area in sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed #334155 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #2563EB !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
}

/* Doc tag pills */
.doc-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.3);
    color: #93C5FD !important;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    margin: 3px 0;
}
.doc-tag-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #3B82F6;
}

/* ── Main content ── */
.block-container {
    padding-top: 2rem !important;
    max-width: 860px !important;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding-bottom: 1.25rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(148,163,184,0.15);
}
.app-wordmark {
    font-size: 1.15rem;
    font-weight: 700;
    color: #2563EB;
    letter-spacing: -0.03em;
}
.app-tagline {
    font-size: 0.82rem;
    color: #64748B;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #64748B;
}
.empty-state-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94A3B8;
    margin-bottom: 0.5rem;
}
.empty-state-sub {
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.6;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    background: rgba(34,197,94,0.1);
    color: #86EFAC;
    border: 1px solid rgba(34,197,94,0.2);
    margin-bottom: 1rem;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22C55E;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    border: 1px solid rgba(148,163,184,0.15) !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid #2563EB !important;
    background: rgba(37,99,235,0.04) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid rgba(148,163,184,0.25) !important;
}

/* ── Source expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(148,163,184,0.15) !important;
    border-radius: 6px !important;
    margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Sidebar label ── */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    display: block;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


# ── Cache heavy objects globally (model downloads once) ────────────────────────
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


# ── Build per-session pipeline when files change ───────────────────────────────
def build_pipeline_from_files(uploaded_files):
    """
    Processes uploaded PDFs and returns a retriever + answer chain.
    Caches in session_state so it doesn't reprocess on every rerun.
    """
    # Use file names + sizes as a cache key
    file_key = "_".join(f"{f.name}_{f.size}" for f in uploaded_files)

    if st.session_state.get("file_key") == file_key:
        return (
            st.session_state["retriever"],
            st.session_state["chain"],
            st.session_state["doc_names"]
        )

    # Load PDFs via temp files
    all_docs = []
    doc_names = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()
        for pg in pages:
            pg.metadata["doc_name"] = uf.name.replace(".pdf", "").replace("_", " ")
        all_docs.extend(pages)
        doc_names.append(uf.name.replace(".pdf", "").replace("_", " "))
        os.unlink(tmp_path)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)

    # Embed and index
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.7}
    )

    # Generic RAG prompt — works for any document type
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a precise document assistant.
Answer the user's question using ONLY the information in the provided document context below.

RULES:
1. Start directly with the answer — no preamble like "According to the document" or "Based on the context".
2. Include every specific number, date, percentage, or figure mentioned in the context.
3. Use bullet points for multi-part answers, lists, or step-by-step processes.
4. If the answer is not in the context, respond with exactly:
   "This information is not covered in the uploaded documents."
5. Never guess or add information from outside the provided context.

Context:
{context}"""),
        ("human", "{question}")
    ])

    llm   = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()

    # Store in session
    st.session_state["file_key"]  = file_key
    st.session_state["retriever"] = retriever
    st.session_state["chain"]     = chain
    st.session_state["doc_names"] = doc_names

    return retriever, chain, doc_names


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def ask(question, retriever, chain):
    docs    = retriever.invoke(question)
    context = format_docs(docs)
    sources = list({
        d.metadata.get("doc_name", os.path.basename(d.metadata.get("source", "")))
        for d in docs
    })
    answer = chain.invoke({"context": context, "question": question})
    return answer, sources


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding: 0.25rem 0 1.25rem;">
            <div style="font-size:1rem; font-weight:700; color:#F1F5F9;">DocTalk</div>
            <div style="font-size:0.72rem; color:#475569; margin-top:2px;
                        text-transform:uppercase; letter-spacing:0.08em;">
                Document Q&A
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-label">Upload Documents</span>',
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more PDF files to start asking questions."
    )

    if uploaded_files:
        st.markdown('<span class="sidebar-label" style="margin-top:1rem;">Loaded</span>',
                    unsafe_allow_html=True)
        for f in uploaded_files:
            name = f.name.replace(".pdf", "").replace("_", " ")
            st.markdown(
                f'<div class="doc-tag"><span class="doc-tag-dot"></span>{name}</div>',
                unsafe_allow_html=True
            )

    st.markdown("<hr style='margin:1.25rem 0'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size:0.77rem; color:#475569; line-height:1.6;">
            Answers are generated exclusively from your uploaded documents.
            No data is stored after your session ends.
        </div>
    """, unsafe_allow_html=True)

    # Clear session button
    if st.session_state.get("messages"):
        st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
        if st.button("Clear conversation", use_container_width=True):
            for key in ["messages", "file_key", "retriever", "chain", "doc_names"]:
                st.session_state.pop(key, None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="app-header">
        <span class="app-wordmark">DocTalk</span>
        <span class="app-tagline">Upload any PDF &mdash; ask anything about it</span>
    </div>
""", unsafe_allow_html=True)


# ── States: no upload vs ready ─────────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-state-title">No documents uploaded yet</div>
            <div class="empty-state-sub">
                Upload one or more PDF files from the sidebar to get started.<br>
                Works with HR policies, research papers, legal contracts,<br>
                product manuals — any text-based PDF.
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Build pipeline ─────────────────────────────────────────────────────────────
with st.spinner("Processing documents..."):
    retriever, chain, doc_names = build_pipeline_from_files(uploaded_files)

n   = len(uploaded_files)
label = f"{n} document loaded" if n == 1 else f"{n} documents loaded"
st.markdown(f"""
    <div class="status-badge">
        <span class="status-dot"></span>
        {label} &mdash; ready
    </div>
""", unsafe_allow_html=True)


# ── Chat ───────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    doc_list = ", ".join(doc_names[:3])
    if len(doc_names) > 3:
        doc_list += f" and {len(doc_names) - 3} more"
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            f"I have processed **{doc_list}**. "
            "Ask me anything about the content — I will answer only from what is in the documents."
        ),
        "sources": []
    }]

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(s)

# Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = ask(prompt, retriever, chain)
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
