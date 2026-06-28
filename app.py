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
    page_title="DocTalk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & base ── */
*, html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    box-sizing: border-box;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Hide sidebar entirely (fixes the collapse bug) ── */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Layout ── */
.block-container {
    max-width: 860px !important;
    padding: 0 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Mono label style (sci-bot inspired) ── */
.mono {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    color: #71717A;
    text-transform: lowercase;
}

/* ── Top nav bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1.2rem;
    border-bottom: 1px solid rgba(0,0,0,0.07);
    margin-bottom: 1.5rem;
}
.top-bar-left { display: flex; align-items: center; gap: 14px; }
.wordmark {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem;
    font-weight: 600;
    color: #0E7490;
    letter-spacing: -0.02em;
}
.doc-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(14,116,144,0.08);
    border: 1px solid rgba(14,116,144,0.2);
    color: #0E7490;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
}
.top-bar-right button {
    background: none;
    border: 1px solid rgba(0,0,0,0.12);
    color: #71717A;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.76rem;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Upload state ── */
.upload-page {
    min-height: 80vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
}
.upload-hero-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
    color: #0E7490;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.upload-hero-title {
    font-size: 2.4rem;
    font-weight: 600;
    color: #18181B;
    letter-spacing: -0.04em;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.upload-hero-sub {
    font-size: 0.9rem;
    color: #71717A;
    line-height: 1.6;
    margin-bottom: 2rem;
    max-width: 400px;
}

/* ── Scattered background symbols ── */
.bg-symbols {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(14,116,144,0.06);
    font-size: 1.4rem;
    user-select: none;
    overflow: hidden;
}
.bg-symbols span {
    position: absolute;
    font-weight: 500;
}

/* ── Upload zone ── */
.upload-zone-wrap { position: relative; z-index: 1; }
[data-testid="stFileUploader"] {
    background: rgba(14,116,144,0.03) !important;
    border: 1.5px dashed rgba(14,116,144,0.35) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(14,116,144,0.06) !important;
    border-color: rgba(14,116,144,0.6) !important;
}
[data-testid="stFileUploader"] button {
    background: #0E7490 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p {
    font-size: 0.8rem !important;
    color: #71717A !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Example chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 2rem; }
.chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 5px 12px;
    border-radius: 4px;
    border: 1px solid rgba(0,0,0,0.1);
    color: #52525B;
    background: rgba(0,0,0,0.02);
}
.chip-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #A1A1AA;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 6px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.5rem !important;
    background: transparent !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 2px solid #0E7490 !important;
    background: rgba(14,116,144,0.03) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 2px solid rgba(0,0,0,0.08) !important;
}
[data-testid="stChatMessage"] p {
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    color: #27272A !important;
}

/* ── Input ── */
[data-testid="stChatInput"] {
    border-radius: 6px !important;
    border-color: rgba(0,0,0,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 0.88rem !important;
    color: #18181B !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #A1A1AA !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Sources expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 5px !important;
    margin-top: 0.4rem !important;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #A1A1AA !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #71717A !important;
}

/* ── Streamlit button (clear chat) ── */
[data-testid="stBaseButton-secondary"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 4px !important;
    border-color: rgba(0,0,0,0.1) !important;
    color: #71717A !important;
}

/* ── Dark mode ── */
@media (prefers-color-scheme: dark) {
    .upload-hero-title { color: #F4F4F5; }
    .chip { border-color: rgba(255,255,255,0.1); color: #A1A1AA; background: rgba(255,255,255,0.03); }
    [data-testid="stChatMessage"] { border-color: rgba(255,255,255,0.06) !important; }
    [data-testid="stChatMessage"] p { color: #E4E4E7 !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(14,116,144,0.08) !important;
    }
    .upload-hero-title { color: #FAFAFA; }
    .wordmark { color: #22D3EE; }
    .bg-symbols { color: rgba(34,211,238,0.04); }
    .doc-badge { background: rgba(34,211,238,0.1); border-color: rgba(34,211,238,0.2); color: #22D3EE; }
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
        ("system", """you are a precise document assistant.
answer using only information found in the context below.

rules:
1. answer directly — skip phrases like "according to the document" or "based on the context".
2. include every exact number, date, name, or figure from the context.
3. use bullet points for lists, steps, or conditions.
4. if the answer isn't in the documents, say exactly: "not covered in your document."
5. never guess or add outside information.

context:
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


# ══════════════════════════════════════════════════════════
# STATE A — No document uploaded
# ══════════════════════════════════════════════════════════
if not st.session_state.get("file_key"):

    # Minimal top wordmark
    st.markdown("""
        <div style="padding:1.4rem 0 0">
            <span style="font-family:'JetBrains Mono',monospace;
                         font-size:.9rem;font-weight:600;color:#0E7490;">
                DocTalk
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Scattered background symbols
    st.markdown("""
    <div class="bg-symbols" aria-hidden="true">
        <span style="top:8%;right:12%">{ }</span>
        <span style="top:15%;right:28%">∑</span>
        <span style="top:22%;right:6%">[ ]</span>
        <span style="top:35%;right:18%">λ</span>
        <span style="top:50%;right:8%">∂</span>
        <span style="top:65%;right:22%">//</span>
        <span style="top:75%;right:10%">→</span>
        <span style="top:85%;right:30%">π</span>
        <span style="top:12%;right:40%">≡</span>
        <span style="top:60%;right:38%">∞</span>
        <span style="top:42%;right:42%">&&</span>
        <span style="top:28%;right:48%">Σ</span>
        <span style="top:70%;right:45%">||</span>
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout: left = hero + upload, right = empty (for breathing room)
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("""
            <div style="padding:3.5rem 0 2rem">
                <div class="upload-hero-label">document intelligence</div>
                <div class="upload-hero-title">ask anything.<br>about any doc.</div>
                <div class="upload-hero-sub">
                    upload a pdf — research paper, policy, contract, book —
                    and ask questions in plain english.
                </div>
            </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "drop your pdf here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        st.markdown("""
            <div style="margin-top:2.5rem">
                <div class="chip-label">works great with</div>
                <div class="chip-row">
                    <span class="chip">harry potter</span>
                    <span class="chip">terms &amp; conditions</span>
                    <span class="chip">offer letters</span>
                    <span class="chip">research papers</span>
                    <span class="chip">cricket rulebook</span>
                    <span class="chip">college syllabus</span>
                    <span class="chip">legal contracts</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    if uploaded_files:
        with st.spinner("reading your document..."):
            build_pipeline(uploaded_files)
        st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════
# STATE B — Document loaded, show chat
# ══════════════════════════════════════════════════════════
retriever = st.session_state["retriever"]
chain     = st.session_state["chain"]
doc_names = st.session_state["doc_names"]

# Top bar
badges = "".join(
    f'<span class="doc-badge">◈ {n}</span>' for n in doc_names
)
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown(f"""
        <div class="top-bar">
            <div class="top-bar-left">
                <span class="wordmark">DocTalk</span>
                <span style="color:rgba(0,0,0,.15);font-size:.9rem">/</span>
                {badges}
            </div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("<div style='padding-top:1.3rem'>", unsafe_allow_html=True)
    if st.button("↩ new document", use_container_width=True):
        for k in ["messages","file_key","retriever","chain","doc_names"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Welcome message
if "messages" not in st.session_state:
    names = " + ".join(doc_names[:2])
    extra = f" + {len(doc_names)-2} more" if len(doc_names) > 2 else ""
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"**{names}{extra}** loaded. what do you want to know?",
        "sources": []
    }]

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("sources"):
                for s in msg["sources"]:
                    st.markdown(f"`{s}`")

# Input
if prompt := st.chat_input("ask anything about your document..."):
    st.session_state.messages.append({"role":"user","content":prompt,"sources":[]})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("searching..."):
            answer, sources = ask(prompt, retriever, chain)
        st.markdown(answer)
        if sources:
            with st.expander("sources"):
                for s in sources:
                    st.markdown(f"`{s}`")

    st.session_state.messages.append({
        "role":"assistant","content":answer,"sources":sources
    })
