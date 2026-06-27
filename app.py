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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #09090B !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #71717A !important; }
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FAFAFA !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(124,58,237,0.4) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: rgba(124,58,237,0.8) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #71717A !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #FAFAFA !important;
}

/* ── Main ── */
.block-container {
    padding-top: 2rem !important;
    max-width: 820px !important;
}

/* ── Gradient wordmark ── */
.wordmark {
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.04em;
    display: inline-block;
}
.tagline {
    font-size: 0.8rem;
    color: #52525B;
    letter-spacing: 0.01em;
}
.app-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding-bottom: 1.25rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(148,163,184,0.1);
}
.header-divider {
    width: 1px;
    height: 16px;
    background: rgba(148,163,184,0.2);
    display: inline-block;
    vertical-align: middle;
}

/* ── Status ── */
.status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.25);
    color: #A78BFA;
    margin-bottom: 1.25rem;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #8B5CF6;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

/* ── Empty state ── */
.empty-wrap {
    margin-top: 4rem;
    text-align: center;
}
.empty-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #3F3F46;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.empty-sub {
    font-size: 0.85rem;
    color: #52525B;
    line-height: 1.7;
    max-width: 380px;
    margin: 0 auto;
}
.empty-examples {
    margin-top: 2rem;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
}
.example-chip {
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.2);
    color: #A78BFA;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,184,0.1) !important;
    padding: 1rem !important;
    margin-bottom: 0.5rem !important;
    background: transparent !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid #7C3AED !important;
    background: rgba(124,58,237,0.04) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid rgba(6,182,212,0.4) !important;
}

/* ── Input ── */
[data-testid="stChatInput"] {
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 0.88rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(148,163,184,0.1) !important;
    border-radius: 8px !important;
    margin-top: 0.4rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #52525B !important;
}

/* ── Doc pill ── */
.doc-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.25);
    color: #A78BFA !important;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.74rem;
    font-weight: 500;
    margin: 3px 0;
    width: 100%;
}
.dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #8B5CF6;
    flex-shrink: 0;
}

/* ── Sidebar label ── */
.s-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3F3F46;
    display: block;
    margin-bottom: 0.6rem;
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
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()
        name   = uf.name.replace(".pdf", "").replace("_", " ")
        for pg in pages:
            pg.metadata["doc_name"] = name
        all_docs.extend(pages)
        doc_names.append(name)
        os.unlink(tmp_path)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    ).split_documents(all_docs)

    vs = FAISS.from_documents(chunks, get_embeddings())
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.7}
    )

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a sharp, precise document assistant.
Answer using ONLY the information in the context below.

Rules:
1. Get straight to the answer — no "Based on the document" or "According to...". Just facts.
2. Include every exact number, date, name, or figure from the context.
3. Bullet points for lists or multi-step things.
4. If the answer genuinely isn't in the documents, say exactly:
   "not in your docs — try asking something else."
5. Never make things up.

Context:
{context}"""),
        ("human", "{question}")
    ])

    chain = RAG_PROMPT | get_llm() | StrOutputParser()

    st.session_state.update({
        "file_key": file_key,
        "retriever": retriever,
        "chain": chain,
        "doc_names": doc_names
    })
    return retriever, chain, doc_names


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

def ask(question, retriever, chain):
    docs    = retriever.invoke(question)
    context = format_docs(docs)
    sources = list({
        d.metadata.get("doc_name",
            os.path.basename(d.metadata.get("source", "")).replace(".pdf",""))
        for d in docs
    })
    return chain.invoke({"context": context, "question": question}), sources


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding:.25rem 0 1.5rem">
            <div style="font-size:1.1rem;font-weight:700;
                        background:linear-gradient(135deg,#7C3AED,#06B6D4);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;letter-spacing:-.03em">
                DocTalk ⚡
            </div>
            <div style="font-size:.7rem;color:#3F3F46;margin-top:3px;
                        text-transform:uppercase;letter-spacing:.1em">
                drop a doc. get answers.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="s-label">your files</span>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "drop pdfs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown('<span class="s-label" style="margin-top:.8rem">in context</span>',
                    unsafe_allow_html=True)
        for f in uploaded_files:
            name = f.name.replace(".pdf","").replace("_"," ")
            st.markdown(
                f'<div class="doc-pill"><span class="dot"></span>{name}</div>',
                unsafe_allow_html=True
            )

    st.markdown("<hr style='margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size:.76rem;color:#3F3F46;line-height:1.65">
            answers come only from what you upload.<br>
            nothing is stored after you close this tab.
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("messages"):
        st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
        if st.button("clear chat", use_container_width=True):
            for k in ["messages","file_key","retriever","chain","doc_names"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
    <div class="app-header">
        <span class="wordmark">DocTalk</span>
        <span class="header-divider"></span>
        <span class="tagline">upload any pdf — ask anything</span>
    </div>
""", unsafe_allow_html=True)


# ── No files state ───────────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
        <div class="empty-wrap">
            <div class="empty-title">nothing here yet</div>
            <div class="empty-sub">
                drop a PDF in the sidebar and start asking questions.<br>
                works on anything — books, contracts, policies, papers.
            </div>
            <div class="empty-examples">
                <span class="example-chip">harry potter</span>
                <span class="example-chip">terms & conditions</span>
                <span class="example-chip">job offer letter</span>
                <span class="example-chip">research paper</span>
                <span class="example-chip">cricket rulebook</span>
                <span class="example-chip">syllabus</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Build pipeline ───────────────────────────────────────────────────
with st.spinner("reading your doc..."):
    retriever, chain, doc_names = build_pipeline(uploaded_files)

n = len(uploaded_files)
st.markdown(f"""
    <div class="status">
        <span class="status-dot"></span>
        {n} doc{"s" if n > 1 else ""} loaded — ask away
    </div>
""", unsafe_allow_html=True)


# ── Chat ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    names = ", ".join(doc_names[:2])
    if len(doc_names) > 2:
        names += f" +{len(doc_names)-2} more"
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"got **{names}**. what do you want to know?",
        "sources": []
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("sources"):
                for s in msg["sources"]:
                    st.write(s)

if prompt := st.chat_input("what do you want to know?"):
    st.session_state.messages.append({"role":"user","content":prompt,"sources":[]})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("digging through your docs..."):
            answer, sources = ask(prompt, retriever, chain)
        st.markdown(answer)
        if sources:
            with st.expander("sources"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
