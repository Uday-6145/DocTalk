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
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

*, html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}

.stApp {
    background-color: #111110 !important;
    color: #E8E6E1 !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    max-width: 680px !important;
    padding: 2.5rem 1.5rem 5rem !important;
    margin: 0 auto !important;
}

/* HEADER */
.dt-logo {
    font-size: 0.95rem;
    font-weight: 600;
    color: #E8E6E1;
    letter-spacing: 0.02em;
}
.dt-prompt { color: #3D3D3A; font-weight: 300; }
.dt-cursor {
    display: inline-block;
    width: 7px; height: 0.9em;
    background: #E8A623;
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 1.1s step-end infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* DIVIDER */
.rule { border: none; border-top: 1px solid #252523; margin: 1rem 0 2rem; }

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #161614 !important;
    border: 1px dashed #2E2E2B !important;
    border-radius: 2px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover,
[data-testid="stFileUploader"]:focus-within {
    border-color: rgba(232,166,35,0.5) !important;
}
[data-testid="stFileUploader"] label { color: #3D3D3A !important; font-size: 0.75rem !important; }
[data-testid="stFileUploader"] button {
    background: transparent !important;
    border: 1px solid #2E2E2B !important;
    color: #7A7A76 !important;
    border-radius: 2px !important;
    font-size: 0.7rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    transition: all 0.15s !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: #E8A623 !important;
    color: #E8A623 !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p { color: #3D3D3A !important; font-size: 0.67rem !important; }

/* FILE PILLS */
.file-row { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:1.75rem; padding-bottom:1.5rem; border-bottom:1px solid #252523; }
.fpill { display:inline-flex; align-items:center; gap:6px; background:#161614; border:1px solid #252523; color:#7A7A76; padding:3px 10px; border-radius:2px; font-size:0.7rem; }
.fpill-dot { width:4px; height:4px; border-radius:50%; background:#E8A623; flex-shrink:0; }

/* STATUS */
.status-line { display:flex; align-items:center; gap:8px; font-size:0.62rem; color:#3D3D3A; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:1.75rem; }
.status-pip { width:6px; height:6px; border-radius:50%; background:#E8A623; animation:pop 2s ease-in-out infinite; }
@keyframes pop { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.7)} }

/* SECTION LABEL */
.sec { font-size:0.6rem; font-weight:600; text-transform:uppercase; letter-spacing:0.15em; color:#2E2E2B; margin-bottom:0.75rem; display:block; }

/* EMPTY STATE */
.empty { padding:3.5rem 0 2rem; }
.empty-h { font-size:1.6rem; font-weight:700; color:#252523; letter-spacing:-0.03em; line-height:1.25; margin-bottom:0.6rem; }
.empty-sub { font-size:0.73rem; color:#3D3D3A; line-height:1.85; font-weight:300; }
.eg-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:1.75rem; }
.eg { background:#161614; border:1px solid #1E1E1C; color:#3D3D3A; padding:3px 10px; border-radius:2px; font-size:0.67rem; letter-spacing:0.03em; }

/* CHAT MESSAGES — annotation style */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid #252523 !important;
    border-radius: 0 !important;
    padding: 0.75rem 0 0.75rem 1.25rem !important;
    margin-bottom: 0.25rem !important;
    margin-left: 0.25rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left-color: #E8A623 !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
[data-testid="stChatMessage"] p {
    font-size: 0.82rem !important;
    line-height: 1.8 !important;
    color: #E8E6E1 !important;
    margin: 0 !important;
}

/* CHAT INPUT */
[data-testid="stChatInput"] {
    background: #161614 !important;
    border: 1px solid #252523 !important;
    border-radius: 2px !important;
    margin-top: 1.5rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(232,166,35,0.4) !important;
    box-shadow: 0 0 0 1px rgba(232,166,35,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #E8E6E1 !important;
    font-size: 0.8rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3D3D3A !important; }

/* EXPANDER */
[data-testid="stExpander"] {
    background: #161614 !important;
    border: 1px solid #252523 !important;
    border-radius: 2px !important;
    margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary { font-size:0.62rem !important; color:#3D3D3A !important; text-transform:uppercase !important; letter-spacing:0.12em !important; }
[data-testid="stExpander"] p { font-size:0.72rem !important; color:#7A7A76 !important; }

/* BUTTONS */
.stButton > button {
    background: transparent !important;
    border: 1px solid #252523 !important;
    color: #3D3D3A !important;
    border-radius: 2px !important;
    font-size: 0.62rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.3rem 0.75rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { border-color:#E8A623 !important; color:#E8A623 !important; }

/* SPINNER */
[data-testid="stSpinner"] p { font-size:0.7rem !important; color:#3D3D3A !important; font-family:'IBM Plex Mono',monospace !important; letter-spacing:0.05em !important; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #111110; }
::-webkit-scrollbar-thumb { background: #252523; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────
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
        return st.session_state["retriever"], st.session_state["chain"], st.session_state["doc_names"]

    all_docs, doc_names = [], []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        pages = PyPDFLoader(tmp_path).load()
        name  = uf.name.replace(".pdf","").replace("_"," ")
        for pg in pages:
            pg.metadata["doc_name"] = name
        all_docs.extend(pages)
        doc_names.append(name)
        os.unlink(tmp_path)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n","\n",". "," ",""]
    ).split_documents(all_docs)

    vs        = FAISS.from_documents(chunks, get_embeddings())
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k":8,"fetch_k":30,"lambda_mult":0.7}
    )

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a precise document assistant.
Answer using ONLY the information in the provided context.

Rules:
1. Start with the answer. No "According to..." or "Based on...".
2. Include every exact number, date, name, or figure.
3. Bullet points for lists or step-by-step answers.
4. If not in the context: "not in your document."
5. Never make things up.

Context:
{context}"""),
        ("human", "{question}")
    ])

    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    st.session_state.update({"file_key":file_key,"retriever":retriever,"chain":chain,"doc_names":doc_names})
    return retriever, chain, doc_names

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

def ask(question, retriever, chain):
    docs    = retriever.invoke(question)
    context = format_docs(docs)
    sources = list({d.metadata.get("doc_name", os.path.basename(d.metadata.get("source","")).replace(".pdf","")) for d in docs})
    return chain.invoke({"context":context,"question":question}), sources


# ═══ UI ═══════════════════════════════════════════════════════
# Header
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("""
        <div style="padding:.5rem 0 0">
            <span class="dt-logo">
                <span class="dt-prompt">&gt;&nbsp;</span>DocTalk<span class="dt-cursor"></span>
            </span>
        </div>
    """, unsafe_allow_html=True)
with col2:
    if st.session_state.get("messages"):
        if st.button("clear"):
            for k in ["messages","file_key","retriever","chain","doc_names"]:
                st.session_state.pop(k, None)
            st.rerun()

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# Upload
st.markdown('<span class="sec">/ drop your files</span>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "upload",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# Empty state
if not uploaded_files:
    st.markdown("""
        <div class="empty">
            <div class="empty-h">drop a document.<br>ask anything.</div>
            <div class="empty-sub">
                upload one or more PDFs above.<br>
                works with books, contracts, policies, papers — any text PDF.
            </div>
            <div class="eg-row">
                <span class="eg">harry potter</span>
                <span class="eg">terms &amp; conditions</span>
                <span class="eg">offer letter</span>
                <span class="eg">research paper</span>
                <span class="eg">college syllabus</span>
                <span class="eg">cricket rulebook</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Build pipeline
with st.spinner("reading..."):
    retriever, chain, doc_names = build_pipeline(uploaded_files)

# File pills
pills = "".join(f'<span class="fpill"><span class="fpill-dot"></span>{n}</span>' for n in doc_names)
st.markdown(f'<div class="file-row" style="margin-top:1rem">{pills}</div>', unsafe_allow_html=True)

n     = len(uploaded_files)
label = f"{n} file" if n == 1 else f"{n} files"
st.markdown(f'<div class="status-line"><span class="status-pip"></span>{label} in context &mdash; ready</div>', unsafe_allow_html=True)

# Chat
st.markdown('<span class="sec">/ conversation</span>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    preview = ", ".join(doc_names[:2]) + (f" +{len(doc_names)-2}" if len(doc_names)>2 else "")
    st.session_state.messages = [{"role":"assistant","content":f"loaded {preview}. what do you want to know?","sources":[]}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("sources"):
                for s in msg["sources"]:
                    st.write(s)

if prompt := st.chat_input("ask a question about your document..."):
    st.session_state.messages.append({"role":"user","content":prompt,"sources":[]})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("searching..."):
            answer, sources = ask(prompt, retriever, chain)
        st.markdown(answer)
        if sources:
            with st.expander("sources"):
                for s in sources: st.write(s)
    st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
