import streamlit as st
import os
import tempfile

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

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}

/* ── PAGE ── */
.stApp { background-color: #0C0C0B !important; }

/* kill chrome */
#MainMenu, footer, header            { visibility: hidden !important; }
[data-testid="stDecoration"],
[data-testid="stToolbar"]            { display: none !important; }

/* ── LOCK SIDEBAR — no collapse button at all ── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stBaseButton-headerNoPadding"] { display: none !important; }

[data-testid="stSidebar"] {
    background-color: #0F0F0E !important;
    border-right: 1px solid #1A1A18 !important;
    min-width: 260px !important;
    max-width: 260px !important;
    transform: none !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding: 2rem 1.4rem !important;
    position: relative;
    min-height: 100vh;
}

/* Sidebar text overrides */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #2A2A28 !important; }

/* ── SIDEBAR WORDMARK ── */
.wm          { line-height:0.85; margin-bottom:0.4rem; }
.wm-top      { font-size:2.6rem; font-weight:700; color:#E8A623; letter-spacing:-0.05em; display:block; }
.wm-bot      { font-size:2.6rem; font-weight:700; color:#222220; letter-spacing:-0.05em; display:block; }
.wm-sub      { font-size:0.58rem; color:#2A2A28; text-transform:uppercase; letter-spacing:0.16em; display:block; margin-top:0.7rem; }

/* Background deco in sidebar */
.sb-deco {
    position:absolute; bottom:5rem; right:1rem;
    font-size:4.5rem; font-weight:700; line-height:0.88;
    color:#161614; pointer-events:none; user-select:none;
    text-align:right;
}

/* ── SIDEBAR RULE ── */
.sb-rule { border:none; border-top:1px solid #1A1A18; margin:1.2rem 0; }

/* ── SIDEBAR LABEL ── */
.sb-lbl { font-size:0.58rem !important; font-weight:600 !important; text-transform:uppercase !important;
           letter-spacing:0.16em !important; color:#222220 !important; display:block; margin-bottom:0.5rem; }

/* ── FILE LIST ── */
.f-item { font-size:0.7rem !important; color:#4A4A46 !important; padding:0.3rem 0;
          border-bottom:1px solid #161614; display:flex; align-items:center; gap:7px; }
.f-dot  { width:4px; height:4px; border-radius:50%; background:#E8A623; flex-shrink:0; }
.f-none { font-size:0.68rem !important; color:#1E1E1C !important; }

/* ── SIDEBAR UPLOAD ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background:#131312 !important;
    border:1px dashed #222220 !important;
    border-radius:2px !important;
    padding:0.25rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover,
[data-testid="stSidebar"] [data-testid="stFileUploader"]:focus-within {
    border-color:rgba(232,166,35,0.45) !important;
}
/* The "Browse files" button */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background:transparent !important; border:1px solid #1E1E1C !important;
    color:#3A3A38 !important; border-radius:2px !important;
    font-size:0.67rem !important; font-family:'IBM Plex Mono',monospace !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    border-color:#E8A623 !important; color:#E8A623 !important;
}
/* All small text inside uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span:not(button span),
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {
    color:#1E1E1C !important; font-size:0.62rem !important;
}

/* ── SIDEBAR FOOTER ── */
.sb-foot { font-size:0.58rem !important; color:#1A1A18 !important; letter-spacing:0.06em; }

/* ── MAIN CONTENT ── */
.block-container {
    max-width:760px !important;
    padding:2.5rem 2rem 6rem !important;
    margin:0 auto !important;
}

/* ── MAIN HEADER ── */
.mh {
    display:flex; justify-content:space-between; align-items:center;
    padding-bottom:1.4rem; margin-bottom:2rem;
    border-bottom:1px solid #1A1A18;
}
.mh-tag     { font-size:0.6rem; color:#2A2A28; text-transform:uppercase; letter-spacing:0.15em; }
.mh-status  { display:flex; align-items:center; gap:7px; font-size:0.6rem; color:#2E2E2B;
               text-transform:uppercase; letter-spacing:0.1em; }
.mh-dot     { width:5px; height:5px; border-radius:50%; background:#E8A623; animation:pip 2s ease-in-out infinite; }
@keyframes pip { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── EMPTY STATE ── */
.empty      { padding:4rem 0 3rem; }
.empty-h    { font-size:2rem; font-weight:700; color:#202020; letter-spacing:-0.04em;
               line-height:1.2; margin-bottom:0.6rem; }
.empty-sub  { font-size:0.72rem; color:#2A2A28; line-height:1.9; font-weight:300; }
.eg-row     { display:flex; flex-wrap:wrap; gap:5px; margin-top:1.75rem; }
.eg         { background:#111110; border:1px solid #1A1A18; color:#2A2A28;
               padding:3px 10px; border-radius:2px; font-size:0.65rem; }

/* ── Q&A FEED ── */
.qa-block   { margin-bottom:1.75rem; }
.qa-q       { font-size:0.82rem; font-weight:500; color:#E8A623;
               margin-bottom:0.8rem; letter-spacing:-0.01em; }
.qa-pre     { color:#3A3A36; margin-right:5px; font-weight:300; }
.qa-a       { font-size:0.8rem; color:#CCCAC3; line-height:1.85; font-weight:300;
               padding-left:1.2rem; border-left:2px solid #1E1E1C; }
.qa-hr      { border:none; border-top:1px solid #161614; margin:1.75rem 0 0; }

/* Generating indicator */
.qa-gen     { font-size:0.72rem; color:#2E2E2B; padding-left:1.2rem;
               border-left:2px solid #E8A623; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background:#131312 !important; border:1px solid #1E1E1C !important;
    border-radius:2px !important; margin-top:1.5rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color:rgba(232,166,35,0.5) !important;
    box-shadow:0 0 0 1px rgba(232,166,35,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    background:transparent !important; color:#CCCAC3 !important;
    font-size:0.8rem !important; font-family:'IBM Plex Mono',monospace !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#252523 !important; }

/* ── BUTTONS ── */
.stButton > button {
    background:transparent !important; border:1px solid #1A1A18 !important;
    color:#252523 !important; border-radius:2px !important;
    font-size:0.58rem !important; font-family:'IBM Plex Mono',monospace !important;
    text-transform:uppercase !important; letter-spacing:0.1em !important;
    transition:all 0.15s !important;
}
.stButton > button:hover { border-color:#E8A623 !important; color:#E8A623 !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] p {
    font-size:0.65rem !important; color:#2A2A28 !important;
    font-family:'IBM Plex Mono',monospace !important;
}

/* scrollbar */
::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:#0C0C0B; }
::-webkit-scrollbar-thumb { background:#1A1A18; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_api_key():
    try:    return st.secrets["GROQ_API_KEY"]
    except: return os.environ.get("GROQ_API_KEY", "")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )

@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0,
                    max_tokens=1024, groq_api_key=get_api_key())

def build_pipeline(uploaded_files):
    key = "_".join(f"{f.name}_{f.size}" for f in uploaded_files)
    if st.session_state.get("fkey") == key:
        return st.session_state["ret"], st.session_state["chain"], st.session_state["names"]
    docs, names = [], []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
            t.write(uf.read()); path = t.name
        pages = PyPDFLoader(path).load()
        nm = uf.name.replace(".pdf","").replace("_"," ")
        for pg in pages: pg.metadata["doc_name"] = nm
        docs.extend(pages); names.append(nm); os.unlink(path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n","\n",". "," ",""]
    ).split_documents(docs)
    vs  = FAISS.from_documents(chunks, get_embeddings())
    ret = vs.as_retriever(search_type="mmr",
                          search_kwargs={"k":8,"fetch_k":30,"lambda_mult":0.7})
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise document assistant.
Answer using ONLY the information in the context below.
1. Start directly with the answer — no "According to..." or "Based on...".
2. Include every exact number, date, name, or figure.
3. Bullet points for lists or step-by-step answers.
4. If the answer isn't in the context: "not in your document."
5. Never fabricate.
Context: {context}"""),
        ("human", "{question}")
    ])
    chain = prompt | get_llm() | StrOutputParser()
    st.session_state.update({"fkey":key,"ret":ret,"chain":chain,"names":names})
    return ret, chain, names

def fmt(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

def ask(q, ret, chain):
    docs = ret.invoke(q)
    ans  = chain.invoke({"context":fmt(docs),"question":q})
    srcs = list({d.metadata.get("doc_name","") for d in docs})
    return ans, srcs

def render_feed(msgs):
    if not msgs: return ""
    html = '<div>'
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "user":
            q = m["content"].replace("<","&lt;").replace(">","&gt;")
            html += f'<div class="qa-block"><div class="qa-q"><span class="qa-pre">//</span>{q}</div>'
            if i+1 < len(msgs) and msgs[i+1]["role"] == "assistant":
                a = msgs[i+1]["content"].replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
                html += f'<div class="qa-a">{a}</div>'
                i += 1
            html += '<div class="qa-hr"></div></div>'
        i += 1
    html += '</div>'
    return html


# ══ SIDEBAR ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Background decoration
    st.markdown('<div class="sb-deco">01<br>—<br>∑</div>', unsafe_allow_html=True)

    # Wordmark
    st.markdown("""
        <div class="wm">
            <span class="wm-top">doc</span>
            <span class="wm-bot">talk</span>
            <span class="wm-sub">document assistant</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)

    # Upload
    st.markdown('<span class="sb-lbl">/ files</span>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "x", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # File list
    if uploaded_files:
        items = "".join(
            f'<div class="f-item"><span class="f-dot"></span>'
            f'{f.name.replace(".pdf","").replace("_"," ")}</div>'
            for f in uploaded_files
        )
        st.markdown(f'<div style="margin-top:.4rem">{items}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="f-none">no files loaded yet</div>',
                    unsafe_allow_html=True)

    st.markdown('<hr class="sb-rule" style="margin-top:auto">', unsafe_allow_html=True)
    st.markdown('<span class="sb-foot">powered by llama 3.3 · groq</span>',
                unsafe_allow_html=True)


# ══ MAIN ══════════════════════════════════════════════════════════════════════

# Header row
hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    if uploaded_files:
        n = len(uploaded_files)
        label = f"{n} file{'s' if n>1 else ''} in context"
        st.markdown(f"""
            <div class="mh">
                <span class="mh-tag">DocTalk</span>
                <span class="mh-status">
                    <span class="mh-dot"></span>{label}
                </span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="mh">
                <span class="mh-tag">DocTalk</span>
                <span class="mh-status">waiting for files</span>
            </div>
        """, unsafe_allow_html=True)

with hcol2:
    if st.session_state.get("messages"):
        if st.button("clear"):
            for k in ["messages","fkey","ret","chain","names"]:
                st.session_state.pop(k, None)
            st.rerun()

# Empty state
if not uploaded_files:
    st.markdown("""
        <div class="empty">
            <div class="empty-h">drop a document.<br>ask anything.</div>
            <div class="empty-sub">
                upload a PDF in the sidebar to begin.<br>
                books · contracts · policies · research · rulebooks
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
with st.spinner("indexing your document..."):
    ret, chain, names = build_pipeline(uploaded_files)

# Init messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Q&A feed
feed_html = render_feed(st.session_state.messages)
if feed_html:
    st.markdown(feed_html, unsafe_allow_html=True)
elif not st.session_state.messages:
    preview = ", ".join(names[:2]) + (f" +{len(names)-2}" if len(names)>2 else "")
    st.markdown(f"""
        <div style="font-size:0.72rem;color:#2A2A28;padding:.5rem 0 1rem;font-weight:300">
            loaded {preview} — ask me anything about it
        </div>
    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("ask me anything..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.spinner("searching..."):
        answer, srcs = ask(prompt, ret, chain)
    st.session_state.messages.append({"role":"assistant","content":answer,"sources":srcs})
