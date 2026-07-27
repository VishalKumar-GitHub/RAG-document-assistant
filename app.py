"""Streamlit RAG Document Assistant with a polished offline-friendly UI."""
import os
import streamlit as st

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional dependency
    Anthropic = None

from rag import read_file, chunk_text, VectorStore, answer

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #fef3c7 0%, #dcfce7 100%); }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stSidebar"] { background: rgba(255,255,255,0.78); backdrop-filter: blur(8px); }
    .stButton > button { background: linear-gradient(90deg, #4ade80, #86efac); color: #052e16; border: none; }
    .stTextInput > div > div > input { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 RAG Document Assistant")
st.caption("A polished knowledge assistant that can index documents and answer questions end to end, even in offline demo mode.")

SAMPLE_TEXT = """Acme Analytics is a fictional product company that helps teams organize knowledge. The assistant supports document upload, indexing, retrieval, and grounded answers. It is useful for onboarding guides, support articles, and internal policy documents. Customers can ask questions such as 'What does Acme Analytics do?' or 'How do I onboard a new teammate?' and receive answers with source citations."""


try:
    secrets = st.secrets
    secrets.get("ANTHROPIC_API_KEY", "")
except Exception:  # pragma: no cover - Streamlit secrets may be unavailable
    secrets = {}

api_key = os.getenv("ANTHROPIC_API_KEY") or secrets.get("ANTHROPIC_API_KEY", "")
client = None
if api_key and Anthropic is not None:
    try:
        client = Anthropic(api_key=api_key)
    except Exception:
        client = None

if "store" not in st.session_state:
    st.session_state.store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = 0
if "initialized" not in st.session_state:
    st.session_state.initialized = False


def build_store(files):
    all_chunks = []
    if files:
        for uploaded_file in files:
            text = read_file(uploaded_file)
            all_chunks.extend(chunk_text(text, uploaded_file.name))
    else:
        all_chunks.extend(chunk_text(SAMPLE_TEXT, "sample-docs.txt"))

    store = VectorStore(client)
    store.build(all_chunks)
    return store, all_chunks


with st.sidebar:
    st.header("Documents")
    st.caption("Upload PDFs or text files, or start with a built-in demo dataset.")
    files = st.file_uploader(
        "Upload PDFs or .txt", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Build index"):
        with st.spinner("Reading and indexing..."):
            store, all_chunks = build_store(files)
            st.session_state.store = store
            st.session_state.indexed_count = len(all_chunks)
            st.session_state.initialized = True
        st.success(f"Indexed {len(all_chunks)} chunks from {len(files) or 1} source(s).")

    if st.session_state.store is not None:
        st.metric("Indexed chunks", st.session_state.indexed_count)
    else:
        st.info("No knowledge base loaded yet. Use the demo dataset or upload documents.")

st.subheader("Knowledge workspace")
if st.session_state.store is None and not st.session_state.initialized:
    with st.spinner("Preparing the demo workspace..."):
        store, all_chunks = build_store([])
        st.session_state.store = store
        st.session_state.indexed_count = len(all_chunks)
        st.session_state.initialized = True

if st.session_state.store is not None:
    st.success("The assistant is ready. Ask a question to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    if st.session_state.store is None:
        st.info("Build an index first, then ask a question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            contexts = st.session_state.store.search(prompt)
            reply = answer(client, prompt, contexts)
            st.markdown(reply)
            with st.expander("Sources retrieved"):
                if contexts:
                    for context in contexts:
                        snippet = context["text"][:220].replace("\n", " ")
                        st.markdown(f"**{context['source']}** — {snippet}...")
                else:
                    st.markdown("No relevant sources were found.")

    st.session_state.messages.append({"role": "assistant", "content": reply})
