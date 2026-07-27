"""Streamlit RAG Document Assistant powered by Claude + FAISS."""
import os
import streamlit as st
from anthropic import Anthropic
from rag import read_file, chunk_text, VectorStore, answer

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")
st.title("📄 RAG Document Assistant")
st.caption("Upload PDFs or text files, then ask questions. Answers are grounded in your documents and cite their sources.")

# --- API key ---
api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
if not api_key:
    st.warning("Set ANTHROPIC_API_KEY (and VOYAGE_API_KEY) in Streamlit secrets or your environment.")
    st.stop()
client = Anthropic(api_key=api_key)

# --- Session state ---
if "store" not in st.session_state:
    st.session_state.store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: upload & index ---
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Upload PDFs or .txt", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Build index", disabled=not files):
        with st.spinner("Reading and indexing..."):
            all_chunks = []
            for f in files:
                text = read_file(f)
                all_chunks.extend(chunk_text(text, f.name))
            store = VectorStore(client)
            store.build(all_chunks)
            st.session_state.store = store
        st.success(f"Indexed {len(all_chunks)} chunks from {len(files)} file(s).")

# --- Chat ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about your documents..."):
    if st.session_state.store is None:
        st.info("Upload documents and click **Build index** first.")
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
                for c in contexts:
                    st.markdown(f"**{c['source']}** — {c['text'][:200]}...")
    st.session_state.messages.append({"role": "assistant", "content": reply})
