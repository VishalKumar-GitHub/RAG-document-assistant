"""Core RAG pipeline: load docs -> chunk -> embed -> FAISS -> retrieve -> answer with Claude."""
import os
import numpy as np
import faiss
from pypdf import PdfReader
from anthropic import Anthropic

EMBED_MODEL = "voyage-3"  # via Anthropic-compatible embeddings; see note in README
CHAT_MODEL = "claude-sonnet-4-5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def read_file(file) -> str:
    """Extract text from an uploaded PDF or txt file-like object."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, source: str):
    """Split text into overlapping chunks, tagged with their source filename."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append({"text": text[start:end], "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class VectorStore:
    """FAISS-backed store of chunk embeddings using Voyage embeddings."""

    def __init__(self, client):
        self.client = client
        self.index = None
        self.chunks = []

    def _embed(self, texts):
        import voyageai
        vo = voyageai.Client()  # reads VOYAGE_API_KEY
        result = vo.embed(texts, model="voyage-3", input_type="document")
        return np.array(result.embeddings, dtype="float32")

    def build(self, chunks):
        self.chunks = chunks
        vecs = self._embed([c["text"] for c in chunks])
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query, k=TOP_K):
        import voyageai
        vo = voyageai.Client()
        q = np.array(
            vo.embed([query], model="voyage-3", input_type="query").embeddings,
            dtype="float32",
        )
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, k)
        return [self.chunks[i] for i in idx[0] if i != -1]


def answer(client: Anthropic, query: str, contexts):
    """Ask Claude to answer using only the retrieved contexts, citing sources."""
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in contexts
    )
    system = (
        "You are a document assistant. Answer the user's question using ONLY the "
        "provided context. Cite the source filename for each claim. If the context "
        "does not contain the answer, say so plainly."
    )
    msg = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {query}",
        }],
    )
    return msg.content[0].text
