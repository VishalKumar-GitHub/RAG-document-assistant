"""Core RAG pipeline with an offline fallback for local demos and testing."""
import re
from collections import Counter
from typing import Optional

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional dependency
    Anthropic = None

CHAT_MODEL = "claude-sonnet-4-5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def read_file(file) -> str:
    """Extract text from an uploaded PDF or txt file-like object."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        if PdfReader is None:
            return "PDF extraction is unavailable in this environment."
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, source: str):
    """Split text into overlapping chunks, tagged with their source filename."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append({"text": text[start:end], "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class VectorStore:
    """Lightweight local vector store that works without external embeddings APIs."""

    def __init__(self, client=None):
        self.client = client
        self.chunks = []
        self.vectors = []

    def _tokenize(self, text: str):
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]

    def _vectorize(self, text: str):
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        return counts

    def build(self, chunks):
        self.chunks = chunks
        self.vectors = [self._vectorize(c["text"]) for c in chunks]

    def search(self, query, k=TOP_K):
        query_vec = self._vectorize(query)
        if not self.vectors:
            return []

        scored = []
        for index, chunk_vec in enumerate(self.vectors):
            if not query_vec:
                score = 0.0
            else:
                shared_terms = set(query_vec) & set(chunk_vec)
                score = sum(query_vec[token] * chunk_vec[token] for token in shared_terms)
            scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_indices = [index for _, index in scored[:k]]
        return [self.chunks[index] for index in top_indices if index < len(self.chunks)]


def answer(client: Optional[Anthropic], query: str, contexts):
    """Answer using the retrieved contexts, with a local fallback when no API client is available."""
    if not contexts:
        return "I could not find relevant context in the uploaded documents."

    if client is not None:
        try:
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
        except Exception:
            pass

    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_terms:
        return "Please ask a question about the indexed documents."

    best_context = None
    best_score = -1
    for context in contexts:
        text = context["text"].lower()
        score = sum(1 for term in query_terms if term in text)
        if score > best_score:
            best_score = score
            best_context = context

    if best_context is None:
        return "I could not find a confident answer in the available context."

    snippet = best_context["text"].strip()
    if len(snippet) > 260:
        snippet = snippet[:257].rstrip() + "..."
    return f"Based on the available documents, {snippet} [Source: {best_context['source']}]"
