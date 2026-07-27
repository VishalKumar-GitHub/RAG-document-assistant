from rag import chunk_text, answer


def test_chunk_text_splits_input_into_overlapping_chunks():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk_text(text, "demo.txt")
    assert len(chunks) >= 1
    assert chunks[0]["source"] == "demo.txt"
    assert "alpha" in chunks[0]["text"]


def test_answer_falls_back_to_context_when_no_client_is_available():
    contexts = [{"source": "demo.txt", "text": "Paris is the capital of France."}]
    reply = answer(None, "What is the capital of France?", contexts)
    assert "Paris" in reply
    assert "demo.txt" in reply
