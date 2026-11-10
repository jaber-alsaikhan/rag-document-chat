"""Step 3 of the pipeline: turn text into vectors (embeddings).

An embedding model maps a text to a point in a high-dimensional space so that
texts with similar MEANING land close together, even when they share no words:

    "Why are transformers slow on long documents?"
    "Self-attention ... cost grows quadratically with input length."  <- closer
    "The recipe needs two eggs and a cup of flour."                   <- far away

We use all-MiniLM-L6-v2: a small (about 90 MB) BERT-style model that runs fine
on a CPU and outputs 384 numbers per text.

Important limit: the model reads at most 256 tokens (roughly 190 English
words). Anything longer is silently cut off before embedding.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and reuse it.

    Loading takes a few seconds (and a download on the very first run), so we
    must not do it for every question. `lru_cache` remembers the returned
    object: the first call loads the model, later calls get the same one.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed many texts. Returns an array of shape (len(texts), 384).

    - Texts are processed in batches, which is much faster than one by one.
    - normalize_embeddings=True scales every vector to length 1. Then cosine
      similarity is just a dot product, and scores from different texts are
      directly comparable.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 100,
    )


def embed_query(question: str) -> np.ndarray:
    """Embed a single question. Returns an array of shape (384,).

    The question MUST go through the same model as the chunks: vectors from
    two different models live in unrelated spaces and cannot be compared.
    """
    return embed_texts([question])[0]


if __name__ == "__main__":
    # Quick manual test:  python -m rag.embedder
    sentences = [
        "Why are transformers slow on long documents?",
        "Self-attention compares every token with every other token, "
        "so its cost grows quadratically with input length.",
        "The recipe needs two eggs and a cup of flour.",
    ]
    vectors = embed_texts(sentences)
    print(f"shape: {vectors.shape}, vector length: {np.linalg.norm(vectors[0]):.3f}")
    print(f"similarity(question, related):   {vectors[0] @ vectors[1]:.3f}")
    print(f"similarity(question, unrelated): {vectors[0] @ vectors[2]:.3f}")
