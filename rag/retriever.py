"""Step 4 of the pipeline: store chunks and find the best ones for a question.

Two search modes are available:

- "vector": embed the question and return the chunks whose vectors are
  closest (cosine similarity). Good at matching MEANING, even with different
  wording. Weak on exact terms such as names, acronyms and numbers.

- "hybrid": run the vector search AND a BM25 keyword search, then merge the
  two ranked lists with Reciprocal Rank Fusion. BM25 is strong exactly where
  vectors are weak (rare exact words), so the combination is more robust.

The vectors live in ChromaDB, which saves them to a folder on disk, so the
index survives restarting the app.
"""

from __future__ import annotations

import re
import weakref
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi

from rag.chunker import Chunk
from rag.embedder import embed_query, embed_texts

# How many candidates each search contributes before fusion in hybrid mode.
CANDIDATES_PER_SEARCH = 20
# Standard constant from the Reciprocal Rank Fusion paper (Cormack et al., 2009).
RRF_K = 60


@dataclass
class RetrievedChunk:
    """A chunk returned by a search, with its relevance score."""

    id: str
    text: str
    source: str
    page: int | None
    score: float  # vector mode: cosine similarity; hybrid mode: RRF score

    @property
    def citation(self) -> str:
        """Human-readable source label, e.g. "[paper.pdf, page 3]"."""
        if self.page is None:
            return f"[{self.source}]"
        return f"[{self.source}, page {self.page}]"


def _tokenize(text: str) -> list[str]:
    """Lower-case and split into words for BM25."""
    return re.findall(r"\w+", text.lower())


def _drop_collection(client, name: str) -> None:
    """Delete a collection, ignoring the case where it is already gone."""
    try:
        client.delete_collection(name)
    except Exception:
        pass


class VectorStore:
    """A thin wrapper around one ChromaDB collection."""

    def __init__(self, persist_dir: str | None = "chroma_db", collection_name: str = "documents"):
        # persist_dir=None keeps everything in memory (used by evaluate.py so
        # that experiments never touch the app's real index).
        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=persist_dir)
        self._name = collection_name
        self._collection = self._open_collection()
        self._bm25_cache = None  # (BM25 index, ids, texts, metadatas) or None
        if persist_dir is None:
            # In-memory collections would otherwise pile up for as long as the
            # program runs. This deletes the collection automatically once the
            # VectorStore object itself is no longer used by anyone.
            weakref.finalize(self, _drop_collection, self._client, self._name)

    def _open_collection(self):
        # "hnsw:space": "cosine" tells Chroma to compare vectors with cosine
        # distance. HNSW is the approximate-nearest-neighbour index it builds:
        # a graph that finds close vectors without comparing against all of them.
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------- indexing

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 256) -> int:
        """Embed chunks and store them. Returns the number of chunks stored.

        `upsert` = update or insert. Because chunk ids are stable, indexing
        the same file twice overwrites the old entries instead of duplicating.
        """
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=texts,
                embeddings=embed_texts(texts).tolist(),
                # Chroma metadata cannot hold None, so "no page" is stored as 0.
                metadatas=[{"source": c.source, "page": c.page or 0} for c in batch],
            )
        self._bm25_cache = None  # the keyword index is now out of date
        return len(chunks)

    # --------------------------------------------------------------- search

    def search(self, question: str, top_k: int = 4, mode: str = "vector") -> list[RetrievedChunk]:
        """Return the `top_k` most relevant chunks, best first."""
        if mode not in ("vector", "hybrid"):
            raise ValueError("mode must be 'vector' or 'hybrid'")
        if not question.strip() or self.count() == 0:
            return []
        if mode == "vector":
            return self._vector_search(question, top_k)
        return self._hybrid_search(question, top_k)

    def _vector_search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[embed_query(question).tolist()],
            n_results=min(top_k, self.count()),
        )
        # Chroma returns one list per query; we sent one query, so take [0].
        return [
            RetrievedChunk(
                id=chunk_id,
                text=text,
                source=meta["source"],
                page=meta["page"] or None,
                # Chroma returns cosine DISTANCE (0 = identical).
                # similarity = 1 - distance, so higher is better.
                score=1.0 - distance,
            )
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def _keyword_search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """BM25: score chunks by the query words they contain.

        A word counts for more when it is rare across all chunks (so "HNSW"
        outweighs "the"), and long chunks are not favoured just for being long.
        """
        if self._bm25_cache is None:
            stored = self._collection.get(include=["documents", "metadatas"])
            index = BM25Okapi([_tokenize(text) for text in stored["documents"]])
            self._bm25_cache = (index, stored["ids"], stored["documents"], stored["metadatas"])
        index, ids, texts, metas = self._bm25_cache

        scores = index.get_scores(_tokenize(question))
        best = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                id=ids[i],
                text=texts[i],
                source=metas[i]["source"],
                page=metas[i]["page"] or None,
                score=float(scores[i]),
            )
            for i in best
            if scores[i] > 0  # 0 means no query word appears in the chunk
        ]

    def _hybrid_search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Merge vector and keyword results with Reciprocal Rank Fusion.

        Cosine similarities and BM25 scores are on different scales, so they
        cannot simply be added. RRF ignores the scores and uses only RANKS:

            rrf(chunk) = sum over both lists of  1 / (60 + rank in that list)

        A chunk ranked well in both lists beats one that is top in only one.
        """
        ranked_lists = [
            self._vector_search(question, CANDIDATES_PER_SEARCH),
            self._keyword_search(question, CANDIDATES_PER_SEARCH),
        ]
        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}
        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
                chunks_by_id[chunk.id] = chunk

        best_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        results = []
        for chunk_id in best_ids:
            chunk = chunks_by_id[chunk_id]
            chunk.score = fused_scores[chunk_id]
            results.append(chunk)
        return results

    # ----------------------------------------------------------- management

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """File names of all indexed documents, sorted."""
        stored = self._collection.get(include=["metadatas"])
        return sorted({meta["source"] for meta in stored["metadatas"]})

    def reset(self) -> None:
        """Delete everything in the collection.

        We drop the collection through Chroma instead of deleting the folder:
        on Windows the database files are locked while the app is running.
        """
        self._client.delete_collection(self._name)
        self._collection = self._open_collection()
        self._bm25_cache = None
