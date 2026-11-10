"""Streamlit UI: upload documents, index them, and chat with them.

Run with:  streamlit run app.py

How Streamlit works (important for reading this file): the whole script runs
again from top to bottom on EVERY interaction (a click, a new message...).
Two tools keep things alive between those reruns:
- st.cache_resource : create an expensive object once (the vector store).
- st.session_state  : a per-user dictionary (the chat history).
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from rag.chunker import chunk_pages
from rag.generator import LLMError, is_not_found, stream_answer
from rag.loader import SUPPORTED_EXTENSIONS, load_document
from rag.retriever import RetrievedChunk, VectorStore

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "75"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
PER_SESSION_INDEX = os.getenv("PER_SESSION_INDEX", "false").lower() == "true"

st.set_page_config(page_title="Chat with your documents", page_icon="📚")


@st.cache_resource(show_spinner="Loading the vector database...")
def get_shared_store() -> VectorStore:
    """Local use: one database on disk, opened once and shared across reruns."""
    return VectorStore(persist_dir=CHROMA_DIR)


def get_store() -> VectorStore:
    """Return the index this visitor should use.

    cache_resource is shared by EVERY visitor of the app, which is right on
    your own machine but wrong on a public deployment: people would see each
    other's documents. With PER_SESSION_INDEX=true each browser session gets
    its own in-memory index instead, kept in st.session_state. It disappears
    when the session ends.
    """
    if not PER_SESSION_INDEX:
        return get_shared_store()
    if "store" not in st.session_state:
        st.session_state.store = VectorStore(
            persist_dir=None, collection_name=f"session_{uuid.uuid4().hex}"
        )
    return st.session_state.store


def index_files(store: VectorStore, uploaded_files) -> None:
    """Run load -> chunk -> embed + store for every uploaded file."""
    progress = st.progress(0.0)
    for position, uploaded in enumerate(uploaded_files, start=1):
        progress.progress(position / len(uploaded_files), text=f"Indexing {uploaded.name}...")
        try:
            pages = load_document(uploaded)  # an upload is already a file object
            chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
            store.add_chunks(chunks)
            st.success(f"{uploaded.name}: {len(chunks)} chunks")
        except ValueError as error:  # unsupported type or no extractable text
            st.error(str(error))
        except Exception as error:  # e.g. a corrupted file; keep going with the rest
            st.error(f"Could not read {uploaded.name}: {error}")
    progress.empty()


def show_sources(chunks: list[RetrievedChunk]) -> None:
    """Expandable list of the chunks that were given to the LLM."""
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)} retrieved chunks)"):
        for chunk in chunks:
            st.markdown(f"**{chunk.citation}** · relevance score {chunk.score:.3f}")
            st.caption(chunk.text)


store = get_store()
if "messages" not in st.session_state:
    # Each message: {"role": "user" | "assistant", "content": str, "sources": list}
    st.session_state.messages = []

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT or DOCX files",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    # Indexing happens on a button click, not on upload: otherwise every
    # rerun of the script would index the same files again.
    if st.button("Index documents", type="primary", disabled=not uploaded_files):
        index_files(store, uploaded_files)

    sources = store.list_sources()
    st.caption(f"{len(sources)} document(s), {store.count()} chunks indexed")
    for name in sources:
        st.markdown(f"- {name}")
    if PER_SESSION_INDEX:
        st.caption("Your documents are visible only to you and are removed when you close this tab.")

    st.header("Search settings")
    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=5, value=min(max(DEFAULT_TOP_K, 3), 5))
    mode = st.radio(
        "Search mode",
        options=["vector", "hybrid"],
        format_func=lambda m: "Vector (meaning)" if m == "vector" else "Hybrid (meaning + keywords)",
    )

    st.divider()
    if st.button("Clear database and chat"):
        store.reset()
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------- chat
st.title("📚 Chat with your documents")
st.caption("Answers come only from the documents you upload, with citations.")

if store.count() == 0:
    st.info("Upload documents in the sidebar and click **Index documents** to start.")

# Redraw the conversation so far (remember: the script reruns every time).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

question = st.chat_input("Ask a question about your documents", disabled=store.count() == 0)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            retrieved = store.search(question, top_k=top_k, mode=mode)
        try:
            # write_stream shows the text as it arrives and returns all of it.
            answer = st.write_stream(stream_answer(question, retrieved))
        except LLMError as error:
            answer = f"⚠️ {error}"
            st.error(str(error))
        # When the model says "not found", the retrieved chunks were not used
        # as evidence, so we do not present them as sources.
        used = [] if is_not_found(answer) else retrieved
        show_sources(used)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": used})
