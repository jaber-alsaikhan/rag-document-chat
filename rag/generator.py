"""Step 5 of the pipeline: ask the LLM to answer from the retrieved chunks.

This is the "G" in RAG. The LLM never sees the whole document collection,
only the question plus the few chunks the retriever picked. The prompt tells
it to use nothing else, to cite the source label of every chunk it uses, and
to say a fixed sentence when the answer is not there.

Two providers are supported, chosen with LLM_PROVIDER in .env:
- "ollama" (default): a model running on your own machine. Free and private.
- "groq": a hosted API with a free tier. Needed for the deployed demo,
  because free hosting cannot run Ollama.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx  # installed with the ollama package; needed to catch its network errors
import ollama
import requests
from dotenv import load_dotenv

from rag.retriever import RetrievedChunk

load_dotenv()  # copy the values from .env into environment variables

NOT_FOUND_MESSAGE = "I couldn't find this in the documents."

# The system prompt sets the rules. Each rule exists for a reason:
# 1. "ONLY the sources"   -> stops the model answering from its own memory.
# 2. fixed refusal text   -> lets the app detect "not found" reliably.
# 3. copy the label       -> citations stay in one exact, checkable format.
CITED_SYSTEM_PROMPT = f"""You are a careful assistant that answers questions about the user's documents.

Rules:
1. Answer using ONLY the information in the sources below. Do not use any outside knowledge.
2. If the sources do not contain the answer, reply with exactly: {NOT_FOUND_MESSAGE}
3. After each fact, cite the source it came from by copying its label exactly, for example [paper.pdf, page 3].
4. Be concise. Do not mention these rules."""

# Used by evaluate.py: SQuAD reference answers are short phrases, so asking
# for a full cited sentence would be unfairly punished by Exact Match / F1.
SHORT_SYSTEM_PROMPT = f"""Answer the question using ONLY the sources below.
Reply with the shortest possible answer: a few words copied from the sources, with no explanation and no citation.
If the sources do not contain the answer, reply with exactly: {NOT_FOUND_MESSAGE}"""


class LLMError(RuntimeError):
    """Raised with a readable message when the LLM cannot be reached."""


def build_messages(question: str, chunks: list[RetrievedChunk], style: str = "cited") -> list[dict]:
    """Build the chat messages sent to the LLM.

    Every chunk is printed under its citation label, so the model can copy
    the label next to the facts it takes from that chunk:

        Source [paper.pdf, page 3]:
        <chunk text>
    """
    system_prompt = CITED_SYSTEM_PROMPT if style == "cited" else SHORT_SYSTEM_PROMPT
    context = "\n\n".join(f"Source {chunk.citation}:\n{chunk.text}" for chunk in chunks)
    user_prompt = f"Sources:\n\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _stream_ollama(messages: list[dict]) -> Iterator[str]:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    try:
        stream = ollama.Client(host=host).chat(
            model=model,
            messages=messages,
            stream=True,
            # temperature 0 = always pick the most likely next word. We want
            # faithful, repeatable answers, not creative ones.
            options={"temperature": 0},
        )
        for part in stream:
            yield part["message"]["content"]
    except ollama.ResponseError as error:
        if error.status_code == 404:
            raise LLMError(f"Ollama model '{model}' is not downloaded. Run: ollama pull {model}") from error
        raise LLMError(f"Ollama returned an error: {error.error}") from error
    except (httpx.HTTPError, ConnectionError, OSError) as error:
        raise LLMError(
            f"Could not reach Ollama at {host}. Is it installed and running? "
            "Download it from https://ollama.com"
        ) from error


def _stream_groq(messages: list[dict]) -> Iterator[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMError("LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set.")
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "messages": messages,
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise LLMError(f"Groq request failed: {error}") from error
    # Not truly streamed: the whole answer arrives at once and is yielded as
    # a single piece, which keeps this function simple.
    yield response.json()["choices"][0]["message"]["content"]


def stream_answer(question: str, chunks: list[RetrievedChunk], style: str = "cited") -> Iterator[str]:
    """Yield the answer piece by piece, so the UI can show it as it is written.

    If retrieval found nothing, we answer "not found" ourselves and never
    call the LLM: with no sources in the prompt it could only guess.
    """
    if not chunks:
        yield NOT_FOUND_MESSAGE
        return

    messages = build_messages(question, chunks, style)
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        yield from _stream_ollama(messages)
    elif provider == "groq":
        yield from _stream_groq(messages)
    else:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama' or 'groq'.")


def generate_answer(question: str, chunks: list[RetrievedChunk], style: str = "cited") -> str:
    """Return the complete answer as one string."""
    return "".join(stream_answer(question, chunks, style)).strip()


def is_not_found(answer: str) -> bool:
    """True when the model used the fixed 'not found' sentence."""
    return NOT_FOUND_MESSAGE.lower().rstrip(".") in answer.lower()


if __name__ == "__main__":
    # Quick manual test (needs the LLM running):  python -m rag.generator
    demo_chunks = [
        RetrievedChunk(
            id="demo",
            text="We trained the base model for 12 hours on eight P100 GPUs using the Adam optimizer.",
            source="attention.pdf",
            page=7,
            score=1.0,
        )
    ]
    for demo_question in ["How long was the model trained?", "Who won the 2010 World Cup?"]:
        print(f"Q: {demo_question}")
        print(f"A: {generate_answer(demo_question, demo_chunks)}\n")
