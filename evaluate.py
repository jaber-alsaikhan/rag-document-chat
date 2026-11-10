"""Evaluate the RAG pipeline on SQuAD.

    python evaluate.py                    # 200 questions, all settings
    python evaluate.py --n 50             # quicker run
    python evaluate.py --skip-generation  # retrieval metrics only (no LLM)

SQuAD is a reading-comprehension dataset: each question comes with the
Wikipedia paragraph that contains the answer, plus the answer text itself.
That gives us ground truth for both halves of RAG:

  Retrieval  - we index ALL paragraphs of the validation set (about 2,000)
               and check whether the question's own paragraph is found.
               Hit@k = share of questions whose paragraph is in the top k.

  Generation - the LLM answers from the retrieved chunks and we compare with
               the reference answers.
               Exact Match = answer identical after normalisation.
               F1          = word overlap between answer and reference.

The script imports the SAME chunker, retriever and generator as the app, so
the numbers describe the real system.
"""

from __future__ import annotations

import argparse
import re
import string
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from rag.chunker import chunk_pages
from rag.generator import LLMError, generate_answer, is_not_found
from rag.loader import Page
from rag.retriever import RetrievedChunk, VectorStore

# The settings to compare: every chunking option is run with every mode.
CHUNKINGS = [
    {"chunk_size": 500, "overlap": 75},   # the app's default
    {"chunk_size": 100, "overlap": 20},   # small chunks, fully inside the embedder's 256-token limit
]
MODES = ["vector", "hybrid"]
HIT_KS = [1, 3, 5]
GENERATION_TOP_K = 4  # chunks given to the LLM, same as the app's default


# ------------------------------------------------------------ answer metrics
# These follow the official SQuAD evaluation script.

def normalize_answer(text: str) -> str:
    """Lower-case, remove punctuation, articles and extra spaces.

    Without this, "The Eiffel Tower." and "eiffel tower" would count as
    different answers.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1_score(prediction: str, reference: str) -> float:
    """Word-level F1: harmonic mean of precision and recall.

    precision = shared words / words in the prediction
    recall    = shared words / words in the reference
    """
    pred_words = normalize_answer(prediction).split()
    ref_words = normalize_answer(reference).split()
    shared = sum((Counter(pred_words) & Counter(ref_words)).values())
    if shared == 0:
        return 0.0
    precision = shared / len(pred_words)
    recall = shared / len(ref_words)
    return 2 * precision * recall / (precision + recall)


def best_over_references(metric, prediction: str, references: list[str]) -> float:
    """SQuAD has several acceptable answers per question; take the best match."""
    return max(metric(prediction, reference) for reference in references)


# ------------------------------------------------------------------- data

def load_squad(n_questions: int, seed: int):
    """Return (pages, questions).

    pages     : one Page per unique paragraph in the whole validation set.
    questions : a random sample; each knows which paragraph holds its answer.
    """
    dataset = load_dataset("rajpurkar/squad", split="validation")

    # Many questions share a paragraph, so give each unique paragraph one id.
    context_ids: dict[str, str] = {}
    for context in dataset["context"]:
        if context not in context_ids:
            context_ids[context] = f"squad_{len(context_ids):05d}"
    pages = [Page(text=text, source=source, page=None) for text, source in context_ids.items()]

    sample = dataset.shuffle(seed=seed).select(range(n_questions))
    questions = [
        {
            "question": row["question"],
            "gold_source": context_ids[row["context"]],
            "references": row["answers"]["text"],
        }
        for row in sample
    ]
    return pages, questions


# -------------------------------------------------------------- evaluation

def llm_is_available() -> bool:
    """Send one tiny request so we fail early instead of after indexing."""
    probe = RetrievedChunk(id="probe", text="The sky is blue.", source="probe", page=None, score=1.0)
    try:
        generate_answer("What colour is the sky?", [probe], style="short")
        return True
    except LLMError as error:
        print(f"\n[!] LLM not available, answer metrics will be skipped.\n    {error}\n")
        return False


def evaluate_setting(store: VectorStore, questions: list[dict], mode: str, with_generation: bool) -> dict:
    """Run every question through one setting and average the metrics."""
    hits = {k: 0 for k in HIT_KS}
    em_total = f1_total = 0.0
    not_found = 0
    start = time.time()

    for number, item in enumerate(questions, start=1):
        retrieved = store.search(item["question"], top_k=max(HIT_KS), mode=mode)

        # Hit@k: is any of the first k chunks cut from the correct paragraph?
        sources = [chunk.source for chunk in retrieved]
        for k in HIT_KS:
            hits[k] += item["gold_source"] in sources[:k]

        if with_generation:
            answer = generate_answer(item["question"], retrieved[:GENERATION_TOP_K], style="short")
            if is_not_found(answer):
                not_found += 1  # scores 0 for EM and F1: SQuAD v1 questions are all answerable
            else:
                em_total += best_over_references(exact_match, answer, item["references"])
                f1_total += best_over_references(f1_score, answer, item["references"])
            if number % 25 == 0:
                print(f"    answered {number}/{len(questions)} ({time.time() - start:.0f}s)")

    n = len(questions)
    row = {f"hit@{k}": round(100 * hits[k] / n, 1) for k in HIT_KS}
    if with_generation:
        row["exact_match"] = round(100 * em_total / n, 1)
        row["f1"] = round(100 * f1_total / n, 1)
        row["not_found_%"] = round(100 * not_found / n, 1)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on SQuAD.")
    parser.add_argument("--n", type=int, default=200, help="number of questions (default 200)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the question sample")
    parser.add_argument("--skip-generation", action="store_true", help="only measure retrieval")
    parser.add_argument("--output", default="results/eval_results.csv", help="where to save the CSV")
    args = parser.parse_args()

    with_generation = not args.skip_generation and llm_is_available()

    print("Loading SQuAD...")
    pages, questions = load_squad(args.n, args.seed)
    print(f"{len(pages)} paragraphs to index, {len(questions)} questions to ask.")

    rows = []
    for chunking in CHUNKINGS:
        size, overlap = chunking["chunk_size"], chunking["overlap"]
        chunks = chunk_pages(pages, size, overlap)
        print(f"\nIndexing with chunk_size={size}, overlap={overlap}: {len(chunks)} chunks")
        # persist_dir=None -> in-memory index, so the app's database is untouched.
        store = VectorStore(persist_dir=None, collection_name=f"eval_{size}_{overlap}")
        store.add_chunks(chunks)

        for mode in MODES:
            print(f"  Evaluating mode={mode}...")
            metrics = evaluate_setting(store, questions, mode, with_generation)
            rows.append({"chunk_size": size, "overlap": overlap, "mode": mode, "chunks": len(chunks), **metrics})
        store.reset()  # free the memory before building the next index

    results = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\nResults on {len(questions)} SQuAD questions over {len(pages)} paragraphs (all values in %):\n")
    print(results.to_string(index=False))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
