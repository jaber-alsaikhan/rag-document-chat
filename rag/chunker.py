"""Step 2 of the pipeline: split pages into overlapping chunks of words.

Why chunk at all?
- The embedding model turns a whole text into ONE vector. A vector for a full
  10-page paper is a blur of every topic in it; a vector for ~500 words is
  about one thing, so it can match a specific question.
- The LLM only gets the few chunks we retrieve, so each chunk has to be small
  enough to fit several in the prompt but big enough to make sense alone.

Why overlap?
- A fixed-size cut can land in the middle of an explanation. Repeating the
  last N words of one chunk at the start of the next means a sentence that
  straddles the cut is still complete in at least one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.loader import Page


@dataclass
class Chunk:
    """A piece of a page, small enough to embed and to show as a source."""

    id: str            # unique and stable, e.g. "paper.pdf::p3::c0"
    text: str
    source: str        # file name, copied from the Page
    page: int | None   # page number, copied from the Page


def split_words(text: str, chunk_size: int = 500, overlap: int = 75) -> list[str]:
    """Split text into pieces of at most `chunk_size` words.

    Consecutive pieces share `overlap` words. The window moves forward by
    (chunk_size - overlap) words each time:

        chunk_size=6, overlap=2  ->  step=4
        words:   w0 w1 w2 w3 w4 w5 w6 w7 w8 w9
        chunk 0: w0 w1 w2 w3 w4 w5
        chunk 1:             w4 w5 w6 w7 w8 w9
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()  # splits on any whitespace, including newlines
    step = chunk_size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + chunk_size]))
        # Stop once a window has reached the end of the text. Without this we
        # would add a tiny last piece made only of words already covered.
        if start + chunk_size >= len(words):
            break
    return pieces


def chunk_pages(pages: list[Page], chunk_size: int = 500, overlap: int = 75) -> list[Chunk]:
    """Chunk every page separately and keep its source metadata.

    Chunks never cross a page boundary. That costs a little context when a
    sentence runs over two pages, but it means every chunk belongs to exactly
    one page, so the citation "[paper.pdf, page 3]" is always exact.
    """
    chunks = []
    for page in pages:
        page_label = page.page if page.page is not None else 0
        for index, text in enumerate(split_words(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    # The id is built from file name + page + position, so
                    # indexing the same file twice produces the same ids and
                    # the database overwrites instead of storing duplicates.
                    id=f"{page.source}::p{page_label}::c{index}",
                    text=text,
                    source=page.source,
                    page=page.page,
                )
            )
    return chunks


if __name__ == "__main__":
    # Quick manual test:  python -m rag.chunker path/to/file.pdf
    import sys

    from rag.loader import load_document

    for file_path in sys.argv[1:]:
        result = chunk_pages(load_document(file_path))
        sizes = [len(c.text.split()) for c in result]
        print(f"{file_path}: {len(result)} chunks, {min(sizes)}-{max(sizes)} words each")
        print(f"  first id: {result[0].id}")
