"""Step 1 of the pipeline: turn uploaded files into plain text.

Every loader returns a list of `Page` objects. Keeping the file name and page
number next to the text from the very start is what makes citations possible
later: the metadata simply travels with the text through every stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@dataclass
class Page:
    """One unit of text plus where it came from."""

    text: str
    source: str        # file name, e.g. "attention.pdf"
    page: int | None   # 1-based page number; None when the format has no pages


def _clean(text: str) -> str:
    """Light cleanup of extracted text.

    PDFs break words across lines with a hyphen ("retrie-\\nval"). Joining them
    back matters because a broken word matches nothing at search time.
    """
    text = re.sub(r"-\n(?=[a-z])", "", text)   # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse runs of blank lines
    return text.strip()


def _load_pdf(file: BinaryIO, name: str) -> list[Page]:
    reader = PdfReader(file)
    pages = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = _clean(pdf_page.extract_text() or "")
        if text:  # skip blank pages and pages that are only images
            pages.append(Page(text=text, source=name, page=number))
    return pages


def _load_docx(file: BinaryIO, name: str) -> list[Page]:
    # A .docx file stores paragraphs, not pages: page breaks are only decided
    # when Word renders the document. So we return one Page with page=None.
    document = DocxDocument(file)
    text = _clean("\n".join(p.text for p in document.paragraphs))
    return [Page(text=text, source=name, page=None)] if text else []


def _load_txt(file: BinaryIO, name: str) -> list[Page]:
    raw = file.read()
    # "utf-8-sig" also strips the invisible BOM that Windows Notepad adds.
    text = _clean(raw.decode("utf-8-sig", errors="replace"))
    return [Page(text=text, source=name, page=None)] if text else []


_LOADERS = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt}


def load_document(file: str | Path | BinaryIO, name: str | None = None) -> list[Page]:
    """Load one document and return its text as a list of Pages.

    `file` can be a path on disk OR an open binary file object. The second
    form is what Streamlit's uploader gives us, so uploads never need to be
    written to disk first.

    Raises ValueError for unsupported types and for files with no extractable
    text (for example a scanned PDF, which is just images of pages).
    """
    if isinstance(file, (str, Path)):
        path = Path(file)
        with open(path, "rb") as handle:
            return load_document(handle, name or path.name)

    name = name or getattr(file, "name", None)
    if not name:
        raise ValueError("A file name is required to detect the document type.")
    name = Path(name).name  # drop any folder part, keep "paper.pdf"

    extension = Path(name).suffix.lower()
    if extension not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{extension}' for {name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    pages = _LOADERS[extension](file, name)
    if not pages:
        raise ValueError(
            f"No text could be extracted from {name}. "
            "If it is a scanned PDF, it needs OCR first."
        )
    return pages


if __name__ == "__main__":
    # Quick manual test:  python -m rag.loader path/to/file.pdf
    import sys

    for file_path in sys.argv[1:]:
        loaded = load_document(file_path)
        words = sum(len(p.text.split()) for p in loaded)
        print(f"{file_path}: {len(loaded)} page(s), {words} words")
        print(f"  first 200 chars: {loaded[0].text[:200]!r}")
