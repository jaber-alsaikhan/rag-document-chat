"""Download a small set of AI papers from arXiv to try the app with.

    python demo_docs/download_papers.py

The PDFs are saved next to this script. They are not stored in the git
repository (see .gitignore): each person downloads their own copy.
To use other papers, edit the PAPERS list below: the key is the arXiv id,
which is the number at the end of a paper's URL (arxiv.org/abs/<id>).
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

# arXiv id -> file name. All eight are about retrieval and RAG, so the demo
# doubles as reading material for this project.
PAPERS = {
    "1706.03762": "attention_is_all_you_need.pdf",
    "1810.04805": "bert.pdf",
    "1908.10084": "sentence_bert.pdf",
    "2004.04906": "dense_passage_retrieval.pdf",
    "2005.11401": "retrieval_augmented_generation.pdf",
    "2212.10496": "hyde_zero_shot_dense_retrieval.pdf",
    "2310.11511": "self_rag.pdf",
    "2312.10997": "rag_survey.pdf",
}

FOLDER = Path(__file__).parent
# arXiv asks automated clients to pause between requests.
SECONDS_BETWEEN_DOWNLOADS = 3


def main() -> None:
    for arxiv_id, file_name in PAPERS.items():
        target = FOLDER / file_name
        if target.exists():
            print(f"already here   {file_name}")
            continue

        request = urllib.request.Request(
            f"https://arxiv.org/pdf/{arxiv_id}",
            headers={"User-Agent": "rag-demo-downloader"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        except OSError as error:  # no internet, timeout, HTTP error...
            print(f"FAILED         {file_name}: {error}")
            continue

        if not data.startswith(b"%PDF"):  # e.g. an error page instead of a PDF
            print(f"FAILED         {file_name}: the server did not return a PDF")
            continue

        target.write_bytes(data)
        print(f"downloaded     {file_name} ({len(data) / 1_000_000:.1f} MB)")
        time.sleep(SECONDS_BETWEEN_DOWNLOADS)

    print(f"\nDone. Upload the PDFs from {FOLDER} in the app's sidebar.")


if __name__ == "__main__":
    main()
