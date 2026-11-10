# Demo documents

Eight AI papers from arXiv to try the app with. The PDFs are not stored in
this repository; download your own copy with one command.

## Download

From the project root:

```bash
python demo_docs/download_papers.py
```

This saves the PDFs into this folder (a few MB each; it takes around half a
minute because the script pauses between requests, as arXiv asks).
Running it again skips files that are already there.

## The papers

| File | Paper | arXiv |
|---|---|---|
| `attention_is_all_you_need.pdf` | Attention Is All You Need | [1706.03762](https://arxiv.org/abs/1706.03762) |
| `bert.pdf` | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | [1810.04805](https://arxiv.org/abs/1810.04805) |
| `sentence_bert.pdf` | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | [1908.10084](https://arxiv.org/abs/1908.10084) |
| `dense_passage_retrieval.pdf` | Dense Passage Retrieval for Open-Domain Question Answering | [2004.04906](https://arxiv.org/abs/2004.04906) |
| `retrieval_augmented_generation.pdf` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [2005.11401](https://arxiv.org/abs/2005.11401) |
| `hyde_zero_shot_dense_retrieval.pdf` | Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE) | [2212.10496](https://arxiv.org/abs/2212.10496) |
| `self_rag.pdf` | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | [2310.11511](https://arxiv.org/abs/2310.11511) |
| `rag_survey.pdf` | Retrieval-Augmented Generation for Large Language Models: A Survey | [2312.10997](https://arxiv.org/abs/2312.10997) |

To use different papers, edit the `PAPERS` list at the top of
`download_papers.py`. You can also download any PDF by hand from
`https://arxiv.org/pdf/<arxiv id>` and save it here.

## Try it

1. Start the app: `streamlit run app.py`
2. In the sidebar, upload the PDFs from this folder and click **Index documents**.
3. Ask questions.

Questions the papers can answer:

- How many attention heads does the base Transformer use?
- What two pre-training tasks does BERT use?
- Why is plain BERT slow for finding the most similar sentence pair, and how does Sentence-BERT fix it?
- What is the difference between RAG-Sequence and RAG-Token?
- How does HyDE build the vector it searches with?
- What are reflection tokens in Self-RAG?

A question the papers cannot answer, to check that the app refuses instead of guessing:

- What is the boiling point of ethanol?
