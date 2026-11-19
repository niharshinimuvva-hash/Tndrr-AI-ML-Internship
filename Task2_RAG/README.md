# Task 2: Retrieval-Augmented Generation (RAG) from Scratch

## 1. Objective

Build a Retrieval-Augmented Generation (RAG) retrieval pipeline over a public document without using LangChain or LlamaIndex. The experiment compares chunk sizes and retrieval methods and evaluates retrieval quality.

## 2. Document and Dataset

- Source document: NIST AI Risk Management Framework (AI RMF 1.0)
- Document length: 48 pages
- Questions evaluated: 15
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Chunking strategies: 300 words and 600 words
- Vector similarity: dense embedding retrieval

The PDF was extracted into page-level text and divided into chunks. Embeddings were generated and compared using NumPy-based similarity search.

## 3. RAG Pipeline

The pipeline consists of:

1. Extract text from the source PDF.
2. Split the extracted text into chunks.
3. Generate dense vector embeddings.
4. Retrieve the top five relevant chunks for each question.
5. Compare retrieval results across chunk sizes.
6. Evaluate retrieval using Recall@5.

The experiment also explored BM25 keyword retrieval and dense retrieval with cross-encoder reranking.

## 4. Evaluation Results

Dense retrieval was evaluated using 15 questions and manually selected primary evidence pages as relevance labels.

| Chunk size | Recall@5 |
|---|---:|
| 300 words | 65.44% |
| 600 words | 49.33% |

The 300-word chunk configuration performed better by 16.11 percentage points in this evaluation.

The relevance labels were manually selected and are provisional. The reported comparison is specifically for dense retrieval.

## 5. Retrieval Failure Analysis

One genuine retrieval failure and its successful fix were documented.

**Question:** What does the framework recommend when an AI system operates beyond its knowledge limits?

- Original retrieval did not retrieve the relevant passage on page 35 in its top five results.
- The query was expanded with related terms, including fail safely, residual negative risk, reliability, robustness, and safe operation.
- After query expansion, page 35 appeared at rank 3 with a similarity score of 0.7101.

The relevant passage discusses demonstrating that an AI system is safe, keeping residual negative risk within tolerance, and ensuring that systems can fail safely beyond their knowledge limits.

Only one genuine failure and successful fix was verified. A second failure was not established, so no additional example is claimed.

## 6. BM25 and Reranking

BM25 is a keyword-based retrieval method that ranks documents using term frequency and inverse document frequency, with document-length normalization.

A cross-encoder reranker evaluates query-passage pairs jointly and reorders retrieved candidates based on their relevance. It can improve ranking quality at the cost of additional computation.

BM25 and cross-encoder reranking were explored in this project. The quantitative Recall@5 comparison reported above uses dense retrieval.

## 7. Limitations

- The evaluation contains only 15 questions.
- Relevance labels are manually selected and provisional.
- Only one genuine retrieval failure and successful fix was verified.
- Recall@5 measures retrieval quality, not the factual accuracy of generated answers.
- The evaluation does not establish performance on other documents or domains.

## 8. Conclusion

The experiment implemented a RAG retrieval pipeline over the NIST AI RMF document and compared 300-word and 600-word chunking strategies. The 300-word configuration achieved a higher Recall@5 score in the current evaluation.

Query expansion also demonstrated a practical way to address a retrieval failure by introducing semantically related terms.

## Project Files

- `Task2_RAG_From_Scratch.ipynb`: implementation and experiments
- `source_document.pdf`: source document
- `extracted_pages.json`: extracted page text
- `chunks_300.json`, `chunks_600.json`: chunked documents
- `evaluation_questions.json`: evaluation questions and relevance labels
- `retrieval_comparison.csv`: retrieval comparison results
- `rag_experiment_results.json`: experiment outputs
- `final_task2_summary.json`: final evaluation summary
