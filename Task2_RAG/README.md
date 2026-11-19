# Task 2: Retrieval-Augmented Generation from Scratch

## 1. Objective

Build a retrieval pipeline from scratch over a public document, using text extraction, chunking, embeddings, and similarity search without LangChain or LlamaIndex.

## 2. Source Document and Environment

- Source: NIST AI Risk Management Framework (AI RMF 1.0)
- Document length: 48 pages
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Similarity search: cosine similarity using NumPy
- Retrieval approaches explored: dense retrieval, BM25, and dense retrieval with cross-encoder reranking
- Evaluation questions: 15

## 3. Chunking

| Configuration | Chunk count |
|---|---:|
| 300-word chunks | 82 |
| 600-word chunks | 48 |

Two chunk sizes were evaluated to examine how chunk granularity affects retrieval performance.

## 4. Evaluation Method

The evaluation uses manually selected primary evidence pages for each of the 15 questions. These relevance labels were corrected after inspecting the source document.

The reported metric is question-level Hit Rate@5, defined as:

`Hit Rate@5 = (Number of questions with at least one relevant page in the top 5 / Total questions) × 100`

A question is counted as a hit when at least one manually labelled relevant evidence page appears among the five highest-ranked retrieved results. It does not measure the proportion of all relevant pages retrieved, so it is not conventional Recall@5.

## 5. Corrected Evaluation Results

| Chunk size | Questions | Hit Rate@5 |
|---|---:|---:|
| 300 words | 15 | 100.00% (15/15) |
| 600 words | 15 | 93.33% (14/15) |

The 300-word configuration retrieved at least one relevant page in the top five for all 15 questions. The 600-word configuration did so for 14 of the 15 questions.

## 6. Retrieval Failure Analysis

Two retrieval failures were investigated. Both were improved using query expansion. The second failure was observed with 600-word chunks.

### Failure 1: AI systems operating beyond their knowledge limits

**Question:** What does the framework recommend when an AI system operates beyond its knowledge limits?

- Original top-five retrieved pages: 44, 6, 7, 43, 30
- Relevant evidence page: 35
- Problem: The original query did not retrieve the key evidence page in the top five.

**Fix:** Expanded the query with related concepts, including fail-safe operation, safety risks, residual negative risk, reliability, robustness, monitoring, and failure response.

- Expanded-query top-five pages: 19, 20, 35, 43, 9
- Relevant page 35 was retrieved at rank 3.
- Page 35 discusses demonstrating system safety, residual risk within tolerance, and the ability to fail safely beyond knowledge limits.

### Failure 2: Purpose of the MANAGE function

**Question:** What is the purpose of the MANAGE function?

- Chunk configuration: 600 words
- Original top-five retrieved pages: 28, 46, 25, 26, 27
- Expected evidence pages: 37, 38, 39, 40
- Problem: None of the expected evidence pages appeared in the original top five.

**Fix:** Expanded the query with risk treatment, prioritization, response and recovery, incident management, post-deployment monitoring, risk mitigation, documentation, and communication.

- Expanded-query top-five pages: 25, 37, 38, 47, 8
- Page 37 was retrieved at rank 2 (score 0.7662).
- Page 38 was retrieved at rank 3 (score 0.7528).
- Page 37 discusses risk prioritization, treatment, and response. Page 38 discusses monitoring, incident response, recovery, and communication.

This second failure was observed with 600-word chunks. The 300-word evaluation had no misses.

## 7. Relevance Labels and Evaluation Caveats

The evaluation uses manually selected primary evidence pages, corrected after source inspection. These labels may not capture every relevant passage. Hit Rate@5 only checks whether at least one labelled relevant page appears in the top five. Query expansion was tested on the identified failure queries and should not be interpreted as a broad benchmark improvement.

The 15-question evaluation is relatively small. Results may vary with different questions, relevance labels, chunking choices, and embedding models.

## 8. Technologies

- Python
- Sentence Transformers
- NumPy
- Pandas
- BM25
- Cross-encoder reranking
- Google Colab

## 9. Project Files

- `Task2_RAG_From_Scratch.ipynb`: implementation and evaluation notebook.
- `source_document.pdf`: NIST AI RMF source document.
- `extracted_pages.json`: extracted document text.
- `chunks_300.json` and `chunks_600.json`: chunked document data.
- `evaluation_questions.json`: evaluation questions and relevance labels.
- `retrieval_comparison.csv`: corrected chunk-size comparison.
- `corrected_retrieval_comparison.csv`: corrected comparison results.
- `corrected_retrieval_results.json`: corrected retrieval evaluation results.
- `second_failure_analysis.json`: MANAGE-function failure analysis.
- `failure_analysis.json`: consolidated record of both retrieval failures.
- `final_task2_summary.json`: final evaluation summary.

## Conclusion

The implementation demonstrates a basic RAG retrieval pipeline and compares two chunk sizes. The corrected question-level Hit Rate@5 was 100.00% for 300-word chunks and 93.33% for 600-word chunks. Two retrieval failures were examined and improved through query expansion. The results are limited by the small evaluation set and manually selected relevance labels.
