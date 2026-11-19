# AI/ML Internship Assignment

This repository contains three practical projects covering parameter-efficient fine-tuning, retrieval-augmented generation, and local LLM inference.

## Task 1: Intent-to-JSON Extraction Using LoRA

- Model: Qwen2.5-0.5B-Instruct
- Dataset: 510 synthetic examples (400 train, 50 validation, 60 test)
- Fine-tuning: LoRA using Hugging Face PEFT
- Trainable parameters: 0.88% of model parameters
- Base model exact-match accuracy: 8.33%
- Fine-tuned exact-match accuracy on original test set: 100%
- Additional unseen evaluation: 3/3 exact matches
- Adversarial evaluation: 70% exact-match accuracy and 95% field accuracy across 10 examples
- Three adversarial failures identified: departure/destination confusion, date wording change, and omitted time preference.

The adversarial results show that perfect accuracy on a small synthetic test set does not guarantee robust generalization.

See [Task 1 README](Task1_Intent2JSON_LoRA/README.md) for configuration, detailed evaluation, failure analysis, and limitations.
## Task 2: Retrieval-Augmented Generation from Scratch

- Source: NIST AI Risk Management Framework (48 pages)
- Evaluation questions: 15
- Embedding model: all-MiniLM-L6-v2
- Dense retrieval Hit Rate@5 with 300-word chunks: 100%
- Dense retrieval Hit Rate@5 with 600-word chunks: 93.33%
- Explored BM25 and cross-encoder reranking.
- Two retrieval failures investigated and successfully fixed using query expansion.

See [Task 2 README](Task2_RAG/README.md).

## Task 3: Local LLM Inference and Quantization

- Model: Qwen2.5-1.5B-Instruct
- Runtime: Hugging Face Transformers on Tesla T4
- FP16 generation speed: 26.51 tokens/second
- 4-bit NF4 generation speed: 12.83 tokens/second
- FP16 peak allocated GPU memory: 2.89 GB
- 4-bit peak allocated GPU memory: 1.092 GB
- Ten prompts manually compared for response quality.

See [Task 3 README](Task3_Local_LLM_Inference/README.md).

## Repository Structure

```text
Tndrr-AI-ML-Internship/
├── Task1_Intent2JSON_LoRA/
├── Task2_RAG/
├── Task3_Local_LLM_Inference/
└── README.md
```

## Technologies

Python, PyTorch, Hugging Face Transformers, PEFT, LoRA, bitsandbytes, Sentence Transformers, NumPy, Pandas, Scikit-learn, BM25, and Google Colab.

## Limitations

- Task 1 uses a small synthetic dataset; its test accuracy does not establish real-world generalization.
- Task 2 uses 15 evaluation questions and manually selected relevance labels. Hit Rate@5 measures whether at least one relevant page appears in the top five. Two retrieval failures and their fixes are documented.
- Task 3 uses ten prompts and qualitative manual quality assessments. Measurements were collected on a Colab Tesla T4, not a production server.

## Author

Niharshini Muvva
