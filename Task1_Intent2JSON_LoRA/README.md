# Task 1: Intent-to-JSON Extraction Using LoRA Fine-Tuning

## 1. Objective

Fine-tune a small open-weight language model to convert natural-language requests into structured JSON by identifying intent and extracting relevant information.

## 2. Model and Dataset

- Base model: Qwen2.5-0.5B-Instruct
- Fine-tuning method: LoRA using PEFT
- Dataset: 510 synthetic examples
- Training: 400 samples
- Validation: 50 samples
- Testing: 60 samples
- Random seed: 42

Six supported intents: restaurant booking, flight search, hotel booking, product search, weather query, and order tracking.

The output schema contains six fields: `intent`, `destination`, `date`, `time`, `party_size`, and `item`. Missing or inapplicable values are represented by JSON null.

## 3. LoRA Configuration

| Parameter | Value |
|---|---|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 0.0002 |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Optimizer | AdamW |

LoRA freezes the original model weights and trains a smaller set of additional parameters, reducing memory and computational requirements compared with full fine-tuning.

## 4. Evaluation Results

Both models were evaluated on the same 60 test examples.

| Metric | Base model | Fine-tuned model |
|---|---:|---:|
| JSON validity | 100% | 100% |
| Exact-match accuracy | 8.33% | 100% |
| Intent accuracy | 76.67% | 100% |
| Field-level accuracy | 52.78% | 100% |

The base model produced 55 incorrect predictions. The fine-tuned model produced zero incorrect predictions on this test set.

## 5. Training Loss

| Epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 0.01604 | 0.00226 |
| 2 | 0.00091 | 0.00168 |
| 3 | 0.00111 | 0.00366 |

Validation loss increased in the third epoch, suggesting some deterioration in validation performance at the end of training.

## 6. Base Model Failure Analysis

Three representative base-model failures:

1. **Incorrect tracking ID extraction:** For "Track my order ORD22280", the base model placed the ID in `destination` instead of `item`. The fine-tuned model correctly extracted it into `item`.
2. **Hallucinated date:** For "I need accommodation in Vijayawada for 3 guests", the base model generated "today" despite no date being specified. The fine-tuned model returned null.
3. **Incorrect intent classification:** For "Search for an air ticket to Chennai for this weekend", the base model generated the unsupported intent `air_ticket`. The fine-tuned model returned `flight_search`.

## 7. Estimated GPU Memory for a 7B Model

| Training approach | Estimated memory |
|---|---:|
| FP16 model weights only | 14 GB |
| Full fine-tuning | 112 GB |
| LoRA | 16–24 GB |
| QLoRA | 6–10 GB |

These are approximate estimates. Actual memory usage depends on sequence length, batch size, optimizer, activations, and implementation.

## 8. Conclusion and Limitations

LoRA fine-tuning substantially improved structured extraction on this experiment's test set, raising exact-match accuracy from 8.33% to 100%.

The dataset was synthetically generated and relatively small. The results do not establish generalization to diverse real-world requests. Further testing on larger and independently collected datasets is necessary.

## Project Files

- `Task1_Intent2JSON_LoRA.ipynb`: training and evaluation notebook
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: dataset splits
- `baseline_results.json`, `finetuned_results.json`: model predictions
- `baseline_metrics.json`, `finetuned_metrics.json`: evaluation metrics
- `training_history.json`: training and validation loss
- `experiment_summary.json`: experiment summary
- `model_comparison.csv`: comparison results
- `intent2json_lora_v2/`: trained LoRA adapter and tokenizer files
