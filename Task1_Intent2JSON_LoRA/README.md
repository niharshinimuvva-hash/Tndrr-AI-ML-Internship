# Task 1: Intent-to-JSON Extraction Using LoRA

## 1. Objective

Fine-tune a small instruction-following language model to convert natural-language requests into structured JSON using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

The system supports six intents: restaurant booking, flight search, hotel booking, product search, weather query, and order tracking.

## 2. Model and Dataset

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Dataset: 510 synthetically generated examples
- Training examples: 400
- Validation examples: 50
- Test examples: 60
- Training framework: Hugging Face Transformers and PEFT
- Fine-tuning method: LoRA
- Runtime: Google Colab with NVIDIA T4 GPU

Each example contains a natural-language request and a JSON output with six fields:

- intent
- destination
- date
- time
- party_size
- item

Inapplicable fields are represented using JSON null. The party_size field is an integer or null.

## 3. LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Bias | none |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Optimizer | AdamW |
| Maximum sequence length | 256 |
| Gradient clipping | 1.0 |

Only 4,399,104 parameters out of 498,431,872 were trainable, approximately 0.88% of the model parameters.

LoRA was selected to reduce the number of trainable parameters and training cost while adapting the model to the structured extraction task.

## 4. Training Results

| Epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 0.01604 | 0.00226 |
| 2 | 0.00091 | 0.00168 |
| 3 | 0.00111 | 0.00366 |

## 5. Base Model vs Fine-Tuned Model

Both models were evaluated on the same 60-example test set.

| Metric | Base model | Fine-tuned model |
|---|---:|---:|
| JSON validity | 100.00% | 100.00% |
| Exact-match accuracy | 8.33% | 100.00% |
| Intent accuracy | 76.67% | 100.00% |
| Field accuracy | 52.78% | 100.00% |

The fine-tuned model substantially improved structured extraction accuracy on the held-out synthetic test set.

## 6. Base Model Error Analysis

The following examples illustrate errors made by the base model and corrected by the fine-tuned model.

### Example 1: Order tracking

Request: Track my order ORD22280.

Base model error: Placed the order ID in destination and left item empty.

Fine-tuned output: Correctly placed ORD22280 in item.

### Example 2: Missing date

Request: I need accommodation in Vijayawada for 3 guests.

Base model error: Invented the date today.

Fine-tuned output: Correctly used null for the unspecified date.

### Example 3: Intent classification

Request: Search for an air ticket to Chennai for this weekend.

Base model error: Generated an unsupported intent label.

Fine-tuned output: Correctly classified the request as flight_search.

## 7. Additional Unseen Evaluation

Three additional manually written requests were evaluated after training. They covered a named restaurant reservation, order tracking, and hotel search.

Result: 3 out of 3 exact matches.

These examples were separate from the original 60-example test set.

## 8. Adversarial Evaluation

A further set of 10 manually written, more challenging requests was used to examine generalization beyond the synthetic training patterns. The examples included corrections, multiple locations, missing details, tracking references, and time preferences.

| Metric | Result |
|---|---:|
| Total requests | 10 |
| Exact matches | 7 |
| Exact-match accuracy | 70.00% |
| Field accuracy | 95.00% |
| Valid JSON outputs | 10/10 |

### Observed Fine-Tuned Model Failures

#### Failure 1: Departure and destination confusion

Request: Find me a flight from Hyderabad to Delhi on December 18.

Expected destination: Delhi.

Predicted destination: Hyderabad.

The model selected the departure city instead of the arrival city.

#### Failure 2: Date wording was changed

Request: Can you get me a table for six at Spice Garden, located in Guntur, at 8:30 tonight?

Expected date: tonight.

Predicted date: today.

The model changed the user's date wording despite the instruction to preserve it.

#### Failure 3: Time preference was omitted

Request: I need a flight to Mumbai, preferably in the evening, but I haven't chosen a travel date.

Expected time: evening.

Predicted time: null.

The training dataset did not contain examples with morning, afternoon, or evening time-of-day preferences. This indicates a coverage gap for expressions not represented in training.

## 9. 7B Model Memory Estimates

Approximate memory requirements for a 7-billion-parameter model:

| Training approach | Estimated GPU memory |
|---|---:|
| FP16 weights for inference | 14 GB |
| Full fine-tuning | 112 GB |
| LoRA | 16–24 GB |
| QLoRA | 6–10 GB |

These are approximate planning estimates, not measurements from this experiment. Actual requirements depend on optimizer states, gradients, activations, sequence length, batch size, and implementation.

## 10. Limitations and Future Improvements

- The dataset is small and synthetically generated, so it may not represent the diversity of real user requests.
- The original test set achieved 100% exact-match accuracy, but the adversarial evaluation achieved 70%. This demonstrates that performance on synthetic test examples does not guarantee robust generalization.
- The model confused departure and arrival cities in a flight request.
- Date expressions and time-of-day preferences require broader training coverage.
- Future work should include more diverse real-world phrasing, multi-location requests, corrections, ambiguous inputs, and additional adversarial examples.
- Further evaluation should use a larger independently created dataset.

## 11. 7B Memory Estimate Discussion

Full fine-tuning requires memory for model weights, gradients, optimizer states, and activations. LoRA reduces trainable parameters by freezing the base model and updating adapter weights. QLoRA further reduces base-model memory by using quantized weights, making fine-tuning more accessible on limited GPU hardware.

## 12. Files

- Task1_Intent2JSON_LoRA.ipynb: training and evaluation notebook.
- train.jsonl: training dataset.
- validation.jsonl: validation dataset.
- test.jsonl: held-out test dataset.
- baseline_results.json and baseline_metrics.json: base model evaluation.
- finetuned_results.json and finetuned_metrics.json: fine-tuned model evaluation.
- training_history.json: training and validation loss history.
- model_comparison.csv: model comparison results.
- experiment_summary.json: experiment configuration and summary.
- intent2json_lora_v2/: saved LoRA adapter and tokenizer files.

## Conclusion

LoRA fine-tuning improved exact-match accuracy from 8.33% to 100% on the original synthetic test set. Additional adversarial testing exposed three genuine extraction weaknesses, demonstrating the importance of evaluating beyond the training data's patterns.
