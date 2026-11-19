
# Task 3: Local LLM Inference and Quantization

## Objective
Compare FP16 and 4-bit quantized inference using the same
model, prompts, generation settings and GPU environment.

## Model and environment
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Runtime: Hugging Face Transformers
- Quantization: bitsandbytes NF4, double quantization enabled
- GPU: Tesla T4
- Evaluation prompts: 10
- Maximum generated tokens: 128
- Sampling: Disabled (greedy decoding)

## Benchmark results

| Metric | FP16 | 4-bit NF4 |
|---|---:|---:|
| Model loading time (seconds) | 82.46 | 14.43 |
| Generation speed (tokens/sec) | 26.51 | 12.83 |
| Peak allocated GPU memory (GB) | 2.89 | 1.092 |
| Peak reserved GPU memory (GB) | 2.934 | 2.918 |

Generation speed is calculated using total generated tokens divided
by total generation time across the ten prompts.

Memory figures are PyTorch GPU allocation measurements and include
the loaded model during inference.

## Quality evaluation

The same ten prompts were evaluated under both configurations
using manual comparison of the generated responses.

| Prompt | Assessment |
|---|---|
| 1. Machine learning | Equivalent |
| 2. AI vs deep learning | Equivalent |
| 3. Prime-checking function | Equivalent |
| 4. REST APIs | 4-bit better |
| 5. Database normalization | FP16 better |
| 6. Overfitting | Equivalent |
| 7. Cloud computing | 4-bit better |
| 8. Binary search | Equivalent |
| 9. AI deployment challenges | 4-bit better |
| 10. Cybersecurity | Equivalent |

Overall, six prompts were assessed as equivalent, three favored
4-bit NF4, and one favored FP16.

These are qualitative judgments based on manual inspection,
not an automated quality score. Responses were limited to
128 generated tokens, so some answers were truncated.
The comparison therefore reflects the visible outputs rather
than complete responses.


## Quantization
FP16 represents model weights using 16-bit floating-point values.
4-bit NF4 stores weights using a lower-precision representation,
reducing model memory requirements.

This can improve memory efficiency and allow larger models to fit
on limited hardware, with a possible trade-off in model quality.

## KV cache
The key-value cache stores attention keys and values from previous
tokens during autoregressive generation. It avoids recalculating
them for every newly generated token.

The cache consumes additional memory, increasing with context
length, concurrent requests and model architecture.

## Deployment plan: 20 concurrent users on a 24GB GPU
- Serve a quantized model using an inference engine such as vLLM.
- Use continuous batching to combine active requests.
- Use PagedAttention or equivalent KV-cache management.
- Apply request queues, concurrency limits and rate limits.
- Set context-length and maximum-generation limits.
- Monitor GPU memory, throughput and latency.
- Load-test with realistic traffic before production deployment.
- Scale to additional GPUs if the measured capacity is insufficient.

For the Qwen2.5-1.5B architecture, assuming an FP16 KV cache,
28 layers, 2 KV heads and 128-dimensional KV heads, 20 users
with 2,048 cached tokens each require approximately
1.09 GiB of KV cache.

This is a theoretical estimate, not a measured concurrent-user test.
The production memory budget must also account for model weights,
runtime overhead, temporary buffers and batching.

## Limitations
- Benchmark conducted on a Colab GPU, not a production 24GB GPU.
- No actual 20-user load test was performed.
- Evaluation used only ten prompts.
- Quality assessment is manual rather than an automated accuracy metric.
