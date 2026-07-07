# Gemma4 12B FP8 Benchmark Scaffold — 26B Alignment Results

This file records the 26B Gemma4 MoE alignment runs used to validate the new `benchmarks/gemma4_12b_fp8/` scaffold before switching to 12B dense FP8 experiments.

## Environment

- Date: 2026-07-07
- Branch: `feat/gemma4-12b-fp8-bench`
- Runtime image note:
  - 26B alignment originally worked on `vllm_gemma4:3`.
  - 12B dense runs require the newer `vllm_gemma4:6` image, based on `vllm/vllm-openai:gemma4-unified`.
  - `vllm_gemma4:3` cannot run the 12B checkpoints because they use newer `gemma4_unified` / `gemma4_unified_assistant` configs.
- Server path used by user:
  ```text
  /scratch/azureml/cr/j/574c7ade511e4bb08b1db2772cdac624/exe/wd/vllm-msn/benchmarks/gemma4_12b_fp8
  ```
- Model paths:
  ```bash
  export GEMMA4_26B_TEXT_ONLY_MODEL_PATH=/tmp/models/gemma4/text_only
  export GEMMA4_26B_ASSISTANT_MODEL_PATH=/tmp/models/gemma4/assistant
  export _ModelDataPath_=/tmp/models/gemma4
  ```
- Dataset:
  ```text
  ../gemma4_moe_fp8/datasets/sc1_delta_v2.jsonl
  ```
- Dataset shape from run output:
  - Requests: 1000
  - Total input tokens: 4,371,285
  - Avg input tokens/request: ~4,371

## Configs

### `26b_e011_mtp`

- Model: `/tmp/models/gemma4/text_only`
- Assistant: `/tmp/models/gemma4/assistant`
- FP8 weights
- CUDA graphs (`enforce_eager=false`)
- MTP speculative decoding with `spec_tokens=5`
- `max_model_len=24576`
- `max_num_seqs=128`
- `max_num_batched_tokens=16384`
- `gpu_memory_utilization=0.95`
- `async_scheduling=true`
- Prefix caching enabled

### `26b_e011_no_mtp`

Same as `26b_e011_mtp`, but speculative decoding disabled:

```json
"assistant_model": "",
"spec_tokens": 0
```

## Offline Results

Commands:

```bash
bash run_offline_align.sh configs/26b_e011_mtp.json --reps 1 --prompts 1000
bash run_offline_align.sh configs/26b_e011_no_mtp.json --reps 1 --prompts 1000
```

Output files on server:

```text
offline_results/26b_e011_mtp_20260707_060751.json
offline_results/26b_e011_no_mtp_20260707_062601.json
```

| Config | Reps | Output tok/s | Total tok/s | Notes |
|---|---:|---:|---:|---|
| `26b_e011_mtp` | 1 | **2023.06** | 8723.79 | Matches previous E011 baseline (~2020) |
| `26b_e011_no_mtp` | 1 | **1474.46** | 6430.13 | Matches previous E012/no-MTP baseline (~1465) |

### Offline alignment conclusion

Both offline alignment points reproduce the previous 26B results within normal run-to-run noise. This validates:

1. Dataset path/content is aligned.
2. `text_only` model path is correct.
3. MTP assistant path is correct.
4. E011 config is faithfully reproduced by the new scaffold.
5. No-MTP baseline is available for future 12B dense comparison.

## Online Result — MTP, A100 80GB, Unlimited Concurrency

The user requested an online MTP vLLM bench with unlimited client concurrency on the 80GB setup.

Server startup:

```bash
bash serve_align.sh configs/26b_e011_mtp.json 8100
```

Bench command semantics:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8100 \
  --endpoint /v1/chat/completions \
  --model gemma4 \
  --tokenizer google/gemma-4-26B-A4B-it \
  --dataset-name custom \
  --dataset-path ../gemma4_moe_fp8/datasets/sc1_delta_v2.jsonl \
  --num-prompts 1000 \
  --output-len 8192 \
  --request-rate inf
```

Note: `--max-concurrency` was intentionally omitted, so vLLM reports:

```text
Maximum request concurrency: None
Peak concurrent requests: 1000
```

### Summary table

| Metric | Value |
|---|---:|
| Successful requests | 1000 |
| Failed requests | 0 |
| Benchmark duration (s) | 675.55 |
| Request throughput (req/s) | 1.48 |
| Total input tokens | 4,371,285 |
| Total generated tokens | 1,427,960 |
| Output token throughput (tok/s) | **2113.78** |
| Peak output token throughput (tok/s) | 1025.00 |
| Total token throughput (tok/s) | **8584.49** |
| Peak concurrent requests | 1000 |

### Latency

| Metric | Mean | Median | P99 |
|---|---:|---:|---:|
| TTFT (ms) | 313,680.33 | 318,780.79 | 617,809.77 |
| TPOT excl. first token (ms) | 58.09 | 56.45 | 122.39 |
| ITL (ms) | 270.74 | 155.00 | 1,155.56 |

High TTFT is expected for unlimited concurrency because all 1000 requests enter the queue at once. The throughput result is the key signal for this run.

### Speculative decoding metrics

| Metric | Value |
|---|---:|
| Acceptance rate (%) | 80.58 |
| Acceptance length | 5.03 |
| Drafts | 284,215 |
| Draft tokens | 1,421,075 |
| Accepted tokens | 1,145,159 |

Per-position acceptance:

| Draft position | Acceptance (%) |
|---:|---:|
| 0 | 93.61 |
| 1 | 87.08 |
| 2 | 80.45 |
| 3 | 74.23 |
| 4 | 67.54 |

## Alignment vs Previous Results

Previous `gemma4_moe_fp8/RESULTS.md` headline numbers:

| Mode | Historical output tok/s | New scaffold output tok/s | Status |
|---|---:|---:|---|
| Offline E011 MTP, A100 80GB | ~2020 | 2023.06 | Reproduced |
| Offline no-MTP, A100 80GB | ~1465 | 1474.46 | Reproduced |
| Online E011 MTP, A100 80GB unlimited | Not previously recorded in this exact 80GB/unlimited form | 2113.78 | New alignment point |

The online 80GB unlimited throughput is slightly above offline E011 in output tok/s, while total tok/s is close. This is acceptable as an online serving alignment point; latency metrics should be interpreted with the unlimited-concurrency queueing behavior in mind.

## Next Step

Add 12B dense FP8 configs beside the 26B alignment configs. Keep dataset, sampling, prompt count, and measurement scripts unchanged for the first 12B run so differences are attributable to the model/config rather than benchmark drift.
