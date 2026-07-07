# Gemma4 12B FP8 Benchmark Scaffold

This directory is the starting point for Gemma4 12B dense FP8 benchmarking.

For the first round, it intentionally keeps the **current 26B Gemma4 MoE E011 alignment configs** so the server can reproduce the known 26B numbers before switching to 12B dense.

## Scope

- Reuse the existing production-shaped dataset from `../gemma4_moe_fp8/datasets/sc1_delta_v2.jsonl`.
- Keep two 26B reference configs:
  - `26b_e011_mtp`: E011, FP8 + CUDA graphs + MTP k=5 + text-only model.
  - `26b_e011_no_mtp`: same as E011 but speculative decoding disabled.
- Provide both offline and online/server scripts.
- Do **not** change the dataset or sampling settings during alignment.

## Expected 26B reference numbers

From `../gemma4_moe_fp8/RESULTS.md`:

| Environment | Config | Output tok/s | Notes |
|---|---|---:|---|
| A100 80GB offline | E011 MTP | ~2020 | `gpu_memory_utilization=0.95` |
| A100 80GB simulated 40GB offline | E011 MTP | ~1255 | `gpu_memory_utilization=0.45` |
| A100 40GB DLIS online | E011 MTP | ~1293 | max concurrency 32 |
| A100 80GB offline | E011 no MTP | ~1465 | E012 isolate, no speculative decoding |

Use these as sanity checks before running 12B dense experiments.

## Directory layout

```text
gemma4_12b_fp8/
├── README.md
├── configs/
│   ├── 26b_e011_mtp.json
│   └── 26b_e011_no_mtp.json
├── run_offline_align.sh
├── bench_offline_align.py
├── serve_align.sh
└── bench_online_align.sh
```

## Dataset prerequisite

This scaffold deliberately reuses the old dataset path:

```text
benchmarks/gemma4_moe_fp8/datasets/sc1_delta_v2.jsonl
```

The dataset is not committed in this clone. On the GPU server, restore or generate the same file before running. The previous result file documents the source as:

```text
https://www.cosmos09.osdinfra.net/cosmos/MSN.DnI/shares/users/zxy/data/mai_profile/sc1_delta_v2.jsonl?property=info
```

Keep the file content unchanged for the first alignment run.

## Offline alignment

Run E011 with MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash run_offline_align.sh configs/26b_e011_mtp.json --reps 2 --prompts 1000
```

Run E011 without MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash run_offline_align.sh configs/26b_e011_no_mtp.json --reps 2 --prompts 1000
```

Outputs are written under:

```text
benchmarks/gemma4_12b_fp8/offline_results/
```

## Online alignment

Start server with E011 MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash serve_align.sh configs/26b_e011_mtp.json 8100
```

In another shell, benchmark the server:

```bash
cd benchmarks/gemma4_12b_fp8
bash bench_online_align.sh --port 8100 --config configs/26b_e011_mtp.json --num-prompts 1000 --max-concurrency 32
```

Start server without MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash serve_align.sh configs/26b_e011_no_mtp.json 8100
```

Then benchmark similarly:

```bash
bash bench_online_align.sh --port 8100 --config configs/26b_e011_no_mtp.json --num-prompts 1000 --max-concurrency 32
```

## Model path overrides

The config files contain Hugging Face defaults, but server runs usually use local model mounts.
Override with environment variables when needed:

```bash
export GEMMA4_26B_TEXT_ONLY_MODEL_PATH=/path/to/gemma4/text_only
export GEMMA4_26B_ASSISTANT_MODEL_PATH=/path/to/gemma4/assistant
```

For Azure/DLIS-style model mounts, `serve_align.sh` also looks for:

```text
${_ModelDataPath_}/text_only
${_ModelDataPath_}/assistant
```

## Next step for 12B dense

After the 26B alignment is reproduced, add 12B dense configs beside the 26B configs, keeping dataset/sampling fixed first. Only change the model path and model-specific options initially so the result is comparable.
