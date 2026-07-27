# Gemma4 MTP layer1_delta — trained draft vs baseline (online benchmark)

**Status: root cause IDENTIFIED (pending probe confirmation).** The trained
draft deploys at ~half the official baseline's acceptance. Evidence points to a
**train/deploy hidden-state mismatch**: training feeds the draft POST-norm target
hidden, vLLM deployment feeds PRE-norm.

## Master table

| run | assistant | quant | accept_rate% | accept_len | pos0 | pos1 | pos2 | pos3 | pos4 | out tok/s |
|-----|-----------|-------|-------------|-----------|------|------|------|------|------|-----------|
| **baseline** (official assistant) | `models/assistant` | fp8 | **65.57** | **4.28** | 86.26 | 75.61 | 64.36 | 54.88 | 46.73 | 1154.1 |
| **trained** (hf_iter_0002269) | `mtp_26b/gemma4_mtp_exp1_hf/hf_iter_0002269` | fp8 | 33.02 | 2.65 | 42.66 | 36.82 | 32.60 | 28.38 | 24.67 | 848.2 |
| **trained** (hf_iter_0002269) | same | **bf16** | 33.50 | 2.67 | 43.03 | 37.20 | 32.90 | 29.03 | 25.34 | 756.6 |

Bench: `run_maiprofile_online.sh`, layer1_delta eval (200 prompts), 26b_e011_mtp
target `models/text_only`, spec_tokens=5, same server/temperature per column.

## What the numbers say

- The trained draft is HALVED at EVERY position vs baseline (uniform ~×0.5),
  including **pos0** — the first drafted token, which has no prefix dependency
  and is NOT affected by teacher-forcing. A uniform per-position halving that
  includes pos0 = **systematic input mismatch**, not a model-quality gap.
- **The draft is NOT bad**: its TRAINING-side eval reported `avg_acc=0.91`
  (pos0≈0.91), which MATCHES the baseline's *deployed* pos0 (0.863). So the
  trained draft is as good as the official one in training — it just isn't fed
  the same thing at deploy time.
- **FP8 is NOT the cause**: bf16 (33.50) ≈ fp8 (33.02). Quantization ruled out.
- **Weight loading is NOT the cause**: server log shows no missing/unexpected
  keys, 0.78 GiB draft loaded, 4 draft layers mapped to target L28/L29.
- **embed_scale is NOT the cause**: vLLM gemma4_mtp normalizer = sqrt(2816) =
  53.07, identical to the training-side scale.

## Root cause (from source, pending live probe)

`torchspec/config/inference_config.py::resolve_last_hidden_states_prenorm`:

```python
"""vLLM's extract_hidden_states connector can only capture raw layer outputs
(pre-norm), while sglang and hf provide post-norm outputs."""
if self.last_hidden_states_prenorm is not None:
    return self.last_hidden_states_prenorm
return self.inference_engine_type == "vllm"
```

- Training config: `last_hidden_states_prenorm: null`, `inference_engine_type: hf`
  -> resolves to `False` -> **POST-norm** hidden (confirmed by
  `gemma4_mtp_target.py:175 last_hidden = out.last_hidden_state`, which for HF
  Gemma4 is after `model.norm`).
- vLLM deployment feeds **PRE-norm** hidden (raw last-layer output).
- => draft trained on RMSNorm'd (~O(1)) hidden but served un-normalized hidden
  -> `pre_projection` sees OOD input -> uniform accept halving including pos0.

The official assistant is unaffected because it was trained against vLLM's
pre-norm convention.

## Confirmation probe

`tools/gemma4_mtp/probe_prenorm_gap.py` (in the TorchSpec repo): loads the target,
runs one real prompt, prints mean per-token L2 norm of PRE-norm vs POST-norm
hidden. A large ratio confirms the gap.

## Fixes (decide after probe)

- **A (clean, retrain):** set training `inference.last_hidden_states_prenorm: true`
  so the HF engine emits pre-norm hidden matching vLLM; retrain. Produces a draft
  compatible with stock vLLM (no patch needed).
- **B (no retrain, patch vLLM):** apply the target's `model.norm` to
  `hidden_states` inside vLLM `gemma4_mtp.py` before `pre_projection`, converting
  the served pre-norm hidden to the post-norm the draft was trained on. Needs
  loading the target's 2816-dim final-norm weight into the MTP module. Faster to
  test the hypothesis, but binds the draft to a patched vLLM.
