"""Parameter space definition for Gemma4 MoE serving optimization on A100 80GB.

Fixed params (always use, not searchable):
  - quantization: "fp8"
  - enforce_eager: False (CUDA graphs on)
  - gpu_memory_utilization: 0.95
  - kv_cache_dtype: "auto"

Searchable params: MoE kernel config, batch/scheduling, MTP depth.
"""
from __future__ import annotations

import random

# --- Searchable parameter space ---
PARAM_SPACE = {
    "max_num_seqs": [64, 96, 128, 192, 256],
    "max_num_batched_tokens": [8192, 16384, 24576, 32768],
    "max_model_len": [16384, 24576],
    "spec_tokens": [3, 5, 7, 9],
    "VLLM_MOE_USE_DEEP_GEMM": ["0", "1"],
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK": ["0", "1"],
    "VLLM_FLASHINFER_MOE_BACKEND": ["throughput", "latency"],
    "VLLM_HUMMING_MOE_GEMM_TYPE": ["indexed", "grouped", "auto"],
}

# --- Fixed params (always applied) ---
FIXED_PARAMS = {
    "quantization": "fp8",
    "kv_cache_dtype": "auto",
    "enforce_eager": False,
    "gpu_memory_utilization": 0.95,
}

# Parameters that are env vars (set before vllm import)
ENV_VAR_PARAMS = {
    "VLLM_MOE_USE_DEEP_GEMM",
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
    "VLLM_FLASHINFER_MOE_BACKEND",
    "VLLM_HUMMING_MOE_GEMM_TYPE",
}

# Fixed env vars for A100 80GB (sm_80)
A100_FIXED_ENV = {
    "VLLM_USE_FLASHINFER_MOE_FP8": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
}

# E011 baseline config (current best: 2020 output_tps)
BASELINE_CONFIG = {
    **FIXED_PARAMS,
    "max_num_seqs": 128,
    "max_num_batched_tokens": 16384,
    "max_model_len": 24576,
    "spec_tokens": 5,
    "VLLM_MOE_USE_DEEP_GEMM": "1",
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK": "1",
    "VLLM_FLASHINFER_MOE_BACKEND": "throughput",
    "VLLM_HUMMING_MOE_GEMM_TYPE": "auto",
}


def validate_config(cfg: dict) -> tuple[bool, str]:
    """Validate a config dict. Returns (is_valid, error_message)."""
    if cfg.get("kv_cache_dtype") == "fp8_e4m3":
        return False, "fp8_e4m3 KV cache not supported on A100 (sm_80)"

    mns = cfg.get("max_num_seqs", 128)
    mnbt = cfg.get("max_num_batched_tokens", 16384)
    if mnbt < mns:
        return False, f"max_num_batched_tokens ({mnbt}) < max_num_seqs ({mns})"

    mml = cfg.get("max_model_len", 24576)
    if mml < 8192:
        return False, f"max_model_len ({mml}) too small for 8192-token outputs"

    spec = cfg.get("spec_tokens", 0)
    if spec > 0 and cfg.get("quantization") is None:
        return False, "MTP requires quantization=fp8 (assistant model is FP8)"

    return True, ""


def config_to_env_vars(cfg: dict) -> dict[str, str]:
    """Extract env var settings from config."""
    env = dict(A100_FIXED_ENV)
    for key in ENV_VAR_PARAMS:
        if key in cfg:
            env[key] = str(cfg[key])
    return env


def config_to_llm_kwargs(cfg: dict, scenario_cfg: dict) -> dict:
    """Convert config dict to vllm.LLM constructor kwargs."""
    kwargs = {
        "trust_remote_code": True,
        "max_model_len": cfg.get("max_model_len", 24576),
        "max_num_seqs": cfg.get("max_num_seqs", 128),
        "max_num_batched_tokens": cfg.get("max_num_batched_tokens",
                                          scenario_cfg.get("max_num_batched_tokens", 16384)),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.95),
        "enforce_eager": cfg.get("enforce_eager", False),
        "seed": 0,
    }
    if cfg.get("quantization"):
        kwargs["quantization"] = cfg["quantization"]
    if cfg.get("kv_cache_dtype", "auto") != "auto":
        kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    return kwargs


def config_summary(cfg: dict) -> str:
    """One-line summary string for results.tsv."""
    parts = []
    q = cfg.get("quantization", "bf16") or "bf16"
    parts.append(q)
    if not cfg.get("enforce_eager", True):
        parts.append("CG")
    spec = cfg.get("spec_tokens", 0)
    if spec > 0:
        parts.append(f"MTP-k{spec}")
    parts.append(f"mns={cfg.get('max_num_seqs', 128)}")
    parts.append(f"mnbt={cfg.get('max_num_batched_tokens', 16384)}")
    mml = cfg.get("max_model_len", 24576)
    if mml != 24576:
        parts.append(f"mml={mml}")
    dg = cfg.get("VLLM_MOE_USE_DEEP_GEMM", "1")
    if dg == "0":
        parts.append("no-DeepGEMM")
    ftk = cfg.get("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1")
    if ftk == "0":
        parts.append("no-FusedTopK")
    backend = cfg.get("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    if backend != "throughput":
        parts.append(f"moe-{backend}")
    humming = cfg.get("VLLM_HUMMING_MOE_GEMM_TYPE", "auto")
    if humming != "auto":
        parts.append(f"humming-{humming}")
    return " | ".join(parts)


def generate_next_configs(history: list[dict], n: int = 4) -> list[dict]:
    """Generate the next batch of configs to try.

    Strategy:
    - If history is empty or small: start with systematic one-at-a-time variations from baseline
    - Otherwise: perturb the best-known config in unexplored directions
    """
    if not history:
        return _initial_exploration(n)

    valid_runs = [h for h in history if h.get("status") == "ok" and h.get("output_tps", 0) > 0]
    if not valid_runs:
        return _initial_exploration(n)

    best = max(valid_runs, key=lambda h: h["output_tps"])
    best_cfg = best.get("config", BASELINE_CONFIG)

    tried_summaries = {h.get("config_summary", "") for h in history}
    candidates = []

    for param, values in PARAM_SPACE.items():
        for val in values:
            if val == best_cfg.get(param):
                continue
            candidate = {**best_cfg, **FIXED_PARAMS, param: val}
            valid, _ = validate_config(candidate)
            if valid and config_summary(candidate) not in tried_summaries:
                candidates.append(candidate)

    if len(candidates) <= n:
        return candidates

    random.shuffle(candidates)
    return candidates[:n]


def _initial_exploration(n: int) -> list[dict]:
    """First round: vary one parameter at a time from baseline."""
    configs = []
    variations = [
        {"VLLM_MOE_USE_DEEP_GEMM": "0"},
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "grouped"},
        {"VLLM_FLASHINFER_MOE_BACKEND": "latency"},
        {"spec_tokens": 7},
        {"max_num_seqs": 192},
        {"max_num_seqs": 96},
        {"max_num_batched_tokens": 24576},
        {"VLLM_USE_FUSED_MOE_GROUPED_TOPK": "0"},
        {"max_model_len": 16384},
        {"spec_tokens": 9},
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "indexed"},
        {"max_num_batched_tokens": 32768},
    ]
    for v in variations[:n]:
        cfg = {**BASELINE_CONFIG, **v}
        valid, _ = validate_config(cfg)
        if valid:
            configs.append(cfg)
    return configs
