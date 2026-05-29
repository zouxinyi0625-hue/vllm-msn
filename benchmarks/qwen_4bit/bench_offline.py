#!/usr/bin/env python3
"""Offline throughput benchmark for Qwen3-4B baseline vs 4-bit quantized model.

Compares:
  - Baseline: Qwen/Qwen3-4B-Instruct-2507 (bf16)
  - 4-bit:    Xinyi0625/train_maiprofile4b_w4 (stored as bf16, 4-bit trained)

Uses the same sc1 dataset as gemma4_moe_fp8 benchmarks.

Usage:
  # Baseline (bf16)
  python bench_offline.py --scenario sc1 --model-tag baseline --max-num-seqs 128 --reps 1

  # 4-bit model
  python bench_offline.py --scenario sc1 --model-tag 4bit --max-num-seqs 128 --reps 1

Environment:
  HF_TOKEN must be set for downloading private models.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT_DIR = Path("bench_results")

MODELS = {
    "baseline": "Qwen/Qwen3-4B-Instruct-2507",
    "4bit": "Xinyi0625/train_maiprofile4b_w4",
}

CSV_FIELDS = [
    "ts", "scenario", "dataset", "model_tag", "model",
    "num_prompts", "output_len_cap",
    "max_model_len", "max_num_batched_tokens", "gpu_mem_util",
    "max_num_seqs", "rep", "seed",
    "chunk_index", "chunk_total",
    "elapsed_time",
    "requests_per_second",
    "prompt_tokens_total", "output_tokens_total", "total_tokens",
    "prompt_tps", "output_tps", "total_tps",
    "out_len_mean", "out_len_stdev", "out_len_p50", "out_len_p90", "out_len_max",
    "finish_stop", "finish_length", "finish_other",
]

SCENARIOS = {
    "sc1": dict(
        dataset="datasets/sc1_delta_v2.jsonl",
        num_prompts=1000,
        output_len=8192,
        max_model_len=16384,
        max_num_batched_tokens=16384,
        gpu_mem_util=0.95,
    ),
}


def ensure_csv_header(csv_path: Path):
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(csv_path: Path, row: dict):
    full = {k: row.get(k, "") for k in CSV_FIELDS}
    with csv_path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(full)


def load_prompts(dataset_path: str, n: int) -> list[str]:
    prompts = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            if n and len(prompts) >= n:
                break
    return prompts


def render_chat(tok, raw_prompts: list[str]) -> list[str]:
    out = []
    for p in raw_prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out.append(text)
    return out


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _aggregate_outputs(outputs):
    out_lens = []
    prompt_total = 0
    output_total = 0
    finish_counts = {"stop": 0, "length": 0, "other": 0}
    for o in outputs:
        prompt_total += len(o.prompt_token_ids)
        for comp in o.outputs:
            n_out = len(comp.token_ids)
            output_total += n_out
            out_lens.append(n_out)
            fr = (comp.finish_reason or "other").lower()
            if fr in finish_counts:
                finish_counts[fr] += 1
            else:
                finish_counts["other"] += 1
    return out_lens, prompt_total, output_total, finish_counts


def _row_from(*, scenario, cfg, model_tag, model, max_num_seqs, rep, seed,
              chunk_index, chunk_total, elapsed, n_prompts, prompt_total,
              output_total, out_lens, finish_counts):
    total = prompt_total + output_total
    out_lens_sorted = sorted(out_lens)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "dataset": cfg["dataset"],
        "model_tag": model_tag,
        "model": model,
        "num_prompts": n_prompts,
        "output_len_cap": cfg["output_len"],
        "max_model_len": cfg["max_model_len"],
        "max_num_batched_tokens": cfg["max_num_batched_tokens"],
        "gpu_mem_util": cfg["gpu_mem_util"],
        "max_num_seqs": max_num_seqs,
        "rep": rep,
        "seed": seed,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "elapsed_time": round(elapsed, 3),
        "requests_per_second": round(n_prompts / elapsed, 4) if elapsed > 0 else 0.0,
        "prompt_tokens_total": prompt_total,
        "output_tokens_total": output_total,
        "total_tokens": total,
        "prompt_tps": round(prompt_total / elapsed, 2) if elapsed > 0 else 0.0,
        "output_tps": round(output_total / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tps": round(total / elapsed, 2) if elapsed > 0 else 0.0,
        "out_len_mean": round(statistics.mean(out_lens), 2) if out_lens else None,
        "out_len_stdev": round(statistics.stdev(out_lens), 2) if len(out_lens) > 1 else 0.0,
        "out_len_p50": int(percentile(out_lens_sorted, 0.5)) if out_lens else None,
        "out_len_p90": int(percentile(out_lens_sorted, 0.9)) if out_lens else None,
        "out_len_max": max(out_lens) if out_lens else None,
        "finish_stop": finish_counts["stop"],
        "finish_length": finish_counts["length"],
        "finish_other": finish_counts["other"],
    }


def run_one_config(*, scenario: str, cfg: dict, model_tag: str, model: str,
                   max_num_seqs: int, reps: int, out_dir: Path, csv_path: Path,
                   chunk_size: int = 0):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n=== build LLM scenario={scenario} model_tag={model_tag} "
          f"max_num_seqs={max_num_seqs} ===", flush=True)
    print(f"    model: {model}", flush=True)

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    raw_prompts = load_prompts(cfg["dataset"], cfg["num_prompts"])
    prompts = render_chat(tok, raw_prompts)

    # Filter out prompts that exceed max_model_len (would OOM or error)
    max_len = cfg["max_model_len"]
    filtered = []
    for p in prompts:
        n_tok = len(tok(p, add_special_tokens=False).input_ids)
        if n_tok < max_len:
            filtered.append(p)
    if len(filtered) < len(prompts):
        print(f"filtered {len(prompts) - len(filtered)} prompts exceeding "
              f"max_model_len={max_len}, keeping {len(filtered)}", flush=True)
    prompts = filtered

    print(f"loaded {len(prompts)} prompts from {cfg['dataset']}", flush=True)

    llm_kwargs = dict(
        model=model,
        trust_remote_code=True,
        max_model_len=cfg["max_model_len"],
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        gpu_memory_utilization=cfg["gpu_mem_util"],
        seed=0,
    )

    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"engine built in {time.time()-t_engine:.1f} s", flush=True)

    rows = []
    for rep in range(1, reps + 1):
        seed = rep
        sampling = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=cfg["output_len"],
            seed=seed,
            ignore_eos=False,
        )

        if chunk_size and chunk_size > 0:
            chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]
        else:
            chunks = [prompts]
        chunk_total = len(chunks)

        for ci, chunk in enumerate(chunks, start=1):
            tag = f"{scenario}_{model_tag}_mns{max_num_seqs}_rep{rep}"
            if chunk_total > 1:
                tag = f"{tag}_chunk{ci:03d}of{chunk_total:03d}"
            print(f"\n--- RUN {tag}  prompts={len(chunk)}  seed={seed} ---", flush=True)
            t0 = time.time()
            outputs = llm.generate(chunk, sampling, use_tqdm=True)
            elapsed = time.time() - t0

            out_lens, prompt_total, output_total, finish_counts = _aggregate_outputs(outputs)
            row = _row_from(
                scenario=scenario, cfg=cfg, model_tag=model_tag, model=model,
                max_num_seqs=max_num_seqs, rep=rep, seed=seed,
                chunk_index=ci if chunk_total > 1 else "",
                chunk_total=chunk_total if chunk_total > 1 else "",
                elapsed=elapsed, n_prompts=len(chunk),
                prompt_total=prompt_total, output_total=output_total,
                out_lens=out_lens, finish_counts=finish_counts,
            )
            append_csv_row(csv_path, row)
            rows.append(row)

            per_run = out_dir / f"{tag}.json"
            with per_run.open("w") as f:
                json.dump({**row, "out_lens": out_lens}, f, indent=2)

            print(
                f"  elapsed={elapsed:.1f}s  req/s={row['requests_per_second']:.3f}  "
                f"out_tok/s={row['output_tps']:.0f}  total_tok/s={row['total_tps']:.0f}  "
                f"out_len(mean+/-sd)={row['out_len_mean']}+/-{row['out_len_stdev']}  "
                f"finish=stop:{finish_counts['stop']}/len:{finish_counts['length']}",
                flush=True,
            )

    summarize(scenario, model_tag, max_num_seqs, rows)

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def summarize(scenario: str, model_tag: str, mns: int, rows: list[dict]):
    if not rows:
        print(f"[SUMMARY] {scenario} {model_tag} mns={mns}: no successful runs", flush=True)
        return

    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    el_m, el_s = m("elapsed_time")
    r_m, r_s = m("requests_per_second")
    o_m, o_s = m("output_tps")
    t_m, t_s = m("total_tps")
    ol_m, ol_s = m("out_len_mean")
    n = len(rows)
    print(
        f"\n[SUMMARY] scenario={scenario}  model={model_tag}  max_num_seqs={mns}  reps={n}\n"
        f"  elapsed_time   : {el_m:.2f} +/- {el_s:.2f} s\n"
        f"  requests/sec   : {r_m:.4f} +/- {r_s:.4f}\n"
        f"  output tokens/s: {o_m:.2f} +/- {o_s:.2f}\n"
        f"  total tokens/s : {t_m:.2f} +/- {t_s:.2f}\n"
        f"  mean out_len   : {ol_m:.1f} +/- {ol_s:.1f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline throughput benchmark: Qwen3-4B baseline vs 4-bit")
    ap.add_argument("--scenario", default="sc1", choices=list(SCENARIOS.keys()))
    ap.add_argument("--model-tag", required=True, choices=list(MODELS.keys()),
                    help="'baseline' for Qwen3-4B-Instruct-2507, '4bit' for 4-bit model")
    ap.add_argument("--model", default=None,
                    help="Override model path (default: looked up from --model-tag)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-num-seqs", default="128",
                    help="Comma list, e.g. 64,128,256")
    ap.add_argument("--dataset", default=None,
                    help="Override the scenario's dataset path.")
    ap.add_argument("--num-prompts", type=int, default=None,
                    help="Override num_prompts. 0 = load entire dataset.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument("--gpu-mem-util", type=float, default=None,
                    help="Override gpu_memory_utilization.")
    args = ap.parse_args()

    # Ensure HF_TOKEN is set for private model access
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set. Private models will fail to download.",
              file=sys.stderr, flush=True)

    model = args.model or MODELS[args.model_tag]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_runs.csv"
    ensure_csv_header(csv_path)

    cfg = dict(SCENARIOS[args.scenario])
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.num_prompts is not None:
        cfg["num_prompts"] = args.num_prompts
    if args.gpu_mem_util is not None:
        cfg["gpu_mem_util"] = args.gpu_mem_util

    sweep = [int(x) for x in args.max_num_seqs.split(",")]
    failures = []
    for mns in sweep:
        try:
            run_one_config(
                scenario=args.scenario, cfg=cfg, model_tag=args.model_tag,
                model=model, max_num_seqs=mns, reps=args.reps,
                out_dir=out_dir, csv_path=csv_path, chunk_size=args.chunk_size)
        except Exception as e:
            print(f"!!! config max_num_seqs={mns} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            failures.append((mns, repr(e)))

    if failures:
        print(f"\nDone with {len(failures)} failure(s):", flush=True)
        for mns, err in failures:
            print(f"  - max_num_seqs={mns}: {err}", flush=True)
        return 1
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
