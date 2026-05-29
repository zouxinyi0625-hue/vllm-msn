#!/usr/bin/env python3
"""Quick check: print token length statistics of the sc1 dataset prompts.

Usage:
    export HF_TOKEN=...
    python check_prompt_lengths.py [--model Qwen/Qwen3-4B-Instruct-2507]
"""
import json
import statistics
import argparse
import os

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="datasets/sc1_delta_v2.jsonl")
    ap.add_argument("--num-prompts", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    lengths = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Render to text first, then tokenize (same as bench_offline.py)
            rendered_text = tok.apply_chat_template(
                [{"role": "user", "content": d["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            token_ids = tok(rendered_text, add_special_tokens=False).input_ids
            lengths.append(len(token_ids))
            if args.num_prompts and len(lengths) >= args.num_prompts:
                break

    lengths_sorted = sorted(lengths)
    print(f"Prompts: {len(lengths)}")
    print(f"Min:  {min(lengths)}")
    print(f"Max:  {max(lengths)}")
    print(f"Mean: {statistics.mean(lengths):.0f}")
    print(f"P50:  {lengths_sorted[len(lengths_sorted)//2]}")
    print(f"P90:  {lengths_sorted[int(len(lengths_sorted)*0.9)]}")
    print(f"P95:  {lengths_sorted[int(len(lengths_sorted)*0.95)]}")
    print(f"P99:  {lengths_sorted[int(len(lengths_sorted)*0.99)]}")

    print(f"\nSuggested max_model_len = P99 + output_len_cap(8192) = {lengths_sorted[int(len(lengths_sorted)*0.99)] + 8192}")


if __name__ == "__main__":
    main()
