#!/usr/bin/env python3
"""Convert EdgeRazor QAT-trained model (stored as bf16) to AWQ 4-bit for vLLM.

This script takes the dequantized bf16 model from EdgeRazor training and
re-quantizes it to AWQ INT4 format that vLLM can accelerate with Marlin/GEMM kernels.

Only requires: pip install autoawq (pre-built wheel, no CUDA compilation needed)

Usage:
    python convert_to_awq.py \
        --model Xinyi0625/train_maiprofile4b_w4 \
        --output ./Qwen3-4B-AWQ-Int4 \
        --group-size 256 \
        --dataset datasets/sc1_delta_v2.jsonl \
        --num-samples 128

After conversion, run vLLM with:
    python bench_offline.py --model-tag 4bit-awq --model ./Qwen3-4B-AWQ-Int4 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_calibration_data(dataset_path: str, tokenizer, num_samples: int,
                          seqlen: int = 2048) -> list[str]:
    """Load calibration texts and pre-tokenize to the format AWQ expects.

    AWQ internally concatenates all texts, tokenizes, then splits into seqlen chunks.
    We need to provide enough text to fill num_samples * seqlen tokens.
    """
    if dataset_path and Path(dataset_path).exists():
        texts = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                texts.append(d["prompt"])
                if len(texts) >= num_samples * 4:  # load extra to ensure enough tokens
                    break
        return texts
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return [t for t in ds["text"] if len(t.strip()) > 100][:num_samples * 4]


def main():
    ap = argparse.ArgumentParser(description="Convert bf16 model to AWQ 4-bit")
    ap.add_argument("--model", required=True,
                    help="HF model id or local path (bf16 EdgeRazor output)")
    ap.add_argument("--output", required=True, help="Output directory for AWQ model")
    ap.add_argument("--bits", type=int, default=4, choices=[4])
    ap.add_argument("--group-size", type=int, default=256,
                    help="Quantization group size (256 to match EdgeRazor w_block_size)")
    ap.add_argument("--dataset", default=None,
                    help="Calibration dataset (jsonl with 'prompt' field)")
    ap.add_argument("--num-samples", type=int, default=128,
                    help="Number of calibration samples")
    args = ap.parse_args()

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("ERROR: autoawq not installed. Run: pip install autoawq", file=sys.stderr)
        return 1

    from transformers import AutoTokenizer

    print(f"Loading model from {args.model}...", flush=True)
    model = AutoAWQForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading tokenizer from {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading calibration data...", flush=True)
    calib_data = load_calibration_data(args.dataset, tokenizer, args.num_samples)
    print(f"  Got {len(calib_data)} calibration texts", flush=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": "GEMM",
    }

    print(f"Quantizing to {args.bits}-bit AWQ (group_size={args.group_size})...", flush=True)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    print(f"Saving AWQ model to {args.output}...", flush=True)
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nDone! Model saved to: {args.output}", flush=True)
    print(f"\nTo benchmark with vLLM:", flush=True)
    print(f"  python bench_offline.py --scenario sc1 --model-tag 4bit-awq "
          f"--model {args.output} --max-num-seqs 512 --reps 1", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
