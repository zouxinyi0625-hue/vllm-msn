#!/usr/bin/env python3
"""Convert EdgeRazor QAT-trained model (stored as bf16) to GPTQ 4-bit for vLLM.

This script takes the dequantized bf16 model from EdgeRazor training and
re-quantizes it to GPTQ INT4 format that vLLM can accelerate with Marlin kernels.

Usage:
    pip install auto-gptq transformers torch

    python convert_to_gptq.py \
        --model Xinyi0625/train_maiprofile4b_w4 \
        --output ./Qwen3-4B-GPTQ-Int4 \
        --bits 4 \
        --group-size 128 \
        --dataset datasets/sc1_delta_v2.jsonl \
        --num-samples 128

After conversion, run vLLM with:
    python bench_offline.py --model-tag 4bit-gptq --model ./Qwen3-4B-GPTQ-Int4 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_calibration_data(dataset_path: str, tokenizer, num_samples: int,
                          max_length: int = 2048) -> list:
    """Load calibration data from the benchmark dataset."""
    from auto_gptq.utils import get_calib_dataset

    # If dataset is provided, use it; otherwise fall back to auto_gptq default
    if dataset_path and Path(dataset_path).exists():
        texts = []
        with open(dataset_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                texts.append(d["prompt"])

        # Tokenize
        examples = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            examples.append({"input_ids": encoded["input_ids"][0],
                             "attention_mask": encoded["attention_mask"][0]})
        return examples
    else:
        # Use default wikitext2 calibration
        return get_calib_dataset(
            "wikitext", tokenizer=tokenizer,
            n_samples=num_samples, max_length=max_length,
        )


def main():
    ap = argparse.ArgumentParser(description="Convert bf16 model to GPTQ 4-bit")
    ap.add_argument("--model", required=True,
                    help="HF model id or local path (bf16 EdgeRazor output)")
    ap.add_argument("--output", required=True, help="Output directory for GPTQ model")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    ap.add_argument("--group-size", type=int, default=256,
                    help="Quantization group size (256 to match EdgeRazor w_block_size)")
    ap.add_argument("--dataset", default=None,
                    help="Calibration dataset (jsonl with 'prompt' field)")
    ap.add_argument("--num-samples", type=int, default=128,
                    help="Number of calibration samples")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Max sequence length for calibration")
    ap.add_argument("--desc-act", action="store_true",
                    help="Use descending activation order (slower but sometimes better)")
    args = ap.parse_args()

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("ERROR: auto-gptq not installed. Run: pip install auto-gptq", file=sys.stderr)
        return 1

    print(f"Loading tokenizer from {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model from {args.model} (bf16)...", flush=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        BaseQuantizeConfig(
            bits=args.bits,
            group_size=args.group_size,
            desc_act=args.desc_act,
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading calibration data ({args.num_samples} samples)...", flush=True)
    calib_data = load_calibration_data(
        args.dataset, tokenizer, args.num_samples, args.max_length
    )

    print(f"Quantizing to {args.bits}-bit (group_size={args.group_size})...", flush=True)
    model.quantize(calib_data)

    print(f"Saving GPTQ model to {args.output}...", flush=True)
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nDone! Model saved to: {args.output}", flush=True)
    print(f"\nTo benchmark with vLLM:", flush=True)
    print(f"  python bench_offline.py --scenario sc1 --model-tag 4bit-gptq "
          f"--model {args.output} --max-num-seqs 512 --reps 1", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
