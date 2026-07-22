#!/usr/bin/env python3
"""Convert the NEW MAI Profile eval split (regen/split_regen output) into the
per-layer benchmark prompt files that run_maiprofile_online.sh consumes.

The training pipeline (gemma4-mtp-trainer/split_regen.py) writes ONE mixed file
  eval_maiprofile_26b.jsonl
with rows:
  {"conversations": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}],
   "id": "<layer>:<hash>", "source_layer": "<layer>"}
The assistant turn is the 26B answer (used for training); the benchmark only
needs the PROMPT (system+user), and generates its own answer to measure accept.

This splits by source_layer and writes, per layer:
  <out-dir>/sc1_maiprofile_<layer>.jsonl   each line {"prompt": "<system+user folded>"}
plus sc1_maiprofile_all.jsonl (mixed). Prompt folding matches
convert_maiprofile_eval_to_sc1.py:fold_messages (system first, then user,
blank-line separated) so numbers are comparable to the existing bench.

USAGE
  python convert_new_eval_to_sc1.py \
      --eval-file "$AZURE_ML_INPUT_ukwdata/maiprofile/mtp_26b/split/eval_maiprofile_26b.jsonl" \
      --output-dir ./maiprofile_bench_prompts_new

Then point run_maiprofile_online.sh at it via PROMPTS_DIR, or run
bench_online_align.sh --dataset-path <layer file> per layer.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def fold_messages(messages) -> str:
    """system first, then user, blank-line separated (matches the existing
    convert_maiprofile_eval_to_sc1.py so bench prompts are identical in form)."""
    system_parts, user_parts = [], []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    parts = []
    if system_parts:
        parts.append("\n\n".join(system_parts).strip())
    if user_parts:
        parts.append("\n\n".join(user_parts).strip())
    return "\n\n".join(p for p in parts if p)


def layer_of(rec) -> str:
    lyr = rec.get("source_layer")
    if lyr:
        return lyr
    rid = rec.get("id") or ""
    return rid.split(":", 1)[0] if ":" in rid else "all"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-file", required=True,
                    help="mixed eval split jsonl (eval_maiprofile_26b.jsonl)")
    ap.add_argument("--output-dir", default="./maiprofile_bench_prompts_new")
    ap.add_argument("--num-prompts", type=int, default=None,
                    help="cap prompts PER LAYER (default: all)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_layer = defaultdict(list)
    n_read = n_bad = n_empty = 0
    with open(args.eval_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            try:
                rec = json.loads(line)
                convs = rec.get("conversations") or rec.get("prompt_messages")
            except Exception:
                n_bad += 1
                continue
            prompt = fold_messages(convs)
            if not prompt.strip():
                n_empty += 1
                continue
            by_layer[layer_of(rec)].append(prompt)

    all_path = out_dir / "sc1_maiprofile_all.jsonl"
    all_f = all_path.open("w", encoding="utf-8")
    per_layer_counts = {}
    for layer in sorted(by_layer):
        prompts = by_layer[layer]
        if args.num_prompts:
            prompts = prompts[: args.num_prompts]
        lp = out_dir / f"sc1_maiprofile_{layer}.jsonl"
        with lp.open("w", encoding="utf-8") as lf:
            for p in prompts:
                row = json.dumps({"prompt": p}, ensure_ascii=False) + "\n"
                lf.write(row)
                all_f.write(row)
        per_layer_counts[layer] = len(prompts)
    all_f.close()

    print(f"=== convert new eval -> sc1 prompts ===")
    print(f"  input : {args.eval_file}")
    print(f"  read={n_read:,} bad={n_bad} empty_prompt={n_empty}")
    print(f"  output: {out_dir}")
    for layer, c in sorted(per_layer_counts.items()):
        print(f"    sc1_maiprofile_{layer}.jsonl : {c:,} prompts")
    print(f"    sc1_maiprofile_all.jsonl : {sum(per_layer_counts.values()):,} prompts")


if __name__ == "__main__":
    main()
