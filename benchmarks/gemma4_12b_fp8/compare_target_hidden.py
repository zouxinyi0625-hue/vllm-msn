#!/usr/bin/env python3
"""对比 HF(训练侧) vs vLLM(推理侧) 在 target_layer_ids 处的 hidden。
一锤定音判断 26B MoE target 的中间层 hidden 是否对齐。

用法（服务器，vLLM venv）:
  python3 compare_target_hidden.py \
    --target /tmp/models/gemma4/text_only \
    --layers 4,12,20,29 \
    --prompt "The quick brown fox jumps over the lazy dog. Tell me a story."
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--layers", default="4,12,20,29", help="vLLM 语义: layer_idx+1 后的值")
ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
args = ap.parse_args()
layers = [int(x) for x in args.layers.split(",")]

# ---------- HF 侧（训练 cache 的抓法：layers[i] 的 forward_hook 输出）----------
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.target)
ids = tok(args.prompt, return_tensors="pt").input_ids
hf = AutoModel.from_pretrained(args.target, dtype=torch.bfloat16,
                               attn_implementation="sdpa").eval().cuda()
# backbone
bb = hf.language_model if hasattr(hf, "language_model") else hf
cap = {}
# 训练存的是 layers[3,11,19,28] 输出；vLLM 语义 [4,12,20,29] = layers[3,11,19,28] 之后
hf_layer_ids = [l - 1 for l in layers]
hs = []
def mk(i):
    def hook(_m, _i, o):
        cap[i] = (o[0] if isinstance(o, tuple) else o).detach()
    return hook
for i in hf_layer_ids:
    bb.layers[i].register_forward_hook(mk(i))
with torch.no_grad():
    hf(input_ids=ids.cuda(), use_cache=False, output_hidden_states=False)
hf_vecs = {l: cap[l - 1] for l in layers}
del hf; torch.cuda.empty_cache()

# ---------- vLLM 侧 ----------
from vllm import LLM
llm = LLM(model=args.target, enforce_eager=True, gpu_memory_utilization=0.6,
          max_model_len=2048, trust_remote_code=True)
mr = llm.llm_engine.model_executor.driver_worker.model_runner
model = mr.get_model()
model.set_aux_hidden_state_layers(tuple(layers))
tokens = ids[0].tolist()
from vllm import SamplingParams
# 跑一步 prefill 拿 aux hidden：直接调 forward 较繁琐，改用 hook
inner = model.model
grab = {}
orig = inner._maybe_add_hidden_state
def patched(aux, idx, h, res):
    r = orig(aux, idx, h, res)
    if idx in layers:
        grab[idx] = (h + res if res is not None else h).detach().clone()
    return r
inner._maybe_add_hidden_state = patched
llm.generate([{"prompt_token_ids": tokens}], SamplingParams(max_tokens=1))

print("\n===== HF vs vLLM cosine per layer =====")
for l in layers:
    a = hf_vecs[l].float().reshape(-1, hf_vecs[l].shape[-1])
    b = grab[l].float().reshape(-1, grab[l].shape[-1])
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n].cuda(), b[:n].cuda()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(f"layer {l:3d}: cos mean={cos.mean():.4f} min={cos.min():.4f} "
          f"| HF|x|={a.norm(dim=-1).mean():.2f} vLLM|x|={b.norm(dim=-1).mean():.2f}")
print("\ncos≈1 → hidden 对齐，问题在 draft/采样；cos<0.9 → MoE hidden 抽取不一致(根因)")
