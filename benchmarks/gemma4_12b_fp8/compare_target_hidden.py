#!/usr/bin/env python3
"""对比 HF(训练侧) vs vLLM(推理侧) 在 target_layer_ids 处的 hidden。
两阶段分离，避免 HF+vLLM 同占一张卡导致 OOM。

先杀掉占卡的旧 server：
  pkill -9 -f "vllm serve"; pkill -9 -f serve_align; sleep 5; nvidia-smi

STEP 1 (HF, 存盘):
  python3 compare_target_hidden.py hf  --target /tmp/models/gemma4/text_only
STEP 2 (vLLM, 存盘 + 对比):
  python3 compare_target_hidden.py vllm --target /tmp/models/gemma4/text_only
"""
import argparse, torch

PROMPT = "The quick brown fox jumps over the lazy dog. Tell me a short story about a robot."
LAYERS = [4, 12, 20, 29]          # vLLM 语义 = HF layers[3,11,19,28] 之后
HF_LAYERS = [l - 1 for l in LAYERS]
HF_PT = "/tmp/hf_target_hidden.pt"
IDS_PT = "/tmp/target_hidden_ids.pt"

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["hf", "vllm"])
ap.add_argument("--target", required=True)
args = ap.parse_args()


def run_hf():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.target)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    torch.save(ids, IDS_PT)
    m = AutoModel.from_pretrained(args.target, dtype=torch.bfloat16,
                                  attn_implementation="sdpa").eval().cuda()
    bb = m.language_model if hasattr(m, "language_model") else m
    cap = {}
    def mk(i):
        def hook(_mod, _in, out):
            cap[i] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
        return hook
    for i in HF_LAYERS:
        bb.layers[i].register_forward_hook(mk(i))
    with torch.no_grad():
        m(input_ids=ids.cuda(), use_cache=False, output_hidden_states=False)
    torch.save({l: cap[l - 1] for l in LAYERS}, HF_PT)
    print("saved HF hidden ->", HF_PT, "| shapes:",
          {l: list(cap[l - 1].shape) for l in LAYERS})


def run_vllm():
    from vllm import LLM, SamplingParams
    ids = torch.load(IDS_PT)
    tokens = ids[0].tolist()
    llm = LLM(model=args.target, enforce_eager=True, gpu_memory_utilization=0.85,
              max_model_len=2048, trust_remote_code=True)
    mr = llm.llm_engine.model_executor.driver_worker.model_runner
    model = mr.get_model()
    model.set_aux_hidden_state_layers(tuple(LAYERS))
    inner = model.model
    grab = {}
    orig = inner._maybe_add_hidden_state
    def patched(aux, idx, h, res):
        r = orig(aux, idx, h, res)
        if idx in LAYERS:
            grab[idx] = (h + res if res is not None else h).detach().float().cpu().clone()
        return r
    inner._maybe_add_hidden_state = patched
    llm.generate([{"prompt_token_ids": tokens}], SamplingParams(max_tokens=1))

    hf = torch.load(HF_PT)
    print("\n===== HF vs vLLM cosine per layer =====")
    for l in LAYERS:
        a = hf[l].float().reshape(-1, hf[l].shape[-1])
        b = grab[l].reshape(-1, grab[l].shape[-1])
        n = min(a.shape[0], b.shape[0])
        cos = torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=-1)
        print(f"layer {l:3d}: cos mean={cos.mean():.4f} min={cos.min():.4f} "
              f"| HF|x|={a[:n].norm(dim=-1).mean():.2f} "
              f"vLLM|x|={b[:n].norm(dim=-1).mean():.2f}")
    print("\ncos≈1 → hidden 对齐(问题在 draft/采样); cos<0.9 → MoE hidden 抽取不一致(根因)")


run_hf() if args.stage == "hf" else run_vllm()
