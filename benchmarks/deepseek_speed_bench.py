#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def one_request(args, idx):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        return {
            "ok": False,
            "idx": idx,
            "latency": time.perf_counter() - t0,
            "error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:300]}",
        }
    except Exception as e:
        return {
            "ok": False,
            "idx": idx,
            "latency": time.perf_counter() - t0,
            "error": repr(e),
        }

    t1 = time.perf_counter()
    latency = t1 - t0

    try:
        data = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception as e:
        return {
            "ok": False,
            "idx": idx,
            "latency": latency,
            "error": f"Invalid JSON: {e}",
        }

    if status != 200:
        return {
            "ok": False,
            "idx": idx,
            "latency": latency,
            "error": f"HTTP {status}: {str(data)[:300]}",
        }

    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    completion_tokens = usage.get("completion_tokens", 0) or 0
    prompt_tokens = usage.get("prompt_tokens", 0) or 0

    return {
        "ok": True,
        "idx": idx,
        "latency": latency,
        "completion_tokens": int(completion_tokens),
        "prompt_tokens": int(prompt_tokens),
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM OpenAI chat endpoint speed")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-local")
    parser.add_argument("--prompt", default="你好，输出一句测试。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("-n", "--num-requests", type=int, default=20)
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    print(f"[bench] url={args.base_url} model={args.model}")
    print(f"[bench] requests={args.num_requests} concurrency={args.concurrency} max_tokens={args.max_tokens}")

    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(one_request, args, i) for i in range(args.num_requests)]
        done = 0
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            done += 1
            if done % max(1, args.num_requests // 10) == 0 or done == args.num_requests:
                print(f"[progress] {done}/{args.num_requests}")

    wall = time.perf_counter() - start

    oks = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]

    lat = [r["latency"] for r in oks]
    out_toks = [r["completion_tokens"] for r in oks]
    in_toks = [r["prompt_tokens"] for r in oks]

    total_out = sum(out_toks)
    total_in = sum(in_toks)
    req_s = len(oks) / wall if wall > 0 else 0.0
    out_tok_s = total_out / wall if wall > 0 else 0.0

    print("\n=== SUMMARY ===")
    print(f"success={len(oks)} failed={len(errs)} total={len(results)}")
    print(f"wall_time_s={wall:.3f}")
    print(f"req_per_s={req_s:.3f}")
    print(f"total_prompt_tokens={total_in}")
    print(f"total_completion_tokens={total_out}")
    print(f"completion_tokens_per_s={out_tok_s:.3f}")

    if lat:
        print("\n=== LATENCY (s) ===")
        print(f"mean={statistics.mean(lat):.3f}")
        print(f"p50={percentile(lat, 0.50):.3f}")
        print(f"p90={percentile(lat, 0.90):.3f}")
        print(f"p95={percentile(lat, 0.95):.3f}")
        print(f"p99={percentile(lat, 0.99):.3f}")
        print(f"max={max(lat):.3f}")

    if oks:
        per_req_tps = [
            (r["completion_tokens"] / r["latency"]) if r["latency"] > 0 else 0.0
            for r in oks
        ]
        print("\n=== PER-REQUEST OUTPUT TPS ===")
        print(f"mean_toks_per_s={statistics.mean(per_req_tps):.3f}")
        print(f"p50_toks_per_s={percentile(per_req_tps, 0.50):.3f}")

    if errs:
        print("\n=== ERRORS (first 10) ===")
        for e in errs[:10]:
            print(f"idx={e['idx']} latency={e['latency']:.3f}s error={e['error']}")


if __name__ == "__main__":
    main()
