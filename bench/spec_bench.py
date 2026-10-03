"""Decode throughput for speculative-decoding A/B: fixed prompts, several rounds, aggregate
tokens / wall time (robust against per-prompt acceptance noise). Uses lane2 bench prompts plus
a few agent-style ones. Usage: spec_bench.py PORT TAG [ROUNDS]"""
import json, statistics as st, sys, time, urllib.request

port, tag = int(sys.argv[1]), sys.argv[2]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
PROMPTS = [
    "用 Python 实现一个线程安全的 LRU 缓存类，支持 get/put，并写 3 个单元测试。",
    "解释混合专家模型（MoE）在推理时的访存特征，以及为什么它在 CPU 卸载场景下比稠密模型更难加速。",
    "把下面的列表原样重复三遍，每遍之间空一行：\n" + "\n".join(f"- item_{i}: value_{i}" for i in range(20)),
    "一个数列前三项是 2, 6, 14，第 n 项满足 a(n) = 2*a(n-1) + c。求 c 和 a(10)，写出推导过程。",
    "Write a Go HTTP middleware that adds request IDs, logs latency, and recovers from panics. Include tests.",
    "阅读这段报错并给出修复：TypeError: Cannot read properties of undefined (reading 'map') at TodoList.render",
]
total_tok = total_s = 0.0
per = []
for r in range(rounds):
    for i, p in enumerate(PROMPTS):
        body = json.dumps(dict(model="GLM-5.3-Flash-NVFP4", messages=[{"role": "user", "content": p}],
                               max_tokens=700, temperature=0)).encode()
        t0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as resp:
            u = json.loads(resp.read())["usage"]
        dt = time.time() - t0
        n = u["completion_tokens"]
        total_tok += n; total_s += dt; per.append(n / dt)
print(json.dumps(dict(tag=tag, rounds=rounds, requests=len(per), agg_tok_s=round(total_tok / total_s, 2),
                      median_tok_s=round(st.median(per), 2), min=round(min(per), 1), max=round(max(per), 1))))
