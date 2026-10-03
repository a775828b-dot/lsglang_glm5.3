#!/usr/bin/env python3
"""Post-start warmup for the GLM-5.3 lane2 service: wait for /health, then send one long, one
medium and one short prompt (max_tokens 4) so the first real request does not pay Triton JIT
compilation of the GPU / hybrid / CPU prefill paths (~7 s for the first long prompt otherwise).
Usage: warmup.py [PORT]  (stdlib only; logs to stdout)."""
import json, sys, time, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 18085
base = f'http://127.0.0.1:{port}'
deadline = time.time() + 3600
while time.time() < deadline:
    try:
        urllib.request.urlopen(base + '/health', timeout=5); break
    except Exception:
        time.sleep(10)
else:
    print('warmup: server never became healthy'); sys.exit(0)
words = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'.split()
for n_words, label in ((18000, 'long'), (1800, 'medium'), (450, 'short')):
    text = ' '.join(words[(i * 7) % len(words)] + str(i % 97) for i in range(n_words))
    body = json.dumps(dict(model='GLM-5.3-Flash-NVFP4', max_tokens=4, temperature=0,
                           messages=[{'role': 'user', 'content': f'warmup {label} {time.time()}\n' + text}])).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(base + '/v1/chat/completions', body, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=1800) as r:
            usage = json.loads(r.read()).get('usage', {})
        print(f'warmup {label}: prompt_tokens={usage.get("prompt_tokens")} {time.time() - t0:.1f} s', flush=True)
    except Exception as exc:
        print(f'warmup {label} failed: {exc!r}', flush=True)
