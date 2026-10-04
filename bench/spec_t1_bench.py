#!/usr/bin/env python3
"""Decode speed and draft acceptance at temperature 1.0 (real usage) and 0 on the running GLM service.

Usage: spec_t1_bench.py --tokenizer MODEL_DIR --longdoc TEXT_FILE [--port 18085] [--label NAME] [--out results.json]
Streams sglang /generate with the GLM chat template (thinking on), so meta_info carries the request's own
spec_verify_ct. Prompts: Chinese explanation, Chinese coding, Chinese bug finding (short context) and a question
about a ~100K-token document (TEXT_FILE is repeated / cut to 100,000 tokens; any long text works).
Each prompt: temperature 0 once, temperature 1.0 / top_p 0.95 three times, 512 new tokens.
Decode rate = (completion_tokens - 1) / (last token time - first token time)."""
import argparse
import http.client
import json
import os
import statistics as st
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from transformers import AutoTokenizer  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument('--tokenizer', required=True)
P.add_argument('--longdoc', required=True)
P.add_argument('--port', type=int, default=18085)
P.add_argument('--label', default='run')
P.add_argument('--out')
A = P.parse_args()
tok = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)

BUGGY = '''下面这段 Python 代码想统计每个用户的订单总额，但结果不对。找出所有 bug，解释原因并给出修正后的完整代码。

def total_by_user(orders, cache={}):
    for o in orders:
        uid = o["user"]
        if uid not in cache:
            cache[uid] = 0
        cache[uid] =+ o["amount"]
    result = sorted(cache.items(), key=lambda x: x[1])
    return dict(result[:10])
'''
SHORT = [
    ('zh_explain', '请解释为什么天空是蓝色的，以及日落时天空为什么会变红。分点说明，语言通俗。'),
    ('zh_code', '用 Python 写一个异步网页下载器，要求有并发限制、失败重试和断点续传，并用中文解释关键设计。'),
    ('zh_debug', BUGGY),
]


def long_prompt():
    ids = tok.encode(open(A.longdoc, encoding='utf-8', errors='replace').read(), add_special_tokens=False)
    while len(ids) < 100_000:
        ids = ids + ids
    return tok.decode(ids[:100_000]) + '\n\n根据上面的文档，概括其主要内容，并指出三个可能出问题的地方。'


def chat(text):
    return tok.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, tokenize=False)


def run(text, params, max_new=512):
    body = json.dumps({'text': chat(text), 'stream': True,
                       'sampling_params': dict(params, max_new_tokens=max_new)}).encode()
    conn = http.client.HTTPConnection('127.0.0.1', A.port, timeout=1800)
    t0 = time.time()
    conn.request('POST', '/generate', body=body, headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f'HTTP {resp.status}: {resp.read()[:300]}')
    first = last = None
    meta, buf = {}, b''
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            event, buf = buf.split(b'\n\n', 1)
            line = event.decode('utf-8', 'replace').strip()
            if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                continue
            m = json.loads(line[5:].strip()).get('meta_info') or {}
            if m.get('completion_tokens', 0) >= 1:
                last = time.time()
                first = first or last
                meta = m
    conn.close()
    out, ver = meta.get('completion_tokens', 0), meta.get('spec_verify_ct') or 0
    return {'out': out, 'verify': ver, 'accept': out / ver if ver else None, 'ttft': (first - t0) if first else None,
            'decode_tps': (out - 1) / (last - first) if first and last and last > first else float('nan')}


T0 = {'temperature': 0.0}
T1 = {'temperature': 1.0, 'top_p': 0.95}
prompts = SHORT + [('long_100k', long_prompt())]
results = {'label': A.label, 'time': time.strftime('%F %T'), 'runs': []}
run('你好', T0, 16)  # warm-up
for name, text in prompts:
    if name == 'long_100k':
        r = run(text, T0, 16)  # prefill once; the measured runs hit the prefix cache
        print(f'{name:10s} prefill-only     ttft {r["ttft"]:.1f}s', flush=True)
    for temp_label, params, reps in (('T0', T0, 1), ('T1', T1, 3)):
        for i in range(reps):
            r = run(text, params)
            r.update(prompt=name, temp=temp_label, rep=i)
            results['runs'].append(r)
            acc = f"{r['accept']:.2f}" if r['accept'] else '  - '
            print(f"{name:10s} {temp_label} #{i}  out {r['out']:4d}  accept {acc}  decode {r['decode_tps']:6.1f} tok/s", flush=True)
print(f'== {A.label}')
for temp_label in ('T0', 'T1'):
    sel = [r for r in results['runs'] if r['temp'] == temp_label]
    print(f"   {temp_label} decode {st.mean(r['decode_tps'] for r in sel):6.1f} tok/s  "
          f"accept {st.mean(r['accept'] for r in sel if r['accept']):.2f}")
if A.out:
    with open(A.out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
