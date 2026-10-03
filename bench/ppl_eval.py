"""Fixed-text perplexity of a running sglang server (precision A/B between configurations).

Usage: ppl_eval.py --port 18085 --tokenizer DIR --tag NAME --corpus docs=a.md --corpus zh=b.txt
                   [--segments 6 --tokens 4096]
Segments are cut deterministically from fixed corpora (the README numbers: Markdown docs, agent
sessions, a Chinese document; not included), so every configuration scores exactly the same token ids. Uses /generate with return_logprob and
logprob_start_len=0 (prefill logprobs of the given ids). Reports mean NLL / PPL overall and per
corpus, and per-segment NLL so two runs can be compared pairwise.
"""
import argparse, json, math, os, urllib.request
from transformers import AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=18085)
P.add_argument('--tokenizer', required=True)
P.add_argument('--tag', required=True)
P.add_argument('--segments', type=int, default=8, help='segments per corpus')
P.add_argument('--tokens', type=int, default=2048)
P.add_argument('--corpus', action='append', required=True, help='NAME=PATH, repeatable')
P.add_argument('--out', default='ppl-results')
A = P.parse_args()
tok = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)
CORPORA = dict(c.split('=', 1) for c in A.corpus)
segs = []
for name, path in CORPORA.items():
    text = open(path, encoding='utf-8', errors='ignore').read()
    ids = tok.encode(text[:4_000_000], add_special_tokens=False)
    n = min(A.segments, max(1, len(ids) // A.tokens))
    stride = max(A.tokens, (len(ids) - A.tokens) // max(1, n))
    for i in range(n):
        s = ids[i * stride: i * stride + A.tokens]
        if len(s) == A.tokens:
            segs.append((name, i, s))
rows = []
for name, i, s in segs:
    # no prefix-cache reuse between segments or runs: every segment is a cold prefill
    urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{A.port}/flush_cache', b''), timeout=60)
    body = json.dumps(dict(input_ids=s, sampling_params=dict(max_new_tokens=1, temperature=0),
                           return_logprob=True, logprob_start_len=0)).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{A.port}/generate', body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        meta = json.loads(r.read())['meta_info']
    lps = [x[0] for x in meta['input_token_logprobs'] if x[0] is not None]
    nll = -sum(lps) / len(lps)
    rows.append(dict(corpus=name, seg=i, n=len(lps), nll=nll))
    print(json.dumps(rows[-1]), flush=True)
summary = {}
for name in list(CORPORA) + ['all']:
    sel = [r for r in rows if name == 'all' or r['corpus'] == name]
    if sel:
        nll = sum(r['nll'] * r['n'] for r in sel) / sum(r['n'] for r in sel)
        summary[name] = dict(segments=len(sel), nll=round(nll, 5), ppl=round(math.exp(nll), 4))
print('SUMMARY', json.dumps(summary), flush=True)
os.makedirs(A.out, exist_ok=True)
json.dump(dict(tag=A.tag, rows=rows, summary=summary), open(os.path.join(A.out, f'{A.tag}.json'), 'w'), indent=1)
