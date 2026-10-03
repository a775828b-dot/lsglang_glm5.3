"""Build a draft hot-token map for sglang --speculative-token-map.

Ranking: special/added tokens, then tokens by frequency in each corpus in the order
given (put real model outputs of the target workload first, general text after),
then remaining ids in ascending order. The target model still verifies every token,
so the map changes only draft acceptance, never outputs.

python build_token_map.py --model MODEL --size 49152 --output hot.pt outputs.txt docs.txt
"""
import argparse
import collections

import torch
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--size', type=int, default=49152)
p.add_argument('--output', required=True)
p.add_argument('--max-chars', type=int, default=0, help='per corpus limit, 0 = whole file')
p.add_argument('--heldout', help='optional text file to report token coverage on')
p.add_argument('corpora', nargs='+')
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
vocab = max(len(tok), max(tok.get_vocab().values()) + 1)


def tokens(path):
    text = open(path, encoding='utf-8', errors='ignore').read()
    if a.max_chars:
        text = text[:a.max_chars]
    ids = []
    for i in range(0, len(text), 200000):
        ids += tok.encode(text[i:i + 200000], add_special_tokens=False)
    return ids


order = list(sorted(set(tok.all_special_ids) | set(tok.added_tokens_decoder)))
for path in a.corpora:
    counts = collections.Counter(tokens(path))
    order += [t for t, _ in counts.most_common()]
    print(f'{path}: {sum(counts.values())} tokens, {len(counts)} distinct', flush=True)
hot = list(dict.fromkeys(order))[:a.size]
seen = set(hot)
for t in range(vocab):
    if len(hot) >= a.size:
        break
    if t not in seen:
        hot.append(t)
        seen.add(t)
torch.save(hot, a.output)
print(f'saved {len(hot)} ids to {a.output}')
if a.heldout:
    held = tokens(a.heldout)
    print(f'held-out coverage {sum(t in seen for t in held) / max(1, len(held)):.4f} over {len(held)} tokens')
