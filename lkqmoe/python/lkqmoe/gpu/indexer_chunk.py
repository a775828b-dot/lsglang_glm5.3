"""Opt-in (LKQMOE_INDEXER_Q_CHUNK=N, e.g. 4096): DSA kpool indexer prefill in query blocks.

During extend (prefill) the kpool indexer of sglang's DSA attention computes FP32 logits for every
query of the chunk against the whole pooled key history in one deep_gemm.fp8_mqa_logits call, then
selects top-k per row. The logits matrix is [chunk, context/pool]: 7.8 GiB for a 16K chunk at 512K
context (15.5 GiB at 32K), the largest transient of a long prefill. Every row is independent (own
key range ks/ke, group length, tail, page-table row, offset), so this computes logits and top-k in
blocks of N queries and concatenates: identical indices, transient N/chunk of the original.

Mechanism: while IndexerKPool._get_topk_ragged_kpool_plan runs, _fp8_mqa_logits returns a lazy
handle for batches above N rows; _topk_from_kpool_logits evaluates it block by block.
"""
import importlib.abc
import importlib.machinery
import os
import sys
import threading

import torch

_TARGET = 'sglang.srt.layers.attention.dsa.dsa_indexer_kpool'
_state = threading.local()
stats = {'blocked_calls': 0, 'max_rows': 0}


class _LazyLogits:
    def __init__(self, fn, q, k, k_scale, w, ks, ke, clean):
        self.fn, self.q, self.k, self.k_scale, self.w, self.ks, self.ke, self.clean = fn, q, k, k_scale, w, ks, ke, clean
        self.shape = (q.shape[0], k.shape[0])
        self.device = q.device


def _rows(t, r0, r1, n):
    return t[r0:r1] if t is not None and t.shape[0] == n else t


def patch(module):
    cls = getattr(module, 'IndexerKPool', None)
    if cls is None or getattr(cls, '_lkqmoe_q_chunk', False):
        return
    for name in ('_get_topk_ragged_kpool_plan', '_fp8_mqa_logits', '_topk_from_kpool_logits'):
        if not callable(getattr(cls, name, None)):
            raise RuntimeError('Unsupported sglang DSA kpool indexer interface')
    block = int(os.environ.get('LKQMOE_INDEXER_Q_CHUNK', '4096'))
    orig_plan = cls._get_topk_ragged_kpool_plan
    orig_logits = cls.__dict__['_fp8_mqa_logits']  # staticmethod object
    orig_logits_fn = orig_logits.__func__ if isinstance(orig_logits, staticmethod) else orig_logits
    orig_topk = cls._topk_from_kpool_logits

    def plan(self, *args, **kwargs):
        _state.lazy = True
        try:
            return orig_plan(self, *args, **kwargs)
        finally:
            _state.lazy = False

    def logits(q_fp8, k_fp8, k_scale, weights, starts, ends, clean_logits=True):
        if getattr(_state, 'lazy', False) and q_fp8.shape[0] > block:
            return _LazyLogits(orig_logits_fn, q_fp8, k_fp8, k_scale, weights, starts, ends, clean_logits)
        return orig_logits_fn(q_fp8, k_fp8, k_scale, weights, starts, ends, clean_logits=clean_logits)

    def topk(self, logits, pool_lens, seq_lens=None, page_table=None, topk_offsets=None,
             row_starts=None, out_rows=None, page_table_row_index=None):
        if not isinstance(logits, _LazyLogits):
            return orig_topk(self, logits, pool_lens, seq_lens=seq_lens, page_table=page_table,
                             topk_offsets=topk_offsets, row_starts=row_starts, out_rows=out_rows,
                             page_table_row_index=page_table_row_index)
        L = logits; n = L.shape[0]
        stats['blocked_calls'] += 1; stats['max_rows'] = max(stats['max_rows'], n)
        parts = []
        for r0 in range(0, n, block):
            r1 = min(n, r0 + block)
            lg = L.fn(L.q[r0:r1], L.k, L.k_scale, L.w[r0:r1], L.ks[r0:r1], L.ke[r0:r1], clean_logits=L.clean)
            # page_table: per-row unless rows are addressed through page_table_row_index
            pt = page_table if page_table_row_index is not None else _rows(page_table, r0, r1, n)
            parts.append(orig_topk(self, lg, pool_lens[r0:r1], seq_lens=_rows(seq_lens, r0, r1, n),
                                   page_table=pt, topk_offsets=_rows(topk_offsets, r0, r1, n),
                                   row_starts=_rows(row_starts, r0, r1, n), out_rows=None,
                                   page_table_row_index=_rows(page_table_row_index, r0, r1, n)))
            del lg
        result = torch.cat(parts)
        if out_rows is None or out_rows == result.shape[0]:
            return result
        padded = torch.full((out_rows, result.shape[1]), -1, dtype=result.dtype, device=result.device)
        padded[:result.shape[0]] = result
        return padded

    cls._get_topk_ragged_kpool_plan = plan
    cls._fp8_mqa_logits = staticmethod(logits)
    cls._topk_from_kpool_logits = topk
    cls._lkqmoe_q_chunk = True
    print(f'[lkqmoe] DSA indexer prefill in query blocks of {block}', file=sys.stderr, flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        fn = getattr(self.delegate, 'create_module', None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    if _TARGET in sys.modules:
        patch(sys.modules[_TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
