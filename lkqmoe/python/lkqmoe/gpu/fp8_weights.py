"""Opt-in (LKQMOE_FP8_WEIGHTS=1): FP8 E4M3 (W8A16) or NVFP4 (LKQMOE_NVFP4_INCLUDE: W4A16 decode,
W4A4 prefill on FP4 tensor cores) storage for selected BF16 linear weights.
LKQMOE_PREFILL_W4A4_INCLUDE: FP8-stored layers whose prefill (M > 16) is quantized per call to NVFP4
(weights FP8 -> BF16 -> NVFP4 with a load-time global scale, activations NVFP4) and run on mm_fp4.

For checkpoints whose attention / dense layers are BF16 (GLM-5.3 Flash NVFP4: ~15 GB, read in
full every decode step), weights of selected UnquantizedLinearMethod layers are quantized after
loading to FP8 E4M3 with one FP32 scale per output row and 128-column group (amax / 448), and the
BF16 copy is freed. Activations stay BF16:
- M <= 16 rows (decode/verify): split-K Triton kernel; FP8 tiles are dequantized in registers to
  BF16, FP32 accumulation, K slices reduced in a fixed order (deterministic), half the bytes;
- larger M (prefill): layers matching the runtime-file key "w8a8_prefill" (LKQMOE_RUNTIME_FILE) run a
  Triton W8A8 GEMM (activations quantized to E4M3 per token x 128 columns, FP32 accumulation of
  per-group scaled FP8 dots); the others dequantize the weight to a temporary BF16 tensor and use the
  original F.linear (same values as the decode path).
Numerical changes: the weight rounding to FP8 (per-group scaled), plus the activation rounding where
W8A8 prefill is enabled.

Selection (after DefaultModelLoader.load_weights_and_postprocess, by full module name): matches LKQMOE_FP8_INCLUDE and not LKQMOE_FP8_EXCLUDE, BF16 2-D weight,
K % 128 == 0, at least LKQMOE_FP8_MIN_ELEMS elements. Excluded by default: weights read directly
by other kernels (fused_qkv_a_proj -> dsv3_fused_a_gemm; kv_b_proj -> MLA absorbed matrices),
shared experts (overlapped with the CPU experts, no speed gain), routers, vision.
"""
import importlib.abc
import importlib.machinery
import os
import re
import sys

import torch
import triton
import triton.language as tl

_TARGET = 'sglang.srt.layers.quantization.unquant'
GROUP = 128
MAX_M = 16
stats = {'layers': 0, 'bf16_bytes': 0, 'fp8_bytes': 0, 'decode_calls': 0, 'prefill_calls': 0, 'names': []}


@triton.jit
def _w8_partial(X, W, S, P, M, N, K, SPAN, NG, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = ps * SPAN
    for kk in range(0, SPAN, BK):
        ks = k0 + kk + tl.arange(0, BK)
        kmask = ks < K
        x = tl.load(X + rows[:, None] * K + ks[None, :], mask=(rows[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * K + ks[None, :], mask=cmask[:, None] & kmask[None, :], other=0.0)
        # the last split-K slice may extend past K: mask the group scale too (an unmasked read ran past the
        # end of the scale tensor for the last row; harmless unless it hit unmapped memory or Inf/NaN bytes)
        s = tl.load(S + cols * NG + (k0 + kk) // 128, mask=cmask & (k0 + kk < K), other=0.0)
        wb = (w.to(tl.float32) * s[:, None]).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(wb), acc)
    tl.store(P + (ps * BM + rows[:, None]) * N + cols[None, :], acc, mask=(rows[:, None] < M) & cmask[None, :])


@triton.jit
def _w8_reduce(P, BIAS, Y, M, N, SLICES: tl.constexpr, BM: tl.constexpr, HAS_BIAS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < M * N
    r = i // N
    c = i % N
    acc = tl.zeros((B,), dtype=tl.float32)
    for s in tl.static_range(SLICES):   # fixed order: deterministic
        acc += tl.load(P + (s * BM + r) * N + c, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(BIAS + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + i, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _w8_dequant(W, S, D, N, K, NG, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (rows[:, None] < N) & (ks[None, :] < K)
    w = tl.load(W + rows[:, None] * K + ks[None, :], mask=mask, other=0.0)
    s = tl.load(S + rows * NG + (tl.program_id(1) * BK) // 128, mask=rows < N, other=0.0)
    tl.store(D + rows[:, None] * K + ks[None, :], (w.to(tl.float32) * s[:, None]).to(tl.bfloat16), mask=mask)


def quantize(w):
    """w [N,K] bf16 -> (fp8 [N,K], fp32 scale [N, K/128])."""
    n, k = w.shape
    g = w.float().view(n, k // GROUP, GROUP)
    s = (g.abs().amax(-1) / 448.0).clamp_min(1e-12)
    q = (g / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(n, k)
    return q.contiguous(), s.contiguous()


def _config(n, k):
    block_n = 32 if n <= 512 else 64
    tiles = triton.cdiv(n, block_n)
    slices = max(1, min(k // 128, triton.cdiv(440, tiles)))
    return block_n, 128, slices


@triton.jit
def _w8_post(X, W, S, P, CNT, BIAS, Y, M, N, K, SPAN, NG, SLICES: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Decode W8A16: FP8 -> BF16 (exact), one dot per 128-column group, the group sum scaled in FP32 (the
    dequantized weight is never rounded to BF16). Split-K slices: the last CTA of a column tile adds the
    slice partials in slice order (fixed order, deterministic) and resets the tile counter."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = ps * SPAN
    for kk in range(0, SPAN, 128):
        ks = k0 + kk + tl.arange(0, 128)
        kmask = ks < K
        x = tl.load(X + rows[:, None] * K + ks[None, :], mask=(rows[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * K + ks[None, :], mask=cmask[:, None] & kmask[None, :], other=0.0)
        s = tl.load(S + cols * NG + (k0 + kk) // 128, mask=cmask & (k0 + kk < K), other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    omask = (rows[:, None] < M) & cmask[None, :]
    if SLICES == 1:
        r = acc
        if HAS_BIAS:
            r += tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + rows[:, None] * N + cols[None, :], r.to(tl.bfloat16), mask=omask)
    else:
        tl.store(P + (ps * BM + rows[:, None]) * N + cols[None, :], acc, mask=omask)
        tl.debug_barrier()
        done = tl.atomic_add(CNT + pn, 1, sem='acq_rel')
        if done == SLICES - 1:
            r = tl.zeros((BM, BN), dtype=tl.float32)
            for sl in range(SLICES):
                r += tl.load(P + (sl * BM + rows[:, None]) * N + cols[None, :], mask=omask, other=0.0,
                             cache_modifier='.cg')
            if HAS_BIAS:
                r += tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            tl.store(Y + rows[:, None] * N + cols[None, :], r.to(tl.bfloat16), mask=omask)
            tl.atomic_xchg(CNT + pn, 0)


@triton.jit
def _w8_fused(X, W, S, P, CNT, BIAS, Y, M, N, K, SPAN, NG, SLICES: tl.constexpr, BM: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Decode W8A16, same arithmetic as _w8_partial + _w8_reduce (dequantized weight rounded to BF16, same
    split-K slices and slice order, so bitwise identical) in one launch: the last CTA of a column tile adds
    the slice partials in slice order and resets the tile counter."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = ps * SPAN
    for kk in range(0, SPAN, BK):
        ks = k0 + kk + tl.arange(0, BK)
        kmask = ks < K
        x = tl.load(X + rows[:, None] * K + ks[None, :], mask=(rows[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * K + ks[None, :], mask=cmask[:, None] & kmask[None, :], other=0.0)
        s = tl.load(S + cols * NG + (k0 + kk) // 128, mask=cmask & (k0 + kk < K), other=0.0)
        wb = (w.to(tl.float32) * s[:, None]).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(wb), acc)
    omask = (rows[:, None] < M) & cmask[None, :]
    if SLICES == 1:
        r = acc
        if HAS_BIAS:
            r += tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + rows[:, None] * N + cols[None, :], r.to(tl.bfloat16), mask=omask)
    else:
        tl.store(P + (ps * BM + rows[:, None]) * N + cols[None, :], acc, mask=omask)
        tl.debug_barrier()
        done = tl.atomic_add(CNT + pn, 1, sem='acq_rel')
        if done == SLICES - 1:
            r = tl.zeros((BM, BN), dtype=tl.float32)
            for sl in range(SLICES):   # fixed order, as _w8_reduce
                r += tl.load(P + (sl * BM + rows[:, None]) * N + cols[None, :], mask=omask, other=0.0,
                             cache_modifier='.cg')
            if HAS_BIAS:
                r += tl.load(BIAS + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            tl.store(Y + rows[:, None] * N + cols[None, :], r.to(tl.bfloat16), mask=omask)
            tl.atomic_xchg(CNT + pn, 0)


# (N, K) -> (BN, warps, stages) of _w8_fused; slices stay those of _config (bitwise identical to the
# two-kernel path). RTX PRO 5000, M=5, cold L2 (bench/w8_tune.py); other shapes: _config BN, 4 warps, 3 stages.
_W8_FUSED_CFG = {(4096, 8192): (64, 4, 2), (16384, 1536): (64, 4, 2), (4096, 16384): (64, 4, 2),
                 (4096, 1536): (32, 4, 2), (4096, 12288): (64, 4, 2), (4096, 4096): (128, 8, 2),
                 (4096, 2048): (32, 4, 2)}


# (N, K) -> (BN, slices, warps, stages): RTX PRO 5000, M=5, cold L2 (bench/w8_post.py)
_W8_CFG = {(24576, 4096): (64, 1, 4, 4), (4096, 8192): (32, 4, 4, 3), (16384, 1536): (64, 1, 8, 3),
           (4096, 16384): (32, 4, 4, 3), (4096, 1536): (32, 4, 4, 2), (4096, 12288): (64, 4, 8, 3),
           (4096, 4096): (32, 4, 4, 2), (4096, 2048): (32, 4, 4, 2)}


def _w8_config(n, k):
    cfg = _W8_CFG.get((n, k))
    if cfg is None:
        cfg = (64, 1, 4, 4) if n >= 16384 else (32, min(4, k // 128), 4, 3)
    return cfg


def gemm_w8(x, q, s, bias=None, cnt=None):
    """x [M,K] bf16 (M <= 16), q fp8 [N,K], s fp32 [N,K/128] -> [M,N] bf16.
    cnt: int32 [>= N/32] zeros, the split-K tile counters (per layer, so CUDA graphs need no memset).
    LKQMOE_FP8_DECODE_KERNEL: fused (default; one launch, bitwise identical to legacy), post (group sums
    scaled after the dot: no BF16 rounding of the dequantized weight, finer but different from the prefill
    dequant path, which lowered draft acceptance), legacy (split-K partial + reduce kernels)."""
    mode = os.environ.get('LKQMOE_FP8_DECODE_KERNEL', 'fused')
    if mode == 'fused':
        m, k = x.shape
        n = q.shape[0]
        block_n, block_k, slices = _config(n, k)
        span = triton.cdiv(triton.cdiv(k, slices), block_k) * block_k
        slices = triton.cdiv(k, span)
        bn, warps, stages = _W8_FUSED_CFG.get((n, k), (block_n, 4, 3))
        if cnt is None or cnt.numel() < triton.cdiv(n, bn):
            cnt = torch.zeros(triton.cdiv(n, bn), dtype=torch.int32, device=x.device)
        part = torch.empty((slices, MAX_M, n), device=x.device, dtype=torch.float32) if slices > 1 else x
        y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _w8_fused[(triton.cdiv(n, bn), slices)](x, q, s, part, cnt, bias if bias is not None else x, y, m, n, k, span,
                                                k // GROUP, slices, MAX_M, bn, block_k, bias is not None,
                                                num_warps=warps, num_stages=stages)
        return y
    if mode == 'post':
        m, k = x.shape
        n = q.shape[0]
        bn, slices, warps, stages = _w8_config(n, k)
        span = triton.cdiv(triton.cdiv(k, slices), 128) * 128
        slices = triton.cdiv(k, span)
        if cnt is None:
            cnt = torch.zeros(triton.cdiv(n, bn), dtype=torch.int32, device=x.device)
        part = torch.empty((slices, MAX_M, n), device=x.device, dtype=torch.float32) if slices > 1 else x
        y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _w8_post[(triton.cdiv(n, bn), slices)](x, q, s, part, cnt, bias if bias is not None else x, y, m, n, k, span,
                                               k // GROUP, slices, MAX_M, bn, bias is not None,
                                               num_warps=warps, num_stages=stages)
        return y
    m, k = x.shape
    n = q.shape[0]
    block_n, block_k, slices = _config(n, k)
    span = triton.cdiv(triton.cdiv(k, slices), block_k) * block_k
    slices = triton.cdiv(k, span)
    part = torch.empty((slices, MAX_M, n), device=x.device, dtype=torch.float32)
    _w8_partial[(triton.cdiv(n, block_n), slices)](x, q, s, part, m, n, k, span, k // GROUP,
                                                    MAX_M, block_n, block_k, num_warps=4)
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _w8_reduce[(triton.cdiv(m * n, 1024),)](part, bias if bias is not None else part, y, m, n, slices, MAX_M,
                                            bias is not None, 1024)
    return y


def dequant(q, s):
    n, k = q.shape
    d = torch.empty((n, k), device=q.device, dtype=torch.bfloat16)
    _w8_dequant[(triton.cdiv(n, 32), triton.cdiv(k, 128))](q, s, d, n, k, k // GROUP, 32, 128)
    return d


# ---- NVFP4 (LKQMOE_NVFP4_INCLUDE): E2M1 codes packed two per byte (low nibble first), E4M3 scale per
# 16 columns, FP32 global scale gs (value = code * scale / gs). Decode: W4A16 split-K kernel below;
# prefill: activations quantized to NVFP4 per call (dynamic global scale) and flashinfer mm_fp4 (W4A4).

@triton.jit
def _w4_partial(X, Q, S, GINV, P, M, N, K, SPAN, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    cmask = cols < N
    ginv = tl.load(GINV)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = ps * SPAN
    for kk in range(0, SPAN, BK):
        kb = k0 + kk
        ks = kb + tl.arange(0, BK)
        x = tl.load(X + rows[:, None] * K + ks[None, :], mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        pcol = kb // 2 + tl.arange(0, BK // 2)
        pb = tl.load(Q + cols[:, None] * (K // 2) + pcol[None, :],
                     mask=cmask[:, None] & (pcol[None, :] < K // 2), other=0)
        codes = tl.reshape(tl.join(pb & 15, pb >> 4), (BN, BK))
        m = (codes & 7).to(tl.float32)
        v = tl.where(m < 4.0, m * 0.5, tl.where(m < 6.0, m - 2.0, m * 2.0 - 8.0))
        v = tl.where((codes & 8) != 0, -v, v)
        scol = kb // 16 + tl.arange(0, BK // 16)
        sb = tl.load(S + cols[:, None] * (K // 16) + scol[None, :],
                     mask=cmask[:, None] & (scol[None, :] < K // 16), other=0)
        sc = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32) * ginv
        sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 16, 16)), (BN, BK))
        acc = tl.dot(x, tl.trans((v * sc).to(tl.bfloat16)), acc)
    tl.store(P + (ps * BM + rows[:, None]) * N + cols[None, :], acc, mask=(rows[:, None] < M) & cmask[None, :])


def quantize_nvfp4(w):
    """w [N,K] bf16 -> dict(q [N,K/2] u8, sf_lin [N,K/16] u8, sf_sw swizzled u8, gs [1], ginv [1])."""
    from flashinfer import fp4_quantize
    gs = (448.0 * 6.0 / w.float().abs().amax().clamp_min(1e-12)).reshape(1).float()
    q, sf_sw = fp4_quantize(w, gs, 16, False, True)
    _, sf_lin = fp4_quantize(w, gs, 16, False, False)
    return dict(q=q.contiguous(), sf_lin=sf_lin.view(torch.uint8).reshape(w.shape[0], -1)[:, :w.shape[1] // 16].contiguous(),
                sf_sw=sf_sw, gs=gs, ginv=(1.0 / gs).float())


def gemm_w4(x, nv, n, bias=None):
    """x [M,K] bf16 (M <= 16) -> [M,N] bf16 (W4A16)."""
    m, k = x.shape
    block_n, block_k, slices = _config(n, k)
    span = triton.cdiv(triton.cdiv(k, slices), block_k) * block_k
    slices = triton.cdiv(k, span)
    part = torch.empty((slices, MAX_M, n), device=x.device, dtype=torch.float32)
    _w4_partial[(triton.cdiv(n, block_n), slices)](x, nv['q'], nv['sf_lin'], nv['ginv'], part, m, n, k, span,
                                                    MAX_M, block_n, block_k, num_warps=4)
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _w8_reduce[(triton.cdiv(m * n, 1024),)](part, bias if bias is not None else part, y, m, n, slices, MAX_M,
                                            bias is not None, 1024)
    return y


def gemm_w4a4(x, nv, bias=None):
    """x [M,K] bf16 (prefill) -> [M,N] bf16: NVFP4 activations (dynamic global scale) x NVFP4 weights."""
    from flashinfer import fp4_quantize, mm_fp4
    gx = (448.0 * 6.0 / x.float().abs().amax().clamp_min(1e-12)).reshape(1)
    xq, xs = fp4_quantize(x, gx, 16, False, True)
    y = mm_fp4(xq, nv['q'].T, xs, nv['sf_sw'].T, (1.0 / (gx * nv['gs'])).float(), torch.bfloat16, backend='cutlass')
    return y if bias is None else y + bias


# ---- FP8 W8A8 prefill (runtime key "w8a8_prefill"): activations quantized per token x 128 columns to E4M3,
# FP8 weights (per row x 128 columns) as stored; per K-group FP8 tensor-core dot, FP32 accumulation of
# partial * (activation scale x weight scale).

@triton.jit
def _act_quant_fp8(X, Q, S, M, K, NG, BK: tl.constexpr):
    r = tl.program_id(0)
    g = tl.program_id(1)
    ks = g * BK + tl.arange(0, BK)
    x = tl.load(X + r * K + ks, mask=ks < K, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), 0) / 448.0, 1e-12)
    tl.store(Q + r * K + ks, (x / s).to(tl.float8e4nv), mask=ks < K)
    tl.store(S + r * NG + g, s)


@triton.jit
def _w8a8_gemm(A, SA, B, SB, C, M, N, K, NG, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    gsize = tl.minimum(num_m - first_m, GROUP_M)
    pm = first_m + (pid % (GROUP_M * num_n)) % gsize
    pn = (pid % (GROUP_M * num_n)) // gsize
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, NG):
        ks = g * BK + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + ks[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(B + rn[:, None] * K + ks[None, :], mask=rn[:, None] < N, other=0.0)
        sa = tl.load(SA + rm * NG + g, mask=rm < M, other=0.0)
        sb = tl.load(SB + rn * NG + g, mask=rn < N, other=0.0)
        acc += tl.dot(a, tl.trans(b)) * (sa[:, None] * sb[None, :])
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemm_w8a8(x, q, s, bias=None, bm=128, bn=128, warps=8, stages=3):
    """x [M,K] bf16 (prefill), q fp8 [N,K], s fp32 [N,K/128] -> [M,N] bf16."""
    m, k = x.shape
    n = q.shape[0]
    ng = k // GROUP
    xq = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    xs = torch.empty((m, ng), device=x.device, dtype=torch.float32)
    _act_quant_fp8[(m, ng)](x, xq, xs, m, k, ng, GROUP)
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _w8a8_gemm[grid](xq, xs, q, s, y, m, n, k, ng, bm, bn, GROUP, 8, num_warps=warps, num_stages=stages)
    return y if bias is None else y + bias


_w8a8_cache = {'key': None, 'rx': None}


def _w8a8_enabled(layer):
    try:
        from lkqmoe.standalone import runtime_config
        pattern = runtime_config().get('w8a8_prefill')
    except Exception:
        pattern = None
    if not pattern:
        return False
    if _w8a8_cache['key'] != pattern:
        _w8a8_cache['key'] = pattern
        _w8a8_cache['rx'] = re.compile(pattern)
    return _w8a8_cache['rx'].search(getattr(layer, '_lkqmoe_name', '')) is not None


_w4a4_cache = {'key': None, 'rx': None}


def _w4a4_enabled(layer):
    """Runtime narrowing of W4A4 prefill: key "w4a4_prefill" in LKQMOE_RUNTIME_FILE (regex on the module
    name; "" = none). Absent key = every layer selected at load time."""
    try:
        from lkqmoe.standalone import runtime_config
        pattern = runtime_config().get('w4a4_prefill')
    except Exception:
        pattern = None
    if pattern is None:
        return True
    if _w4a4_cache['key'] != pattern:
        _w4a4_cache['key'] = pattern
        _w4a4_cache['rx'] = re.compile(pattern) if pattern else None
    rx = _w4a4_cache['rx']
    return rx is not None and rx.search(getattr(layer, '_lkqmoe_name', '')) is not None


def _forward(layer, x, bias):
    nv = getattr(layer, '_lkqmoe_nvfp4', None)
    if nv is not None:
        k = x.shape[-1]
        m = x.numel() // k
        x2 = x.reshape(m, k)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        n = nv['q'].shape[0]
        if m <= MAX_M:
            stats['decode_calls'] += 1
            y = gemm_w4(x2, nv, n, bias)
        else:
            stats['prefill_calls'] += 1
            y = gemm_w4a4(x2, nv, bias)
        return y.view(*x.shape[:-1], n)
    q, s = layer.weight, layer._lkqmoe_fp8_scale
    k = x.shape[-1]
    m = x.numel() // k
    x2 = x.reshape(m, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if m <= MAX_M:
        stats['decode_calls'] += 1
        y = gemm_w8(x2, q, s, bias, getattr(layer, '_lkqmoe_w8_cnt', None))
    else:
        stats['prefill_calls'] += 1
        gs = getattr(layer, '_lkqmoe_w4a4_gs', None)
        if gs is not None and not _w4a4_enabled(layer):
            gs = None
        if gs is None and _w8a8_enabled(layer):
            y = gemm_w8a8(x2, q, s, bias)
            return y.view(*x.shape[:-1], q.shape[0])
        if gs is not None:
            # FP8 storage (decode precision), FP4 tensor cores for prefill: FP8 -> BF16 -> NVFP4 per call
            from flashinfer import fp4_quantize
            wq, wsf = fp4_quantize(dequant(q, s), gs, 16, False, True)
            y = gemm_w4a4(x2, dict(q=wq, sf_sw=wsf, gs=gs), bias)
        else:
            y = torch.nn.functional.linear(x2, dequant(q, s), bias)
    return y.view(*x.shape[:-1], q.shape[0])


def _selectors():
    include = re.compile(os.environ.get('LKQMOE_FP8_INCLUDE', r'\.(self_attn|mlp)\.'))
    exclude = re.compile(os.environ.get('LKQMOE_FP8_EXCLUDE',
                                        r'(fused_qkv_a_proj|kv_b_proj|shared_experts|\.gate$|visual|vision|weights_proj)'))
    return include, exclude, int(os.environ.get('LKQMOE_FP8_MIN_ELEMS', str(4 << 20)))


def quantize_model(model):
    """After all weights are loaded and post-processed: quantize selected layers by full module name
    (LinearBase.prefix is empty for many layers, so names come from named_modules())."""
    unquant = sys.modules.get(_TARGET)
    method_cls = getattr(unquant, 'UnquantizedLinearMethod', None) if unquant else None
    if method_cls is None:
        return
    include, exclude, min_elems = _selectors()
    nv_pattern = os.environ.get('LKQMOE_NVFP4_INCLUDE', '')
    nv_include = re.compile(nv_pattern) if nv_pattern else None
    nv_exclude = re.compile(os.environ.get('LKQMOE_NVFP4_EXCLUDE',
                                           r'(fused_qkv_a_proj|kv_b_proj|\.gate$|visual|vision|weights_proj)'))
    w4_pattern = os.environ.get('LKQMOE_PREFILL_W4A4_INCLUDE', '')
    w4a4_include = re.compile(w4_pattern) if w4_pattern else None
    count = nv_count = 0
    for name, layer in model.named_modules():
        if not isinstance(getattr(layer, 'quant_method', None), method_cls):
            continue
        w = getattr(layer, 'weight', None)
        if (os.environ.get('LKQMOE_SMALL_M_GEMM') == '1' and isinstance(w, torch.Tensor) and w.is_cuda
                and w.dtype == torch.bfloat16 and w.dim() == 2 and getattr(layer, '_lkqmoe_sg_cnt', None) is None):
            # split-K tile counters for the one-launch small-M BF16 GEMM (small_gemm.gemm) of layers that stay
            # BF16; zero, each use resets its entries (unused if the layer is converted below)
            layer._lkqmoe_sg_cnt = torch.zeros(triton.cdiv(w.shape[0], 16), dtype=torch.int32, device=w.device)
        if (w is None or not isinstance(w, torch.Tensor) or not w.is_cuda or w.dtype != torch.bfloat16
                or w.dim() != 2 or w.shape[1] % GROUP or w.numel() < min_elems
                or getattr(layer, '_lkqmoe_fp8_scale', None) is not None):
            continue
        if nv_include is not None and nv_include.search(name) and not nv_exclude.search(name):
            nv = quantize_nvfp4(w.data)
            layer.weight = torch.nn.Parameter(nv['q'], requires_grad=False)
            layer._lkqmoe_nvfp4 = nv
            layer._lkqmoe_fp8_scale = 'nvfp4'   # marker: handled by this module
            stats['nvfp4_layers'] = stats.get('nvfp4_layers', 0) + 1; nv_count += 1
            stats['bf16_bytes'] += w.numel() * 2
            stats['fp8_bytes'] += nv['q'].numel() + nv['sf_lin'].numel() + nv['sf_sw'].numel()
            if len(stats['names']) < 400:
                stats['names'].append('nvfp4:' + name)
            del w
            continue
        if not include.search(name) or exclude.search(name):
            continue
        if w4a4_include is not None and w4a4_include.search(name):
            # prefill W4A4: NVFP4 global scale from the original BF16 weight (per-call NVFP4 of the FP8 copy);
            # which of these layers actually use it can be narrowed at runtime (runtime file "w4a4_prefill")
            layer._lkqmoe_w4a4_gs = (448.0 * 6.0 / w.data.float().abs().amax().clamp_min(1e-12)).reshape(1).float()
            layer._lkqmoe_name = name
            stats['w4a4_layers'] = stats.get('w4a4_layers', 0) + 1
        q, sc = quantize(w.data)
        layer.weight = torch.nn.Parameter(q, requires_grad=False)
        layer._lkqmoe_fp8_scale = sc
        # split-K tile counters of the decode kernel (zero; each use resets its own entries)
        layer._lkqmoe_w8_cnt = torch.zeros(triton.cdiv(q.shape[0], 32), dtype=torch.int32, device=q.device)
        layer._lkqmoe_name = name
        stats['layers'] += 1; count += 1
        stats['bf16_bytes'] += w.numel() * 2
        stats['fp8_bytes'] += q.numel() + sc.numel() * 4
        if len(stats['names']) < 400:
            stats['names'].append(name)
        del w
    torch.cuda.empty_cache()
    print(f'[lkqmoe] low-bit weights: {nv_count} NVFP4 + {count} FP8 layers in {type(model).__name__}; total {summary()}; '
          f'e.g. {stats["names"][:3]}', file=sys.stderr, flush=True)


def patch(module):
    method = getattr(module, 'UnquantizedLinearMethod', None)
    if method is None or getattr(method, '_lkqmoe_fp8_weights', False):
        return
    orig_apply = method.apply
    orig_apply_into = getattr(method, 'apply_into', None)

    def apply(self, layer, x, bias=None):
        if getattr(layer, '_lkqmoe_fp8_scale', None) is not None:
            return _forward(layer, x, bias)
        return orig_apply(self, layer, x, bias)

    method.apply = apply
    if orig_apply_into is not None:
        def apply_into(self, layer, x, output, bias=None):
            if getattr(layer, '_lkqmoe_fp8_scale', None) is not None:
                output.copy_(_forward(layer, x, bias).view_as(output))
                return output
            return orig_apply_into(self, layer, x, output, bias)
        method.apply_into = apply_into
    method._lkqmoe_fp8_weights = True
    print('[lkqmoe] FP8 (W8A16) weight storage for selected BF16 linear layers active', file=sys.stderr, flush=True)
    # Only the first lkqmoe import hook for a module runs; chain the small-M BF16 GEMM patch for the
    # same module (it wraps this apply and leaves FP8 layers to it).
    if os.environ.get('LKQMOE_SMALL_M_GEMM') == '1':
        from . import small_gemm
        small_gemm.patch(module)


def patch_loader(module):
    cls = getattr(module, 'DefaultModelLoader', None)
    if cls is None or getattr(cls, '_lkqmoe_fp8_weights', False):
        return
    original = cls.__dict__['load_weights_and_postprocess']
    fn = original.__func__ if isinstance(original, staticmethod) else original

    def load_weights_and_postprocess(model, weights, target_device):
        result = fn(model, weights, target_device)
        quantize_model(model)
        return result

    cls.load_weights_and_postprocess = staticmethod(load_weights_and_postprocess)
    cls._lkqmoe_fp8_weights = True


def summary():
    return dict(fp8_layers=stats['layers'], nvfp4_layers=stats.get('nvfp4_layers', 0),
                w4a4_prefill_layers=stats.get('w4a4_layers', 0), bf16_gib=round(stats['bf16_bytes'] / 2**30, 2),
                fp8_gib=round(stats['fp8_bytes'] / 2**30, 2), decode_calls=stats['decode_calls'],
                prefill_calls=stats['prefill_calls'])


_LOADER_TARGET = 'sglang.srt.model_loader.loader'
_PATCHES = {_TARGET: patch, _LOADER_TARGET: patch_loader}


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate, fn):
        self.delegate = delegate
        self.fn = fn

    def create_module(self, spec):
        fn = getattr(self.delegate, 'create_module', None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        self.fn(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        fn = _PATCHES.get(fullname)
        if fn is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader, fn)
        return spec


def install():
    for name, fn in _PATCHES.items():
        if name in sys.modules:
            fn(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
