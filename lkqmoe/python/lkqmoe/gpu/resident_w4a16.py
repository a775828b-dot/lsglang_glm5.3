"""Opt-in (LKQMOE_RESIDENT_W4A16=1): GPU-resident NVFP4 MoE layers decode as W4A16.

sglang runs GPU-resident NVFP4 experts through flashinfer's CUTLASS FP4 MoE, which also quantizes
the activations to NVFP4 (W4A4). For small batches (decode / speculative verify, M <= 16 tokens)
this module computes the same layer with BF16 activations instead (W4A16), directly from the
weights already prepared for flashinfer (packed E2M1 [E, 2I, H/2] / [E, H, I/2], E4M3 block scales
in the swizzled 128x4 layout, per-expert global scales), so no extra memory is used. Larger batches
(prefill) keep the original W4A4 path.

Kernels (CUDA-graph capturable, deterministic):
1. used experts -> slots (static shapes, no host sync); each slot reads its expert once and serves
   all M tokens (tokens not routed to it get weight 0);
2. gate/up: dequantize FP4 tiles in registers (SM100/SM120 hardware cvt.rn.f16x2.e2m1x2, FP16 tiles;
   software BF16 path otherwise), tensor-core dot, FP32 accumulation, global scales, clamped SwiGLU
   (gate <= L, |up| <= L), intermediate rounded to BF16;
3. down per slot into an FP32 partial buffer; 4. per token, sum of its top-k routes in route order.
Routing weights already include routed_scaling_factor (sglang fuses it into top-k for this backend).
LKQMOE_RESIDENT_W4A16_SKIP_DRAFT=1 keeps the NextN draft layer on W4A4: its KV / extend path runs W4A4, and
a W4A16 draft decode lowered GLM-5.3 acceptance (multi-turn 53 vs 62 tok/s).
"""
import importlib.abc
import importlib.machinery
import os
import sys

import torch
import triton
import triton.language as tl

_TARGET = 'sglang.srt.layers.quantization.modelopt_quant'
MAX_M = 16
stats = {'calls': 0, 'layers': 0}
# launch configuration (benchmarked on RTX PRO 5000 with GLM-5.3 shapes, real layer, CUDA graph, cold L2)
CONFIG = dict(hw=True, gbn=32, dbn=64, bk=256, intdeq=True, gw=4, gs=2, dw=4, ds=2, ks=1)
# hw: SM100/SM120 cvt.rn.f16x2.e2m1x2 dequantization with FP16 tiles (layer 20, M=5, 23 experts: 352 us,
# ~0.92 TB/s; the software paths reach 488-547 us)


@triton.jit
def _fp4_tile(Q, S, e, rows, k0, QE, QR, ROWS_P, COLS_P, BN: tl.constexpr, BK: tl.constexpr,
              INTDEQ: tl.constexpr = False):
    """Dequantized (without the global scale) tile [BN, BK] of expert e: rows x [k0, k0+BK).
    QE / QR: expert and row strides (bytes) of the packed weights; ROWS_P x COLS_P: padded scale matrix."""
    pb = tl.load(Q + e * QE + rows[:, None] * QR + (k0 // 2 + tl.arange(0, BK // 2))[None, :])
    codes = tl.reshape(tl.join(pb & 15, pb >> 4), (BN, BK))
    # swizzled 128x4 E4M3 scales: [rb][cb][32][4][4] per expert (ROWS_P x COLS_P padded)
    c = k0 // 16 + tl.arange(0, BK // 16)
    off = ((rows[:, None] // 128) * (COLS_P // 4) + c[None, :] // 4) * 512 + (rows[:, None] % 32) * 16 \
        + ((rows[:, None] % 128) // 32) * 4 + c[None, :] % 4
    sb = tl.load(S + e * ROWS_P * COLS_P + off)
    if INTDEQ:
        # E2M1 -> BF16 bits: m=0 -> 0, m=1 -> 0.5 (0x3F00), m>=2 -> exponent 126+(m>>1), mantissa bit (m&1);
        # value x E4M3 scale is exact in BF16 (2 + 4 significant bits)
        cw = codes.to(tl.int32)
        m = cw & 7
        bits = tl.where(m < 2, m * 0x3F00, ((126 + (m >> 1)) << 7) | ((m & 1) << 6)) | ((cw & 8) << 12)
        v = bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
        sc = sb.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 16, 16)), (BN, BK))
        return v * sc
    m = (codes & 7).to(tl.float32)
    v = tl.where(m < 4.0, m * 0.5, tl.where(m < 6.0, m - 2.0, m * 2.0 - 8.0))
    v = tl.where((codes & 8) != 0, -v, v)
    sc = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 16, 16)), (BN, BK))
    return v * sc


@triton.jit
def _fp4_halves(Q, S, e, rows, k0, QE, QR, ROWS_P, COLS_P, BN: tl.constexpr, BK: tl.constexpr):
    """Two BF16 tiles [BN, BK/2] of expert e (without the global scale): columns k0+2j (low nibbles) and
    k0+2j+1 (high nibbles). Values are built from E2M1 bits directly (no int->float conversion)."""
    pb = tl.load(Q + e * QE + rows[:, None] * QR + (k0 // 2 + tl.arange(0, BK // 2))[None, :]).to(tl.int32)
    c = k0 // 16 + tl.arange(0, BK // 16)
    off = ((rows[:, None] // 128) * (COLS_P // 4) + c[None, :] // 4) * 512 + (rows[:, None] % 32) * 16         + ((rows[:, None] % 128) // 32) * 4 + c[None, :] % 4
    sb = tl.load(S + e * ROWS_P * COLS_P + off)
    sc = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32).to(tl.bfloat16)
    sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 16, 8)), (BN, BK // 2))
    lo = pb & 15
    hi = pb >> 4
    ml = lo & 7
    mh = hi & 7
    bl = tl.where(ml < 2, ml * 0x3F00, ((126 + (ml >> 1)) << 7) | ((ml & 1) << 6)) | ((lo & 8) << 12)
    bh = tl.where(mh < 2, mh * 0x3F00, ((126 + (mh >> 1)) << 7) | ((mh & 1) << 6)) | ((hi & 8) << 12)
    wl = bl.to(tl.int16).to(tl.bfloat16, bitcast=True) * sc
    wh = bh.to(tl.int16).to(tl.bfloat16, bitcast=True) * sc
    return wl, wh


@triton.jit
def _gate_up(X, SLOT_E, Q13, S13, G13, MID, M, H, I, LIMIT, R13P, C13P,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, INTDEQ: tl.constexpr):
    slot = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(SLOT_E + slot)
    if e < 0:
        return
    e = e.to(tl.int64)
    tok = tl.arange(0, BM)
    n = nb * BN + tl.arange(0, BN)
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    qe = 2 * I * (H // 2)
    for k0 in range(0, H, BK):
        if INTDEQ:
            ke = k0 + 2 * tl.arange(0, BK // 2)
            xe = tl.load(X + tok[:, None] * H + ke[None, :], mask=tok[:, None] < M, other=0.0)
            xo = tl.load(X + tok[:, None] * H + ke[None, :] + 1, mask=tok[:, None] < M, other=0.0)
            gl, gh = _fp4_halves(Q13, S13, e, n, k0, qe, H // 2, R13P, C13P, BN, BK)
            ul, uh = _fp4_halves(Q13, S13, e, n + I, k0, qe, H // 2, R13P, C13P, BN, BK)
            accg = tl.dot(xe, tl.trans(gl), accg)
            accg = tl.dot(xo, tl.trans(gh), accg)
            accu = tl.dot(xe, tl.trans(ul), accu)
            accu = tl.dot(xo, tl.trans(uh), accu)
        else:
            ks = k0 + tl.arange(0, BK)
            x = tl.load(X + tok[:, None] * H + ks[None, :], mask=tok[:, None] < M, other=0.0)
            wg = _fp4_tile(Q13, S13, e, n, k0, qe, H // 2, R13P, C13P, BN, BK).to(tl.bfloat16)
            wu = _fp4_tile(Q13, S13, e, n + I, k0, qe, H // 2, R13P, C13P, BN, BK).to(tl.bfloat16)
            accg = tl.dot(x, tl.trans(wg), accg)
            accu = tl.dot(x, tl.trans(wu), accu)
    g = accg * tl.load(G13 + e * 2)
    u = accu * tl.load(G13 + e * 2 + 1)
    if LIMIT > 0:
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        g = tl.minimum(g, LIMIT)
    val = g / (1.0 + tl.exp(-g)) * u
    tl.store(MID + (slot * BM + tok[:, None]) * I + n[None, :], val.to(tl.bfloat16), mask=tok[:, None] < M)


@triton.jit
def _down(MID, SLOT_E, Q2, S2, G2, PART, M, H, I, R2P, C2P,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, INTDEQ: tl.constexpr):
    slot = tl.program_id(0)
    hb = tl.program_id(1)
    e = tl.load(SLOT_E + slot)
    if e < 0:
        return
    e = e.to(tl.int64)
    tok = tl.arange(0, BM)
    h = hb * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    qe = H * (I // 2)
    for k0 in range(0, I, BK):
        if INTDEQ:
            ke = k0 + 2 * tl.arange(0, BK // 2)
            ae = tl.load(MID + (slot * BM + tok[:, None]) * I + ke[None, :], mask=tok[:, None] < M, other=0.0)
            ao = tl.load(MID + (slot * BM + tok[:, None]) * I + ke[None, :] + 1, mask=tok[:, None] < M, other=0.0)
            wl, wh = _fp4_halves(Q2, S2, e, h, k0, qe, I // 2, R2P, C2P, BN, BK)
            acc = tl.dot(ae, tl.trans(wl), acc)
            acc = tl.dot(ao, tl.trans(wh), acc)
        else:
            ks = k0 + tl.arange(0, BK)
            a = tl.load(MID + (slot * BM + tok[:, None]) * I + ks[None, :], mask=tok[:, None] < M, other=0.0)
            w = _fp4_tile(Q2, S2, e, h, k0, qe, I // 2, R2P, C2P, BN, BK).to(tl.bfloat16)
            acc = tl.dot(a, tl.trans(w), acc)
    acc = acc * tl.load(G2 + e)
    tl.store(PART + (slot * BM + tok[:, None]) * H + h[None, :], acc, mask=tok[:, None] < M)


@triton.jit
def _slot_map(IDS, NIDS, USED, RANK, SLOT_E, ROUTE_SLOT, E, NSLOTS, BE: tl.constexpr, BI: tl.constexpr):
    """One program: used experts -> slots (rank among used experts), slot -> expert (-1 when empty,
    entry NSLOTS is the sink) and route -> slot. Same integers as the torch scatter / cumsum sequence."""
    e = tl.arange(0, BE)
    emask = e < E
    tl.store(USED + e, 0, mask=emask)
    j = tl.arange(0, 2 * BI)
    tl.store(SLOT_E + j, -1, mask=j < NSLOTS + 1)
    tl.debug_barrier()
    i = tl.arange(0, BI)
    imask = i < NIDS
    ids = tl.load(IDS + i, mask=imask, other=0).to(tl.int32)
    tl.store(USED + ids, 1, mask=imask)
    tl.debug_barrier()
    used = tl.load(USED + e, mask=emask, other=0)
    rank = tl.cumsum(used, 0) - 1
    tl.store(RANK + e, rank, mask=emask)
    tl.store(SLOT_E + rank, e, mask=emask & (used > 0))
    tl.debug_barrier()
    tl.store(ROUTE_SLOT + i, tl.load(RANK + ids, mask=imask, other=0), mask=imask)


@triton.jit
def _combine(PART, ROUTE_SLOT, W, OUT, M, H, K: tl.constexpr, BM: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), dtype=tl.float32)
    for k in tl.static_range(K):   # route order: deterministic
        s = tl.load(ROUTE_SLOT + t * K + k)
        w = tl.load(W + t * K + k)
        acc += w * tl.load(PART + (s * BM + t) * H + h, mask=h < H, other=0.0)
    tl.store(OUT + t * H + h, acc.to(OUT.dtype.element_ty), mask=h < H)


@triton.jit
def _fp4_halves_hw(Q, S, e, rows, k0, QE, QR, ROWS_P, COLS_P, BN: tl.constexpr, BK: tl.constexpr):
    """Like _fp4_halves but with the SM100/SM120 hardware conversion cvt.rn.f16x2.e2m1x2 (one instruction per
    packed byte) and FP16 tiles: values x E4M3 scale are exact in FP16."""
    pb = tl.load(Q + e * QE + rows[:, None] * QR + (k0 // 2 + tl.arange(0, BK // 2))[None, :]).to(tl.uint32)
    r = tl.inline_asm_elementwise(
        asm="{ .reg .b8 t; cvt.u8.u32 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
        constraints="=r,r", args=[pb], dtype=tl.uint32, is_pure=True, pack=1)
    lo = (r & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (r >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    c = k0 // 16 + tl.arange(0, BK // 16)
    off = ((rows[:, None] // 128) * (COLS_P // 4) + c[None, :] // 4) * 512 + (rows[:, None] % 32) * 16 + ((rows[:, None] % 128) // 32) * 4 + c[None, :] % 4
    sb = tl.load(S + e * ROWS_P * COLS_P + off)
    sc = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32).to(tl.float16)
    sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 16, 8)), (BN, BK // 2))
    return lo * sc, hi * sc


@triton.jit
def _gate_up_hw(X, SLOT_E, Q13, S13, G13, MID, M, H, I, LIMIT, R13P, C13P,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    slot = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(SLOT_E + slot)
    if e < 0:
        return
    e = e.to(tl.int64)
    tok = tl.arange(0, BM)
    n = nb * BN + tl.arange(0, BN)
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    qe = 2 * I * (H // 2)
    for k0 in range(0, H, BK):
        ke = k0 + 2 * tl.arange(0, BK // 2)
        xe = tl.load(X + tok[:, None] * H + ke[None, :], mask=tok[:, None] < M, other=0.0).to(tl.float16)
        xo = tl.load(X + tok[:, None] * H + ke[None, :] + 1, mask=tok[:, None] < M, other=0.0).to(tl.float16)
        gl, gh = _fp4_halves_hw(Q13, S13, e, n, k0, qe, H // 2, R13P, C13P, BN, BK)
        ul, uh = _fp4_halves_hw(Q13, S13, e, n + I, k0, qe, H // 2, R13P, C13P, BN, BK)
        accg = tl.dot(xe, tl.trans(gl), accg)
        accg = tl.dot(xo, tl.trans(gh), accg)
        accu = tl.dot(xe, tl.trans(ul), accu)
        accu = tl.dot(xo, tl.trans(uh), accu)
    g = accg * tl.load(G13 + e * 2)
    u = accu * tl.load(G13 + e * 2 + 1)
    if LIMIT > 0:
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        g = tl.minimum(g, LIMIT)
    val = g / (1.0 + tl.exp(-g)) * u
    tl.store(MID + (slot * BM + tok[:, None]) * I + n[None, :], val.to(tl.bfloat16), mask=tok[:, None] < M)


@triton.jit
def _down_hw(MID, SLOT_E, Q2, S2, G2, PART, M, H, I, R2P, C2P,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    slot = tl.program_id(0)
    hb = tl.program_id(1)
    e = tl.load(SLOT_E + slot)
    if e < 0:
        return
    e = e.to(tl.int64)
    tok = tl.arange(0, BM)
    h = hb * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    qe = H * (I // 2)
    for k0 in range(0, I, BK):
        ke = k0 + 2 * tl.arange(0, BK // 2)
        ae = tl.load(MID + (slot * BM + tok[:, None]) * I + ke[None, :], mask=tok[:, None] < M, other=0.0).to(tl.float16)
        ao = tl.load(MID + (slot * BM + tok[:, None]) * I + ke[None, :] + 1, mask=tok[:, None] < M, other=0.0).to(tl.float16)
        wl, wh = _fp4_halves_hw(Q2, S2, e, h, k0, qe, I // 2, R2P, C2P, BN, BK)
        acc = tl.dot(ae, tl.trans(wl), acc)
        acc = tl.dot(ao, tl.trans(wh), acc)
    acc = acc * tl.load(G2 + e)
    tl.store(PART + (slot * BM + tok[:, None]) * H + h[None, :], acc, mask=tok[:, None] < M)


# ---- split-K variant (CONFIG['ks'] > 1): more programs per layer; deterministic fixed-order reductions.

@triton.jit
def _gate_up_sk(X, SLOT_E, Q13, S13, P13, M, H, I, R13P, C13P, NSLOTS, KSPAN,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    slot = tl.program_id(0)
    nb = tl.program_id(1)
    ks_id = tl.program_id(2)
    e = tl.load(SLOT_E + slot)
    if e < 0:
        return
    e = e.to(tl.int64)
    tok = tl.arange(0, BM)
    n = nb * BN + tl.arange(0, BN)
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    qe = 2 * I * (H // 2)
    for kk in range(0, KSPAN, BK):
        k0 = ks_id * KSPAN + kk
        ke = k0 + 2 * tl.arange(0, BK // 2)
        xe = tl.load(X + tok[:, None] * H + ke[None, :], mask=tok[:, None] < M, other=0.0)
        xo = tl.load(X + tok[:, None] * H + ke[None, :] + 1, mask=tok[:, None] < M, other=0.0)
        gl, gh = _fp4_halves(Q13, S13, e, n, k0, qe, H // 2, R13P, C13P, BN, BK)
        ul, uh = _fp4_halves(Q13, S13, e, n + I, k0, qe, H // 2, R13P, C13P, BN, BK)
        accg = tl.dot(xe, tl.trans(gl), accg)
        accg = tl.dot(xo, tl.trans(gh), accg)
        accu = tl.dot(xe, tl.trans(ul), accu)
        accu = tl.dot(xo, tl.trans(uh), accu)
    base = (ks_id * NSLOTS + slot) * BM
    tl.store(P13 + (base + tok[:, None]) * (2 * I) + n[None, :], accg, mask=tok[:, None] < M)
    tl.store(P13 + (base + tok[:, None]) * (2 * I) + I + n[None, :], accu, mask=tok[:, None] < M)


@triton.jit
def _swiglu_sk(P13, SLOT_E, G13, MID, M, I, LIMIT, NSLOTS, KS: tl.constexpr, BM: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.load(SLOT_E + slot)
    if e < 0:
        return
    if t >= M:
        return
    e = e.to(tl.int64)
    n = tl.program_id(2) * B + tl.arange(0, B)
    g = tl.zeros((B,), dtype=tl.float32)
    u = tl.zeros((B,), dtype=tl.float32)
    for k in tl.static_range(KS):   # fixed order
        row = ((k * NSLOTS + slot) * BM + t) * (2 * I)
        g += tl.load(P13 + row + n, mask=n < I, other=0.0)
        u += tl.load(P13 + row + I + n, mask=n < I, other=0.0)
    g = g * tl.load(G13 + e * 2)
    u = u * tl.load(G13 + e * 2 + 1)
    if LIMIT > 0:
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        g = tl.minimum(g, LIMIT)
    tl.store(MID + (slot * BM + t) * I + n, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=n < I)


@triton.jit
def _down_sk(MID, SLOT_E, Q2, S2, G2, PART, M, H, I, R2P, C2P, NSLOTS, KSPAN,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    slot = tl.program_id(0)
    hb = tl.program_id(1)
    ks_id = tl.program_id(2)
    e = tl.load(SLOT_E + slot)
    if e < 0:
        return
    e = e.to(tl.int64)
    tok = tl.arange(0, BM)
    h = hb * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    qe = H * (I // 2)
    for kk in range(0, KSPAN, BK):
        k0 = ks_id * KSPAN + kk
        ke = k0 + 2 * tl.arange(0, BK // 2)
        ae = tl.load(MID + (slot * BM + tok[:, None]) * I + ke[None, :], mask=tok[:, None] < M, other=0.0)
        ao = tl.load(MID + (slot * BM + tok[:, None]) * I + ke[None, :] + 1, mask=tok[:, None] < M, other=0.0)
        wl, wh = _fp4_halves(Q2, S2, e, h, k0, qe, I // 2, R2P, C2P, BN, BK)
        acc = tl.dot(ae, tl.trans(wl), acc)
        acc = tl.dot(ao, tl.trans(wh), acc)
    acc = acc * tl.load(G2 + e)
    tl.store(PART + ((ks_id * NSLOTS + slot) * BM + tok[:, None]) * H + h[None, :], acc, mask=tok[:, None] < M)


@triton.jit
def _combine_sk(PART, ROUTE_SLOT, W, OUT, M, H, NSLOTS, K: tl.constexpr, KS: tl.constexpr, BM: tl.constexpr,
                B: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), dtype=tl.float32)
    for k in tl.static_range(K):          # route order, then K-slice order: deterministic
        s = tl.load(ROUTE_SLOT + t * K + k)
        w = tl.load(W + t * K + k)
        part = tl.zeros((B,), dtype=tl.float32)
        for q in tl.static_range(KS):
            part += tl.load(PART + ((q * NSLOTS + s) * BM + t) * H + h, mask=h < H, other=0.0)
        acc += w * part
    tl.store(OUT + t * H + h, acc.to(OUT.dtype.element_ty), mask=h < H)


def moe_w4a16(x, topk_ids, topk_w, q13, s13, g13, q2, s2, g2, limit):
    """x [M,H] bf16, topk_ids [M,K], topk_w [M,K] -> [M,H] (x.dtype). Weights in flashinfer CUTLASS layout."""
    m, h = x.shape
    k = topk_ids.shape[1]
    E, two_i, _ = q13.shape
    i = two_i // 2
    dev = x.device
    nslots = m * k
    if os.environ.get('LKQMOE_RESIDENT_W4A16_TORCH_MAP') != '1' and E <= 1024 and nslots <= 256:
        # used experts -> slots in one kernel (static shapes): slot index = rank of the expert among used ones
        ids = topk_ids.contiguous()
        used = torch.empty(E, dtype=torch.int32, device=dev)
        rank = torch.empty(E, dtype=torch.int32, device=dev)
        slot_e = torch.empty((nslots + 1,), dtype=torch.int32, device=dev)   # last entry: sink for unused experts
        route_slot = torch.empty((m, k), dtype=torch.int32, device=dev)
        _slot_map[(1,)](ids, nslots, used, rank, slot_e, route_slot, E, nslots, triton.next_power_of_2(E),
                        triton.next_power_of_2(nslots), num_warps=4)
    else:
        ids = topk_ids.to(torch.int64)
        used = torch.zeros(E, dtype=torch.int32, device=dev)
        used.scatter_(0, ids.reshape(-1), 1)
        rank = torch.cumsum(used, 0) - 1                      # slot of every used expert
        slot_e = torch.full((nslots + 1,), -1, dtype=torch.int32, device=dev)   # last entry: sink for unused experts
        ar = torch.arange(E, dtype=torch.int32, device=dev)
        slot_e.scatter_(0, torch.where(used.bool(), rank, torch.full_like(rank, nslots)).long(),
                        torch.where(used.bool(), ar, torch.full_like(ar, -1)))
        route_slot = rank[ids].to(torch.int32).contiguous()
    # every slot serves all tokens; weights of tokens not routed to it are applied as 0 in _combine
    xb = x.to(torch.bfloat16).contiguous()
    cfg = CONFIG
    if cfg.get('ks', 1) > 1:
        return _moe_w4a16_sk(xb, m, h, i, k, nslots, slot_e, route_slot, topk_w, q13, s13, g13, q2, s2, g2, limit, x.dtype)
    mid = torch.empty((nslots, MAX_M, i), dtype=torch.bfloat16, device=dev)
    part = torch.empty((nslots, MAX_M, h), dtype=torch.float32, device=dev)
    if cfg.get('hw', False):
        _gate_up_hw[(nslots, i // cfg['gbn'])](xb, slot_e, q13.view(torch.uint8), s13.view(torch.uint8), g13, mid,
                                               m, h, i, float(limit), s13.shape[1], s13.shape[2], MAX_M, cfg['gbn'],
                                               cfg['bk'], num_warps=cfg['gw'], num_stages=cfg['gs'])
        _down_hw[(nslots, h // cfg['dbn'])](mid, slot_e, q2.view(torch.uint8), s2.view(torch.uint8), g2, part,
                                            m, h, i, s2.shape[1], s2.shape[2], MAX_M, cfg['dbn'], cfg['bk'],
                                            num_warps=cfg['dw'], num_stages=cfg['ds'])
        out = torch.empty((m, h), dtype=x.dtype, device=dev)
        _combine[(m, triton.cdiv(h, 1024))](part, route_slot, topk_w.to(torch.float32).contiguous(), out, m, h, k,
                                            MAX_M, 1024)
        return out
    r13p, c13p = s13.shape[1], s13.shape[2]
    r2p, c2p = s2.shape[1], s2.shape[2]
    s13u, s2u = s13.view(torch.uint8), s2.view(torch.uint8)
    _gate_up[(nslots, i // cfg['gbn'])](xb, slot_e, q13.view(torch.uint8), s13u, g13, mid, m, h, i, float(limit),
                                        r13p, c13p, MAX_M, cfg['gbn'], cfg['bk'], cfg['intdeq'],
                                        num_warps=cfg['gw'], num_stages=cfg['gs'])
    _down[(nslots, h // cfg['dbn'])](mid, slot_e, q2.view(torch.uint8), s2u, g2, part, m, h, i, r2p, c2p,
                                     MAX_M, cfg['dbn'], cfg['bk'], cfg['intdeq'], num_warps=cfg['dw'], num_stages=cfg['ds'])
    out = torch.empty((m, h), dtype=x.dtype, device=dev)
    _combine[(m, triton.cdiv(h, 1024))](part, route_slot, topk_w.to(torch.float32).contiguous(), out, m, h, k, MAX_M, 1024)
    return out


def patch(module):
    cls = getattr(module, 'ModelOptNvFp4FusedMoEMethod', None)
    if cls is None or getattr(cls, '_lkqmoe_resident_w4a16', False):
        return
    original = cls.apply
    max_m = int(os.environ.get('LKQMOE_RESIDENT_W4A16_MAX_M', str(MAX_M)))
    # LKQMOE_RESIDENT_W4A16_SKIP_DRAFT=1: keep the NextN/MTP draft layer (layer_name model.decoder...) on W4A4
    skip_draft = os.environ.get('LKQMOE_RESIDENT_W4A16_SKIP_DRAFT') == '1'

    def apply(self, layer, dispatch_output):
        topk = getattr(dispatch_output, 'topk_output', None)
        x = getattr(dispatch_output, 'hidden_states', None)
        if (getattr(layer, 'is_gpu_resident_layer', False) and topk is not None and x is not None
                and not (skip_draft and 'decoder' in (getattr(layer, 'layer_name', '') or ''))
                and isinstance(x, torch.Tensor) and x.dim() == 2 and 0 < x.shape[0] <= max_m
                and hasattr(topk, 'topk_ids') and hasattr(topk, 'topk_weights')
                and getattr(layer, 'w13_blockscale_swizzled', None) is not None
                and getattr(layer, 'w13_weight_scale_2', None) is not None
                and layer.w13_weight_scale_2.dim() == 2):
            if not getattr(layer, '_lkqmoe_rw_logged', False):
                layer._lkqmoe_rw_logged = True
                stats['layers'] += 1
                print(f'[lkqmoe] resident W4A16 decode: layer {getattr(layer, "layer_id", "?")} '
                      f'w13 {tuple(layer.w13_weight.shape)} sf {tuple(layer.w13_blockscale_swizzled.shape)}',
                      file=sys.stderr, flush=True)
            stats['calls'] += 1
            limit = getattr(self.moe_runner_config, 'swiglu_limit', None) or 0.0
            out = moe_w4a16(x, topk.topk_ids, topk.topk_weights,
                            layer.w13_weight, layer.w13_blockscale_swizzled, layer.w13_weight_scale_2.float().contiguous(),
                            layer.w2_weight, layer.w2_blockscale_swizzled, layer.w2_weight_scale_2.float().contiguous(),
                            limit)
            from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
            return StandardCombineInput(hidden_states=out)
        return original(self, layer, dispatch_output)

    cls.apply = apply
    cls._lkqmoe_resident_w4a16 = True
    print('[lkqmoe] resident NVFP4 MoE: W4A16 for decode batches (<= %d tokens), W4A4 prefill unchanged' % max_m,
          file=sys.stderr, flush=True)


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



def _moe_w4a16_sk(xb, m, h, i, k, nslots, slot_e, route_slot, topk_w, q13, s13, g13, q2, s2, g2, limit, out_dtype):
    cfg = CONFIG
    ks, ks2, bk = cfg['ks'], cfg.get('ks2', cfg['ks']), cfg['bk']
    dev = xb.device
    kspan = h // ks
    kspan2 = i // ks2
    p13 = torch.empty((ks, nslots, MAX_M, 2 * i), dtype=torch.float32, device=dev)
    _gate_up_sk[(nslots, i // cfg['gbn'], ks)](xb, slot_e, q13.view(torch.uint8), s13.view(torch.uint8), p13,
                                               m, h, i, s13.shape[1], s13.shape[2], nslots, kspan,
                                               MAX_M, cfg['gbn'], bk, num_warps=cfg['gw'], num_stages=cfg['gs'])
    mid = torch.empty((nslots, MAX_M, i), dtype=torch.bfloat16, device=dev)
    _swiglu_sk[(nslots, MAX_M, triton.cdiv(i, 1024))](p13, slot_e, g13, mid, m, i, float(limit), nslots, ks, MAX_M, 1024)
    part = torch.empty((ks2, nslots, MAX_M, h), dtype=torch.float32, device=dev)
    _down_sk[(nslots, h // cfg['dbn'], ks2)](mid, slot_e, q2.view(torch.uint8), s2.view(torch.uint8), g2, part,
                                             m, h, i, s2.shape[1], s2.shape[2], nslots, kspan2,
                                             MAX_M, cfg['dbn'], bk, num_warps=cfg['dw'], num_stages=cfg['ds'])
    out = torch.empty((m, h), dtype=out_dtype, device=dev)
    _combine_sk[(m, triton.cdiv(h, 1024))](part, route_slot, topk_w.to(torch.float32).contiguous(), out, m, h,
                                           nslots, k, ks2, MAX_M, 1024)
    return out
