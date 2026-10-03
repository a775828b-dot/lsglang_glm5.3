"""Opt-in (LKQMOE_GLM53_MHC=1): fully fused mHC pre for GLM-5.3 on SM120.

sglang's SM120 path (`mhc_sm120_triton.mhc_pre_triton`) fuses only the part after the
projection. The projection itself, F.linear(x_flat [s, n*h] FP32, fn [2n+n*n, n*h] FP32),
runs in cuBLAS; for decode (s<=8) cuBLAS picks gemmSN with 3 CTAs: ~92 us per call, 90 calls
per GLM verify step (~8 ms). The RMS statistics and the final pre-weighted sum over the n
streams are further torch kernels (x.float(), square, mean, rsqrt, mul, sum, cast).

Here: one split-K projection kernel that reads the residual once (any float dtype), produces
deterministic FP32 partial sums of the projection and of x^2, and one tail kernel that adds the
partials in a fixed order, applies the unchanged sigmoid / softmax / Sinkhorn math and writes
layer_input = sum_j pre_j * residual_j. Arithmetic stays FP32 (IEEE dot, no TF32); only the
summation order differs from cuBLAS/torch. Same signature and outputs as mhc_pre_triton.
"""
import importlib.abc
import importlib.machinery
import os
import sys

import torch
import triton
import triton.language as tl

_TARGET = 'sglang.kernels.ops.layernorm.mhc_sm120_triton'


@triton.jit
def _proj_partial_kernel(x_ptr, fn_ptr, part_ptr, sq_ptr, s, K, k_chunk, x_stride,
                         M: tl.constexpr, MP: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_s * BS + tl.arange(0, BS)
    cols = tl.arange(0, MP)
    rmask = rows < s
    cmask = cols < M
    acc = tl.zeros((BS, MP), dtype=tl.float32)
    sq = tl.zeros((BS,), dtype=tl.float32)
    k0 = pid_k * k_chunk
    for kk in range(0, k_chunk, BK):
        ks = k0 + kk + tl.arange(0, BK)
        kmask = ks < K
        x = tl.load(x_ptr + rows[:, None] * x_stride + ks[None, :],
                    mask=rmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
        w = tl.load(fn_ptr + cols[:, None] * K + ks[None, :],
                    mask=cmask[:, None] & kmask[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc, input_precision='ieee')
        sq += tl.sum(x * x, axis=1)
    base = pid_k * s
    tl.store(part_ptr + (base + rows[:, None]) * MP + cols[None, :], acc, mask=rmask[:, None])
    tl.store(sq_ptr + base + rows, sq, mask=rmask)


@triton.jit
def _tail_kernel(part_ptr, sq_ptr, scale_ptr, base_ptr, res_ptr, res_stride,
                 pre_ptr, post_ptr, comb_ptr, li_ptr,
                 s, split, inv_k, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, H,
                 SINKHORN: tl.constexpr, N: tl.constexpr, MP: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    ij = offs[:, None] * N + offs[None, :]
    pre_raw = tl.zeros((N,), dtype=tl.float32)
    post_raw = tl.zeros((N,), dtype=tl.float32)
    comb_raw = tl.zeros((N, N), dtype=tl.float32)
    sq = 0.0
    for p in range(0, split):  # fixed order: deterministic
        b = (p * s + row) * MP
        pre_raw += tl.load(part_ptr + b + offs)
        post_raw += tl.load(part_ptr + b + N + offs)
        comb_raw += tl.load(part_ptr + b + 2 * N + ij)
        sq += tl.load(sq_ptr + p * s + row)
    r = 1.0 / tl.sqrt(sq * inv_k + rms_eps)

    sc0 = tl.load(scale_ptr + 0)
    sc1 = tl.load(scale_ptr + 1)
    sc2 = tl.load(scale_ptr + 2)
    pre = tl.sigmoid(pre_raw * r * sc0 + tl.load(base_ptr + offs)) + hc_pre_eps
    post = hc_post_mult * tl.sigmoid(post_raw * r * sc1 + tl.load(base_ptr + N + offs))
    comb = comb_raw * r * sc2 + tl.load(base_ptr + 2 * N + ij)
    mx = tl.max(comb, axis=1)[:, None]
    e = tl.exp(comb - mx)
    comb = e / tl.sum(e, axis=1)[:, None]
    comb = comb + hc_sinkhorn_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_sinkhorn_eps)
    for _ in tl.static_range(SINKHORN - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_sinkhorn_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_sinkhorn_eps)
    tl.store(pre_ptr + row * N + offs, pre)
    tl.store(post_ptr + row * N + offs, post)
    tl.store(comb_ptr + row * N * N + ij, comb)

    # layer_input[row] = sum_j pre_j * residual[row, j, :]
    for h0 in range(0, H, BH):
        hs = h0 + tl.arange(0, BH)
        hm = hs < H
        res = tl.load(res_ptr + row * res_stride + offs[:, None] * H + hs[None, :],
                      mask=hm[None, :], other=0.0).to(tl.float32)
        li = tl.sum(pre[:, None] * res, axis=0)
        tl.store(li_ptr + row * H + hs, li.to(li_ptr.dtype.element_ty), mask=hm)


def mhc_pre_fused(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                  hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    """Drop-in for mhc_pre_triton. residual: (s, n, h); returns (post (s,n,1), comb (s,n,n), layer_input (s,h))."""
    s, n, h = residual.shape
    assert n & (n - 1) == 0, f'n must be a power of two, got {n}'
    M, K = fn.shape
    assert K == n * h and M == 2 * n + n * n and fn.dtype == torch.float32
    residual = residual.contiguous()
    fn = fn.contiguous()
    MP = max(16, triton.next_power_of_2(M))
    BS, BK = 16, 64
    if s <= 64:
        split = 64 if K % (64 * BK) == 0 else 1          # decode/verify: spread fn over 64 CTAs
    else:
        split = max(1, min(8, 2048 // triton.cdiv(s, BS)))  # long batches already fill the GPU
        while split > 1 and K % (split * BK):
            split //= 2
        BS = 32 if s >= 4096 else 16
    k_chunk = K // split
    dev = residual.device
    part = torch.empty((split, s, MP), device=dev, dtype=torch.float32)
    sq = torch.empty((split, s), device=dev, dtype=torch.float32)
    x2d = residual.view(s, K)
    _proj_partial_kernel[(triton.cdiv(s, BS), split)](
        x2d, fn, part, sq, s, K, k_chunk, x2d.stride(0),
        M=M, MP=MP, BS=BS, BK=BK, num_warps=4)
    pre = torch.empty((s, n), device=dev, dtype=torch.float32)
    post = torch.empty((s, n), device=dev, dtype=torch.float32)
    comb = torch.empty((s, n, n), device=dev, dtype=torch.float32)
    layer_input = torch.empty((s, h), device=dev, dtype=residual.dtype)
    _tail_kernel[(s,)](
        part, sq, hc_scale, hc_base, residual, residual.stride(0),
        pre, post, comb, layer_input,
        s, split, 1.0 / K, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, h,
        SINKHORN=sinkhorn_repeat, N=n, MP=MP, BH=1024, num_warps=4)
    return post.unsqueeze(-1), comb, layer_input


@triton.jit
def _post_proj_partial_kernel(x_ptr, res_ptr, post_ptr, comb_ptr, out_ptr, fn_ptr, part_ptr, sq_ptr, s, H, K, k_chunk,
                              N: tl.constexpr, M: tl.constexpr, MP: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr):
    """mHC post of the attention sublayer fused into the projection of the following MLP-sublayer pre.
    For its slice of the flattened (stream, hidden) axis each program computes the post output
    out[b,h] = post[b]*x[h] + sum_a comb[a,b]*res[a,h] (as _mhc_post_kernel), stores it in the residual
    dtype and feeds the stored value to the same FP32 projection / sum of squares as _proj_partial_kernel."""
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_s * BS + tl.arange(0, BS)
    cols = tl.arange(0, MP)
    na = tl.arange(0, N)
    rmask = rows < s
    cmask = cols < M
    acc = tl.zeros((BS, MP), dtype=tl.float32)
    sq = tl.zeros((BS,), dtype=tl.float32)
    k0 = pid_k * k_chunk
    for kk in range(0, k_chunk, BK):
        ks = k0 + kk + tl.arange(0, BK)
        kmask = ks < K
        b = ks // H
        hc = ks - b * H
        m2 = rmask[:, None] & kmask[None, :]
        m3 = m2[:, None, :]
        xv = tl.load(x_ptr + rows[:, None] * H + hc[None, :], mask=m2, other=0.0).to(tl.float32)
        pb = tl.load(post_ptr + rows[:, None] * N + b[None, :], mask=m2, other=0.0).to(tl.float32)
        cab = tl.load(comb_ptr + rows[:, None, None] * (N * N) + na[None, :, None] * N + b[None, None, :],
                      mask=m3, other=0.0).to(tl.float32)
        ra = tl.load(res_ptr + rows[:, None, None] * K + na[None, :, None] * H + hc[None, None, :],
                     mask=m3, other=0.0).to(tl.float32)
        o = tl.sum(cab * ra, axis=1)
        o = o + pb * xv
        o = o.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + rows[:, None] * K + ks[None, :], o, mask=m2)
        xr = o.to(tl.float32)
        w = tl.load(fn_ptr + cols[:, None] * K + ks[None, :], mask=cmask[:, None] & kmask[None, :], other=0.0)
        acc = tl.dot(xr, tl.trans(w), acc, input_precision='ieee')
        sq += tl.sum(xr * xr, axis=1)
    base = pid_k * s
    tl.store(part_ptr + (base + rows[:, None]) * MP + cols[None, :], acc, mask=rmask[:, None])
    tl.store(sq_ptr + base + rows, sq, mask=rmask)


def mhc_post_pre_fused(x, residual, h_post, h_res, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                       hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n):
    """hc_post(x, residual, h_post, h_res) followed by hc_pre of the next sublayer, decode sizes (s <= 64).
    x [s,h], residual [s,n*h], h_post [s,n], h_res [s,n*n] -> (new residual [s,n*h], layer_input [s,h],
    comb [s,n*n], post [s,n]): the same values as the two separate calls."""
    s, h = x.shape
    K = n * h
    M = fn.shape[0]
    assert fn.dtype == torch.float32 and M == 2 * n + n * n and fn.shape[1] == K
    MP = max(16, triton.next_power_of_2(M))
    BS, BK = 16, 64
    split = 64 if K % (64 * BK) == 0 else 1
    k_chunk = K // split
    dev = x.device
    out = torch.empty((s, n, h), device=dev, dtype=x.dtype)
    part = torch.empty((split, s, MP), device=dev, dtype=torch.float32)
    sq = torch.empty((split, s), device=dev, dtype=torch.float32)
    _post_proj_partial_kernel[(triton.cdiv(s, BS), split)](
        x.contiguous(), residual.contiguous(), h_post.contiguous(), h_res.contiguous(), out, fn.contiguous(),
        part, sq, s, h, K, k_chunk, N=n, M=M, MP=MP, BS=BS, BK=BK, num_warps=4)
    pre = torch.empty((s, n), device=dev, dtype=torch.float32)
    post = torch.empty((s, n), device=dev, dtype=torch.float32)
    comb = torch.empty((s, n, n), device=dev, dtype=torch.float32)
    layer_input = torch.empty((s, h), device=dev, dtype=x.dtype)
    _tail_kernel[(s,)](
        part, sq, hc_scale, hc_base, out, K, pre, post, comb, layer_input,
        s, split, 1.0 / K, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, h,
        SINKHORN=sinkhorn_repeat, N=n, MP=MP, BH=1024, num_warps=4)
    return out.view(s, K), layer_input, comb.view(s, n * n), post


_COMM = 'sglang.srt.layers.communicator_mhc'


def patch_communicator(module):
    """LKQMOE_GLM53_MHC_FUSE_POST=1: MHCState.attn_to_mlp (attention post + MLP pre of one decoder layer)
    runs mhc_post_pre_fused for decode sizes; other sizes / layers keep the original two calls."""
    state = getattr(module, 'MHCState', None)
    if state is None or getattr(state, '_lkqmoe_fuse_post', False):
        return
    original = state.attn_to_mlp

    def attn_to_mlp(self, hidden_states, residual, out_norm=None):
        layer = getattr(self.hc_ffn_pre, '__self__', None)
        cfg = getattr(layer, 'config', None)
        if (cfg is not None and getattr(layer, 'hc_ffn_fn', None) is not None and hidden_states.dim() == 2
                and 1 <= hidden_states.shape[0] <= 64 and self.h_res is not None and self.h_post is not None
                and hidden_states.is_cuda and self.hc_mult & (self.hc_mult - 1) == 0):
            residual, hidden_states, self.h_res, self.h_post = mhc_post_pre_fused(
                hidden_states, residual, self.h_post, self.h_res, layer.hc_ffn_fn, layer.hc_ffn_scale,
                layer.hc_ffn_base, cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_eps, 2.0, cfg.hc_sinkhorn_iters, self.hc_mult)
            if out_norm is not None:
                hidden_states = out_norm(hidden_states)
            return hidden_states, residual
        return original(self, hidden_states, residual, out_norm)

    state.attn_to_mlp = attn_to_mlp
    state._lkqmoe_fuse_post = True
    print('[lkqmoe] GLM-5.3 mHC post fused into the next pre (attention -> MLP)', file=sys.stderr, flush=True)


def patch(module):
    if getattr(module, '_lkqmoe_glm53_mhc', False):
        return
    if not callable(getattr(module, 'mhc_pre_triton', None)):
        raise RuntimeError('Unsupported sglang mHC SM120 interface')
    module._lkqmoe_original_mhc_pre_triton = module.mhc_pre_triton
    module.mhc_pre_triton = mhc_pre_fused
    module._lkqmoe_glm53_mhc = True
    print('[lkqmoe] GLM-5.3 fused mHC pre active', file=sys.stderr, flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        fn = getattr(self.delegate, 'create_module', None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        (patch_communicator if module.__name__ == _COMM else patch)(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET and not (fullname == _COMM and os.environ.get('LKQMOE_GLM53_MHC_FUSE_POST') == '1'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    if _TARGET in sys.modules:
        patch(sys.modules[_TARGET])
    if os.environ.get('LKQMOE_GLM53_MHC_FUSE_POST') == '1' and _COMM in sys.modules:
        patch_communicator(sys.modules[_COMM])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
