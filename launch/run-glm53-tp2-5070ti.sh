#!/usr/bin/env bash
# GLM-5.3 Flash NVFP4 + EAGLE (NextN) on 2x RTX 5070 Ti 16GB (SM120, TP=2) + 2x EPYC 7642 (Zen2, AVX2, NPS4 = 8 NUMA nodes).
# Based on lsglang_glm5.3 launch/run-glm53-nvfp4.sh (1x RTX PRO 5000) and guqiong96's TP2 recipes (2x 5060 Ti / 3090).
# Routed experts of all 42 MoE layers on lkqmoe 0.5.3 (AVX2 kernels picked at run time; one process per TP rank, each
# with half of every expert on the NUMA nodes of its GPU's socket) + LKQstrata expert pool on each GPU. No GPU-resident
# MoE layer: the NextN draft layer's experts run on lkqmoe too (saves ~1.9 GiB per GPU for the 256K KV cache and the pool).
# Per-GPU budget (GiB, estimated): weights ~8, KV 256K ~2.35 (FP8 MLA, replicated on both GPUs), mamba ~0.85,
# graphs/activations ~2-2.5 -> ~1.6-2.1 left for the pool (~0.28 GiB per slot). Start at 4 slots, raise after
# reading 'avail mem' in the startup log.
set -Eeuo pipefail

# Measured 2026-10-17 (TP2, 256K context, 4 pool slots): prefill is PCIe-bound (every chunk uploads all experts,
# ~80 GiB per GPU over PCIe 4.0 ~23 GB/s), so prefill speed scales with CHUNK. CHUNK 5120 + EXPERT_CHUNK 2 +
# INDEXER_Q_CHUNK 2048: 64K TTFT 60.6 s (1083 tok/s), 256K TTFT 237.8 s (1089 tok/s), decode 33-34 tok/s,
# VRAM peak 15770 / 16303 MiB at 256K. (CHUNK 2048: 531 tok/s; CHUNK 4096 + EXPERT_CHUNK 4: 940 tok/s.)
# ---- paths (edit, or set as environment variables) -------------------------------------------
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}   # this repository (launch/, lkqmoe/)
LSGLANG=${LSGLANG:-/opt/Lsglang}       # guqiong96/Lsglang @ cca56c1f36 + patches/lsglang-cca56c1f-glm53.patch
VENV=${VENV:-$LSGLANG/env}             # python 3.12, requirements-lock.txt (see README); env/.../nvidia/cu13 needs
                                       # lib64 -> lib and lib/libcudart.so -> libcudart.so.13 for the JIT kernels
LKQ=${LKQ:-$REPO/lkqmoe}               # lkqmoe 0.5.3+ (AVX512 + AVX2 builds, TP ranks)
MODEL=${MODEL:-/models/LibertAIDAI/GLM-5.3-Flash-NVFP4}
PORT=${PORT:-18085}
RUN=${RUN:-$REPO/run}

# ---- sizes (16 GB per GPU) ------------------------------------------------------------------
CTX=${CTX:-262144}                     # context length; KV (FP8 MLA) is replicated on both GPUs
MAXREQ=${MAXREQ:-1}                    # running requests (2 needs MAMBA=10)
MAMBA=${MAMBA:-6}                      # mamba state slots (5 per request + 1)
CHUNK=${CHUNK:-5120}                   # chunked prefill size (activations and lkqmoe route buffers scale with it;
                                       # lkqmoe's own GPU prefill block stays 4096, the smallest it supports)
MEMFRAC=${MEMFRAC:-0.90}
LKQSTRATA=${LKQSTRATA:-1}
LKQSTRATA_SLOTS=${LKQSTRATA_SLOTS:-4}  # expert slots per layer per GPU (half experts: ~6.75 MiB x 42 layers = 0.28 GiB per slot)
RESIDENT=${RESIDENT:-}                 # GPU-resident MoE layers; "0" = NextN draft layer on the GPU (+~1.9 GiB per GPU)
THREADS=${THREADS:-44}                 # lkqmoe workers per TP rank (4 nodes x 11 of 12 cores; the rest for dispatcher / main)
PREFETCH_CHUNKS=${PREFETCH_CHUNKS:-2}  # GPU prefill staging ring of packed expert chunks (12 on the 72 GB card)
EXPERT_CHUNK=${EXPERT_CHUNK:-2}        # experts per GPU prefill upload chunk (16 on the 72 GB card); BF16 unpack buffers scale with it
INDEXER_Q_CHUNK=${INDEXER_Q_CHUNK:-2048} # DSA indexer prefill query block; its long-context scores scale with it
# Hot-vocab draft head (49152 tokens): off under TP. The draft head is built by indexing the target lm_head
# with full-vocab ids, but with TP each rank holds only its vocab shard (index out of bounds). Without it the
# draft shares the target head (full vocab); rejection sampling handles both cases.
TOKEN_MAP=${TOKEN_MAP:-}

cuda_root="$VENV/lib/python3.12/site-packages/nvidia/cu13"
mkdir -p "$RUN/lkqmoe-stats" "$RUN/cache"
# Post-start warmup (waits for /health): compiles the CPU / hybrid / GPU prefill paths before the first request.
nohup "$VENV/bin/python" "$REPO/launch/warmup.py" "$PORT" > "$RUN/warmup.log" 2>&1 &
cd "$LSGLANG"
exec env \
  LKQMOE_MODE=standalone \
  LKQMOE_LIBRARY="$LKQ/liblkqmoe.so" \
  LKQMOE_STATS_DIR="$RUN/lkqmoe-stats" LKQMOE_ORIGINAL_WARMUP=0 LKQMOE_ZERO_COPY=1 LKQMOE_SPIN_COUNT=1048576 LKQMOE_ACTIVE_FLAGS=1 \
  LKQMOE_THREADS="$THREADS" LKQMOE_DYNAMIC=2 LKQMOE_DOWN_BF16=1 \
  LKQMOE_TP_NODES=auto LKQMOE_MAIN_CPUS=auto \
  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=512 LKQMOE_RUNTIME_FILE="$REPO/launch/lkqmoe-runtime.json" \
  LKQMOE_CPU_PREFILL_BATCH=1024 LKQMOE_PREFILL_UNPACK=1 LKQMOE_GPU_PREFILL_CHUNK=4096 \
  LKQMOE_PREFILL_RELEASE_WORKSPACE=1 LKQMOE_PREFILL_PREFETCH=1 LKQMOE_PREFILL_OVERLAP=1 \
  LKQMOE_PREFILL_EXPERT_CHUNK="$EXPERT_CHUNK" LKQMOE_PREFILL_ROUTE_ACCUM=1 LKQMOE_PREFILL_PREFETCH_CHUNKS="$PREFETCH_CHUNKS" \
  LKQMOE_PREFILL_TILE_M=64 LKQMOE_GATE_BN=128 LKQMOE_DOWN_BN=128 LKQMOE_DOWN_BK=64 \
  LKQMOE_GATE_LAUNCH=8,3 LKQMOE_DOWN_LAUNCH=8,3 LKQMOE_DOWN_PRECISION=bf16 \
  LVLLM_GPU_RESIDENT_MOE_LAYERS="$RESIDENT" LKQMOE_RESIDENT_W4A16=1 LKQMOE_RESIDENT_W4A16_SKIP_DRAFT=1 \
  LKQSTRATA="$LKQSTRATA" LKQSTRATA_SLOTS="$LKQSTRATA_SLOTS" LKQSTRATA_PROFILE="$LKQ/lkqstrata-glm53.json" LKQSTRATA_LOG=500 \
  LKQMOE_FP8_WEIGHTS=1 'LKQMOE_FP8_INCLUDE=[.](self_attn|mlp)[.]|eh_proj' \
  'LKQMOE_FP8_EXCLUDE=(fused_qkv_a_proj|kv_b_proj|[.]gate$|visual|vision|weights_proj)' \
  'LKQMOE_PREFILL_W4A4_INCLUDE=[.](self_attn|mlp)[.]|eh_proj' \
  LKQMOE_GLM53_MHC=1 LKQMOE_SMALL_M_GEMM=1 LKQMOE_INDEXER_Q_CHUNK="$INDEXER_Q_CHUNK" LKQMOE_GLM53_CPU_IMAGE=1 \
  SGLANG_MHC_PRE_TRITON=1 SGLANG_MHC_POST_TRITON=1 \
  SGLANG_RS_DRAFT_TEMP_SCALE=0.8 SGLANG_SPEC_TRUNCATE_Q=0.2 \
  SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1 SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA=1 \
  SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
  PYTHONPATH="$LKQ/python:$LSGLANG/python" \
  CUDA_HOME="$cuda_root" PATH="$cuda_root/bin:$VENV/bin:$PATH" \
  LD_LIBRARY_PATH="$cuda_root/lib:${LD_LIBRARY_PATH:-}" \
  TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0 \
  TILELANG_DEFAULT_TARGET='{"kind":"cuda","arch":"sm_120"}' \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MAX_JOBS=8 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  LVLLM_MOE_NUMA_ENABLED=1 LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1 \
  LVLLM_ENABLE_NUMA_INTERLEAVE=1 LK_THREADS=48 OMP_NUM_THREADS=1 \
  LK_THREAD_BINDING=CPU_CORE LK_POWER_SAVING=1 \
  LVLLM_GPU_PREFETCH_WINDOW=1 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  XDG_CACHE_HOME="$RUN/cache" \
  "$VENV/bin/sglang" serve \
    --model-path "$MODEL" \
    --served-model-name GLM-5.3-Flash-NVFP4 \
    --host 127.0.0.1 --port "$PORT" --trust-remote-code \
    --tp-size 2 --disable-custom-all-reduce \
    --weight-loader-drop-cache-after-load \
    --context-length "$CTX" --max-running-requests "$MAXREQ" \
    --max-total-tokens "$((CTX + 4096))" --max-mamba-cache-size "$MAMBA" \
    --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$CHUNK" \
    --mem-fraction-static "$MEMFRAC" --page-size 64 \
    --quantization modelopt_fp4 --moe-runner-backend flashinfer_cutlass \
    --kv-cache-dtype fp8_e4m3 \
    --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
    --linear-attn-backend triton \
    --mamba-ssm-dtype bfloat16 --mamba-scheduler-strategy extra_buffer \
    --mm-feature-transport cpu --image-processor-backend pil \
    --disable-shared-experts-fusion \
    --disable-prefill-cuda-graph \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --watchdog-timeout 3600 --enable-metrics --enable-cache-report \
    --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 --speculative-adaptive \
    --speculative-adaptive-config "$REPO/launch/adaptive-steps-1to5.json" \
    ${TOKEN_MAP:+--speculative-token-map "$TOKEN_MAP"} \
    --speculative-use-rejection-sampling \
    ${EXTRA_ARGS:-}
