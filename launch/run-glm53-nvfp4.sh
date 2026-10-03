#!/usr/bin/env bash
# GLM-5.3 Flash NVFP4 + EAGLE (NextN) on one RTX PRO 5000 72GB (SM120) + 2x EPYC 9334 (4 NUMA nodes).
# MoE layers 3-11 on the GPU; routed experts of the other 33 MoE layers run on lkqmoe (compiled, ../lkqmoe).
# This is the launch used for the measurements in README.md; edit the paths block first.
set -Eeuo pipefail

# ---- paths ------------------------------------------------------------------------------
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
LSGLANG=${LSGLANG:-/opt/Lsglang}      # guqiong96/Lsglang @ cca56c1f36 (lkmoe-glm5.3-flash-sm80plus) + patches/lsglang-cca56c1f-glm53.patch
VENV=${VENV:-$LSGLANG/env}            # python 3.12, see README "环境搭建"
MODEL=${MODEL:-/models/LibertAIDAI/GLM-5.3-Flash-NVFP4}
PORT=${PORT:-18085}
STATS_DIR=${STATS_DIR:-$REPO/run/lkqmoe-stats}

cuda_root="$VENV/lib/python3.12/site-packages/nvidia/cu13"
lkq="$REPO/lkqmoe"
mkdir -p "$STATS_DIR" "$REPO/run/cache"
# Post-start warmup (waits for /health): compiles the CPU / hybrid / GPU prefill paths before the first request.
nohup python3 "$REPO/launch/warmup.py" "$PORT" > "$REPO/run/warmup.log" 2>&1 &
cd "$LSGLANG"
exec env \
  LKQMOE_MODE=standalone \
  LKQMOE_LIBRARY="$lkq/liblkqmoe.so" \
  LKQMOE_STATS_DIR="$STATS_DIR" LKQMOE_ORIGINAL_WARMUP=0 LKQMOE_ZERO_COPY=1 LKQMOE_SPIN_COUNT=1048576 \
  LKQMOE_THREADS=112 LKQMOE_DYNAMIC=2 LKQMOE_DOWN_BF16=1 \
  LKQMOE_MAIN_CPUS=76-79,92-94,108-111,124-127 LKQMOE_PROBE_DISPATCH_CPU=95 \
  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=512 LKQMOE_RUNTIME_FILE="$REPO/launch/lkqmoe-runtime.json" \
  LKQMOE_CPU_PREFILL_BATCH=1024 LKQMOE_PREFILL_UNPACK=1 LKQMOE_GPU_PREFILL_CHUNK=8192 \
  LKQMOE_PREFILL_RELEASE_WORKSPACE=1 LKQMOE_PREFILL_PREFETCH=1 LKQMOE_PREFILL_OVERLAP=1 \
  LKQMOE_PREFILL_EXPERT_CHUNK=16 LKQMOE_PREFILL_ROUTE_ACCUM=1 \
  LKQMOE_PREFILL_TILE_M=64 LKQMOE_GATE_BN=128 LKQMOE_DOWN_BN=128 LKQMOE_DOWN_BK=64 \
  LKQMOE_GATE_LAUNCH=8,3 LKQMOE_DOWN_LAUNCH=8,3 LKQMOE_DOWN_PRECISION=bf16 \
  LVLLM_GPU_RESIDENT_MOE_LAYERS=0,3-11 LKQMOE_RESIDENT_W4A16=1 LKQMOE_RESIDENT_W4A16_SKIP_DRAFT=1 \
  LKQMOE_FP8_WEIGHTS=1 'LKQMOE_FP8_INCLUDE=[.](self_attn|mlp)[.]|eh_proj' \
  'LKQMOE_FP8_EXCLUDE=(fused_qkv_a_proj|kv_b_proj|[.]gate$|visual|vision|weights_proj)' \
  'LKQMOE_PREFILL_W4A4_INCLUDE=[.](self_attn|mlp)[.]|eh_proj' \
  LKQMOE_GLM53_MHC=1 LKQMOE_SMALL_M_GEMM=1 LKQMOE_INDEXER_Q_CHUNK=4096 LKQMOE_GLM53_CPU_IMAGE=1 \
  SGLANG_MHC_PRE_TRITON=1 SGLANG_MHC_POST_TRITON=1 \
  SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 \
  PYTHONPATH="$lkq/python:$LSGLANG/python" \
  CUDA_HOME="$cuda_root" PATH="$cuda_root/bin:$PATH" \
  LD_LIBRARY_PATH="$cuda_root/lib:${LD_LIBRARY_PATH:-}" \
  TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0 \
  TILELANG_DEFAULT_TARGET='{"kind":"cuda","arch":"sm_120"}' \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MAX_JOBS=8 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  LVLLM_MOE_NUMA_ENABLED=1 LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1 \
  LVLLM_ENABLE_NUMA_INTERLEAVE=1 LK_THREADS=64 OMP_NUM_THREADS=1 \
  LK_THREAD_BINDING=CPU_CORE LK_POWER_SAVING=1 \
  LVLLM_GPU_PREFETCH_WINDOW=2 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  XDG_CACHE_HOME="$REPO/run/cache" \
  "$VENV/bin/sglang" serve \
    --model-path "$MODEL" \
    --served-model-name GLM-5.3-Flash-NVFP4 \
    --host 127.0.0.1 --port "$PORT" --trust-remote-code \
    --tp-size 1 \
    --weight-loader-drop-cache-after-load \
    --context-length 524288 --max-running-requests 1 \
    --max-total-tokens 528384 --max-mamba-cache-size 8 \
    --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
    --mem-fraction-static 0.95 --page-size 64 \
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
    --speculative-token-map "$lkq/draft-token-map-49152.pt"
