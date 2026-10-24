# 版本不再更新，请移步最新 XTier 仓库：[https://github.com/a775828b-dot/xtier](https://github.com/a775828b-dot/xtier)

# lsglang_glm5.3

GLM-5.3 Flash NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，512K 上下文、最多 2 并发，EAGLE（NextN）投机解码（无损拒绝采样），
**LKQstrata 热专家池**（每个 MoE 层的热门专家常驻 GPU、自适应换入换出，与 CPU 同时计算），温度 1.0 下 decode 约 85 tok/s，支持图片。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本 0.5.6）。

*English: a single-GPU GLM-5.3 Flash NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 512K context, two concurrent requests,
EAGLE/NextN speculative decoding with lossless rejection sampling, LKQstrata GPU expert pool for the hot experts of every MoE layer,
~85 tok/s decode at temperature 1.0, image input) on guqiong96/Lsglang, with the CPU-side routed experts on lkqmoe 0.5.6, an
NVFP4-specialised MoE kernel inspired by lk_moe (by guqiong96, the author of lsglang); a compiled, closed-source build is included here.
With 24 memory channels and NPS4 use `launch/run-glm53-nvfp4-24ch-nps4.sh` (host CPUs resolved by lkqmoe); release notes are under
"版本更新", details in `reports/`.*

## 版本更新

- **2026-10-22**：lkqmoe 0.5.6（与 lsglang_ds4.1 同一份编译版）；内存补满 24 通道、BIOS 改 NPS4，新增 `launch/run-glm53-nvfp4-24ch-nps4.sh`
  （`LKQMOE_MAIN_CPUS=auto`，默认 2 并发）。单请求 decode 平均 100.0 → 106.9 tok/s（+7%），温度 1.0 为 84.8 tok/s，2 并发合计 101.6 → 112.4 tok/s。
  详见 [reports/2026-10-22-24ch-nps4-lkqmoe056.md](reports/2026-10-22-24ch-nps4-lkqmoe056.md)。
- **2026-10-17**：lkqmoe 0.5.3：只有 AVX2 的 CPU 与多卡张量并行；双卡 TP2 实测（2× EPYC 7642 + 2× RTX 5070 Ti，256K）。
  详见 [reports/2026-10-17-lkqmoe053-avx2-tp2.md](reports/2026-10-17-lkqmoe053-avx2-tp2.md)。
- **2026-10-16**：lkqmoe 0.5.2：NUMA 拓扑从系统读取（NPS2 / NPS4），`LKQMOE_MAIN_CPUS=auto`。详见 [reports/2026-10-16-lkqmoe052-numa.md](reports/2026-10-16-lkqmoe052-numa.md)。
- **2026-10-14**：LKQstrata 热专家池（lkqmoe 0.5.0）：温度 1.0 decode 62.5 → 83.3 tok/s，2 并发合计 77.9 → 101.6 tok/s。
  详见 [reports/2026-10-14-lkqstrata.md](reports/2026-10-14-lkqstrata.md)。
- **2026-10-04**：温度 1.0 decode +8.5–10%：无损拒绝采样配合草稿热词表、草稿温度 ×0.8、按草稿置信度截断验证、DSA 元数据融合。
  做法见本页「运行方式」，数据 `results/temperature1-20261004.json`。
- **2026-10-03**：首次发布（lkqmoe 0.4.2 / 0.4.3）。详见 [reports/2026-10-03-release-lkqmoe043.md](reports/2026-10-03-release-lkqmoe043.md)。

## 组成

| 部分 | 来源 | 本仓库内容 |
|---|---|---|
| 模型 | [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)：LibertAIDAI 对智谱 **GLM-5.3 Flash** 的 NVFP4 量化（路由专家 NVFP4，注意力、稠密层与共享专家 BF16，含 1 层 NextN 草稿），188 GB | — |
| 推理框架 | guqiong96/Lsglang 分支 `lkmoe-glm5.3-flash-sm80plus` @ [`cca56c1f36`](https://github.com/guqiong96/Lsglang/commit/cca56c1f36b0326e4d5cb7c6878bf29c064e39d0)（GLM-5.3 支持来自 sglang PR #36507）；补丁含 [usrlocalben/Lsglang](https://github.com/usrlocalben/Lsglang) `pr-20-22-contd`（HiCache DSA 修复、原生图片预处理、NextN 多模态等）与 SM120 上融合的 mHC pre/post Triton 内核；投机解码：拒绝采样配合草稿高频词表、草稿温度系数、按草稿置信度截断验证 | `patches/lsglang-cca56c1f-glm53.patch` |
| CPU 专家内核 | **lkqmoe**（闭源，编译版）：受 lk_moe 启发、针对 NVFP4 格式专项加速的 CPU/GPU 混合 MoE 内核。lk_moe 来自 lsglang 作者 guqiong96（[guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)）；lkqmoe 独立实现其 `MOE_NVFP4` 接口，不含 lk_moe 代码。0.5.0 起含 **LKQstrata 热专家池**：GPU 上缓存每层的热门专家，换入换出策略参考 [Strata](https://github.com/Niko1221/Strata)（MIT，独立实现） | `lkqmoe/` |
| 热专家池排名表 | 从 GLM 实际 decode 路由统计的每层专家使用排名（32 条中英文、代码、数学、agent 等提示，温度 1.0）；启动时按它装填，之后自适应 | `lkqmoe/lkqstrata-glm53.json` |
| GPU 侧补丁 | 开源：mHC 融合、BF16 权重转 FP8（W8A16 decode / W8A8 预填充）、GPU 常驻 MoE 层 W4A16 decode、DSA 索引器分块预填充、图片 CPU 预处理 | `lkqmoe/python/lkqmoe/gpu/` |
| 草稿热词表 | 从 GLM 实际输出、代理会话与文档统计的 49152 个高频 token（词表 154880；只影响草稿接受率，目标模型仍验证每个 token） | `lkqmoe/draft-token-map-49152.pt` |
| 启动 | 测速所用的完整参数 | `launch/run-glm53-nvfp4-24ch-nps4.sh`（24 通道 / NPS4，2 并发）；`launch/run-glm53-nvfp4.sh`（20 通道 / NPS2，单请求） |
| 双卡启动 | 2× RTX 5070 Ti 16GB + 2× EPYC 7642（AVX2）的 TP2 参数，见 `reports/2026-10-17-lkqmoe053-avx2-tp2.md` | `launch/run-glm53-tp2-5070ti.sh` |

## 本机硬件

| | |
|---|---|
| CPU | 2× AMD EPYC 9334（共 64 核 128 线程），AVX512-BF16/VBMI |
| 内存 / NUMA | 24 × 16 GiB DDR5-4800（384 GiB），BIOS NPS4 → 8 个 NUMA 节点（2026-10-22 起；之前为 20 × 16 GB、320 GB、NPS2 4 节点） |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（SM120），驱动 595.99，CUDA 13.2；功耗墙 350 W |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

单请求：温度 0，每点输出 1024 token，输入是埋了一条事实的长文本（`bench/speed_points.py`），所有长度都找回事实。
并发：短提示，温度 1.0，每个请求 512 token，同时开始，各 4 轮。

### 24 通道 / NPS4（2026-10-22，lkqmoe 0.5.6，`launch/run-glm53-nvfp4-24ch-nps4.sh`，2 并发配置）

| 输入 token | decode tok/s | 接受长度 | 首 token |
|---:|---:|---:|---:|
| 4K | 117.0 | 4.51 | 3.24 s |
| 16K | 102.8 | 4.05 | 6.07 s |
| 33K | 108.8 | 3.98 | 12.98 s |
| 66K | 126.5 | 4.18 | 27.66 s |
| 133K | 104.9 | 4.10 | 53.62 s |
| 269K | 92.9 | 3.94 | 106.03 s |
| 413K | 96.9 | 3.83 | 166.42 s |
| 500K | 105.4 | 3.70 | 200.98 s |

- 平均 decode 106.9 tok/s；500K 一点用 `--exact` 在 WebUI 正式启动后测得，其余各点同一配置。
- 温度 1.0（实际使用，4 类中文任务各 3 次，`bench/spec_t1_bench.py`）：**84.8 tok/s**。
- 并发：1 个请求 82.5 tok/s；2 个请求每个 58.0、合计 **112.4 tok/s**。
- 困惑度 4.0598（6 × 4096 token）。
- 原始数据：`results/speed-points-20261022-24ch-nps4.json`、`results/concurrency-20261022-24ch-nps4.json`、`results/temperature1-20261022-24ch-nps4.json`。

### 20 通道 / NPS2（2026-10-14，lkqmoe 0.5.0，`launch/run-glm53-nvfp4.sh`）

| 输入 token | decode tok/s | 首 token |
|---:|---:|---:|
| 4K | 102.1 | 3.23 s |
| 17K | 110.0 | 8.71 s |
| 33K | 114.7 | 12.90 s |
| 66K | 95.6 | 27.34 s |
| 136K | 102.4 | 53.32 s |
| 274K | 96.2 | 107.66 s |
| 413K | 85.5 | 165.98 s |
| 515K | 93.2 | 208.46 s |

- 平均 decode 100.0 tok/s；温度 1.0：83.3 tok/s；并发：1 个请求 78.1 tok/s，2 个请求合计 101.6 tok/s（60 槽、`--max-running-requests 2 --max-mamba-cache-size 10`）。
- 原始数据：`results/speed-points-20261014.json`、`results/temperature1-20261014.json`。

## 运行方式

- 模型 45 层：前 3 层稠密，其余 42 层 MoE（288 专家 top-8 + 1 个共享专家，hidden 4096，专家中间维 2048，SwiGLU 限幅 10）；
  注意力 34 层 KDA 线性注意力 + 11 层 DSA 稀疏 MLA（每 4 层一层）；mHC 残差（hc_mult 4）；1 层 NextN 草稿。
- 42 个 MoE 层的路由专家都由 lkqmoe 承担：CPU（112 线程，按节点动态领取行）加上 GPU 上的 LKQstrata 热专家池（每层 62 个槽位，
  约 34.3 GiB）；草稿层在 GPU。
- LKQstrata：每层 decode 时，一个打包内核读本层的驻留表（显存，CUDA graph 重放时读取，驻留变化不用重新抓图），把路由分成两半：
  在槽位里的专家交给 GPU（W4A16），其余写给 CPU；GPU 在通知 CPU 之后、等待 CPU 之前计算，最后把两部分相加。CPU 侧的信箱线程
  统计每层路由（带衰减的计数），策略线程每 4 个验证窗口把最常用的未驻留专家换进槽位、换出最少用的（收益门槛 1.5，每窗口最多
  16 个）；换入换出经映射内存里的命令环和每窗口开头的提交内核生效：换出立即生效，换入在拷贝落地后生效，正在写的槽位不会被读取。
- 预填充：640 token 以下走 CPU，640–1792 CPU+GPU 混合，更长走 lkqmoe 的 Triton GPU 内核（按 16 个专家一块上传权重，
  解包与上传在独立流上流水，路由结果按块 FP32 累加；12 个暂存块组成跨层环，当前层最后 12 块一解包完，下一层的前 12 块
  就开始上传，正好落在下一层注意力计算期间，预填充时多占约 2.1 GiB）；16K 分块预填充；DSA 索引器的预填充 logits 按 4096 个查询一块计算
  （512K 时临时显存 7.8 GiB → 约 2 GiB）。
- mHC：SM120 上 pre（含 FP32 投影）与 post 各融合成 Triton 内核（decode 每次 82 → 18 µs）。
- KV 缓存 FP8；单请求并发；图片在 CPU 上解码与预处理（`--image-processor-backend pil`）。
- EAGLE（NextN）：自适应草稿步数（候选 1–5 步，`launch/adaptive-steps-1to5.json`），草稿热词表 49152。
- 拒绝采样（`--speculative-use-rejection-sampling`，Leviathan 式，无损）：草稿按 softmax(logits / T) 采样，验证时以
  coin·q(x) < p(x) 接受，拒绝时从 (p − q)⁺ 补采，输出分布与不用投机解码相同。上游 sglang 要求草稿与主模型词表一致，
  本补丁把只覆盖高频词表的草稿分布 q 在验证前填回完整词表（表外 q = 0），两者可以同时用；贪心请求（温度 0，sglang 内部是
  top_k = 1）草稿直接取 argmax，温度 0 的接受率不受影响。
- 草稿温度 = 请求温度 × 0.8（`SGLANG_RS_DRAFT_TEMP_SCALE`）：对任意草稿分布都无损，0.8 时期望接受最高；系数是用验证时
  直接计算的期望接受率 Σmin(p, q) 选的（q 换温度可由原 q 精确换算），在线扫参噪声太大不可用。
- 按草稿置信度截断验证（`SGLANG_SPEC_TRUNCATE_Q=0.2`）：草稿完成后，在 GPU 上找出第一个 q(x) < 0.2 的草稿 token，
  验证最多接受它之前的部分（全部接受时奖励 token 取自主模型 p，与更短的草稿链完全等价，仍然无损）；被截掉的 token 由 lkqmoe
  跳过 CPU 专家计算（每丢一个约省 4.9 ms）。贪心请求不截断。
- DSA 注意力元数据（`SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION`、`SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA`，
  分支自带、默认关闭）：kpool 元数据刷新融合、多步草稿元数据去重、验证元数据录进 CUDA graph，只改索引，不改数值。
- 启动后自动发 3 条预热请求（`launch/warmup.py`），首个真实请求不再付 Triton 编译时间。

各部分的数值精度：

| 部分 | 权重 | 激活 / 计算 |
|---|---|---|
| 路由专家 decode，CPU 部分（不在热专家池的专家） | NVFP4 | BF16 输入，FP32 Gate/Up 与 SiLU，Down 用 BF16 点积，FP32 累加（W4A16） |
| 路由专家 decode，GPU 部分（热专家池里的专家） | NVFP4 | SM120 硬件 FP4→FP16 转换，FP32 累加（W4A16，与上一版常驻层相同） |
| 路由专家的长预填充（GPU Triton） | NVFP4 → BF16 | BF16（W4A16），FP32 路由累加 |
| 草稿层 MoE（以及选用常驻层时的常驻层预填充） | NVFP4 | NVFP4（W4A4，flashinfer CUTLASS）；常驻层 decode 为 W4A16（`resident_w4a16.py`） |
| 注意力投影、稠密 MLP、共享专家、`eh_proj` | FP8 E4M3（每行 × 128 列一个 FP32 scale） | decode BF16（W8A16）；预填充 FP8，每 128 列一组 scale（W8A8），`eh_proj` 反量化后 BF16 |
| `fused_qkv_a_proj`、`kv_b_proj`、路由、DSA `weights_proj`、视觉 | BF16 | BF16 |

## 环境搭建

1. 框架：
   ```sh
   git clone https://github.com/guqiong96/Lsglang /opt/Lsglang && cd /opt/Lsglang
   git checkout cca56c1f36b0326e4d5cb7c6878bf29c064e39d0
   git apply /path/to/lsglang_glm5.3/patches/lsglang-cca56c1f-glm53.patch
   ```
2. Python 3.12 虚拟环境（`/opt/Lsglang/env`），按 Lsglang v1.4.12 的 wheel 发布包安装，版本见 `requirements-lock.txt`
   （torch 2.13.0、triton 3.7.1、flashinfer 0.6.17、transformers 5.12.1、tilelang 0.1.11）。不需要安装 `lk_moe`：lkqmoe 以
   `LKQMOE_MODE=standalone` 直接提供 `lk_moe` 模块。
3. 下载模型，按需修改启动脚本开头的路径后启动：24 通道 / NPS4 用 `bash launch/run-glm53-nvfp4-24ch-nps4.sh`，20 通道 / NPS2 用 `bash launch/run-glm53-nvfp4.sh`。
   服务在 `127.0.0.1:18085`，OpenAI 兼容接口，模型名 `GLM-5.3-Flash-NVFP4`，思考输出由 `glm45` 解析器分离，工具调用 `glm47`。
   冷加载约 4–5 分钟（模型文件已在页缓存时）。

启动前需要约 200 GB 可用内存（42 层专家都在内存里）、70 GB 空闲显存。

## 主要启动参数

```
--context-length 524288 --max-total-tokens 528384 --max-running-requests 2 --max-mamba-cache-size 10 --chunked-prefill-size 16384
--mem-fraction-static 0.95 --page-size 64 --quantization modelopt_fp4 --moe-runner-backend flashinfer_cutlass
--kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla
--linear-attn-backend triton --mamba-ssm-dtype bfloat16 --mamba-scheduler-strategy extra_buffer
--disable-shared-experts-fusion --disable-prefill-cuda-graph --mm-feature-transport cpu --image-processor-backend pil
--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--speculative-adaptive --speculative-adaptive-config launch/adaptive-steps-1to5.json
--speculative-token-map lkqmoe/draft-token-map-49152.pt --speculative-use-rejection-sampling
--reasoning-parser glm45 --tool-call-parser glm47
LVLLM_GPU_RESIDENT_MOE_LAYERS=0  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=512  LVLLM_GPU_PREFETCH_WINDOW=2
LKQSTRATA=1  LKQSTRATA_SLOTS=60  LKQSTRATA_PROFILE=lkqmoe/lkqstrata-glm53.json
SGLANG_MHC_PRE_TRITON=1  SGLANG_MHC_POST_TRITON=1
LKQMOE_MODE=standalone  LKQMOE_THREADS=112  LKQMOE_PROBE_DISPATCH_CPU=95  LKQMOE_MAIN_CPUS=auto  LKQMOE_DYNAMIC=2  LKQMOE_DOWN_BF16=1  LKQMOE_PREFILL_OVERLAP=1
LKQMOE_PREFILL_EXPERT_CHUNK=16  LKQMOE_PREFILL_ROUTE_ACCUM=1  LKQMOE_PREFILL_PREFETCH_CHUNKS=12  LKQMOE_DOWN_PRECISION=bf16
LKQMOE_RESIDENT_W4A16=1  LKQMOE_RESIDENT_W4A16_SKIP_DRAFT=1  LKQMOE_FP8_WEIGHTS=1  LKQMOE_GLM53_MHC=1
LKQMOE_SMALL_M_GEMM=1  LKQMOE_INDEXER_Q_CHUNK=4096  LKQMOE_GLM53_CPU_IMAGE=1  LKQMOE_ACTIVE_FLAGS=1
SGLANG_RS_DRAFT_TEMP_SCALE=0.8  SGLANG_SPEC_TRUNCATE_Q=0.2
SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1  SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA=1
```
完整列表见 `launch/run-glm53-nvfp4-24ch-nps4.sh`（单请求：`MAX_REQUESTS=1 LKQSTRATA_SLOTS=62 MAMBA=8`）；预填充 CPU / 混合 / GPU 的分界与 W8A8 的作用范围在 `launch/lkqmoe-runtime.json`（运行中修改即生效）。

## lkqmoe（`lkqmoe/`）

- 编译版：`liblkqmoe.so`（分发库）+ `liblkqmoe_avx512.so` / `liblkqmoe_avx2.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥与 LKQstrata 换入换出管理）、`python/lkqmoe/*.pyc`（lsglang 适配层、Triton GPU 预填充、
  LKQstrata 热专家池 `pool.pyc`）、
  `python/lkqmoe/gpu/small_gemm.pyc`（decode 小批量 BF16 GEMM）；含 Triton 内核的模块内嵌压缩源码，因为 Triton 编译时要读源码。
  许可见 `lkqmoe/LICENSE`：可免费使用、原样再分发。
- 版本 0.5.6（2026-10-22，与 lsglang_ds4.1 是同一份编译版）：CPU 内核按 NUMA 拓扑自动分配（NPS2 4 节点、NPS4 8 节点），
  `LKQMOE_MAIN_CPUS=auto` 自动取"允许的 CPU − 专家线程 − 信箱调度线程"；同时支持只有 AVX2 的 CPU 与多卡张量并行（见下面的使用建议）；
  CPU Down 一次读一行（`-DLKQMOE_DOWN_R4=1`），本模型实权重单层与四行版在 ±3% 内持平、输出逐位相同。各版本的改动见上面「版本更新」。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`；`python/lkqmoe/gpu/`：
  - `glm53_mhc.py`：mHC pre 整体融合（split-K FP32 投影 + RMS + 加权和），替换 cuBLAS gemmSN（decode 只用 3 个 CTA）。
    另有可选的“注意力 post 并入 MLP pre”（`LKQMOE_GLM53_MHC_FUSE_POST`，默认关闭：累加顺序不同，偶有 1 ulp 差异）。
  - `fp8_weights.py`：加载后把选中的 BF16 线性层存成 FP8 E4M3（显存 11.55 → 5.95 GiB），decode 用 split-K W8A16 内核
    （split-K 归约由每个列块最后完成的 CTA 按固定顺序完成，一次启动、结果确定），预填充用 W8A8 Triton 内核。
    省下的显存给了 GPU 侧的 MoE 专家（上一版是多一个常驻层，本版是热专家池）。
  - `resident_w4a16.py`（选用常驻层时）：GPU 常驻 NVFP4 MoE 层的 decode 改用 BF16 激活（W4A16），直接读 flashinfer 已排好的权重与 swizzle scale，
    用 SM120 的硬件 FP4→FP16 转换；专家到槽位的映射一个内核完成；静态形状，可进 CUDA graph，结果确定；对 FP64 误差 2.2e-3，
    速度与 flashinfer W4A4 相当。
  - `indexer_chunk.py`：DSA 索引器预填充 logits 分块，512K 不再 OOM。
  - `glm53_cpu_image.py`：图片解码放 CPU（GPU 解码只多占显存再拷回 CPU）。
- 精度：CPU 专家 BF16 Down（单层对 FP64 相对误差约 1.5e-3）；GPU 预填充对 FP64 误差 2.3e-3。
- 硬件要求：x86-64，AVX512-BF16/VBMI 或 AVX2/FMA（运行时自动选择），NUMA 拓扑自动识别（已在本机 NPS2 的 4 节点 × 16 核与 NPS4 的 8 节点 × 8 核上实测；hidden / intermediate 须是 16 × 节点数的倍数）。`launch/run-glm53-nvfp4-24ch-nps4.sh` 用 `LKQMOE_MAIN_CPUS=auto`，`launch/run-glm53-nvfp4.sh` 的列表按本机 NPS2 写好，SM120 GPU（`resident_w4a16.py` 的
  硬件 FP4 转换需要 sm_100a / sm_120a）。

- AVX2 / 多卡使用建议：
  - 只有 AVX2 的 CPU 上专家计算更慢（同一台 Zen4 上实权重单层的 AVX2 路径约慢 1.4–1.6 倍），decode 更依赖热专家池，显存允许时多给槽位；
    `LKQMOE_THREADS` 取“每个 NUMA 节点的物理核数 − 1”乘节点数，留出信箱调度线程和主线程。
  - 多卡：设 `LKQMOE_TP_NODES=auto` 与 `LKQMOE_MAIN_CPUS=auto`，去掉写死的 `LKQMOE_PROBE_DISPATCH_CPU`；
    每张卡对应的 NUMA 节点最好属于同一颗 CPU（`nvidia-smi topo -m` 可查）。消费级卡之间没有 P2P，加 `--disable-custom-all-reduce`。
  - TP 下草稿热词表（`--speculative-token-map`）不能用：草稿 lm_head 按完整词表的 id 从目标 lm_head 取行，而 TP 下每个 rank
    只有自己那一半词表，会越界；去掉后草稿直接共用目标 lm_head，投机解码仍然无损。
  - lkqmoe 的 GPU 预填充每个分块都要把全部专家权重传到 GPU，PCIe 4.0 主机上 prefill 速度大致与分块大小成正比，显存不够时
    可减小 `LKQMOE_PREFILL_EXPERT_CHUNK` 和 DSA/索引器类的分块参数来换更大的预填充分块。

## 测试脚本（`bench/`）

- `speed_points.py`：任意长度打点测速（本页单请求表）；`--exact` 精确输入长度，`--stop-on-fail` 遇到失败点即停。
- `spec_bench.py`：多轮对话 decode 速度（6 类提示）。
- `spec_t1_bench.py`：温度 1.0 / 0 的 decode 速度与接受长度（本页温度 1.0 表；长文档用任意文本文件）。
- `ppl_eval.py`：困惑度（逐 token 对数概率，分段、每段清缓存）；语料路径用参数指定。
- `gen_corpus.py`、`build_token_map.py`：收集模型输出、生成草稿热词表。

## 许可

本仓库开源部分按 Apache-2.0（与 Lsglang/sglang 相同）；`lkqmoe/` 下的二进制见其 LICENSE。模型权重遵循各自许可。
