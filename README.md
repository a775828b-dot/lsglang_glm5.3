# lsglang_glm5.3

GLM-5.3 Flash NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，512K 上下文，EAGLE（NextN）投机解码（无损拒绝采样），
**LKQstrata 热专家池**（每个 MoE 层的热门专家常驻 GPU、自适应换入换出，与 CPU 同时计算），温度 1.0 下 decode 约 83 tok/s，支持图片。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本）。

*English: a single-GPU GLM-5.3 Flash NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 512K context, EAGLE/NextN speculative
decoding with lossless rejection sampling, LKQstrata GPU expert pool for the hot experts of every MoE layer, ~83 tok/s decode at temperature 1.0,
image input) on guqiong96/Lsglang, with the CPU-side routed experts on lkqmoe, an NVFP4-specialised MoE kernel inspired by lk_moe
(by guqiong96, the author of lsglang); a compiled, closed-source build is included here.*

## 组成

| 部分 | 来源 | 本仓库内容 |
|---|---|---|
| 模型 | [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)：LibertAIDAI 对智谱 **GLM-5.3 Flash** 的 NVFP4 量化（路由专家 NVFP4，注意力、稠密层与共享专家 BF16，含 1 层 NextN 草稿），188 GB | — |
| 推理框架 | guqiong96/Lsglang 分支 `lkmoe-glm5.3-flash-sm80plus` @ [`cca56c1f36`](https://github.com/guqiong96/Lsglang/commit/cca56c1f36b0326e4d5cb7c6878bf29c064e39d0)（GLM-5.3 支持来自 sglang PR #36507）；补丁含 [usrlocalben/Lsglang](https://github.com/usrlocalben/Lsglang) `pr-20-22-contd`（HiCache DSA 修复、原生图片预处理、NextN 多模态等）与 SM120 上融合的 mHC pre/post Triton 内核；投机解码：拒绝采样配合草稿高频词表、草稿温度系数、按草稿置信度截断验证 | `patches/lsglang-cca56c1f-glm53.patch` |
| CPU 专家内核 | **lkqmoe**（闭源，编译版）：受 lk_moe 启发、针对 NVFP4 格式专项加速的 CPU/GPU 混合 MoE 内核。lk_moe 来自 lsglang 作者 guqiong96（[guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)）；lkqmoe 独立实现其 `MOE_NVFP4` 接口，不含 lk_moe 代码。0.5.0 起含 **LKQstrata 热专家池**：GPU 上缓存每层的热门专家，换入换出策略参考 [Strata](https://github.com/Niko1221/Strata)（MIT，独立实现） | `lkqmoe/` |
| 热专家池排名表 | 从 GLM 实际 decode 路由统计的每层专家使用排名（32 条中英文、代码、数学、agent 等提示，温度 1.0）；启动时按它装填，之后自适应 | `lkqmoe/lkqstrata-glm53.json` |
| GPU 侧补丁 | 开源：mHC 融合、BF16 权重转 FP8（W8A16 decode / W8A8 预填充）、GPU 常驻 MoE 层 W4A16 decode、DSA 索引器分块预填充、图片 CPU 预处理 | `lkqmoe/python/lkqmoe/gpu/` |
| 草稿热词表 | 从 GLM 实际输出、代理会话与文档统计的 49152 个高频 token（词表 154880；只影响草稿接受率，目标模型仍验证每个 token） | `lkqmoe/draft-token-map-49152.pt` |
| 启动 | 测速所用的完整参数 | `launch/run-glm53-nvfp4.sh` |

## 本机硬件

| | |
|---|---|
| CPU | 2× AMD EPYC 9334（共 64 核 128 线程），NPS2 → 4 个 NUMA 节点，AVX512-BF16/VBMI |
| 内存 | 320 GB DDR5-4800（20× 16 GB），实测带宽约 511 GB/s |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（SM120），驱动 595.99，CUDA 13.2；功耗墙 350 W（上一版数据测于 300 W；decode 约 166 W，不受限） |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

**本版升级（2026-10-14，lkqmoe 0.5.0 LKQstrata 热专家池）**

- 上一版把 MoE 第 3–11 层整层放在 GPU（约 34 GB），其余 33 层的路由专家只在 CPU 上算，GPU 和 CPU 轮流空闲。
- 本版把同一块显存改成 42 个 MoE 层各 62 个"热专家"槽位：每层 decode 时，槽位里的专家由 GPU 计算（W4A16），其余由 CPU
  计算，两边同时进行后相加；槽位启动时按排名表装填，运行中按路由统计自适应换入换出。实测每层约一半的专家由 GPU 承担。
- 支持 2 个请求同时运行（见下面并发表），上下文仍是 512K。
- 精度不变：困惑度 4.0698 → 4.0580，所有长度的埋藏事实都找回。代价是首 token 变慢：长输入 2–4%，4K–17K 输入 13–17%（池里的层预填充要从内存上传权重）。
- 回到上一版：`RESIDENT=0,3-11 LKQSTRATA=0 bash launch/run-glm53-nvfp4.sh`。

**单请求：上一版 → 本版**（温度 0，每点输出 1024 token，输入是埋了一条事实的长文本，`bench/speed_points.py`；
上一版 2026-10-04，本版 2026-10-14；原始数据 `results/speed-points-20261004.json`、`results/speed-points-20261014.json`）

| 输入 token | decode tok/s | 首 token 时间 |
|---:|---:|---:|
| 4K | 80.2 → **102.1**（+27%） | 2.77 → 3.23 s |
| 17K | 75.8 → **110.0**（+45%） | 7.68 → 8.71 s |
| 33K | 76.5 → **114.7**（+50%） | 12.44 → 12.90 s |
| 66K | 76.4 → **95.6**（+25%） | 26.31 → 27.34 s |
| 136K | 68.8 → **102.4**（+49%） | 51.94 → 53.32 s |
| 274K | 62.5 → **96.2**（+54%） | 105.16 → 107.66 s |
| 413K | 65.9 → **85.5**（+30%） | 162.45 → 165.98 s |
| 515K | 67.0 → **93.2**（+39%） | 204.00 → 208.46 s |
| **平均** | 71.6 → **100.0（+40%）** | |

温度 1.0（实际使用，4 类中文任务各 3 次，`bench/spec_t1_bench.py`）：62.5 → **83.3 tok/s（+33%）**。

**并发：上一版 → 本版**（短提示，温度 1.0，每个请求 512 token，同时开始，同日测得）

| 并发 | 上一版 单请求 / 总吞吐 | 本版 单请求 / 总吞吐 | 总吞吐 |
|---:|---:|---:|---:|
| 1 | 60.8 / 60.9 tok/s | 78.0 / 78.1 tok/s | **+28%** |
| 2 | 39.4 / 77.9 tok/s | 51.5 / 101.6 tok/s | **+30%** |

2 并发要另加 `LKQSTRATA_SLOTS=60` 和启动参数 `--max-running-requests 2 --max-mamba-cache-size 10`（省出的显存给第二个请求），
512K 上下文不变。

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
3. 下载模型，按需修改 `launch/run-glm53-nvfp4.sh` 开头的路径后启动：`bash launch/run-glm53-nvfp4.sh`。
   服务在 `127.0.0.1:18085`，OpenAI 兼容接口，模型名 `GLM-5.3-Flash-NVFP4`，思考输出由 `glm45` 解析器分离，工具调用 `glm47`。
   冷加载约 4–5 分钟（模型文件已在页缓存时）。

启动前需要约 200 GB 可用内存（42 层专家都在内存里）、70 GB 空闲显存。

## 主要启动参数

```
--context-length 524288 --max-total-tokens 528384 --max-running-requests 1 --chunked-prefill-size 16384
--mem-fraction-static 0.95 --page-size 64 --quantization modelopt_fp4 --moe-runner-backend flashinfer_cutlass
--kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla
--linear-attn-backend triton --mamba-ssm-dtype bfloat16 --mamba-scheduler-strategy extra_buffer
--disable-shared-experts-fusion --disable-prefill-cuda-graph --mm-feature-transport cpu --image-processor-backend pil
--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--speculative-adaptive --speculative-adaptive-config launch/adaptive-steps-1to5.json
--speculative-token-map lkqmoe/draft-token-map-49152.pt --speculative-use-rejection-sampling
--reasoning-parser glm45 --tool-call-parser glm47
LVLLM_GPU_RESIDENT_MOE_LAYERS=0  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=512  LVLLM_GPU_PREFETCH_WINDOW=2
LKQSTRATA=1  LKQSTRATA_SLOTS=62  LKQSTRATA_PROFILE=lkqmoe/lkqstrata-glm53.json
SGLANG_MHC_PRE_TRITON=1  SGLANG_MHC_POST_TRITON=1
LKQMOE_MODE=standalone  LKQMOE_THREADS=112  LKQMOE_DYNAMIC=2  LKQMOE_DOWN_BF16=1  LKQMOE_PREFILL_OVERLAP=1
LKQMOE_PREFILL_EXPERT_CHUNK=16  LKQMOE_PREFILL_ROUTE_ACCUM=1  LKQMOE_PREFILL_PREFETCH_CHUNKS=12  LKQMOE_DOWN_PRECISION=bf16
LKQMOE_RESIDENT_W4A16=1  LKQMOE_RESIDENT_W4A16_SKIP_DRAFT=1  LKQMOE_FP8_WEIGHTS=1  LKQMOE_GLM53_MHC=1
LKQMOE_SMALL_M_GEMM=1  LKQMOE_INDEXER_Q_CHUNK=4096  LKQMOE_GLM53_CPU_IMAGE=1  LKQMOE_ACTIVE_FLAGS=1
SGLANG_RS_DRAFT_TEMP_SCALE=0.8  SGLANG_SPEC_TRUNCATE_Q=0.2
SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1  SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA=1
```
完整列表见 `launch/run-glm53-nvfp4.sh`；预填充 CPU / 混合 / GPU 的分界与 W8A8 的作用范围在 `launch/lkqmoe-runtime.json`（运行中修改即生效）。

## lkqmoe（`lkqmoe/`）

- 编译版：`liblkqmoe.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥与 LKQstrata 换入换出管理）、`python/lkqmoe/*.pyc`（lsglang 适配层、Triton GPU 预填充、
  LKQstrata 热专家池 `pool.pyc`）、
  `python/lkqmoe/gpu/small_gemm.pyc`（decode 小批量 BF16 GEMM）；含 Triton 内核的模块内嵌压缩源码，因为 Triton 编译时要读源码。
  许可见 `lkqmoe/LICENSE`：可免费使用、原样再分发。
- 版本 0.5.0（2026-10-14）：LKQstrata 热专家池。默认关闭，关闭时与 0.4.3 逐位相同（`liblkqmoe.so` 不变）。开关：
  `LKQSTRATA=1`；`LKQSTRATA_SLOTS`（每层槽位，GLM-5.3 用 62 = 上一版 9 个常驻层的显存；保留 3 层常驻时 44）；
  `LKQSTRATA_PROFILE`（排名表，按模型层号取行，任意常驻层组合都能用）；`LKQSTRATA_ADAPT`（自适应，默认 1）与
  `LKQSTRATA_EVERY` / `_SWAPS` / `_DECAY` / `_MIN_GAIN`（默认 4 / 16 / 0.7 / 1.5）；`LKQSTRATA_LOG=N`（每 N 个 decode 窗口打一行命中率）。
  常驻层仍由 `LVLLM_GPU_RESIDENT_MOE_LAYERS` 决定，热专家池只覆盖其余层。精度：GPU 部分对 CPU 结果相对误差 1e-7–3e-4（实权重单层）。
- 版本 0.4.3：与 DeepSeek V4.1、Qwen3.8 管线是同一套通用内核，
  GLM-5.3 的形状（4096 / 2048）在编译期特化；预填充暂存块数可配置（`LKQMOE_PREFILL_PREFETCH_CHUNKS`），
  decode 小批量 BF16 GEMM 的 split-K 归约在同一次启动里完成（结果逐位不变）。2026-10-04 起新增每 token 跳过标志（`LKQMOE_ACTIVE_FLAGS=1`，默认关闭）：调用方把要丢弃的 token 标出后，打包内核把它们的路由写成 −1，CPU 只算其余 token、被跳过的行返回 0（用于上面的截断验证）；CPU 内核 `liblkqmoe.so` 不变，Qwen3.8 实权重 118 例逐位一致。
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
- 硬件要求：x86-64 AVX512-BF16（VBMI 更快），4 个 NUMA 节点 × 16 物理核（其他拓扑未验证），SM120 GPU（`resident_w4a16.py` 的
  硬件 FP4 转换需要 sm_100a / sm_120a）。

## 测试脚本（`bench/`）

- `speed_points.py`：任意长度打点测速（本页速度表）；`--exact` 精确输入长度，`--stop-on-fail` 遇到失败点即停。
- `spec_bench.py`：多轮对话 decode 速度（6 类提示）。
- `spec_t1_bench.py`：温度 1.0 / 0 的 decode 速度与接受长度（本页温度 1.0 表；长文档用任意文本文件）。
- `ppl_eval.py`：困惑度（逐 token 对数概率，分段、每段清缓存）；语料路径用参数指定。
- `gen_corpus.py`、`build_token_map.py`：收集模型输出、生成草稿热词表。

## 许可

本仓库开源部分按 Apache-2.0（与 Lsglang/sglang 相同）；`lkqmoe/` 下的二进制见其 LICENSE。模型权重遵循各自许可。
