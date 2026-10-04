# lsglang_glm5.3

GLM-5.3 Flash NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，512K 上下文，EAGLE（NextN）投机解码（无损拒绝采样，温度 1.0 下 decode 约 61 tok/s），支持图片。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本）。

*English: a single-GPU GLM-5.3 Flash NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 512K context, EAGLE/NextN speculative
decoding with lossless rejection sampling, ~61 tok/s decode at temperature 1.0, image input) on guqiong96/Lsglang, with the CPU-side routed experts on lkqmoe, an NVFP4-specialised MoE kernel inspired by lk_moe
(by guqiong96, the author of lsglang); a compiled, closed-source build is included here.*

## 组成

| 部分 | 来源 | 本仓库内容 |
|---|---|---|
| 模型 | [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)：LibertAIDAI 对智谱 **GLM-5.3 Flash** 的 NVFP4 量化（路由专家 NVFP4，注意力、稠密层与共享专家 BF16，含 1 层 NextN 草稿），188 GB | — |
| 推理框架 | guqiong96/Lsglang 分支 `lkmoe-glm5.3-flash-sm80plus` @ [`cca56c1f36`](https://github.com/guqiong96/Lsglang/commit/cca56c1f36b0326e4d5cb7c6878bf29c064e39d0)（GLM-5.3 支持来自 sglang PR #36507）；补丁含 [usrlocalben/Lsglang](https://github.com/usrlocalben/Lsglang) `pr-20-22-contd`（HiCache DSA 修复、原生图片预处理、NextN 多模态等）与 SM120 上融合的 mHC pre/post Triton 内核；投机解码：拒绝采样配合草稿高频词表、草稿温度系数、按草稿置信度截断验证 | `patches/lsglang-cca56c1f-glm53.patch` |
| CPU 专家内核 | **lkqmoe**（闭源，编译版）：受 lk_moe 启发、针对 NVFP4 格式专项加速的 CPU/GPU 混合 MoE 内核。lk_moe 来自 lsglang 作者 guqiong96（[guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)）；lkqmoe 独立实现其 `MOE_NVFP4` 接口，不含 lk_moe 代码 | `lkqmoe/` |
| GPU 侧补丁 | 开源：mHC 融合、BF16 权重转 FP8（W8A16 decode / W8A8 预填充）、GPU 常驻 MoE 层 W4A16 decode、DSA 索引器分块预填充、图片 CPU 预处理 | `lkqmoe/python/lkqmoe/gpu/` |
| 草稿热词表 | 从 GLM 实际输出、代理会话与文档统计的 49152 个高频 token（词表 154880；只影响草稿接受率，目标模型仍验证每个 token） | `lkqmoe/draft-token-map-49152.pt` |
| 启动 | 测速所用的完整参数 | `launch/run-glm53-nvfp4.sh` |

## 本机硬件

| | |
|---|---|
| CPU | 2× AMD EPYC 9334（共 64 核 128 线程），NPS2 → 4 个 NUMA 节点，AVX512-BF16/VBMI |
| 内存 | 320 GB DDR5-4800（20× 16 GB），实测带宽约 511 GB/s |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（SM120），驱动 595.84，CUDA 13.2；本页测速时功耗墙 300 W（默认值；decode 约 166 W，不受限） |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

### 温度 1.0（实际使用）

客户端不传温度时用模型默认值（温度 1.0，top_p 0.95），这才是 agent / 对话的真实情况。同一时段对比上一版（2026-10-03，
自适应 EAGLE + 高频词表，不用拒绝采样）与本版（`bench/spec_t1_bench.py`；原始数据 `results/temperature1-20261004.json`）：

| 测试 | 上一版 | 本版 | 变化 |
|---|---:|---:|---:|
| 温度 1.0：中文讲解 / 中文写代码 / 中文找 bug / 约 10 万 token 文档问答，各 3 次 × 512 token | 56.8 tok/s（接受长度 2.48） | **61.6**（2.58） | **+8.5%** |
| 温度 1.0：14 个提示（中英文、代码、数学、agent 排错等）× 768 token | 55.3（2.35） | **61.0**（2.50） | **+10.3%** |
| 温度 0：上面 4 个提示各 1 次 | 61.4（2.78） | 61.2（2.72） | 持平 |

提升来自投机解码，全部无损（输出分布与不用投机解码完全相同）：拒绝采样配合草稿高频词表 +5.2%，草稿温度 ×0.8 约 +0.9%，
按草稿置信度截断验证约 +1.75%；另有 DSA 注意力元数据的融合（每步少约 0.7 ms 主机空等）。预填充、温度 0、困惑度、显存不变。

### 长上下文打点（温度 0）

单请求，温度 0，`ignore_eos`，每点输出 1024 token（含思考内容）；输入是长日志文本，在 10% 深度埋一条事实并在末尾提问
（`bench/speed_points.py`）。上下文 524,288，8 个点在 512K 总上下文内取样；显存峰值包含预填充暂存（约 2.1 GiB），
512K 满载时仍余约 2.4 GiB。原始数据 `results/speed-points-20261004.json`（上一版 `results/speed-points-20261003.json`，首 token 时间相同，decode 8 点平均 71.7 vs 本版 71.6，在单次运行的 ±10% 波动内）。

| 输入 token | 首 token 时间 | 预填充 tok/s | decode tok/s | 平均接受长度 | 找回埋藏事实 | 显存峰值 |
|---:|---:|---:|---:|---:|:---:|---:|
| 3,995 | 2.77 s | 1,441 | 80.2 | 3.17 | 是 | 66,634 MiB |
| 16,929 | 7.68 s | 2,204 | 75.8 | 3.27 | 是 | 70,754 MiB |
| 32,873 | 12.44 s | 2,644 | 76.5 | 3.36 | 是 | 70,876 MiB |
| 66,101 | 26.31 s | 2,512 | 76.4 | 3.41 | 是 | 70,878 MiB |
| 135,927 | 51.94 s | 2,617 | 68.8 | 3.41 | 是 | 70,900 MiB |
| 274,390 | 105.16 s | 2,609 | 62.5 | 3.37 | 是 | 70,922 MiB |
| 413,125 | 162.45 s | 2,543 | 65.9 | 3.35 | 是 | 70,944 MiB |
| 514,559 | 204.00 s | 2,522 | 67.0 | 3.36 | 是 | 70,944 MiB |

- 多轮对话（`bench/spec_bench.py`，6 条提示 × 2 轮，每条最多 700 token，温度 0，总输出 token ÷ 总耗时）：63.19 tok/s（上一版 61.66，原版 38.77）。
- 精度：困惑度（`bench/ppl_eval.py`，13 段 × 4096 token，中英文文档与代理会话）4 次平均 4.0698（每次 4.0686–4.0713；上一版 4.0719）；原版 4.0517（2 次），高约 0.45%，主要来自预填充时 GPU 常驻层的 W4A4 与 CPU 专家的 BF16 中间值。
- decode 速度随生成内容变化较大：同一提示两次运行的接受长度可以差 0.4，自适应草稿随之改变每步验证的 token 数，单点 decode
  相差 ±10% 很常见（例如 514K 同一提示 62.0 / 63.2 tok/s），比较时看多点平均或固定草稿步数下的每步耗时。困惑度与多轮测速的原始数据
  `results/ppl-and-multiturn-20261004.json`（上一版 `-20261003.json`）。

## 运行方式

- 模型 45 层：前 3 层稠密，其余 42 层 MoE（288 专家 top-8 + 1 个共享专家，hidden 4096，专家中间维 2048，SwiGLU 限幅 10）；
  注意力 34 层 KDA 线性注意力 + 11 层 DSA 稀疏 MLA（每 4 层一层）；mHC 残差（hc_mult 4）；1 层 NextN 草稿。
- MoE 第 3–11 层（9 层）常驻 GPU，其余 33 层的路由专家在 CPU（lkqmoe，112 线程，按节点动态领取行）；草稿层在 GPU。
- 预填充：640 token 以下走 CPU，640–1792 CPU+GPU 混合，更长走 lkqmoe 的 Triton GPU 内核（按 16 个专家一块上传权重，
  解包与上传在独立流上流水，路由结果按块 FP32 累加；12 个暂存块组成跨层环，当前层最后 12 块一解包完，下一层的前 12 块
  就开始上传，正好落在下一层注意力计算期间，预填充时多占约 2.1 GiB）；16K 分块预填充；DSA 索引器的预填充 logits 按 4096 个查询一块计算
  （512K 时临时显存 7.8 GiB → 约 2 GiB）。
- mHC：SM120 上 pre（含 FP32 投影）与 post 各融合成 Triton 内核（decode 每次 82 → 18 µs）。
- KV 缓存 FP8；单请求并发；图片在 CPU 上解码与预处理（`--image-processor-backend pil`）。
- EAGLE（NextN）：自适应草稿步数（候选 1–5 步，`launch/adaptive-steps-1to5.json`），草稿热词表 49152。
- 拒绝采样（`--speculative-use-rejection-sampling`，Leviathan 式，无损）：草稿按 softmax(logits / T) 采样，验证时以
  coin·q(x) < p(x) 接受，拒绝时从 (p − q)⁺ 补采，输出分布与不用投机解码相同。原版 sglang 要求草稿与主模型词表一致，
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
| CPU 路由专家（33 层）decode | NVFP4 | BF16 输入，FP32 Gate/Up 与 SiLU，Down 用 BF16 点积，FP32 累加（W4A16） |
| CPU 层专家的长预填充（GPU Triton） | NVFP4 → BF16 | BF16（W4A16），FP32 路由累加 |
| GPU 常驻 MoE 层 decode（≤16 token） | NVFP4 | SM120 硬件 FP4→FP16 转换，FP32 累加（W4A16，`resident_w4a16.py`） |
| GPU 常驻 MoE 层预填充、草稿层 MoE | NVFP4 | NVFP4（W4A4，flashinfer CUTLASS） |
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

启动前需要约 160 GB 可用内存、70 GB 空闲显存。

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
LVLLM_GPU_RESIDENT_MOE_LAYERS=0,3-11  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=512  LVLLM_GPU_PREFETCH_WINDOW=2
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

- 编译版：`liblkqmoe.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥）、`python/lkqmoe/*.pyc`（lsglang 适配层与 Triton GPU 预填充）、
  `python/lkqmoe/gpu/small_gemm.pyc`（decode 小批量 BF16 GEMM）；含 Triton 内核的模块内嵌压缩源码，因为 Triton 编译时要读源码。
  许可见 `lkqmoe/LICENSE`：可免费使用、原样再分发。版本 0.4.3：与 DeepSeek V4.1、Qwen3.8 管线是同一套通用内核，
  GLM-5.3 的形状（4096 / 2048）在编译期特化；预填充暂存块数可配置（`LKQMOE_PREFILL_PREFETCH_CHUNKS`），
  decode 小批量 BF16 GEMM 的 split-K 归约在同一次启动里完成（结果逐位不变）。2026-10-04 起新增每 token 跳过标志（`LKQMOE_ACTIVE_FLAGS=1`，默认关闭）：调用方把要丢弃的 token 标出后，打包内核把它们的路由写成 −1，CPU 只算其余 token、被跳过的行返回 0（用于上面的截断验证）；CPU 内核 `liblkqmoe.so` 不变，Qwen3.8 实权重 118 例逐位一致。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`；`python/lkqmoe/gpu/`：
  - `glm53_mhc.py`：mHC pre 整体融合（split-K FP32 投影 + RMS + 加权和），替换 cuBLAS gemmSN（decode 只用 3 个 CTA）。
    另有可选的“注意力 post 并入 MLP pre”（`LKQMOE_GLM53_MHC_FUSE_POST`，默认关闭：累加顺序不同，偶有 1 ulp 差异）。
  - `fp8_weights.py`：加载后把选中的 BF16 线性层存成 FP8 E4M3（显存 11.55 → 5.95 GiB），decode 用 split-K W8A16 内核
    （split-K 归约由每个列块最后完成的 CTA 按固定顺序完成，一次启动、结果确定），预填充用 W8A8 Triton 内核。
    省下的显存换成多一个 GPU 常驻 MoE 层。
  - `resident_w4a16.py`：GPU 常驻 NVFP4 MoE 层的 decode 改用 BF16 激活（W4A16），直接读 flashinfer 已排好的权重与 swizzle scale，
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
