# 2026-10-03：首次发布；lkqmoe 0.4.3

- 版本 0.4.3：与 DeepSeek V4.1、Qwen3.8 管线是同一套通用内核，
  GLM-5.3 的形状（4096 / 2048）在编译期特化；预填充暂存块数可配置（`LKQMOE_PREFILL_PREFETCH_CHUNKS`），
  decode 小批量 BF16 GEMM 的 split-K 归约在同一次启动里完成（结果逐位不变）。2026-10-04 起新增每 token 跳过标志（`LKQMOE_ACTIVE_FLAGS=1`，默认关闭）：调用方把要丢弃的 token 标出后，打包内核把它们的路由写成 −1，CPU 只算其余 token、被跳过的行返回 0（用于上面的截断验证）；CPU 内核 `liblkqmoe.so` 不变，Qwen3.8 实权重 118 例逐位一致。
