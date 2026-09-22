# 融合 Triangle Attention

本改动新增 `triangle_attention_fused`，融合反向计算并调优 TMA 前向。相比 TriFast 主干（`master`，`b4ecec4`），下列实测配置的前向速度为 **1.024–1.105 倍**，前后向合计为 **1.186–1.846 倍**。

## 改了什么

- 反向共享概率和分数梯度的计算，生成 dQ/dK/dV/dBias，避免重复计算，不物化完整概率张量。
- 默认 `chunk_i=128`，限制 FP32 dQ 临时缓冲区；`chunk_i=None` 使用完整 workspace。这是固定默认值，不按空闲显存自动切换。
- BF16/FP16 前向保留 TMA 输入读取，使用普通向量输出写回；新增 D32 调优候选，跳过冗余 padding 拷贝，同时保留地址对齐保护。
- 真实 N 保持运行时参数，沿用主干向上取 2 幂的调优桶，Q/K/V 保留实际尺寸。首次进入桶可能编译多个候选，后续长度复用已选配置。

原 `triangle_attention` API 不变。融合 API 支持一阶梯度；反向使用 FP32 atomic 归约，不保证确定性。真实 masked key 保留有限 −10000 语义，补齐 key 不参与归一化。

## 相比 TriFast 主干的性能

| N | 上游 → 本版前向 (ms) | 上游 → 本版反向 (ms) | 前向速度比 | 上游 → 本版 F+B (ms) | F+B 速度比 |
|---:|---:|---:|---:|---:|---:|
| 500 | 1.709 → 1.547 | 12.662 → 6.494 | 1.105× | 14.370 → 8.037 | 1.788× |
| 512 | 1.490 → 1.413 | 7.556 → 5.557 | 1.054× | 9.053 → 6.977 | 1.298× |
| 513 | 2.129 → 1.945 | 15.625 → 7.675 | 1.095× | 17.757 → 9.622 | 1.846× |
| 640 | 2.820 → 2.676 | 14.377 → 10.593 | 1.054× | 17.198 → 13.276 | 1.295× |
| 768 | 4.734 → 4.537 | 24.498 → 17.885 | 1.043× | 29.229 → 22.438 | 1.303× |
| 800 | 5.942 → 5.802 | 28.994 → 23.691 | 1.024× | 34.903 → 29.426 | 1.186× |
| 1024 | 10.811 → 10.490 | 56.705 → 40.910 | 1.031× | 67.422 → 51.276 | 1.315× |

条件：H20-3e、BF16、B1/H8/D32、默认 chunk128、连续 dO。预热后的 eager 公开 API，3 次预热、3 轮 ABBA、每侧 60 个样本，取耗时中位数。前向包含反向所需统计量，单独反向使用 `retain_graph`，F+B 直接测量。包含 wrapper 成本，不代表完整训练 step，也不推广至未测设备和形状。

长度按表中顺序执行，N500/513 首次选择两个前向桶的配置，其余长度复用；主干及反向使用已预热的持久化配置。[原始样本](benchmarks/evidence/forward_tuning/bench-upstream.log) · [调优配置](benchmarks/evidence/forward_tuning/tuning-configs/)。

“低显存”是相对完整融合 workspace 而言；默认融合实现仍可能比主干使用更多显存。

## 验证

验证实现：`61da9e0`。

- [42 项独立 FP64 检查](benchmarks/evidence/forward_tuning/forced-correctness-final.log)：输出及四类梯度，覆盖 BF16/FP16、mask、长度尾部及 full/chunked workspace；D32 强制新配置，另有 D128 冒烟测试。
- [16 项 TMA 回归测试](benchmarks/evidence/forward_tuning/pytest-tma.log)：包括基址未对齐的连续 bias 视图。
- [Eager](benchmarks/evidence/forward_tuning/proof-eager.log) 和 [torch.compile](benchmarks/evidence/forward_tuning/proof-compiled.log)：各 4 个桶、17 次调用，桶内复用不新增 JIT/PTX 编译；N>1 使用一张 Dynamo 图，N=1 另用一张。
- [memcheck](benchmarks/evidence/forward_tuning/memcheck.log) 与 [initcheck](benchmarks/evidence/forward_tuning/initcheck.log)：各覆盖两个 N65 BF16 强制配置用例，零错误。

[源码及环境记录](benchmarks/evidence/forward_tuning/SOURCE_AND_RUN.json) · [证据校验清单](benchmarks/evidence/forward_tuning/SHA256SUMS.json)。
