# 前向退化定位与优化结果

后续调优见[当前报告](forward_tuning_zh.md)。本报告保留历史 `4b832de` 源码及性能记录。

冻结 pointer 候选 `64148ed` 的前向慢于上游，虽然融合反向使前后向总耗时下降，仍需修复前向退化。[完整基线报告](bucketed_n_zh.md)记录该候选的性能及验证；本报告记录修复后的冻结实现 `4b832de` 及其完整验证和最终性能。

## 已验证的影响因素

[ablation.log](benchmarks/evidence/forward_recovery/ablation.log) 在 H20-3e、BF16、B1/H8/D32 上使用预热后的交错 ABBA 测量，每侧 60 个样本，包含前向 wrapper 成本。

| N | 我们的 runtime N / 精确 N 耗时比，同配置 | 上游关闭 TMA 相比开启 TMA 的耗时增加 |
|---:|---:|---:|
| 512 | 1.080× | 12.1% |
| 800 | 1.062× | 7.4% |
| 1024 | 1.070× | 15.4% |

精确 N 消融复用 runtime 候选已选中的配置，未单独重新调优；它是在同实现、同配置下检查长度特化的影响，**不是历史 `249f` 的重建**。上游的 TMA 开关实验则在另一套实现上检查访存路径。两组结果不能相加或相乘，声称已解释全部退化：pointer 候选与上游之间其他实现和配置差异仍会影响结果。这些证据支持同时保留桶内动态 N、恢复合适的 TMA 路径继续验证。

## 最终实现

新的前向保留 BF16/FP16 的 TMA Q/K/V、bias、mask 读取，输出改用普通向量 store，生成现有融合反向需要的 O、LSE、mx、dn，继续接现有反向；FP32 保留 centered 统计前向，避免改变其数值合同。前向 bias/mask 仅为满足 descriptor 的 16 字节行对齐而补齐；反向 bias 和 FP32 dBias 才补至 CLOSEST_N 桶宽，Q/K/V 不做整桶补齐。长度仍按上游向上 2 幂分桶，真实 N 为运行时参数，不回退到逐精确 N 特化。

尾部补齐 key 的 score 在 softmax 前设为负无穷，其概率显式置零，排除在真实 key 的归一化分母之外。真实 masked key 仍遵守有限 −10000 分数语义；不能将补齐与真实 masked key 混为一类，尤其是全 mask 行。

N 和 stride 使用运行时 int64，descriptor 坐标在调用接口处显式转换为 int32。该接口转换不代表任意大坐标均合法，也不把全局指针地址计算缩窄成 int32。query 尾部无效行不会参与合法行的独立 softmax 归约，且写回受边界保护，因此移除对合法结果没有作用的额外 softmax select。

## 输出写回调查与修复

此前采用 TMA 输出 store 的 `5f1` 候选在 initcheck 中报告 `_preprocess` 读取 O 未初始化。原始 [initcheck.log](benchmarks/evidence/forward_recovery/initcheck.log) 与 [initcheck-interruption.json](benchmarks/evidence/forward_recovery/initcheck-interruption.json) 保留：父 driver 被中断，不能把仍为 running 的旧命令 metadata 解释为成功或伪造退出码。

预热后重新分配输出并填 NaN 的 12 项覆盖检查全部通过；检查的六份 PTX 均包含输出 store、commit 与 `wait_group.read`。[NVIDIA Compute Sanitizer 已知限制](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)说明部分 tensormap/tcgen05 global 指令尚不受支持，可能导致 initcheck 假阳性。这些现象符合已知工具限制，但不足以证明本实例一定是误报。

采用普通向量输出 store 后，TMA 输入读取仍保留。四项定向 initcheck（BF16 D32、FP16 D128，各覆盖 full/chunk）均退出 0、零错误。初步 [bench-pointer-store](benchmarks/evidence/forward_recovery/bench-pointer-store.log) 在 N=512/800/1024 的前向速度比分别为 1.033/1.002/1.014×，直接测量 F+B 为 1.293/1.182/1.311×；仅作为选择替代实现的诊断证据，不是最终提交性能表。冻结 `4b832de` 后已完成完整复用、数值、eager/compile、memcheck/initcheck、workspace 与性能检查。

## 冻结实现与最终验证

最终代码冻结为 **`4b832dee1e34b5303d5b39f9435cc83117064136`**。以下为该版本自身的完整验证；历史 pointer `64148ed` 仅作为对照。性能测量也已完成，见下表。

| 项目 | 结果 |
|---|---|
| 冻结源码 hash / commit | `4b832de`，见 [frozen_sources.json](benchmarks/evidence/forward_recovery/final/frozen_sources.json) |
| 最终性能与显存 | 见下文最终表；`64148ed` 仅作历史 pointer 对照 |
| 桶内 JIT/PTX 复用 | eager/compiled 各 4 桶、17 次调用、46 次目标 PTX 编译；已见桶内新增 0 |
| 完整 eager / compile | 386 passed（220.10 秒）/ 18 passed |
| 大 N FP64 独立参考 | N=500/512/513/800/1024 通过，B1 H8 BF16 D32，每个 N 一个 seed |
| memcheck / initcheck | 各 5 passed、0 errors，命令退出 0 |


所有最终结论对应该候选自身的冻结源码和日志，不能继承 pointer 基线的通过记录。测量口径沿用基线报告：稳态 eager 公开 API，包含 wrapper 成本；不是单 kernel、完整训练 step 或编译性能。

编译模式的 Dynamo 图计数在 N>1 时为 1，首次 N=1 后为 2；重复 N=1 不再增加。kernel 桶内复用不代表所有长度只有一张计算图。复用 proof 的主序列独立参考只覆盖 N≤129，大 N 的数值结论来自单独的 FP64 检查。

实际验证设备为 H20（SM90）。SM<90 的低精度路径在代码中回退 pointer 前向，但本轮没有其他架构的实卡验证。

## 最终性能与低显存取舍

[bench-final](benchmarks/evidence/forward_recovery/final/bench-final.log) 与 [workspace 对照](benchmarks/evidence/forward_recovery/final/bench-workspace.log) 均完成、命令退出 0。环境为 H20-3e，PyTorch 2.13.0a0+9186a08b2c.nv26.07、Triton 3.7.1、CUDA 13.3、Compute Sanitizer 2026.2.1.0；BF16，B1 H8 D32，连续 dO，mask 概率 0.2。上游对照为 `b4ecec4`，融合默认 chunk128。

| N | Fwd upstream → fused (ms) | Bwd upstream → fused (ms) | F+B upstream → fused (ms) | Fwd speedup | F+B speedup |
|---:|---:|---:|---:|---:|---:|
| 500 | 1.711 → 1.594 | 12.670 → 6.495 | 14.384 → 8.092 | 1.073× | 1.777× |
| 512 | 1.494 → 1.445 | 7.563 → 5.532 | 9.066 → 6.979 | 1.034× | 1.299× |
| 513 | 2.134 → 1.999 | 15.617 → 7.688 | 17.761 → 9.678 | 1.067× | 1.835× |
| 640 | 2.828 → 2.733 | 14.377 → 10.599 | 17.201 → 13.316 | 1.035× | 1.292× |
| 768 | 4.742 → 4.608 | 24.516 → 17.940 | 29.231 → 22.523 | 1.029× | 1.298× |
| 800 | 5.946 → 5.919 | 29.251 → 23.870 | 35.207 → 29.842 | 1.005× | 1.180× |
| 1024 | 10.790 → 10.637 | 57.144 → 41.290 | 67.936 → 51.961 | 1.014× | 1.307× |

这组长度的前向已恢复至上游水平或更快；前后向总收益来自融合反向，不能据此推广到未测 dtype/D、设备或完整训练步骤。速度比为上游耗时/融合耗时；完整原始样本包含反向独立速度比。

| N | Full → chunk128 F+B (ms) | Chunk latency change | Full → chunk128 peak increment (MiB) | Saved (MiB) |
|---:|---:|---:|---:|---:|
| 512 | 6.923 → 6.956 | +0.48% | 804.52 → 612.52 | 192.00 |
| 800 | 29.528 → 29.784 | +0.87% | 1985.54 → 1459.54 | 526.00 |
| 1024 | 52.360 → 51.970 | -0.74% | 3217.03 → 2321.03 | 896.00 |

默认低显存相对融合 full workspace 减少 192/526/896 MiB 的 peak increment，F+B 延迟变化在 −0.74% 至 +0.87% 之间。这是相对 full workspace 的取舍；相对上游并不省显存：

| N | Upstream → fused peak increment (MiB) |
|---:|---:|
| 500 | 539.96 → 591.29 |
| 512 | 564.25 → 612.52 |
| 513 | 566.46 → 630.80 |
| 640 | 881.64 → 954.39 |
| 768 | 1269.56 → 1347.77 |
| 800 | 1380.84 → 1459.54 |
| 1024 | 2257.00 → 2321.03 |

MiB=2²⁰ 字节。peak allocated increment 排除预分配输入，包含输出、梯度和临时量，不是总显存。

测量为预热后的 eager 公开 API，3 次预热、3 轮 ABBA、每侧 60 个样本，中位数之比；前向启用梯度，单独反向使用 retain_graph，F+B 直接测量而非两列相加。包含 padding/cast/flush 及 event 区间内主机提交间隙，不是单 kernel 或 compile 性能。

**缓存顺序：** 先执行 workspace 测量 `512,800,1024`，H8 的新 v3 前向调优命名空间由 512/800 分别首次触发 512/1024 桶；随后 final 顺序 `500,512,513,640,768,800,1024` 复用这些选择。上游及反向复用既有 persistent 配置，[最终配置 JSON](benchmarks/evidence/forward_recovery/final/tuning-configs/) 已归档。因此不能说 final 的 500 首次决定了桶内 winner，也不能将这些稳态测量解释为冷启动时间。

最终证据目录为 [forward_recovery/final](benchmarks/evidence/forward_recovery/final/)，包含源码清单、命令状态和日志。大 N 独立参考按 i 覆盖完整张量，限上述配置每个 N 一个 seed；candidate_seconds 不是性能测量。

原始验证日志：[proof eager](benchmarks/evidence/forward_recovery/final/proof-eager.log) · [proof compiled](benchmarks/evidence/forward_recovery/final/proof-compiled.log) · [eager](benchmarks/evidence/forward_recovery/final/pytest-eager.log) · [compile](benchmarks/evidence/forward_recovery/final/pytest-compile.log) · [FP64](benchmarks/evidence/forward_recovery/final/large-reference.log) · [memcheck](benchmarks/evidence/forward_recovery/final/memcheck.log) · [initcheck](benchmarks/evidence/forward_recovery/final/initcheck.log) · [SHA256SUMS.json](benchmarks/evidence/forward_recovery/final/SHA256SUMS.json)
