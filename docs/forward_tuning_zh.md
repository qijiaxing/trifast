# 融合 attention：追加前向调优

本报告描述冻结实现 **`61da9e0`**，在[已验证的 TMA 前向／融合反向实现](forward_recovery_zh.md)上追加小幅前向优化。最终公开 API 测量和本轮验证已完成：相对上一融合版，前向速度提升 1.68–2.74%，直接 F+B 提升 0.35–0.52%。早期 f21 测量仅作诊断。

## 实现变化

融合 TMA 路径拥有独立调优池，从上游配置复制并保留 descriptor 设置 hook。仅 D=32 新增 64×64 tile、4 warps、2 stages、maxnreg=96 候选；其他 D 剪除此候选，上游与 pointer fallback 保留原配置池。这是可选调优配置，不是对所有输入强制启用。

wrapper 在无需额外列时跳过 padding。mask 仍转换到宽类型，只在宽度已合适时省去额外 no-op pad。bias 则必须同时满足 16 字节行 pitch 和基址对齐：连续 view 也可能带有非对齐 storage offset，因此即使列宽无需补齐，`b.data_ptr() % 16 != 0` 时仍分配新缓冲区。新增四项测试覆盖该保护。

N 规则不变：真实 N 为运行时参数，向上取 2 幂分桶，不把精确 N 放入调优键；同桶共享选择，首次调优仍可能编译多个候选。反向、默认 chunk128、FP32 centered 统计、普通输出 store，以及仅支持一阶／非确定性梯度的合同沿用上版。前向 bias/mask 行对齐补齐与反向 bias/dBias 桶宽存储不同，Q/K/V 不扩展到桶大小。

## 诊断与未采用方案

[诊断归档](benchmarks/evidence/forward_tuning/README.md)保留已完成日志和命令状态。40 配置扫描未找到无寄存器上限时的稳定收益；unified-loop 和 tail-first 均退化，未采用。仅跳过 no-op padding 的收益最多约 0.5%，没有明显一致提升。Nsight Compute 因 `ERR_NVGPUCTRPERM` 无计数器权限失败，因此不宣称已通过计数器确认瓶颈。

寄存器候选在所测 H20 BF16 B1/H8/D32 上，内部 forward wrapper 重复速度比为 1.014–1.042×；早期 f21 公开 API 的前向速度提升较小，为 1.78–2.72%，直接 F+B 为 0.28–0.53%。这些数字不能替代最终 `61da9e0` 测量；缺少计数器也不能断言已证明 occupancy 的因果机制。

编译资源元数据显示，寄存器候选为 96 registers、14 spills、28,856 字节 shared；原无 cap、3 stages 为 111 registers、0 spills、45,408 字节 shared。候选虽然出现 spill，但实测仍更快而保留；未通过硬件计数器证明 occupancy 机制。

## 最终验证

最终源码的 [eager proof](benchmarks/evidence/forward_tuning/proof-eager.log) 与 [compiled proof](benchmarks/evidence/forward_tuning/proof-compiled.log) 均退出 0：各覆盖 4 个桶、17 次目标调用、50 次目标 PTX 编译，复用已见桶时新增 JIT key/PTX 均为 0。compiled 在 N>1 时为一张图，首次 N=1 后为两张图，重复 N=1 不再增加；graph break 记录为空。

[memcheck](benchmarks/evidence/forward_tuning/memcheck.log) 与 [initcheck](benchmarks/evidence/forward_tuning/initcheck.log) 均退出 0、零错误；各覆盖 N65 BF16 的 mixed/all-mask 两个 case，并强制 cap96。这是这两个 case 的 sanitizer 覆盖，不代表全部输入组合。

[最终强制配置正确性](benchmarks/evidence/forward_tuning/forced-correctness-final.log) 42/42 通过、退出 0：BF16/FP16，D32 default N1/65/128/129 的 mixed/all/singleton/sentinel，full N65/129 mixed/all，另各一个 D128 N65 mixed smoke；输出及四梯度对照独立 FP64。D32 强制 cap96，D128 验证未受影响的路径。[本轮 TMA 测试](benchmarks/evidence/forward_tuning/pytest-tma.log) 16 passed，包括非对齐 bias；[上一版对照](benchmarks/evidence/forward_tuning/compare-previous-final.log) 已完成并退出 0。当前源码本轮运行的是上述 42+16 项测试及 proof/sanitizer；此前 `4b832de` 的 386 eager、18 compile 属于[历史验证](forward_recovery_zh.md)，并非本轮重新运行。无 final 后缀的 `forced-correctness.log` 与 `compare-previous.log` 属于 f21，不能转移为本版通过证据。

## 公开 API：上一融合版与本版

| N | 上版 → 本版前向 (ms) | 前向速度比 | 上版 → 本版 F+B (ms) | F+B 速度比 |
|---:|---:|---:|---:|---:|
| 500 | 1.591 → 1.549 | 1.0274× | 8.079 → 8.041 | 1.0047× |
| 512 | 1.444 → 1.414 | 1.0212× | 7.010 → 6.974 | 1.0052× |
| 513 | 1.999 → 1.949 | 1.0255× | 9.671 → 9.622 | 1.0051× |
| 640 | 2.728 → 2.677 | 1.0192× | 13.327 → 13.270 | 1.0043× |
| 768 | 4.620 → 4.529 | 1.0202× | 22.534 → 22.444 | 1.0040× |
| 800 | 5.888 → 5.753 | 1.0233× | 29.546 → 29.431 | 1.0039× |
| 1024 | 10.639 → 10.463 | 1.0168× | 51.563 → 51.384 | 1.0035× |

对照固定相同融合反向，仅替换 `4b832de` 与本版前向。[最终同进程对照](benchmarks/evidence/forward_tuning/compare-previous-final.log) 使用私有调优缓存，shape 顺序 500/512/513/640/768/800/1024，由 500/513 首次触发对应桶，后续同桶复用；旧前向与新前向分别调优，新前向选择 cap96。反向未改，独立反向实测仅有近 1× 的波动。两版测量均为 3 次预热、3 轮 ABBA、每侧 60 样本。

## 公开 API：上游与本版

| N | 上游 → 本版前向 (ms) | 上游 → 本版反向 (ms) | 前向速度比 | 上游 → 本版 F+B (ms) | F+B 速度比 |
|---:|---:|---:|---:|---:|---:|
| 500 | 1.709 → 1.547 | 12.662 → 6.494 | 1.105× | 14.370 → 8.037 | 1.788× |
| 512 | 1.490 → 1.413 | 7.556 → 5.557 | 1.054× | 9.053 → 6.977 | 1.298× |
| 513 | 2.129 → 1.945 | 15.625 → 7.675 | 1.095× | 17.757 → 9.622 | 1.846× |
| 640 | 2.820 → 2.676 | 14.377 → 10.593 | 1.054× | 17.198 → 13.276 | 1.295× |
| 768 | 4.734 → 4.537 | 24.498 → 17.885 | 1.043× | 29.229 → 22.438 | 1.303× |
| 800 | 5.942 → 5.802 | 28.994 → 23.691 | 1.024× | 34.903 → 29.426 | 1.186× |
| 1024 | 10.811 → 10.490 | 56.705 → 40.910 | 1.031× | 67.422 → 51.276 | 1.315× |

测量采用 warm eager 公开 API、ABBA 采样及 CUDA event 中位数之比，包含 wrapper 分配、padding、cast 及区间内主机提交间隙。前向启用梯度，单独反向使用 retain_graph，F+B 直接测量，不把两列相加。它不是单 kernel、compile 或完整训练 step 性能。环境为 H20-3e、PyTorch 2.13.0a0+9186a08b2c.nv26.07、Triton 3.7.1，BF16 B1 H8 D32、默认 chunk128、连续 dO、mask 概率 0.2。源码 hash 与完整原始样本保存在最终日志中。


[最终上游对照](benchmarks/evidence/forward_tuning/bench-upstream.log) 退出 0：H20-3e，BF16 B1 H8 D32，chunk128，连续 dO，mask 概率 0.2；3 次预热、3 轮 ABBA、每侧 60 样本。shape 顺序 500/512/513/640/768/800/1024，新 v4 前向由 500/513 首次触发两个桶并选择 cap96；上游与反向继承 persistent 配置，见[缓存快照](benchmarks/evidence/forward_tuning/tuning-configs/)。前向速度比 1.024–1.105×，直接 F+B 1.186–1.846×。

本轮 GPU 分配实际持续 33 分 06 秒，包含编译、主机工作及等待，不等于 GPU 持续计算时间。见[源码与运行记录](benchmarks/evidence/forward_tuning/SOURCE_AND_RUN.json)及[证据校验清单](benchmarks/evidence/forward_tuning/SHA256SUMS.json)。本轮未重跑 full/chunk 显存对照；相关历史结果及“低显存相对 full fused、并非相对上游”的限制见[上版报告](forward_recovery_zh.md)。
