# 按长度桶复用融合 attention：冻结 pointer 基线

本报告记录冻结 pointer 候选 `64148ed` 的实现、验证和性能。前向退化尚待后续 TMA 工作优化；本报告是对照基线，不是最终提交结论。

## 如何分桶

沿用上游 `b4ecec4` 的长度取整规则：对正整数 N，`CLOSEST_N = 2**ceil(log2(N))`，实现为 `1 << (N - 1).bit_length()`。真实 N 是运行时参数；CLOSEST_N 是编译常量和调优键的一部分。

| 真实长度 N | 桶 CLOSEST_N |
|---|---:|
| 129–256 | 256 |
| 257–512 | 512 |
| 513–1024 | 1024 |
| 1025–2048 | 2048 |

桶不是预先固定的四个：若训练长度限定在 256–2048（含端点），会触及这四个桶；若允许 1–2048，则可能触及 1、2、4……2048，共 12 个桶。只为实际遇到的桶生成所需配置。

例如 500 与 512 都使用 512 桶；513 与 800 都使用 1024 桶。计算和输出仍按真实 N 定义。桶数也不是编译次数：每桶首次 autotune 会编译、测量多个候选配置，前向与反向另有各自 kernel，dtype、H、D、设备等配置变化也可能增加版本。目标是固定这些配置后，不因同桶内每个新 N 重复特化；需要实际 JIT key/PTX 证据确认，不能只凭 autotune key 推断。

## 当前候选的实现

前向和融合反向均将 N 显式声明为运行时 `tl.int64`，并加入 `do_not_specialize`。前向 autotune key 使用 H、DIM、CLOSEST_N、DTYPE_ID；反向使用 H、D、BJ、BK、CENTERED、CLOSEST_N。精确 N 不在这些调优键中。当前前向使用 32×32、64×32、64×64 三种 tile，各配 stages=1/3，共六个候选；FP32 且 D≥64 时剪枝为 32×32、stage=1。反向使用 stages=1/2/3 三个候选，同样在 FP32 且 D≥64 时仅保留 stage=1。已移除未带来收益的前向寄存器上限候选。这些配置选择不改变长度桶规则。

为恢复 bias 行访问的对齐信息，仅将 bias 的物理末维补齐到 CLOSEST_N，形成 `[B,H,N,CLOSEST_N]`，并在反向使用相同物理行宽的 FP32 dBias 累加区。返回时截取真实 N 列并转为连续输出。Q、K、V 以及逻辑输出仍使用真实 N，不把整个 attention 扩展到桶大小。补齐列不是有效 key；尾块概率必须为零，不能参与 softmax 分母。bias 补齐、dBias 截取和 dO 连续化的成本属于完整调用成本。

前向先遍历完整 key tile，再单独处理有 mask 的尾块。融合反向的一个 CTA 拥有 `(batch, head, i, key tile)`，遍历所有 query tile：完整 query 块走无 query 尾部 mask 的循环，最后不足一块时单独处理。运行时分支让完整 key tile 的 CTA 走快路径，仅尾部 key tile 保留必要 mask；这些分支存在于桶对应的 kernel 内，不按精确 N 生成新版本。

反向共享五个矩阵乘法的中间结果：QKᵀ 重建概率、dO Vᵀ、dScore K、dScoreᵀ Q、Pᵀ dO。dK/dV 由所属 CTA 在 FP32 累加后唯一写出；dQ 跨 key tile、dBias 跨 i 通过 FP32 atomic 归约。概率矩阵不完整落地。delta 和 mask 元数据仍由独立预处理计算。

默认 `chunk_i=128`，按 i 切片执行融合反向。FP32 dQ scratch 大小为 `4*B*H*min(128,N)*N*D` 字节；另外还有 FP32 dBias 的 `4*B*H*N*CLOSEST_N` 字节、最终梯度、统计量等，不能把 scratch 大小当成总峰值。显式 full-workspace 模式仍可用。各 chunk 先清零 scratch、执行主 kernel，再 flush 到最终 dQ。

公开 autograd 与不透明 `torch.library.custom_op` 分派边界保留，避免 `torch.compile` 直接按 N 展开 Python chunk 循环。kernel 桶内复用和 Dynamo 计算图复用是不同指标；单元素维度或其他 guard 仍可能触发新的图。

有限 mask 分数替换值 −10000、全 mask 行、singleton 稳定梯度、FP32 centered 统计语义不变。仍只支持一阶梯度，atomic 归约不保证确定性。

## 与历史版本的关系

| 版本 | 长度策略 | 性能证据适用范围 |
|---|---|---|
| 原融合 PR `249f3e2` | 精确 N 特化 | 原报告中的速度仅代表该冻结版本 |
| 严格动态实验 `db7285d` | 移除长度桶，固定配置跨 N 复用 | 见[动态实验报告](dynamic_n_zh.md)，未达到性能目标 |
| 本候选 | 真实 N 为运行时参数，按向上 2 幂桶调优 | 见下文冻结 pointer 基线结果 |

## 已完成的编译复用验证

冻结源码清单见 [frozen_sources.json](benchmarks/evidence/bucketed_n/frozen_sources.json)。以下六次命令均退出 0；环境为 NVIDIA H20-3e、PyTorch 2.13.0a0+9186a08b2c.nv26.07、Triton 3.7.1，B=1、H=2。

| 验证日志 | 配置及长度覆盖 | 目标 kernel PTX 编译总数 | 已见桶内新增编译 |
|---|---|---:|---:|
| [eager](benchmarks/evidence/bucketed_n/proof-eager.log) | BF16 D32，15 步，桶 512/1024/256/1，chunk128 | 38 | 0 |
| [compiled](benchmarks/evidence/bucketed_n/proof-compiled.log) | 同上，dynamic/fullgraph compile | 38 | 0 |
| [full workspace](benchmarks/evidence/bucketed_n/proof-full.log) | BF16 D32，同一 15 步序列 | 37 | 0 |
| [small reference](benchmarks/evidence/bucketed_n/proof-small-reference.log) | BF16 D32，11 步，桶 128/256/1，全部 N≤256 做独立参考 | 29 | 0 |
| [FP16 D128](benchmarks/evidence/bucketed_n/proof-fp16-d128.log) | 7 步，N=65/80/127/128/65/1/1，转置 dO | 20 | 0 |
| [FP32 D128](benchmarks/evidence/bucketed_n/proof-fp32-d128.log) | 同一 7 步序列，连续 dO | 6 | 0 |

主 15 步序列为 `257,300,500,512,257,513,800,1024,513,129,200,256,129,1,1`。默认路径首桶编译 11 个目标版本（6 个前向候选、3 个反向候选、预处理、flush），后续每个新桶新增 9 个候选版本；复用已见桶时新增 JIT key/PTX 均为 0。full workspace 首桶为 10 个，没有 flush。FP32 D128 剪枝后首桶 4 个、下个桶 2 个。进程级 PTX 总数还包含参考实现，本表只计目标 kernel。

六次运行的 `numerical_failures` 均为 0，并各包含两次额外 singleton-valid-key 检查；主序列独立参考仅覆盖 N≤129，不能把大 N 的编译复用检查当成完整独立数值验证。small-reference 和 D128 两次运行覆盖其全部主序列的独立参考。compiled 的 Dynamo `unique_graphs` 在 N>1 时为 1，首次 N=1 后为 2，重复 N=1 不再增加。

## 未完成运行与重试

2026-09-22 的首次冷配置完整 eager 矩阵在 1800 秒命令超时后退出 **124**（`experiments/bucketed-n-20260922/pytest-eager.log` 及 `.command.json`）。日志约完成 100 个测试点、未显示数值失败，但没有完成汇总，**不能记作整套通过**。该矩阵覆盖 12 个 dtype/D 组合及多个长度桶；30 分钟是整套矩阵的超时上限，不能解释为单个训练配置启动耗时。

随后 GPU 作业 73115 达到一小时分配上限，延期尝试时已经过期。随后新的两小时 GPU 会话串行完成 `pytest-eager-warm`，复用工作区保存的 autotune 配置，并重建新容器 `/tmp` Triton 编译缓存。该重试、compile 矩阵和 `bench-final` 的完成结果见下文；首次超时记录单独保留。

## 测量与验证口径

最终 benchmark 使用 eager 公开 API，在预热后按 ABBA 顺序采样，以两侧样本中位数之比报告加速比。前后向组合延迟直接测量，不能用前向与反向两列相加替代；单独反向使用 `retain_graph`，前向测量启用梯度。因此它描述这些 API 调用的稳态成本，不是单 kernel 延迟、完整训练 step 或 `torch.compile` 性能。

完整调用计入 bias padding、cast、dQ flush 等操作以及 CUDA event 测量区间内的主机提交间隙。首次遇到的 N 及现有缓存会影响同桶选择哪一个调优配置；最终表必须列明实际 shape 顺序、预热与缓存条件，不能只列一组无顺序的 N。

显存指标为预分配输入之外的 peak allocated increment，包含输出、梯度及临时量；不是进程总显存，也不包含已经分配的输入。必须明确比较双方使用相同的测量边界。

大 N 的 FP64 独立参考按 i 分片覆盖完整张量，而非抽样行；其覆盖范围仍仅限实际命令中的 B=1、H=8、指定 dtype/D 和单个随机 seed。参考脚本的 `candidate_seconds` 是验证流程耗时记录，不能作为性能 benchmark。warm eager 重试已完成，具体通过数量和耗时见下文。

## 冻结 pointer 候选的性能与完整门禁

本节只描述 **`64148ed4a1825f98d3175ba0267a11101fcc965e`** 的 pointer 前向候选，是后续 TMA 前向修复前的对照。用户要求继续定位并优化前向退化，暂不发布 PR；以下不是后续实现的性能，也不是最终提交结论。

[bench-final 原始记录](benchmarks/evidence/bucketed_n/bench-final.log)：与上游 `b4ecec4` 对比，H20-3e，BF16，B=1、H=8、D=32，默认 chunk128，连续 dO，mask 概率 0.2。实际 shape 顺序为 **500,512,513,640,768,800,1024**；每个模式预热 3 次、3 轮 ABBA，每侧 60 个样本。复用工作区已保存调优配置，不能视为每个 N 都重新冷调优。精确缓存状态和配置选择应结合原日志及调优记录解读。

| N | 前向 上游 → 候选 (ms) | 反向 上游 → 候选 (ms) | 前后向 上游 → 候选 (ms) | 前后向加速比 |
|---:|---:|---:|---:|---:|
| 500 | 1.710 → 2.032 | 12.658 → 6.496 | 14.378 → 8.514 | 1.689× |
| 512 | 1.493 → 1.987 | 7.557 → 5.557 | 9.050 → 7.543 | 1.200× |
| 513 | 2.130 → 2.410 | 15.608 → 7.677 | 17.744 → 10.088 | 1.759× |
| 640 | 2.820 → 3.821 | 14.376 → 10.595 | 17.199 → 14.432 | 1.192× |
| 768 | 4.734 → 6.491 | 24.500 → 17.870 | 29.233 → 24.395 | 1.198× |
| 800 | 5.947 → 7.587 | 29.256 → 23.859 | 35.200 → 31.468 | 1.119× |
| 1024 | 10.715 → 14.796 | 56.645 → 40.989 | 67.427 → 55.818 | 1.208× |

全部七个长度的前向均有退化：上游/候选速度比为 0.724–0.884×；反向为 1.226–2.033×，直接测量的前后向总计为 1.119–1.759×。因此反向收益不能掩盖前向问题，后续优化以该记录作为对照。旧精确 N 特化版本约 1.5× 的数字不替代此表。

同一调用的 peak allocated increment（MiB=2²⁰ 字节）：

| N | 上游 (MiB) | 候选 (MiB) | 增加 (MiB) |
|---:|---:|---:|---:|
| 500 | 539.96 | 591.56 | 51.60 |
| 512 | 564.25 | 612.52 | 48.27 |
| 513 | 566.46 | 630.80 | 64.34 |
| 640 | 881.64 | 954.39 | 72.75 |
| 768 | 1269.56 | 1347.77 | 78.21 |
| 800 | 1380.84 | 1459.54 | 78.70 |
| 1024 | 2257.00 | 2321.03 | 64.03 |

“低显存”指相对融合 full-workspace 路径降低 dQ scratch，并不表示比上游省显存。本表候选在这些形状下比上游多 48.27–78.70 MiB；full-workspace 对照另行补充。

| 已完成检查 | 结果 |
|---|---|
| [完整 eager 重试](benchmarks/evidence/bucketed_n/pytest-eager-warm.log) | 374 passed，18 deselected；pytest 2024.81 秒，命令含启动总计 2026.39 秒；退出 0 |
| [compile 矩阵](benchmarks/evidence/bucketed_n/pytest-compile.log) | 18 passed，331 deselected；退出 0 |
| [大 N FP64 独立参考](benchmarks/evidence/bucketed_n/large-reference.log) | N=500/512/513/800/1024，B1 H8 BF16 D32，逐 i 覆盖完整输出及四梯度；退出 0 |
| [最终 pointer benchmark](benchmarks/evidence/bucketed_n/bench-final.log) | 七个长度，所有测量完成；退出 0 |
| 前述六项 bucket proof | 全部退出 0，已见桶内新增编译 0 |

大 N 每个长度使用一个 seed（`20260921+n`），不是多 seed 稳健性证明。未在这里声明新实现的 sanitizer 门禁通过；历史版本检查不能自动转移至本候选。首次冷矩阵超时记录继续保留。后续 TMA 代码必须另行冻结与验证，不能将这些通过结果套用过去。
