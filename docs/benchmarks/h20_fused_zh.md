# H20 融合注意力性能记录

> 历史记录：本文描述 PR #3 的 `249f3e2` 形状特化版本，不是当前实验分支的运行时 N 版本。当前实现及重新测量见 [分桶报告](../bucketed_n_zh.md)。

本报告仅对比最新 master **b4ecec4c8ac599bf5aa16ae5aeb5ebdc02b7addb** 与本次融合实现，主配置为 `chunk_i=128` 的低显存反向。原 `triangle_attention` 接口仍保留；表格不代表替换原接口的默认分派。

N512～1024 的低显存反向加速 **1.528～1.640×**，完整前后向加速 **1.385～1.494×**。前向在对齐形状提升 **1.022～1.050×**；**N800 前向只有 0.943～0.944×，约慢 6%**。完整前后向数字已包含这一回退。

## 环境与测量边界

GPU 为 NVIDIA H20-3e，计算能力 9.0；PyTorch `2.13.0a0+9186a08b2c.nv26.07`，Triton `3.7.1`。B=1、H=8、D=32、BF16。固定随机种子，mask 的替换概率为 20%，有限替换值为 −10000。这是合成稠密注意力负载，不是全 mask 快捷路径测试。dO 分别为连续布局，以及两个空间轴转置后的非连续布局。

每对实现在同一进程执行三轮 ABBA，每块十次 CUDA event 采样，每个实现先执行三次不计时预热，因此每种模式、每个标签共 60 个样本。耗时取中位数，加速比为基线中位数除以候选中位数。前向包含完整 wrapper 与分配；反向复用已构建的图（`retain_graph=True`）；完整前后向每次重建图。梯度清空、autograd、预处理 kernel、workspace 清零、复制与转换均计入。不使用 CUDA Graph。CUDA event 计时包含设备可见的提交间隙，但不是应用整体墙钟延迟。编译和 autotune 不计入预热后的结果。独立测量的前向、反向中位数不应直接相加替代完整前后向结果。

## 主结果：低显存版本对比 master

单元格为 **master → 候选毫秒数（加速比）**；contiguous 为连续 dO，transposed 为空间转置 dO。

| dO | N | Forward ms (×) | Backward ms (×) | F+B ms (×) |
|---|---:|---:|---:|---:|
| contiguous | 512 | 1.494 → 1.424 (1.049×) | 7.570 → 4.801 (1.577×) | 9.067 → 6.219 (1.458×) |
| contiguous | 640 | 2.791 → 2.721 (1.025×) | 14.386 → 9.129 (1.576×) | 17.165 → 11.853 (1.448×) |
| contiguous | 768 | 4.691 → 4.591 (1.022×) | 24.293 → 15.197 (1.599×) | 28.964 → 19.744 (1.467×) |
| contiguous | 800 | 5.813 → 6.162 (0.943×) | 29.040 → 19.004 (1.528×) | 34.854 → 25.170 (1.385×) |
| contiguous | 1024 | 10.769 → 10.498 (1.026×) | 56.705 → 34.575 (1.640×) | 67.395 → 45.102 (1.494×) |
| transposed | 512 | 1.496 → 1.424 (1.050×) | 7.610 → 4.899 (1.553×) | 9.108 → 6.319 (1.441×) |
| transposed | 640 | 2.792 → 2.722 (1.026×) | 14.456 → 9.190 (1.573×) | 17.241 → 11.820 (1.459×) |
| transposed | 768 | 4.658 → 4.558 (1.022×) | 24.444 → 15.356 (1.592×) | 29.100 → 19.925 (1.460×) |
| transposed | 800 | 5.815 → 6.161 (0.944×) | 29.214 → 19.077 (1.531×) | 35.026 → 25.242 (1.388×) |
| transposed | 1024 | 10.771 → 10.499 (1.026×) | 56.967 → 35.207 (1.618×) | 67.655 → 45.713 (1.480×) |

## 更大形状：低显存版本对比 master

dO 为连续布局，仍为 BF16、B=1/H=8/D=32、chunk128。本次仅测完整前后向，不据此宣称这些形状的单独前向、反向或转置布局收益。

| N | Master F+B ms | Low-memory F+B ms | Speedup | Master peak MiB | Low-memory peak MiB |
|---:|---:|---:|---:|---:|---:|
| 1536 | 229.118 | 151.339 | 1.514× | 5078.25 | 5125.55 |
| 2048 | 532.966 | 348.557 | 1.529× | 9028.00 | 9026.06 |

## 辅助结果：完整 workspace 对比 master

测量边界相同，但采用单独的成对实验。

| dO | N | Forward ms (×) | Backward ms (×) | F+B ms (×) |
|---|---:|---:|---:|---:|
| contiguous | 512 | 1.505 → 1.424 (1.057×) | 7.557 → 4.754 (1.589×) | 9.054 → 6.179 (1.465×) |
| contiguous | 640 | 2.792 → 2.719 (1.027×) | 14.366 → 8.953 (1.605×) | 17.165 → 11.683 (1.469×) |
| contiguous | 768 | 4.704 → 4.592 (1.024×) | 24.505 → 15.113 (1.621×) | 29.190 → 19.708 (1.481×) |
| contiguous | 800 | 5.870 → 6.208 (0.946×) | 29.266 → 18.956 (1.544×) | 35.123 → 25.163 (1.396×) |
| contiguous | 1024 | 10.770 → 10.578 (1.018×) | 56.709 → 34.558 (1.641×) | 67.387 → 45.057 (1.496×) |
| transposed | 512 | 1.493 → 1.423 (1.049×) | 7.614 → 4.806 (1.584×) | 9.110 → 6.234 (1.461×) |
| transposed | 640 | 2.792 → 2.722 (1.026×) | 14.343 → 8.952 (1.602×) | 17.109 → 11.651 (1.468×) |
| transposed | 768 | 4.658 → 4.567 (1.020×) | 24.443 → 15.085 (1.620×) | 29.097 → 19.630 (1.482×) |
| transposed | 800 | 5.814 → 6.172 (0.942×) | 29.206 → 18.869 (1.548×) | 35.025 → 25.042 (1.399×) |
| transposed | 1024 | 10.769 → 10.511 (1.025×) | 56.973 → 34.785 (1.638×) | 67.658 → 45.286 (1.494×) |

## 低显存的配对代价

下表直接比较完整 workspace 与低显存版本，不跨不同实验会话相减。time increase 表示低显存版本的耗时增加比例。

| dO | N | Full → low-memory F+B ms | Low-memory time increase | Peak full → low-memory MiB | Peak reduction |
|---|---:|---:|---:|---:|---:|
| contiguous | 512 | 6.181 → 6.221 | 0.64% | 804.52 → 612.52 | 23.87% |
| contiguous | 800 | 24.907 → 25.151 | 0.98% | 1967.04 → 1441.04 | 26.74% |
| contiguous | 1024 | 44.988 → 45.115 | 0.28% | 3217.03 → 2321.03 | 27.85% |
| transposed | 512 | 6.227 → 6.316 | 1.43% | 804.52 → 612.52 | 23.87% |
| transposed | 800 | 25.163 → 25.375 | 0.84% | 1967.04 → 1441.04 | 26.74% |
| transposed | 1024 | 45.315 → 45.740 | 0.94% | 3217.03 → 2321.03 | 27.85% |

峰值为 PyTorch allocated bytes 相对预分配输入的增量，包含保留的输出与四个梯度，不含 allocator reserved 内存，也不是设备或进程总显存。MiB=2²⁰ 字节。两种候选保留的输出字节数相同。相对完整 workspace 节省显存，**不等于比 master 更省显存**：

| N | Master peak MiB | Low-memory peak MiB |
|---:|---:|---:|
| 512 | 564.25 | 612.52 |
| 640 | 881.64 | 936.89 |
| 768 | 1269.56 | 1329.77 |
| 800 | 1380.84 | 1441.04 |
| 1024 | 2257.00 | 2321.03 |

## 证据、复现与适用范围

最终验证全部通过：374项eager/接口检查；warm与独立调参缓存两轮各18项compile配置（每项复用调用2次）；memcheck/initcheck各5项、0 errors；7项上游精选回归；13项独立FP64目标尺寸检查。上游精选并非完整上游测试集；compile参数配置之间重置Dynamo，不代表单个持续缓存支持无限形状。

可审查证据：[性能汇总](evidence/benchmark_summary.json)、[全部计时样本](evidence/benchmark_samples.jsonl.gz)、[验证汇总](evidence/validation_summary.json)、[源码清单](evidence/source_manifest.json)及[证据说明](evidence/README.md)。

这些证据中的运行标识为 `v4-benchmark-master-{low-memory,full-workspace}-{contiguous,transposed}.log` 、`v4-benchmark-lowmemory-{contiguous,transposed}.log` 和 `v4-benchmark-master-large.log`。日志包含环境、相对仓库路径的源码 SHA256、全部采样、精度诊断及完成记录。master revision 表示基础 checkout，候选源码由对应哈希标识。与 master 的数值一致性诊断不是独立正确性证明；独立 FP64 验证及测试证据应单独审查。

在仓库根目录、相同 CUDA/PyTorch/Triton 环境运行：

```bash
PYTHONPATH=src python scripts/bench_fused.py --baseline original --candidate low-memory --chunk-i 128 --shapes 512,640,768,800,1024 --modes forward,backward,forward_backward --rounds 3 --samples 10 --warmup 3
# Repeat with --noncontiguous-do for the transposed gradient layout.
PYTHONPATH=src python scripts/bench_fused.py --baseline original --candidate low-memory --chunk-i 128 --shapes 1536,2048 --modes forward_backward --rounds 3 --samples 10 --warmup 3
PYTHONPATH=src python scripts/bench_fused.py --baseline original --candidate full-workspace --shapes 512,640,768,800,1024 --modes forward,backward,forward_backward --rounds 3 --samples 10 --warmup 3
PYTHONPATH=src python scripts/bench_fused.py --baseline full-workspace --candidate low-memory --chunk-i 128 --shapes 512,800,1024 --modes forward_backward --rounds 3 --samples 10 --warmup 3
PYTHONPATH=src python scripts/validate_fused.py --api low-memory --chunk-i 128 --shapes 512,800,1024 --dim 32 --dtype bf16 --reference-chunk 8
```

这些结果仅覆盖上述 GPU、dtype、维度与预热负载，不能外推其他 GPU、D、完整模型或分布式训练。FP32 原子归约具有非确定性，API 仅支持一阶梯度。本工作共享 GPU 内部计算并减少中间访存，没有 NCCL 或跨 GPU 通信性能结论，也没有证明达到硬件理论极限。
