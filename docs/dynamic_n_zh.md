# 运行时序列长度：实验分支

> 历史冻结实验 `db7285d`：本文结果仅适用于严格跨长度复用版本，当前按桶实现见 [分桶报告](bucketed_n_zh.md)。

**当前实验分支尚未达到性能目标。原PR #3保留`249f`；原形状特化性能数字不代表此原型。**

融合路径将序列长度 `N` 作为运行时参数。在设备、dtype、head 数、通道维度及 launch 配置固定时，改变 N 复用注意力 kernel，不再按长度生成新的 Triton 特化版本。这是 kernel 复用约束，不代表所有 PyTorch 计算图都只编译一次。

公开输入仍为 `[B,H,N,N,D]`，维度为正，D 支持 {16,32,64,128}，浮点输入类型一致，mask 为布尔类型。有限分数替换值仍为 −10000；FP32 前向统计量继续使用反向匹配的 centered 坐标。原子梯度归约仍具有非确定性，仅支持一阶梯度。改变设备、dtype、D/H 或 launch 配置可能需要其他二进制；下述复用证据不代表覆盖所有配置。

“跨长度”指显存和 launch grid 能容纳的合法形状，不是无限大长度。当前前向／预处理将 N 映射到 grid.y，B×H 映射到 grid.z，两者在 CUDA 上均受 65,535 上限约束；下述测试未覆盖接近该上限的情况。

## 实现变化

`_fused_forward.py` 移除逐N autotune与取整N编译参数，冻结前向使用单个完整key块循环，再由`_fwd_kv_block(K_MASKED=...)`处理独立masked尾块。BF16/FP16固定64×32、4warps、3stages；FP32固定32×32、4warps、1stage；保留必要query mask。N采用显式`tl.int64`标量ABI并列入`do_not_specialize`，避免值为1或对齐引起特化及跨长度标量类型变化。

反向预处理以固定宽mask块作运行时扫描，不再构造`arange(next_power_of_2(N))`；chunk起点、容量、flush长度使用运行时参数。当前原型先对dO执行 **`reshape(B*H,N,N,D).contiguous()`**，随后复用Q的连续地址映射；不是把任意dO stride直接传入attention kernel。非连续及宽stride输入通过这次物化继续可用，其复制时间和显存必须计入调用成本。连续全局基址保留宽算术，singleton稳定梯度及N=1语义不变。主反向在同一binary内按运行时N选择完整tile对齐、N整除8和generic尾块三条路径，不是三种逐N特化launch。已删除无用`DO_TRANSPOSED`参数。

### 与上游的准确比较

上游`b4ecec4`的`_fwd`已经将N声明为运行时参数，而非`tl.constexpr`。但wrapper同时计算`CLOSEST_N=2**ceil(log2(N))`，后者是constexpr且属于autotune key（H、DIM、CLOSEST_N）。因此不能说“上游每一个N都必然重新编译”：它按长度桶选择配置，同桶内可能复用，跨桶可能产生不同特化；其他参数guard和对齐特化也可能影响复用。

本原型去掉这一bucket依赖，目标是在固定配置下跨bucket复用同一kernel。它不是首次把N改成运行时参数，也不凭此宣称速度超过上游或原形状特化融合PR。

反向主 CTA 拥有一个 `(batch, head, i, key tile)`，沿 query tile 循环。五个矩阵乘法分别为：重建概率的 QKᵀ、dO Vᵀ、dScore K、dScoreᵀ Q、Pᵀ dO。各梯度共享 P 与 dScore。dK/dV 由唯一 CTA 写入；dQ 跨 key tile 用 FP32 atomic 归约，dBias 跨 i 用 FP32 atomic 归约。不落地立方规模概率张量。delta 与 mask 元数据仍由独立预处理计算，因此整个反向不是单 launch。

默认低显存路径使用 `chunk_i=128`。FP32 dQ scratch 含 `B × H × min(128,N) × N × D` 个元素，即 **4 × B × H × min(128,N) × N × D 字节**。这只是 dQ scratch，不是总峰值显存；最终 dQ、dK/dV、FP32 dBias、归一化统计及元数据另占空间。各 chunk 顺序执行清零、融合主 kernel 和 dQ flush；同一切片全部 key tile 的贡献先在 FP32 累加，再转换至最终 dtype。

前向 launch 选择与 Python 反向 chunk 循环被封装在不透明的 `torch.library.custom_op` 分派边界内，fake 实现描述输出形状。因此 `torch.compile` 不会按 chunk 数将这个 Python 循环展开为不同图。公开 autograd 函数提供反向并保留一阶梯度合同。该边界不会消除 Dynamo 自身的形状 guard 或单元素维度特化。

## 冻结源码验证

以下相对链接指向冻结证据快照，包含源码hash。旧`eager-fourth`/`compiled-first`仅为历史原型记录，不作为最终源码证据。

| 门禁 | 实际完成结果 |
|---|---|
| [eager](benchmarks/evidence/dynamic_n/pytest-frozen-eager.log) | 374 passed，18编译测试未选 |
| [compile](benchmarks/evidence/dynamic_n/pytest-frozen-compile.log) | 18 passed，331未选；每例两次调用，独立参数变体间隔离 |
| [大形状FP64](benchmarks/evidence/dynamic_n/large-frozen.log) | BF16 B1H8D32，N512/800/1024三例独立FP64全部通过 |
| [eager复用](benchmarks/evidence/dynamic_n/proof-frozen-eager.log) / [compiled复用](benchmarks/evidence/dynamic_n/proof-frozen-compiled.log) | 各12次target调用，4个target PTX事件，首次后新增0，数值失败0 |
| [full工作区eager](benchmarks/evidence/dynamic_n/proof-full.log) / [compiled](benchmarks/evidence/dynamic_n/proof-full-compiled.log) | 各12次target调用，各3个target PTX事件，首次后新增0；N=1可产生第二张Dynamo图 |
| [FP16](benchmarks/evidence/dynamic_n/proof-fp16.log) / [FP32](benchmarks/evidence/dynamic_n/proof-fp32.log) / [D128](benchmarks/evidence/dynamic_n/proof-d128.log) | 各8次target调用，各4个target PTX事件，首次后新增0，数值失败0 |
| [memcheck](benchmarks/evidence/dynamic_n/memcheck-frozen.log) / [initcheck](benchmarks/evidence/dynamic_n/initcheck-frozen.log) | 各5项通过，sanitizer错误均为0 |

BF16低显存复用序列为`65,64,129,128,17,1,500,512,513,800,65`，另加有效key singleton检查；该序列中的独立FP64仅覆盖N≤129，大形状精度由上述独立专项证明。进程总PTX计数还含其他kernel，不可替代target计数。compiled复用运行在N=1产生第二张Dynamo图，目标binary仍不变：**kernel复用不是单图保证**。compile测试不同参数变体间reset Dynamo，每例内部两次调用仍复用同一compiled callable。

## 冻结性能：相对原上游存在回退

[原始benchmark](benchmarks/evidence/dynamic_n/bench-frozen.log)已完成9个shape/mode summary且退出码0。H20-3e，BF16 B1H8D32，连续dO，默认`chunk_i=128`，元素mask概率.2；三轮预热ABBA，每侧每shape/mode 60样本，CUDA events，**不使用CUDA graph**。初始化、复制和autograd成本计入。下表完整前向+反向，A为原上游`b4ecec4`，B为runtime-N低显存候选。

| N | 原上游A ms | 动态B ms | A/B比值 | 延迟增幅B/A−1 |
|---|---:|---:|---:|---:|
| 512 | 9.052 | 10.136 | 0.893× | +12.0% |
| 800 | 35.113 | 67.505 | 0.520× | +92.3% |
| 1024 | 67.910 | 76.276 | 0.890× | +12.3% |

比值小于1表示更慢，不是提速，也不是原PR #3的测量。binary复用减少重复特化，不能据此推断训练step加速或已经达到旧PR性能。

后续可测方向包括TMA搬运及规范化padding stride恢复向量化；这些尚未实现或验证，不承诺未来收益。

```bash
PYTHONPATH=src python scripts/check_dynamic_n.py --shapes 65,64,129,128,17,1,500,512,513,800,65
PYTHONPATH=src python scripts/check_dynamic_n.py --shapes 65,64,129,128,17,1,500,512,513,800,65 --torch-compile
```
