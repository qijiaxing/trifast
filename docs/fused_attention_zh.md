# 可选融合 Triangle Attention

> 历史记录：本文描述 PR #3 的 `249f3e2` 形状特化版本，不是当前实验分支的运行时 N 版本。当前实现及重新测量见 [分桶报告](bucketed_n_zh.md)。

本改动基于 Jiaxing TriFast 的 `b4ecec4`，该版本已经包含 TMA 前向优化。原 `triangle_attention` 及其调参策略不变；新增融合入口默认使用 `chunk_i=128` 的低显存路径。最新基线的性能数据、逐轮采样和验证摘要见[性能报告](benchmarks/h20_fused_zh.md)。此前针对 `33b5c00` 的数字属于历史基线，不能用作相对最新 master 的加速。

## 使用

```python
from trifast import triangle_attention_fused

# q/k/v: [B, H, N, N, D]
# bias:  [B, H, N, N]
# mask:  [B, N, N]，bool；True 表示替换该 score
out = triangle_attention_fused(q, k, v, bias, mask)
out.backward(grad_output)
```

浮点输入必须同 dtype、同 CUDA 设备；支持 BF16、FP16、FP32，D=16/32/64/128，正 B/H/N。允许非连续输入和上游梯度，必要的布局整理可能分配副本。

`triangle_attention_fused(..., chunk_i=128)` 默认采用低显存分块。需要完整工作区模式时显式传 `chunk_i=None`：

```python
out = triangle_attention_fused(q, k, v, bias, mask, chunk_i=None)
```

这是固定默认配置，不根据空闲显存自动切换。原有显式低显存 helper 保留兼容：

```python
from trifast.fused_low_memory_api import triangle_attention_fused_low_memory

out = triangle_attention_fused_low_memory(
    q, k, v, bias, mask, chunk_i=128,
)
out.backward(grad_output)
```

主融合入口的 `chunk_i` 接受正整数或 `None`，不接受 bool；兼容 helper 只接受正整数，不接受 `None`。超过 N 时按一个 chunk 处理。不会因为显存不足静默改算法。

## 改了什么

原实现分别计算 dQ、dK/dV、dBias，多次重算概率及其导数。本实现以 key tile 为工作单元，共享重算的 P 和 dScore，把反向矩阵点积总数从 9 个减为 5 个。dK/dV 使用 FP32 累加并由唯一程序写回；dQ/dBias 使用 FP32 原子加法，归约完成后再转换为输入类型。不创建 N³ 的概率中间张量。

完整调用仍包括前处理、工作区初始化、主反向和类型转换等多个 kernel；“融合”不代表整个调用只有一次 launch。独立前向保留在线 softmax 算法，使用自己的四组 pointer 配置，不借用新上游带 TMA descriptor hook 的配置。实际 N 进入调参 key，避免 N768/N800 等不同尾块形状误用同一桶的配置。D128 采用较小的 query tile，FP32 大 D 另有限制 shared-memory 使用的配置。

低显存实现沿独立的 i 维度分块复用 dQ 的 FP32 工作区。完整工作区模式（`chunk_i=None`）的 dQ 工作区是 `4*B*H*N*N*D` 字节，低显存工作区是 `4*B*H*min(chunk_i,N)*N*D` 字节。它增加清零、主计算和写回的启动次数；dBias 始终跨 chunk 在 FP32 中累加，没有降低精度来节省内存。具体时间代价应以目标形状的完整调用复测为准。

## 数值语义与限制

mask=True 将 score 替换为有限的 −10000，而不是加上偏置或填 −∞。全 mask 行的输出为 V 的均值，dQ/dK/dBias 为零，dV 仍非零。尾块的越界 padding 单独排除，不算作真正的 masked key。只有一个有效 key 时使用稳定的差值表达式，避免两个独立舍入的内积相减产生伪梯度；当 masked key 的有限 sentinel 概率非零时，保留其应有贡献。

FP32 前向先减去公共 bias 偏移，在匹配的中心化域中保存和重建统计量，以保留 sentinel 附近的小 score 差异。内部统计量必须与其配套反向使用，不能只看 shape 一样就在不同前向之间互换。

仅支持一阶梯度；原子归约不保证逐位确定性，forward 和 backward 都拒绝 deterministic 模式，包括 warn-only。未承诺二阶导、JVP/vmap、任意非有限输入或无限形状。已验证硬件为 H20，软件为 PyTorch 2.13 开发版和 Triton 3.7.1；不能据此声称所有 CUDA GPU 或依赖声明的最低版本均已验证。

编译正确不等于编译更快：所测 torch.compile 反向可能在进入内核前把非连续输出梯度复制成连续布局。完整耗时须包含这笔开销。不同 N、stride、chunk 大小也可能触发重新编译。

## 验证与复现

独立 FP64 参考使用有限 mask 的 matmul/softmax/autograd，不以原 TriFast 误差放宽门槛。固定相对 L2 门槛：BF16 0.012、FP16 0.002、FP32 2e-5；参考为零的元素另要求绝对误差不超过 2e-5，所有值必须有限。

```bash
python -m pytest -q tests/unit/test_fused.py tests/unit/test_fused_contracts.py -m 'not compile'
python -m pytest -q tests/unit/test_fused.py -m compile

compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest -q tests/unit/test_fused.py -m sanitizer
compute-sanitizer --tool initcheck --error-exitcode 99 \
  python -m pytest -q tests/unit/test_fused.py -m sanitizer

python scripts/validate_fused.py --shapes 512,800,1024 --api fused
python scripts/validate_fused.py --shapes 512,800,1024 --api low-memory --chunk-i 128

python scripts/bench_fused.py --baseline original --candidate low-memory
python scripts/bench_fused.py --baseline full-workspace --candidate low-memory --chunk-i 128
```

基准默认 `--baseline original --candidate low-memory`；baseline 可选 original/full-workspace，candidate 可选 full-workspace/low-memory。独立验证命令 `python scripts/validate_fused.py --api low-memory` 默认也选择 low-memory，另可指定 original/full-workspace。

基准包含前向、反向和前后向合计；先完成编译/调参/warmup，再做三轮 ABBA，每轮每块 10 个 CUDA-event 样本。反向复用已建图，合计每次重建图；初始化、转换、复制及主机提交空档均在计时口径内。峰值另测，排除预分配输入，包含保留的输出、梯度和临时工作区；不是进程总显存或 reserved 显存。JSONL 保存每个样本、输入配置及源码 SHA256。

这些都是单卡算子结果，不等于整模型训练 step 收益，也没有完成 NCCL/多卡通信与计算融合。全 mask 快捷分支、保存 mask count 等进一步试验保留在开发记录中，不属于这个 PR 的实现和性能结论。
