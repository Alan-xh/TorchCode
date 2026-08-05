'''
在 Triton 中实现 GQA，核心思路是利用 KV 头数少于 Q 头数这一特点，避免在 HBM 中复制 KV 缓存，而是在内核中让多个 Q 头共享同一个 KV 头进行计算，从而显著减少内存访问。

GQA 的典型配置是 num_heads (Q头数) 是 kv_heads (KV头数) 的整数倍，这个倍数称为 gqa_group_size。因此，可以设计一个内核，让每个 KV 头负责计算 gqa_group_size 个 Q 头的注意力输出。

1. 内核中通过 pid 获取当前负责的 KV 头索引 kv_head_idx，进而计算出它需要服务的 Q 头范围。
@triton.jit
def gqa_decode_kernel(...):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    # 计算当前 KV 头负责的 Q 头范围
    head_idx_q_start = kv_head_idx * gqa_group_size
    # 加载这一组 Q 头的数据
    offs_h = tl.arange(0, BLOCK_H)  # BLOCK_H 通常等于 gqa_group_size 或其约数
    q = tl.load(q_ptr + ... + (head_idx_q_start + offs_h)[:, None] * stride_qh + ...)

2. 对于长序列的解码阶段，为了充分利用 GPU 并行能力，会将 KV 序列分成多个块 (num_splits)，每个 split 独立计算部分注意力，最后再合并结果。这个 split_kv 内核会输出部分注意力输出 (o_partial) 和对应的 Log-Sum-Exp (lse_partial)。
# 在循环中处理分配给该 split 的 KV 块
for block_idx in range(loop_range):
    start_n = (start + block_idx) * BLOCK_N
    # 加载 K, V 块
    k = tl.load(k_cache_ptr + start_n * stride_k_s + ...)
    v = tl.load(v_cache_ptr + start_n * stride_v_s + ...)
    # 计算 QK^T, 应用 mask, 执行 online softmax 更新
    # ...

3. 需要一个额外的合并内核，将 num_splits 个部分结果合并成最终的注意力输出。这个内核会对每个 Q 头在不同 split 上计算出的 lse 进行全局 log-sum-exp 合并。
@triton.jit
def _merge_kernel(...):
    # 加载该头在所有 split 上的 lse
    lse = tl.load(lse_partial_ptr + offs_splits * lse_partial_stride_split, ...)
    lse_max = tl.max(lse)
    # 计算归一化权重
    sumexp_normalized = tl.sum(tl.exp(lse - lse_max), axis=0)
    # 加权平均合并输出
    acc = tl.sum(o_partial * tl.exp(lse - lse_max)[:, None], axis=0) / sumexp_normalized
'''

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _gqa_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    output_ptr,
    lse_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_k_d,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_v_d,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    BATCH,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    SEQ_LEN,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """GQA 解码内核，使用 Split-KV 技术"""
    
    # 获取程序 ID
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    
    # 计算 split 数量
    num_splits = tl.num_programs(2)
    
    # 计算这个 KV 头对应的 Q 头范围
    q_head_start = kv_head_idx * (NUM_Q_HEADS // NUM_KV_HEADS)
    
    # 当前 split 处理的 KV 序列范围
    kv_seq_len_per_split = tl.cdiv(SEQ_LEN, num_splits)
    kv_start = split_idx * kv_seq_len_per_split
    kv_end = tl.minimum(kv_start + kv_seq_len_per_split, SEQ_LEN)
    
    # Q 头的偏移
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    
    # 检查是否超出 Q 头范围
    q_head_mask = q_head_start + offs_h < NUM_Q_HEADS
    valid_q_heads = q_head_start + offs_h < NUM_Q_HEADS
    
    # 加载 Q
    q_offsets = (
        batch_idx * stride_q_b +
        (q_head_start + offs_h[:, None]) * stride_q_h +
        offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptr + q_offsets, mask=q_head_mask[:, None], other=0.0)
    # q shape: [BLOCK_H, BLOCK_D]
    
    # 初始化用于 online softmax 的变量
    m_i = tl.full([BLOCK_H], -float('inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    
    # 计算需要处理的 KV 块数
    if kv_end > kv_start:
        # 确保至少处理一个块
        num_blocks = tl.cdiv(kv_end - kv_start, BLOCK_N)
        
        # 初始化 KV 序列指针
        kv_ptr_start = kv_start
        
        # 循环处理每个 KV 块
        for block_idx in range(num_blocks):
            # 当前块的起始位置
            kv_start_n = kv_ptr_start + block_idx * BLOCK_N
            kv_end_n = tl.minimum(kv_start_n + BLOCK_N, kv_end)
            kv_len = kv_end_n - kv_start_n
            
            # K 和 V 的偏移
            offs_n = tl.arange(0, BLOCK_N)
            kv_mask = offs_n < kv_len
            
            # 加载 K
            k_offsets = (
                batch_idx * stride_k_b +
                kv_head_idx * stride_k_h +
                (kv_start_n + offs_n[None, :]) * stride_k_s +
                offs_d[:, None] * stride_k_d
            )
            k = tl.load(k_cache_ptr + k_offsets, mask=kv_mask[None, :], other=0.0)
            # k shape: [BLOCK_D, BLOCK_N]
            
            # 计算 QK^T
            # q: [BLOCK_H, BLOCK_D], k: [BLOCK_D, BLOCK_N]
            # qk: [BLOCK_H, BLOCK_N]
            qk = tl.dot(q, k)
            
            # 应用 mask (因果 mask)
            seq_mask = (kv_start_n + offs_n[None, :]) < SEQ_LEN
            qk = tl.where(seq_mask, qk, -float('inf'))
            
            # Online Softmax 更新
            m_ij = tl.max(qk, axis=1)  # [BLOCK_H]
            # 应用最大值作为 mask，避免无效 Q 头影响
            m_ij = tl.where(q_head_mask, m_ij, -float('inf'))
            
            # 更新 m_i
            m_i_new = tl.maximum(m_i, m_ij)
            
            # 计算 exp
            # 对于 m_i 中为 -inf 的情况，exp(m_i - m_i_new) 会变成 0
            p = tl.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, axis=1)  # [BLOCK_H]
            
            # 更新 acc 和 l_i
            # 注意：需要缩放之前的 acc 和 l_i
            alpha = tl.exp(m_i - m_i_new)
            # 确保 mask 外的 Q 头不会影响结果
            alpha = tl.where(q_head_mask, alpha, 0.0)
            
            # 加载 V
            v_offsets = (
                batch_idx * stride_v_b +
                kv_head_idx * stride_v_h +
                (kv_start_n + offs_n[:, None]) * stride_v_s +
                offs_d[None, :] * stride_v_d
            )
            v = tl.load(v_cache_ptr + v_offsets, mask=kv_mask[:, None], other=0.0)
            # v shape: [BLOCK_N, BLOCK_D]
            
            # 更新 acc: [BLOCK_H, BLOCK_D]
            # p: [BLOCK_H, BLOCK_N], v: [BLOCK_N, BLOCK_D]
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            
            # 更新 l_i 和 m_i
            l_i = l_i * alpha + l_ij
            m_i = m_i_new
            
        # 计算最终输出
        # 归一化：acc / l_i
        # 注意：l_i 可能为 0，需要处理
        l_i = tl.where(l_i == 0, 1.0, l_i)
        output = acc / l_i[:, None]
        
        # 计算 log-sum-exp 用于后续合并
        lse = m_i + tl.log(l_i)
    else:
        # 如果这个 split 没有数据
        output = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        lse = tl.full([BLOCK_H], -float('inf'), dtype=tl.float32)
    
    # 存储输出和 LSE
    # 计算输出偏移
    out_offsets = (
        batch_idx * stride_out_b +
        (q_head_start + offs_h[:, None]) * stride_out_h +
        offs_d[None, :] * stride_out_d
    )
    tl.store(output_ptr + out_offsets, output.to(output_ptr.dtype), mask=q_head_mask[:, None])
    
    # 存储 LSE
    lse_offsets = (
        batch_idx * NUM_Q_HEADS * num_splits +
        (q_head_start + offs_h) * num_splits +
        split_idx
    )
    tl.store(lse_ptr + lse_offsets, lse, mask=q_head_mask)


@triton.jit
def _merge_kernel(
    out_partial_ptr,
    lse_partial_ptr,
    output_ptr,
    stride_out_partial_b,
    stride_out_partial_h,
    stride_out_partial_s,
    stride_out_partial_d,
    stride_lse_partial_b,
    stride_lse_partial_h,
    stride_lse_partial_s,
    NUM_Q_HEADS,
    NUM_SPLITS,
    HEAD_DIM,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """合并多个 split 的结果"""
    
    batch_idx = tl.program_id(0)
    q_head_start = tl.program_id(1) * BLOCK_H
    
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_head_mask = q_head_start + offs_h < NUM_Q_HEADS
    
    # 加载所有 split 的 LSE
    # lse_partial shape: [BATCH, NUM_Q_HEADS, NUM_SPLITS]
    lse_offsets = (
        batch_idx * stride_lse_partial_b +
        (q_head_start + offs_h[:, None]) * stride_lse_partial_h +
        tl.arange(0, NUM_SPLITS)[None, :] * stride_lse_partial_s
    )
    lse = tl.load(lse_partial_ptr + lse_offsets, mask=q_head_mask[:, None], other=-float('inf'))
    # lse shape: [BLOCK_H, NUM_SPLITS]
    
    # 计算全局 LSE
    # 使用 log-sum-exp 合并
    lse_max = tl.max(lse, axis=1)  # [BLOCK_H]
    # 对于无效的 Q 头，设置 lse_max 为 -inf
    lse_max = tl.where(q_head_mask, lse_max, -float('inf'))
    
    # 计算权重
    weights = tl.exp(lse - lse_max[:, None])  # [BLOCK_H, NUM_SPLITS]
    sum_weights = tl.sum(weights, axis=1)  # [BLOCK_H]
    
    # 避免除以 0
    sum_weights = tl.where(sum_weights == 0, 1.0, sum_weights)
    
    # 加载部分输出并合并
    # out_partial shape: [BATCH, NUM_Q_HEADS, NUM_SPLITS, HEAD_DIM]
    out_offsets = (
        batch_idx * stride_out_partial_b +
        (q_head_start + offs_h[:, None, None]) * stride_out_partial_h +
        tl.arange(0, NUM_SPLITS)[None, :, None] * stride_out_partial_s +
        offs_d[None, None, :] * stride_out_partial_d
    )
    out_partial = tl.load(out_partial_ptr + out_offsets, mask=q_head_mask[:, None, None], other=0.0)
    # out_partial shape: [BLOCK_H, NUM_SPLITS, HEAD_DIM]
    
    # 加权平均
    # weights: [BLOCK_H, NUM_SPLITS, 1]
    # out_partial: [BLOCK_H, NUM_SPLITS, HEAD_DIM]
    weights_expanded = weights[:, :, None]
    output = tl.sum(out_partial * weights_expanded, axis=1) / sum_weights[:, None]
    # output shape: [BLOCK_H, HEAD_DIM]
    
    # 存储最终输出
    out_final_offsets = (
        batch_idx * NUM_Q_HEADS * HEAD_DIM +
        (q_head_start + offs_h[:, None]) * HEAD_DIM +
        offs_d[None, :]
    )
    tl.store(output_ptr + out_final_offsets, output.to(output_ptr.dtype), mask=q_head_mask[:, None])


def gqa_decode(
    q: torch.Tensor,          # [BATCH, NUM_Q_HEADS, HEAD_DIM]
    k_cache: torch.Tensor,    # [BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM]
    v_cache: torch.Tensor,    # [BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM]
    num_splits: int = 8,
    block_d: int = 128,
    block_n: int = 64,
    block_h: int = 8,
) -> torch.Tensor:
    """
    GQA 解码的 Triton 实现
    
    Args:
        q: 查询张量 [BATCH, NUM_Q_HEADS, HEAD_DIM]
        k_cache: KV 缓存中的 K [BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM]
        v_cache: KV 缓存中的 V [BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM]
        num_splits: Split-KV 的 splits 数量
        block_d: 每个块的 head dimension
        block_n: 每个块的序列长度
        block_h: 每个块的 Q 头数量
        
    Returns:
        output: [BATCH, NUM_Q_HEADS, HEAD_DIM]
    """
    BATCH, NUM_Q_HEADS, HEAD_DIM = q.shape
    NUM_KV_HEADS = k_cache.shape[1]
    SEQ_LEN = k_cache.shape[2]
    
    # 检查 GQA 配置
    assert NUM_Q_HEADS % NUM_KV_HEADS == 0, f"NUM_Q_HEADS ({NUM_Q_HEADS}) must be divisible by NUM_KV_HEADS ({NUM_KV_HEADS})"
    
    # 分配输出
    output_partial = torch.zeros(
        BATCH, NUM_Q_HEADS, num_splits, HEAD_DIM,
        dtype=torch.float32, device=q.device
    )
    lse_partial = torch.full(
        (BATCH, NUM_Q_HEADS, num_splits),
        -float('inf'), dtype=torch.float32, device=q.device
    )
    
    # 调用内核
    grid = (BATCH, NUM_KV_HEADS, num_splits)
    
    # 自动调优
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_D': 128, 'BLOCK_N': 64, 'BLOCK_H': 4}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_D': 128, 'BLOCK_N': 128, 'BLOCK_H': 4}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_D': 128, 'BLOCK_N': 64, 'BLOCK_H': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_D': 128, 'BLOCK_N': 64, 'BLOCK_H': 4}, num_warps=8, num_stages=4),
            triton.Config({'BLOCK_D': 64, 'BLOCK_N': 64, 'BLOCK_H': 4}, num_warps=4, num_stages=4),
        ],
        key=['HEAD_DIM', 'SEQ_LEN', 'NUM_Q_HEADS', 'NUM_KV_HEADS'],
    )
    def _gqa_decode_kernel_wrapper(
        q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
        stride_q_b, stride_q_h, stride_q_d,
        stride_k_b, stride_k_h, stride_k_s, stride_k_d,
        stride_v_b, stride_v_h, stride_v_s, stride_v_d,
        stride_out_b, stride_out_h, stride_out_d,
        BATCH, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, SEQ_LEN,
        BLOCK_D, BLOCK_N, BLOCK_H
    ):
        _gqa_decode_kernel[
            grid,
            triton.language.num_warps(4)
        ](
            q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
            stride_q_b, stride_q_h, stride_q_d,
            stride_k_b, stride_k_h, stride_k_s, stride_k_d,
            stride_v_b, stride_v_h, stride_v_s, stride_v_d,
            stride_out_b, stride_out_h, stride_out_d,
            BATCH, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, SEQ_LEN,
            BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H
        )
    
    # 调用带自动调优的包装函数
    # 注意：这里为了简化，直接调用内核而不使用 autotune
    # 实际使用中，可以使用 @triton.autotune 装饰器
    
    _gqa_decode_kernel[
        grid,
        triton.language.num_warps(4)
    ](
        q, k_cache, v_cache, output_partial, lse_partial,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        output_partial.stride(0), output_partial.stride(1), output_partial.stride(3),
        BATCH, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, SEQ_LEN,
        BLOCK_D=block_d, BLOCK_N=block_n, BLOCK_H=block_h
    )
    
    # 合并结果
    output = torch.empty(BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=q.dtype, device=q.device)
    
    # 计算合并的 grid
    merge_grid = (BATCH, triton.cdiv(NUM_Q_HEADS, block_h))
    
    @triton.jit
    def _merge_kernel_wrapper(
        out_partial_ptr, lse_partial_ptr, output_ptr,
        stride_out_partial_b, stride_out_partial_h, stride_out_partial_s, stride_out_partial_d,
        stride_lse_partial_b, stride_lse_partial_h, stride_lse_partial_s,
        NUM_Q_HEADS, NUM_SPLITS, HEAD_DIM,
        BLOCK_D, BLOCK_H
    ):
        _merge_kernel[
            merge_grid,
            triton.language.num_warps(4)
        ](
            out_partial_ptr, lse_partial_ptr, output_ptr,
            stride_out_partial_b, stride_out_partial_h, stride_out_partial_s, stride_out_partial_d,
            stride_lse_partial_b, stride_lse_partial_h, stride_lse_partial_s,
            NUM_Q_HEADS, NUM_SPLITS, HEAD_DIM,
            BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H
        )
    
    _merge_kernel_wrapper[
        merge_grid,
        triton.language.num_warps(4)
    ](
        output_partial, lse_partial, output,
        output_partial.stride(0), output_partial.stride(1), output_partial.stride(2), output_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        NUM_Q_HEADS, num_splits, HEAD_DIM,
        BLOCK_D=block_d, BLOCK_H=block_h
    )
    
    return output


# 测试代码
def test_gqa_decode():
    torch.manual_seed(42)
    
    BATCH = 2
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 2  # GQA: 8 Q heads, 2 KV heads
    SEQ_LEN = 2048
    HEAD_DIM = 128
    
    print("Testing GQA implementation...")
    print(f"Configuration: B={BATCH}, Q_heads={NUM_Q_HEADS}, KV_heads={NUM_KV_HEADS}, Seq={SEQ_LEN}, Head_Dim={HEAD_DIM}")
    print(f"GQA Group Size: {NUM_Q_HEADS // NUM_KV_HEADS}")
    
    # 创建输入
    q = torch.randn(BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16, device='cuda')
    k_cache = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device='cuda')
    
    # 运行 Triton 实现
    with torch.no_grad():
        output_triton = gqa_decode(q, k_cache, v_cache, num_splits=4)
    
    # PyTorch 参考实现
    def reference_gqa(q, k_cache, v_cache):
        BATCH, NUM_Q_HEADS, HEAD_DIM = q.shape
        NUM_KV_HEADS = k_cache.shape[1]
        SEQ_LEN = k_cache.shape[2]
        GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
        
        output = torch.zeros(BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=q.dtype, device=q.device)
        
        for b in range(BATCH):
            for kv_h in range(NUM_KV_HEADS):
                q_start = kv_h * GQA_GROUP
                for i in range(GQA_GROUP):
                    q_h = q_start + i
                    # q: [1, HEAD_DIM]
                    # k: [SEQ_LEN, HEAD_DIM]
                    k = k_cache[b, kv_h].transpose(0, 1)  # [HEAD_DIM, SEQ_LEN]
                    v = v_cache[b, kv_h]  # [SEQ_LEN, HEAD_DIM]
                    
                    scores = torch.matmul(q[b, q_h], k)  # [SEQ_LEN]
                    scores = scores / math.sqrt(HEAD_DIM)
                    
                    # 应用因果 mask（可选）
                    # 这里简单使用 softmax
                    attn_weights = torch.softmax(scores, dim=-1)
                    
                    output[b, q_h] = torch.matmul(attn_weights, v)  # [HEAD_DIM]
        return output
    
    with torch.no_grad():
        output_ref = reference_gqa(q, k_cache, v_cache)
    
    # 比较结果
    diff = torch.abs(output_triton - output_ref)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")
    
    return output_triton, output_ref


# 性能基准测试
def benchmark_gqa():
    import time
    
    BATCH = 1
    NUM_Q_HEADS = 32
    NUM_KV_HEADS = 4
    SEQ_LEN = 4096
    HEAD_DIM = 128
    
    print(f"\nBenchmarking GQA: Q={NUM_Q_HEADS}, KV={NUM_KV_HEADS}, Seq={SEQ_LEN}, Dim={HEAD_DIM}")
    
    q = torch.randn(BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16, device='cuda')
    k_cache = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = gqa_decode(q, k_cache, v_cache, num_splits=8)
    
    torch.cuda.synchronize()
    
    # Benchmark
    num_iters = 20
    start = time.time()
    for _ in range(num_iters):
        _ = gqa_decode(q, k_cache, v_cache, num_splits=8)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time = elapsed / num_iters
    print(f"Average time per iteration: {avg_time*1000:.2f} ms")
    print(f"Throughput: {NUM_Q_HEADS * SEQ_LEN * HEAD_DIM / avg_time / 1e9:.2f} GFLOPS")
    
    return avg_time


if __name__ == "__main__":
    # 运行测试
    test_gqa_decode()
    
    # 运行基准测试
    benchmark_gqa()