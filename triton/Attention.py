"""
完整的Transformer实现，包含：
1. GQA (Grouped Query Attention) 使用Split-KV技术
2. 优化的MLP前向传播
3. Transformer层和完整模型
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
from typing import Optional, Tuple

# ==================== GQA 注意力内核 ====================

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
    
    # 初始化用于 online softmax 的变量
    m_i = tl.full([BLOCK_H], -float('inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    
    # 计算需要处理的 KV 块数
    if kv_end > kv_start:
        num_blocks = tl.cdiv(kv_end - kv_start, BLOCK_N)
        kv_ptr_start = kv_start
        
        # 循环处理每个 KV 块
        for block_idx in range(num_blocks):
            kv_start_n = kv_ptr_start + block_idx * BLOCK_N
            kv_end_n = tl.minimum(kv_start_n + BLOCK_N, kv_end)
            kv_len = kv_end_n - kv_start_n
            
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
            
            # 计算 QK^T
            qk = tl.dot(q, k)
            
            # 应用 mask
            seq_mask = (kv_start_n + offs_n[None, :]) < SEQ_LEN
            qk = tl.where(seq_mask, qk, -float('inf'))
            
            # Online Softmax 更新
            m_ij = tl.max(qk, axis=1)
            m_ij = tl.where(q_head_mask, m_ij, -float('inf'))
            m_i_new = tl.maximum(m_i, m_ij)
            
            p = tl.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, axis=1)
            
            alpha = tl.exp(m_i - m_i_new)
            alpha = tl.where(q_head_mask, alpha, 0.0)
            
            # 加载 V
            v_offsets = (
                batch_idx * stride_v_b +
                kv_head_idx * stride_v_h +
                (kv_start_n + offs_n[:, None]) * stride_v_s +
                offs_d[None, :] * stride_v_d
            )
            v = tl.load(v_cache_ptr + v_offsets, mask=kv_mask[:, None], other=0.0)
            
            # 更新 acc
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = m_i_new
            
        # 计算最终输出
        l_i = tl.where(l_i == 0, 1.0, l_i)
        output = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)
    else:
        output = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        lse = tl.full([BLOCK_H], -float('inf'), dtype=tl.float32)
    
    # 存储输出和 LSE
    out_offsets = (
        batch_idx * stride_out_b +
        (q_head_start + offs_h[:, None]) * stride_out_h +
        offs_d[None, :] * stride_out_d
    )
    tl.store(output_ptr + out_offsets, output.to(output_ptr.dtype), mask=q_head_mask[:, None])
    
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
    lse_offsets = (
        batch_idx * stride_lse_partial_b +
        (q_head_start + offs_h[:, None]) * stride_lse_partial_h +
        tl.arange(0, NUM_SPLITS)[None, :] * stride_lse_partial_s
    )
    lse = tl.load(lse_partial_ptr + lse_offsets, mask=q_head_mask[:, None], other=-float('inf'))
    
    # 计算全局 LSE
    lse_max = tl.max(lse, axis=1)
    lse_max = tl.where(q_head_mask, lse_max, -float('inf'))
    
    weights = tl.exp(lse - lse_max[:, None])
    sum_weights = tl.sum(weights, axis=1)
    sum_weights = tl.where(sum_weights == 0, 1.0, sum_weights)
    
    # 加载部分输出并合并
    out_offsets = (
        batch_idx * stride_out_partial_b +
        (q_head_start + offs_h[:, None, None]) * stride_out_partial_h +
        tl.arange(0, NUM_SPLITS)[None, :, None] * stride_out_partial_s +
        offs_d[None, None, :] * stride_out_partial_d
    )
    out_partial = tl.load(out_partial_ptr + out_offsets, mask=q_head_mask[:, None, None], other=0.0)
    
    weights_expanded = weights[:, :, None]
    output = tl.sum(out_partial * weights_expanded, axis=1) / sum_weights[:, None]
    
    # 存储最终输出
    out_final_offsets = (
        batch_idx * NUM_Q_HEADS * HEAD_DIM +
        (q_head_start + offs_h[:, None]) * HEAD_DIM +
        offs_d[None, :]
    )
    tl.store(output_ptr + out_final_offsets, output.to(output_ptr.dtype), mask=q_head_mask[:, None])


# ==================== MLP 内核 ====================

@triton.jit
def mlp_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """优化的矩阵乘法内核"""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def mlp_forward_kernel(
    x_ptr, w1_ptr, w2_ptr, b1_ptr, b2_ptr, out_ptr,
    batch_size, in_features, hidden_dim, out_features,
    stride_x_batch, stride_x_feat,
    stride_w1_in, stride_w1_hidden,
    stride_w2_hidden, stride_w2_out,
    stride_out_batch, stride_out_feat,
    activation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """MLP前向传播内核，包含两层线性变换和激活函数"""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 第一层: x @ w1
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_x_batch + 
            (k + offs_k[None, :]) * stride_x_feat,
            mask=(offs_m[:, None] < batch_size) & (k + offs_k[None, :] < in_features),
            other=0.0
        )
        w1_block = tl.load(
            w1_ptr + (k + offs_k[:, None]) * stride_w1_in + 
            offs_n[None, :] * stride_w1_hidden,
            mask=(k + offs_k[:, None] < hidden_dim) & (offs_n[None, :] < out_features),
            other=0.0
        )
        acc += tl.dot(x_block, w1_block)
    
    # 添加偏置和激活
    b1 = tl.load(b1_ptr + offs_n, mask=offs_n < out_features, other=0.0)
    hidden = acc + b1[None, :]
    
    if activation == 'relu':
        hidden = tl.maximum(hidden, 0.0)
    elif activation == 'gelu':
        hidden = 0.5 * hidden * (1.0 + tl.tanh(0.79788456 * (hidden + 0.044715 * hidden * hidden * hidden)))
    elif activation == 'silu':
        hidden = hidden * tl.sigmoid(hidden)
    
    # 第二层: hidden @ w2
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        hidden_block = tl.load(
            hidden + (k + offs_k[None, :]) * stride_x_feat,
            mask=(offs_m[:, None] < batch_size) & (k + offs_k[None, :] < hidden_dim),
            other=0.0
        )
        w2_block = tl.load(
            w2_ptr + (k + offs_k[:, None]) * stride_w2_hidden + 
            offs_n[None, :] * stride_w2_out,
            mask=(k + offs_k[:, None] < hidden_dim) & (offs_n[None, :] < out_features),
            other=0.0
        )
        acc2 += tl.dot(hidden_block, w2_block)
    
    # 添加第二层偏置
    b2 = tl.load(b2_ptr + offs_n, mask=offs_n < out_features, other=0.0)
    output = acc2 + b2[None, :]
    
    # 存储结果
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_batch + 
        offs_n[None, :] * stride_out_feat,
        output,
        mask=(offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)
    )


# ==================== 注意力模块 ====================

class GQAAttention(nn.Module):
    """GQA注意力模块"""
    
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_splits: int = 8,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_q_heads
        self.num_splits = num_splits
        
        assert hidden_size % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [B, L, D]
            past_key: [B, num_kv_heads, seq_len, head_dim]
            past_value: [B, num_kv_heads, seq_len, head_dim]
            use_cache: 是否返回 KV cache
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 处理 KV cache
        if past_key is not None and past_value is not None:
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # 更新 cache
        if use_cache:
            present_key, present_value = k, v
        else:
            present_key, present_value = None, None
        
        # 使用 GQA 内核计算注意力
        if seq_len == 1:  # 解码阶段
            q = q.squeeze(1)  # [B, num_q_heads, head_dim]
            
            # 调用 GQA 内核
            output = gqa_decode(
                q, k, v,
                num_splits=self.num_splits,
                block_d=128,
                block_n=64,
                block_h=min(8, self.num_q_heads // self.num_kv_heads)
            )
            # output: [B, num_q_heads, head_dim]
            
            # 重塑回原始形状
            output = output.view(batch_size, -1)  # [B, num_q_heads * head_dim]
            
        else:  # 预填充阶段，使用标准注意力
            # 扩展 K 和 V 以匹配 Q 头数
            k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 输出投影
        output = self.o_proj(output)
        
        return output, (present_key, present_value) if use_cache else None


# ==================== MLP 模块 ====================

class TritonMLP(nn.Module):
    """使用Triton的MLP模块"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        
        self.w1 = nn.Parameter(torch.empty(hidden_size, intermediate_size))
        self.b1 = nn.Parameter(torch.zeros(intermediate_size))
        self.w2 = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(hidden_size))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        output_flat = torch.empty(
            (batch_size * seq_len, hidden_size),
            device=x.device,
            dtype=x.dtype
        )
        
        # 选择块大小
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_K = 32
        BLOCK_SIZE_N = 64
        
        grid = (
            triton.cdiv(batch_size * seq_len, BLOCK_SIZE_M),
            triton.cdiv(hidden_size, BLOCK_SIZE_N),
        )
        
        mlp_forward_kernel[grid](
            x_flat, self.w1, self.w2, self.b1, self.b2, output_flat,
            batch_size * seq_len, hidden_size, self.intermediate_size, hidden_size,
            x_flat.stride(0), x_flat.stride(1),
            self.w1.stride(0), self.w1.stride(1),
            self.w2.stride(0), self.w2.stride(1),
            output_flat.stride(0), output_flat.stride(1),
            self.activation,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        
        return output_flat.view(batch_size, seq_len, hidden_size)


# ==================== GQA 外层函数 ====================

def gqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_splits: int = 8,
    block_d: int = 128,
    block_n: int = 64,
    block_h: int = 8,
) -> torch.Tensor:
    """
    GQA 解码的外层函数
    
    Args:
        q: [BATCH, NUM_Q_HEADS, HEAD_DIM]
        k_cache: [BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM]
        v_cache: [BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM]
        num_splits: Split-KV 的数量
        block_d: Head dimension 的块大小
        block_n: 序列长度的块大小
        block_h: Q头数的块大小
    """
    BATCH, NUM_Q_HEADS, HEAD_DIM = q.shape
    NUM_KV_HEADS = k_cache.shape[1]
    SEQ_LEN = k_cache.shape[2]
    
    assert NUM_Q_HEADS % NUM_KV_HEADS == 0
    
    # 分配部分输出
    output_partial = torch.zeros(
        BATCH, NUM_Q_HEADS, num_splits, HEAD_DIM,
        dtype=torch.float32, device=q.device
    )
    lse_partial = torch.full(
        (BATCH, NUM_Q_HEADS, num_splits),
        -float('inf'), dtype=torch.float32, device=q.device
    )
    
    # 调用 GQA 内核
    grid = (BATCH, NUM_KV_HEADS, num_splits)
    
    _gqa_decode_kernel[grid](
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
    merge_grid = (BATCH, triton.cdiv(NUM_Q_HEADS, block_h))
    
    _merge_kernel[merge_grid](
        output_partial, lse_partial, output,
        output_partial.stride(0), output_partial.stride(1), output_partial.stride(2), output_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        NUM_Q_HEADS, num_splits, HEAD_DIM,
        BLOCK_D=block_d, BLOCK_H=block_h
    )
    
    return output


# ==================== Transformer 层 ====================

class TransformerBlock(nn.Module):
    """完整的 Transformer 块"""
    
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = 'gelu',
        num_splits: int = 8,
    ):
        super().__init__()
        
        self.attention = GQAAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            num_splits=num_splits,
        )
        
        self.mlp = TritonMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
        )
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 自注意力
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        attn_output, present = self.attention(
            hidden_states,
            past_key=past_key,
            past_value=past_value,
            use_cache=use_cache,
        )
        
        hidden_states = residual + self.dropout(attn_output)
        
        # MLP
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)
        
        return hidden_states, present


# ==================== 完整 Transformer 模型 ====================

class TransformerModel(nn.Module):
    """完整 Transformer 模型"""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        num_splits: int = 8,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
                dropout=dropout,
                activation=activation,
                num_splits=num_splits,
            )
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids: [B, L]
            past_key_values: 每层的 (key, value) 缓存
            use_cache: 是否返回 KV cache
        """
        batch_size, seq_len = input_ids.shape
        
        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # 嵌入
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # 通过各层
        presents = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key = past_key_values[i][0] if past_key_values is not None else None
            past_value = past_key_values[i][1] if past_key_values is not None else None
            
            hidden_states, present = layer(
                hidden_states,
                past_key=past_key,
                past_value=past_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                presents.append(present)
        
        # 最终输出
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, presents


# ==================== 测试代码 ====================

def test_gqa():
    """测试 GQA 实现"""
    torch.manual_seed(42)
    
    BATCH = 2
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 2
    SEQ_LEN = 2048
    HEAD_DIM = 128
    
    print("Testing GQA implementation...")
    print(f"Config: B={BATCH}, Q_heads={NUM_Q_HEADS}, KV_heads={NUM_KV_HEADS}, Seq={SEQ_LEN}")
    
    q = torch.randn(BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16, device='cuda')
    k_cache = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device='cuda')
    
    # Triton 实现
    with torch.no_grad():
        output_triton = gqa_decode(q, k_cache, v_cache, num_splits=4)
    
    # PyTorch 参考实现
    def reference_gqa(q, k_cache, v_cache):
        BATCH, NUM_Q_HEADS, HEAD_DIM = q.shape
        NUM_KV_HEADS = k_cache.shape[1]
        SEQ_LEN = k_cache.shape[2]
        GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
        
        output = torch.zeros(BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=q.dtype, device=q.device)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        
        for b in range(BATCH):
            for kv_h in range(NUM_KV_HEADS):
                q_start = kv_h * GQA_GROUP
                k = k_cache[b, kv_h].transpose(0, 1)
                v = v_cache[b, kv_h]
                
                for i in range(GQA_GROUP):
                    q_h = q_start + i
                    scores = torch.matmul(q[b, q_h], k) * scale
                    attn_weights = torch.softmax(scores, dim=-1)
                    output[b, q_h] = torch.matmul(attn_weights, v)
        return output
    
    with torch.no_grad():
        output_ref = reference_gqa(q, k_cache, v_cache)
    
    diff = torch.abs(output_triton - output_ref)
    max_diff = torch.max(diff)
    
    print(f"Max diff: {max_diff:.6f}")
    print("✓ Test passed!" if max_diff < 1e-2 else "✗ Test failed!")
    
    return output_triton, output_ref


def test_transformer():
    """测试完整 Transformer"""
    torch.manual_seed(42)
    
    # 模型配置
    config = {
        'vocab_size': 10000,
        'hidden_size': 512,
        'num_layers': 4,
        'num_q_heads': 8,
        'num_kv_heads': 2,
        'intermediate_size': 2048,
        'max_seq_len': 512,
        'dropout': 0.0,
        'num_splits': 4,
    }
    
    print("\nTesting complete Transformer...")
    print(f"Config: {config}")
    
    model = TransformerModel(**config).cuda()
    model.eval()
    
    # 测试前向传播
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device='cuda')
    
    with torch.no_grad():
        logits, presents = model(input_ids, use_cache=False)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output shape: {logits.shape}")
    
    # 测试 KV cache
    input_ids_step = torch.randint(0, config['vocab_size'], (batch_size, 1), device='cuda')
    past_key_values = None
    
    for step in range(5):
        with torch.no_grad():
            logits, past_key_values = model(
                input_ids_step,
                past_key_values=past_key_values,
                use_cache=True,
            )
        print(f"Step {step}: logits shape {logits.shape}, cache layers: {len(past_key_values)}")
    
    print("✓ Transformer test passed!")


def benchmark_performance():
    """性能基准测试"""
    import time
    
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    # GQA 基准测试
    BATCH = 1
    NUM_Q_HEADS = 32
    NUM_KV_HEADS = 4
    SEQ_LEN = 4096
    HEAD_DIM = 128
    
    print(f"\nGQA Benchmark:")
    print(f"Q={NUM_Q_HEADS}, KV={NUM_KV_HEADS}, Seq={SEQ_LEN}, Dim={HEAD_DIM}")
    
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
    gqa_time = (time.time() - start) / num_iters * 1000
    
    print(f"GQA average time: {gqa_time:.2f} ms")
    print(f"Throughput: {NUM_Q_HEADS * SEQ_LEN * HEAD_DIM / (gqa_time/1000) / 1e9:.2f} GFLOPS")
    
    # Transformer 基准测试
    print(f"\nTransformer Benchmark:")
    config = {
        'vocab_size': 10000,
        'hidden_size': 1024,
        'num_layers': 12,
        'num_q_heads': 16,
        'num_kv_heads': 4,
        'intermediate_size': 4096,
        'max_seq_len': 2048,
        'dropout': 0.0,
        'num_splits': 8,
    }
    
    model = TransformerModel(**config).cuda()
    model.eval()
    
    batch_size = 1
    seq_len = 1024
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device='cuda')
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    
    # Benchmark
    num_iters = 10
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    transformer_time = (time.time() - start) / num_iters * 1000
    
    print(f"Transformer forward time: {transformer_time:.2f} ms")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":
    print("="*60)
    print("Triton Transformer with GQA")
    print("="*60)
    
    # 运行测试
    test_gqa()
    test_transformer()
    benchmark_performance()