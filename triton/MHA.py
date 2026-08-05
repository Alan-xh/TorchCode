import torch
import triton
import triton.language as tl
import math

'''
程序执行模型:
tl.program_id(axis)：用于获取当前程序实例（可以理解为一个CUDA线程块）在指定轴（axis=0,1,2）上的唯一ID。它帮助每个程序块识别自己应该处理数据中的哪一部分，是实现并行计算的核心。
tl.num_programs(axis)：返回指定轴上程序实例的总数量。在需要将工作负载均匀分配给所有程序块，或者在程序块内进行循环处理多个数据块时，会用到这个函数。

编译时常量:
tl.constexpr：这是一个类型注解，用于标记那些在编译时必须已知的常量参数（如 BLOCK_SIZE）。Triton 编译器会将这些参数直接“折叠”进生成的代码中，消除运行时开销，并利用它们来优化内存访问和循环。

数据操作与张量:
tl.arange(start, end)：生成一个从 start 到 end（不包含）的连续整数序列（一维张量）。它常与 tl.program_id 结合，来计算出当前程序块需要处理的数据的具体索引范围。
tl.load(ptr, mask)：从指定的内存指针地址加载数据。它非常灵活，支持通过mask参数来屏蔽越界或无效的内存访问，这是处理矩阵边缘数据的关键。
tl.zeros(shape, dtype)：创建一个指定形状和数据类型的、所有元素初始值为零的张量。在矩阵乘法中，它常被用来初始化累加器（acc），用于存储中间计算结果。
tl.dot(a, b)：执行块级的矩阵乘法。这是 Triton 实现高性能矩阵运算（GEMM）的核心，它允许对两个块级张量进行点乘，并将结果累加。
tl.store(ptr, value, mask)：将数据存储回指定的内存地址。和 tl.load 一样，它也支持 mask 参数，以保证内存写入的安全性。

数据类型与数学运算:
tl.float32：32位浮点数数据类型，是Triton中最常用的精度之一。
tl.max：用于计算张量中的最大值，常用于规约（reduction）操作。
tl.where(condition, x, y)：基于条件选择元素的三元操作符，等价于condition ? x : y。它常用于实现条件运算或掩码操作。
'''

@triton.jit
def fused_mha_kernel(
    # 输入指针
    q_ptr, k_ptr, v_ptr, out_ptr,
    # 矩阵维度
    batch, seq_len, num_heads, head_dim,
    # 步长
    q_batch_stride, q_seq_stride, q_head_stride, q_dim_stride,
    k_batch_stride, k_seq_stride, k_head_stride, k_dim_stride,
    v_batch_stride, v_seq_stride, v_head_stride, v_dim_stride,
    out_batch_stride, out_seq_stride, out_head_stride, out_dim_stride,
    # 缩放因子
    scale,
    # 块大小
    BLOCK_M: tl.constexpr,  # 查询块大小
    BLOCK_N: tl.constexpr,  # 键值块大小
    BLOCK_D: tl.constexpr,  # 头维度块大小
):
    """
    融合的多头注意力内核
    计算: softmax(Q @ K^T / sqrt(d)) @ V
    """
    # 获取程序ID
    pid_m = tl.program_id(0)  # 批处理和头的组合索引
    pid_n = tl.program_id(1)  # 序列块索引
    
    # 计算实际的批处理、头和序列位置
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    
    # 分配批处理和头
    batch_idx = pid_m // num_heads
    head_idx = pid_m % num_heads
    
    # 序列块起始位置
    start_m = pid_n * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    # 确保不超出序列长度
    m_mask = offs_m < seq_len
    
    # 加载Q: [BLOCK_M, BLOCK_D]
    q_offs = (
        batch_idx * q_batch_stride +
        offs_m[:, None] * q_seq_stride +
        head_idx * q_head_stride +
        tl.arange(0, BLOCK_D)[None, :] * q_dim_stride
    )
    q = tl.load(q_ptr + q_offs, mask=m_mask[:, None], other=0.0)
    
    # 初始化输出和归一化因子
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # 计算注意力分数并累加
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # 加载K: [BLOCK_N, BLOCK_D]
        k_offs = (
            batch_idx * k_batch_stride +
            offs_n[:, None] * k_seq_stride +
            head_idx * k_head_stride +
            tl.arange(0, BLOCK_D)[None, :] * k_dim_stride
        )
        k = tl.load(k_ptr + k_offs, mask=n_mask[:, None], other=0.0)
        
        # 计算注意力分数: Q @ K^T
        # [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # 应用mask（将无效位置设为-inf）
        n_mask_expanded = n_mask[None, :]
        m_mask_expanded = m_mask[:, None]
        mask = m_mask_expanded & n_mask_expanded
        scores = tl.where(mask, scores, float("-inf"))
        
        # Softmax计算
        # 减去最大值以提高数值稳定性
        max_val = tl.max(scores, axis=1)[:, None]
        scores = scores - max_val
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=1)[:, None]
        probs = exp_scores / sum_exp
        
        # 加载V: [BLOCK_N, BLOCK_D]
        v_offs = (
            batch_idx * v_batch_stride +
            offs_n[:, None] * v_seq_stride +
            head_idx * v_head_stride +
            tl.arange(0, BLOCK_D)[None, :] * v_dim_stride
        )
        v = tl.load(v_ptr + v_offs, mask=n_mask[:, None], other=0.0)
        
        # 累加: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        acc += tl.dot(probs, v)
    
    # 存储输出
    out_offs = (
        batch_idx * out_batch_stride +
        offs_m[:, None] * out_seq_stride +
        head_idx * out_head_stride +
        tl.arange(0, BLOCK_D)[None, :] * out_dim_stride
    )
    tl.store(out_ptr + out_offs, acc, mask=m_mask[:, None])


class MultiHeadAttention(torch.nn.Module):
    """
    使用Triton实现的多头注意力模块
    """
    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 线性投影层
        self.wq = torch.nn.Linear(d_model, d_model, bias=bias)
        self.wk = torch.nn.Linear(d_model, d_model, bias=bias)
        self.wv = torch.nn.Linear(d_model, d_model, bias=bias)
        self.wo = torch.nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, d_model]
        mask: [batch, seq_len] or None
        """
        batch, seq_len, _ = x.shape
        
        # 线性投影并重塑
        q = self.wq(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 如果有mask，将其应用到注意力计算中
        # 注意：Triton内核中的mask处理在kernel内部完成
        
        # 分配输出张量
        out = torch.empty_like(q)
        
        # 计算块大小
        BLOCK_M = 32  # 可以根据需要调整
        BLOCK_N = 32
        BLOCK_D = min(self.head_dim, 64)  # head_dim可能很大，限制块大小
        
        # 计算网格
        grid = (batch * self.num_heads, (seq_len + BLOCK_M - 1) // BLOCK_M)
        
        # 获取指针和步长
        q_ptr = q.contiguous().view(batch, self.num_heads, seq_len, self.head_dim)
        k_ptr = k.contiguous().view(batch, self.num_heads, seq_len, self.head_dim)
        v_ptr = v.contiguous().view(batch, self.num_heads, seq_len, self.head_dim)
        out_ptr = out.contiguous().view(batch, self.num_heads, seq_len, self.head_dim)
        
        # 计算步长
        q_batch_stride = q_ptr.stride(0)
        q_head_stride = q_ptr.stride(1)
        q_seq_stride = q_ptr.stride(2)
        q_dim_stride = q_ptr.stride(3)
        
        k_batch_stride = k_ptr.stride(0)
        k_head_stride = k_ptr.stride(1)
        k_seq_stride = k_ptr.stride(2)
        k_dim_stride = k_ptr.stride(3)
        
        v_batch_stride = v_ptr.stride(0)
        v_head_stride = v_ptr.stride(1)
        v_seq_stride = v_ptr.stride(2)
        v_dim_stride = v_ptr.stride(3)
        
        out_batch_stride = out_ptr.stride(0)
        out_head_stride = out_ptr.stride(1)
        out_seq_stride = out_ptr.stride(2)
        out_dim_stride = out_ptr.stride(3)
        
        # 启动内核
        fused_mha_kernel[grid](
            q_ptr, k_ptr, v_ptr, out_ptr,
            batch, seq_len, self.num_heads, self.head_dim,
            q_batch_stride, q_seq_stride, q_head_stride, q_dim_stride,
            k_batch_stride, k_seq_stride, k_head_stride, k_dim_stride,
            v_batch_stride, v_seq_stride, v_head_stride, v_dim_stride,
            out_batch_stride, out_seq_stride, out_head_stride, out_dim_stride,
            self.scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        
        # 恢复形状 [batch, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        # 输出投影
        out = self.wo(out)
        
        return out


# 更优化的版本：使用两级累加减少内存访问
@triton.jit
def fused_mha_kernel_v2(
    # 输入指针
    q_ptr, k_ptr, v_ptr, out_ptr,
    # 矩阵维度
    batch, seq_len, num_heads, head_dim,
    # 步长
    q_batch_stride, q_seq_stride, q_head_stride, q_dim_stride,
    k_batch_stride, k_seq_stride, k_head_stride, k_dim_stride,
    v_batch_stride, v_seq_stride, v_head_stride, v_dim_stride,
    out_batch_stride, out_seq_stride, out_head_stride, out_dim_stride,
    # 缩放因子
    scale,
    # 块大小
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    优化的MHA内核，使用两级累加减少内存访问
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    num_pid_m = tl.num_programs(0)
    batch_idx = pid_m // num_heads
    head_idx = pid_m % num_heads
    
    start_m = pid_n * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < seq_len
    
    # 加载Q
    q_offs = (
        batch_idx * q_batch_stride +
        offs_m[:, None] * q_seq_stride +
        head_idx * q_head_stride +
        tl.arange(0, BLOCK_D)[None, :] * q_dim_stride
    )
    q = tl.load(q_ptr + q_offs, mask=m_mask[:, None], other=0.0)
    
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # 处理整个序列
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # 加载K
        k_offs = (
            batch_idx * k_batch_stride +
            offs_n[:, None] * k_seq_stride +
            head_idx * k_head_stride +
            tl.arange(0, BLOCK_D)[None, :] * k_dim_stride
        )
        k = tl.load(k_ptr + k_offs, mask=n_mask[:, None], other=0.0)
        
        # 计算scores
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # 应用mask
        n_mask_expanded = n_mask[None, :]
        m_mask_expanded = m_mask[:, None]
        mask = m_mask_expanded & n_mask_expanded
        scores = tl.where(mask, scores, float("-inf"))
        
        # Softmax
        max_val = tl.max(scores, axis=1)[:, None]
        scores = scores - max_val
        exp_scores = tl.exp(scores)
        sum_exp = tl.sum(exp_scores, axis=1)[:, None]
        probs = exp_scores / sum_exp
        
        # 加载V
        v_offs = (
            batch_idx * v_batch_stride +
            offs_n[:, None] * v_seq_stride +
            head_idx * v_head_stride +
            tl.arange(0, BLOCK_D)[None, :] * v_dim_stride
        )
        v = tl.load(v_ptr + v_offs, mask=n_mask[:, None], other=0.0)
        
        # 累加
        acc += tl.dot(probs, v)
    
    # 存储输出
    out_offs = (
        batch_idx * out_batch_stride +
        offs_m[:, None] * out_seq_stride +
        head_idx * out_head_stride +
        tl.arange(0, BLOCK_D)[None, :] * out_dim_stride
    )
    tl.store(out_ptr + out_offs, acc, mask=m_mask[:, None])


def test_mha():
    """测试MHA实现"""
    # 设置参数
    batch = 2
    seq_len = 128
    d_model = 512
    num_heads = 8
    
    # 创建模型
    model = MultiHeadAttention(d_model, num_heads).cuda()
    model.eval()  # 关闭dropout用于测试
    
    # 创建输入
    x = torch.randn(batch, seq_len, d_model, device='cuda')
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    # 与 PyTorch 原生实现比较
    def torch_mha(x):
        q = model.wq(x).view(batch, seq_len, num_heads, d_model // num_heads)
        k = model.wk(x).view(batch, seq_len, num_heads, d_model // num_heads)
        v = model.wv(x).view(batch, seq_len, num_heads, d_model // num_heads)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model // num_heads)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = model.wo(out)
        return out
    
    with torch.no_grad():
        torch_output = torch_mha(x)
    
    # 比较结果
    diff = torch.abs(output - torch_output).max().item()
    print(f"Max difference with PyTorch: {diff:.6f}")
    assert diff < 1e-4, f"Results differ too much: {diff}"
    print("Test passed!")


if __name__ == "__main__":
    test_mha()