import torch
import triton
import triton.language as tl

@triton.jit
def mlp_forward_kernel(
    # 输入矩阵
    x_ptr,  # [batch, in_features]
    w1_ptr, w2_ptr,  # [in_features, hidden], [hidden, out_features]
    b1_ptr, b2_ptr,  # [hidden], [out_features]
    out_ptr,  # [batch, out_features]
    
    # 维度信息
    batch_size,
    in_features,
    hidden_dim,
    out_features,
    
    # 步长
    stride_x_batch,
    stride_x_feat,
    stride_w1_in,
    stride_w1_hidden,
    stride_w2_hidden,
    stride_w2_out,
    stride_out_batch,
    stride_out_feat,
    
    # 激活函数选择
    activation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 获取当前程序id
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # 计算当前块处理的批次和输出维度范围
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, batch_size)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, out_features)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 第一层：x @ w1 + b1
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        # 加载 x 块
        x_block = tl.load(
            x_ptr + m_start * stride_x_batch + 
            k * stride_x_feat + 
            tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_x_batch +
            tl.arange(0, BLOCK_SIZE_K)[None, :],
            mask=(m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) &
                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < hidden_dim),
            other=0.0
        )
        
        # 加载 w1 块
        w1_block = tl.load(
            w1_ptr + k * stride_w1_in + 
            tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_w1_in +
            tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_w1_hidden,
            mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < hidden_dim) &
                 (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < out_features),
            other=0.0
        )
        
        # 矩阵乘法累加
        acc += tl.dot(x_block, w1_block)
    
    # 添加偏置并应用激活函数
    b1 = tl.load(
        b1_ptr + tl.arange(0, BLOCK_SIZE_N),
        mask=tl.arange(0, BLOCK_SIZE_N) < out_features,
        other=0.0
    )
    
    # 应用激活函数（ReLU）
    hidden = acc + b1[None, :]
    if activation == 'relu':
        hidden = tl.maximum(hidden, 0.0)
    elif activation == 'gelu':
        # GELU 近似
        hidden = 0.5 * hidden * (1.0 + tl.tanh(0.79788456 * (hidden + 0.044715 * hidden * hidden * hidden)))
    
    # 第二层：hidden @ w2 + b2
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        hidden_block = tl.load(
            hidden + k * stride_x_feat + 
            tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_x_feat +
            tl.arange(0, BLOCK_SIZE_K)[None, :],
            mask=(m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) &
                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < hidden_dim),
            other=0.0
        )
        
        w2_block = tl.load(
            w2_ptr + k * stride_w2_hidden + 
            tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_w2_hidden +
            tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_w2_out,
            mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < hidden_dim) &
                 (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < out_features),
            other=0.0
        )
        
        acc2 += tl.dot(hidden_block, w2_block)
    
    # 添加第二层偏置
    b2 = tl.load(
        b2_ptr + tl.arange(0, BLOCK_SIZE_N),
        mask=tl.arange(0, BLOCK_SIZE_N) < out_features,
        other=0.0
    )
    output = acc2 + b2[None, :]
    
    # 存储结果
    tl.store(
        out_ptr + m_start * stride_out_batch + 
        n_start * stride_out_feat + 
        tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_out_batch +
        tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_out_feat,
        output,
        mask=(m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) &
             (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < out_features)
    )

class TritonMLP(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, activation='relu'):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.activation = activation
        
        # 初始化权重和偏置
        self.w1 = torch.nn.Parameter(torch.randn(in_features, hidden_dim) * 0.01)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = torch.nn.Parameter(torch.randn(hidden_dim, out_features) * 0.01)
        self.b2 = torch.nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 分配输出张量
        out = torch.empty((batch_size, self.out_features), device=x.device, dtype=x.dtype)
        
        # 选择块大小（可根据实际情况调整）
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_K = 32
        BLOCK_SIZE_N = 64
        
        # 计算网格大小
        grid = (
            triton.cdiv(batch_size, BLOCK_SIZE_M),
            triton.cdiv(self.out_features, BLOCK_SIZE_N),
        )
        
        # 调用内核
        mlp_forward_kernel[grid](
            x,
            self.w1, self.w2,
            self.b1, self.b2,
            out,
            batch_size,
            self.in_features,
            self.hidden_dim,
            self.out_features,
            x.stride(0), x.stride(1),
            self.w1.stride(0), self.w1.stride(1),
            self.w2.stride(0), self.w2.stride(1),
            out.stride(0), out.stride(1),
            self.activation,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        
        return out

@triton.jit
def mlp_matmul_kernel(
    # 指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 元参数
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """优化的矩阵乘法内核"""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # 计算当前块的起始位置
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 创建指针
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # 累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 循环计算
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def mlp_forward(x, w1, b1, w2, b2, activation='relu'):
    """使用两个矩阵乘法实现的 MLP 前向传播"""
    batch_size, in_features = x.shape
    hidden_dim = w1.shape[1]
    out_features = w2.shape[1]
    
    # 第一层
    hidden = torch.empty((batch_size, hidden_dim), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(batch_size, 64), triton.cdiv(hidden_dim, 64))
    
    mlp_matmul_kernel[grid](
        x, w1, hidden,
        batch_size, hidden_dim, in_features,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        hidden.stride(0), hidden.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
    )
    
    # 添加偏置和激活
    hidden = hidden + b1
    if activation == 'relu':
        hidden = torch.relu(hidden)
    elif activation == 'gelu':
        hidden = torch.nn.functional.gelu(hidden)
    
    # 第二层
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(batch_size, 64), triton.cdiv(out_features, 64))
    
    mlp_matmul_kernel[grid](
        hidden, w2, output,
        batch_size, out_features, hidden_dim,
        hidden.stride(0), hidden.stride(1),
        w2.stride(0), w2.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
    )
    
    return output + b2


def test_mlp():
    # 设置参数
    batch_size = 1024
    in_features = 512
    hidden_dim = 1024
    out_features = 256
    
    # 创建模型和数据
    model = TritonMLP(in_features, hidden_dim, out_features, activation='relu')
    model.cuda()
    
    x = torch.randn(batch_size, in_features, device='cuda')
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 与 PyTorch 比较
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, out_features)
    ).cuda()
    
    torch_model[0].weight.data = model.w1.T
    torch_model[0].bias.data = model.b1
    torch_model[2].weight.data = model.w2.T
    torch_model[2].bias.data = model.b2
    
    torch_output = torch_model(x)
    
    # 检查误差
    diff = torch.abs(output - torch_output).max()
    print(f"Maximum difference: {diff.item():.6f}")
    assert diff < 1e-4, f"Difference too large: {diff}"
    
    # 性能测试
    import time
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    
    # 测试 Triton MLP
    start = time.time()
    for _ in range(100):
        _ = model(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100
    
    # 测试 PyTorch MLP
    start = time.time()
    for _ in range(100):
        _ = torch_model(x)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100
    
    print(f"Triton MLP time: {triton_time * 1000:.2f} ms")
    print(f"PyTorch MLP time: {torch_time * 1000:.2f} ms")
    print(f"Speedup: {torch_time / triton_time:.2f}x")


class TritonMLPWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2, activation='relu'):
        ctx.save_for_backward(x, w1, b1, w2, b2)
        ctx.activation = activation
        
        # 前向传播
        hidden = mlp_forward(x, w1, b1, w2, b2, activation)
        return hidden
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w1, b1, w2, b2 = ctx.saved_tensors
        activation = ctx.activation
        
        # 这里需要实现反向传播
        # 为简化，此处省略，实际使用时需要实现梯度计算
        # ...
        
        return grad_x, grad_w1, grad_b1, grad_w2, grad_b2, None

