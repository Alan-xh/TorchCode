import tensorrt as trt
import numpy as np

class AttentionBuilder:
    def __init__(self, network, config):
        self.network = network
        self.config = config
        
    def build_multi_head_attention(self, input_tensor, num_heads, head_dim):
        """
        构建多头注意力机制
        """
        batch_size, seq_len, embed_dim = input_tensor.shape
        
        # 1. 线性变换得到Q, K, V
        q_proj = self.network.add_fully_connected(
            input_tensor, embed_dim, self._get_weights('q_weight')
        )
        k_proj = self.network.add_fully_connected(
            input_tensor, embed_dim, self._get_weights('k_weight')
        )
        v_proj = self.network.add_fully_connected(
            input_tensor, embed_dim, self._get_weights('v_weight')
        )
        
        # 2. 重塑为多头形式 [B, S, H, D] -> [B, H, S, D]
        q_reshaped = self._reshape_for_heads(q_proj.get_output(0), num_heads, head_dim)
        k_reshaped = self._reshape_for_heads(k_proj.get_output(0), num_heads, head_dim)
        v_reshaped = self._reshape_for_heads(v_proj.get_output(0), num_heads, head_dim)
        
        # 3. Scaled Dot-Product Attention
        # Q * K^T
        matmul1 = self.network.add_matrix_multiply(
            q_reshaped, trt.MatrixOperation.NONE,
            k_reshaped, trt.MatrixOperation.TRANSPOSE
        )
        
        # Scale
        scale = self.network.add_constant(
            (1,), np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32)
        )
        scaled = self.network.add_elementwise(
            matmul1.get_output(0), scale.get_output(0),
            trt.ElementWiseOperation.PROD
        )
        
        # Softmax
        softmax = self.network.add_softmax(scaled.get_output(0))
        softmax.axes = 1 << 3  # 沿最后一个维度
        
        # Attention * V
        matmul2 = self.network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE,
            v_reshaped, trt.MatrixOperation.NONE
        )
        
        # 4. 重塑回原始形状
        output = self._reshape_back(matmul2.get_output(0), batch_size, seq_len, embed_dim)
        
        # 5. 输出投影
        out_proj = self.network.add_fully_connected(
            output, embed_dim, self._get_weights('out_weight')
        )
        
        return out_proj.get_output(0)
    
    def _reshape_for_heads(self, tensor, num_heads, head_dim):
        # [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
        shape = self.network.add_reshape(
            tensor, 
            trt.Dims([-1, -1, num_heads, head_dim])
        )
        # 转置
        transpose = self.network.add_transpose(shape.get_output(0))
        transpose.order = [0, 2, 1, 3]  # B, H, S, D
        return transpose.get_output(0)