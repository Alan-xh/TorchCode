import tensorrt as trt
import numpy as np

class AttentionBuilder:
    def __init__(self, network, config):
        '''
        初始化构建器

        参数:
        - network: TensorRT 网络对象
        - config: 模型配置
        '''
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

# 在BERT模型中调用
class BERTModelTRT:
    def __init__(self):
        self.network = ...
        
    def build_bert_layer(self, input_tensor):
        # 调用注意力层
        attention_output = self._build_bert_attention(input_tensor)
        
        # 继续构建FFN层
        ffn_output = self._build_ffn(attention_output)
        
        return ffn_output
    
    def _build_ffn(self, input_tensor):
        """构建前馈网络层"""
        hidden_size = 768
        intermediate_size = 3072  # 4 * hidden_size
        
        # 第一层：放大到4倍
        fc1 = self.network.add_fully_connected(
            input_tensor, intermediate_size, self._load_weights('fc1_weight')
        )
        gelu = self.network.add_activation(
            fc1.get_output(0), trt.ActivationType.GELU
        )
        
        # 第二层：投影回原维度
        fc2 = self.network.add_fully_connected(
            gelu.get_output(0), hidden_size, self._load_weights('fc2_weight')
        )
        
        # 残差连接 + LayerNorm
        add = self.network.add_elementwise(
            input_tensor, fc2.get_output(0), trt.ElementWiseOperation.SUM
        )
        output = self._add_layer_norm(add.get_output(0))
        
        return output

    def _build_bert_attention(self, input_tensor):
        hidden_size = 768          # BERT-base的隐藏层维度
        num_attention_heads = 12   # 注意力头数
        attention_head_size = hidden_size // num_attention_heads  # 64
        
        # ========== 第一步：线性投影得到 Q, K, V ==========
        # 初始化权重（实际使用时需要从预训练模型加载）
        q_weight = self.network.add_constant(
            trt.Dims2(hidden_size, hidden_size),
            self._load_weights('query_weight')
        )
        k_weight = self.network.add_constant(
            trt.Dims2(hidden_size, hidden_size),
            self._load_weights('key_weight')
        )
        v_weight = self.network.add_constant(
            trt.Dims2(hidden_size, hidden_size),
            self._load_weights('value_weight')
        )
        
        # 全连接层投影
        q_proj = self.network.add_fully_connected(
            input_tensor, hidden_size, q_weight.get_output(0)
        )
        k_proj = self.network.add_fully_connected(
            input_tensor, hidden_size, k_weight.get_output(0)
        )
        v_proj = self.network.add_fully_connected(
            input_tensor, hidden_size, v_weight.get_output(0)
        )
        
        # ========== 第二步：重塑为多头格式 ==========
        # [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads, head_size]
        # -> [batch, num_heads, seq_len, head_size]
        
        def reshape_for_multi_head(tensor):
            # Reshape: [B, S, H*D] -> [B, S, H, D]
            reshape = self.network.add_reshape(
                tensor,
                trt.Dims4(0, 0, num_attention_heads, attention_head_size)
            )
            
            # Transpose: [B, S, H, D] -> [B, H, S, D]
            transpose = self.network.add_transpose(reshape.get_output(0))
            transpose.set_order(trt.Permutation([0, 2, 1, 3]))
            
            return transpose.get_output(0)
        
        Q = reshape_for_multi_head(q_proj.get_output(0))
        K = reshape_for_multi_head(k_proj.get_output(0))
        V = reshape_for_multi_head(v_proj.get_output(0))
        
        # ========== 第三步：Scaled Dot-Product Attention ==========
        # 计算 Q * K^T
        matmul_qk = self.network.add_matrix_multiply(
            Q, trt.MatrixOperation.NONE,
            K, trt.MatrixOperation.TRANSPOSE
        )
        
        # 缩放因子: 1/sqrt(head_size)
        scale_factor = 1.0 / np.sqrt(attention_head_size)
        scale_const = self.network.add_constant(
            trt.Dims3(1, 1, 1),
            np.array([scale_factor], dtype=np.float32)
        )
        scaled_scores = self.network.add_elementwise(
            matmul_qk.get_output(0),
            scale_const.get_output(0),
            trt.ElementWiseOperation.PROD
        )
        
        # Softmax (沿seq_len维度)
        softmax = self.network.add_softmax(scaled_scores.get_output(0))
        softmax.set_axes(1 << 3)  # 最后一个维度
        
        # Attention权重 * V
        matmul_attn = self.network.add_matrix_multiply(
            softmax.get_output(0),
            trt.MatrixOperation.NONE,
            V,
            trt.MatrixOperation.NONE
        )
        
        # ========== 第四步：还原维度 ==========
        # [B, H, S, D] -> [B, S, H, D]
        transpose_back = self.network.add_transpose(matmul_attn.get_output(0))
        transpose_back.set_order(trt.Permutation([0, 2, 1, 3]))
        
        # [B, S, H, D] -> [B, S, H*D]
        reshape_back = self.network.add_reshape(
            transpose_back.get_output(0),
            trt.Dims3(0, 0, hidden_size)
        )
        
        # ========== 第五步：输出投影 ==========
        out_weight = self.network.add_constant(
            trt.Dims2(hidden_size, hidden_size),
            self._load_weights('output_weight')
        )
        out_proj = self.network.add_fully_connected(
            reshape_back.get_output(0),
            hidden_size,
            out_weight.get_output(0)
        )
        
        # ========== 第六步：残差连接 + LayerNorm ==========
        # Add残差
        residual_add = self.network.add_elementwise(
            input_tensor,
            out_proj.get_output(0),
            trt.ElementWiseOperation.SUM
        )
        
        # Layer Normalization
        layer_norm = self._add_layer_norm(residual_add.get_output(0))
        
        return layer_norm.get_output(0)