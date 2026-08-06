TensorRT 是 nvidia 厂商专有优化层 sdk，与 ONNX Runtime\ OpenVINO 类似，从 PyTorch/ONNX 导出模型、处理动态形状到量化等完整工作流, 支持几乎相同的 C++ 和 Python API。

### 核心工作流
构建阶段 (Build Phase)：通过 Builder 和 BuilderConfig 接口进行。主要步骤包括创建网络定义（推荐从ONNX导入）、设置优化配置（如精度、工作空间大小），最后生成一个针对特定GPU优化的、序列化的引擎（Engine/Plan File）。

运行时阶段 (Runtime Phase)：应用程序通过 Runtime 接口反序列化引擎文件，创建 ExecutionContext 来执行推理。你可以创建多个上下文来并行运行推理任务。

### 重要特性
精度与量化：广泛支持FP32、FP16、BF16、INT8、FP8，甚至支持 INT4 和 FP4 的权重压缩。所有低精度（INT8/FP8等）类型都需要通过显式的量化-反量化层（Q/DQ） 来使用。自TensorRT 11.0起，强类型网络（Strongly Typed Networks）成为默认模式，旧版API已被移除。

动态形状 (Dynamic Shapes)：如果你的模型输入尺寸（如批量大小、图像尺寸）会变化，可以配置优化档案（Optimization Profiles），让 TensorRT 为不同的形状范围都生成优化的执行方案。

自定义层 (Plugins)：当遇到 TensorRT 原生不支持的算子时，你可以通过插件接口（Plugin Interface）自行实现。TensorRT内置了一个插件库，并提供了详细的插件编写指南。

多GPU与并行：支持在多GPU上并行化工作负载。可以通过 cudaSetDevice() 为 Builder 或 Engine 指定运行的GPU。新版本还增强了多设备推理（Multi-Device Inference）功能。

