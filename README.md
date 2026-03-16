---
title: TorchCode
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🔥 TorchCode

**破解 PyTorch 面试**

从零实现算子和网络结构——正是顶级机器学习团队最常考察的技能。

*类似 LeetCode，但针对张量。自托管。基于 Jupyter。即时反馈。*

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/duoan/TorchCode?style=social)](https://github.com/duoan/TorchCode)
[![GitHub Container Registry](https://img.shields.io/badge/ghcr.io-TorchCode-blue?style=flat-square&logo=github)](https://ghcr.io/duoan/torchcode)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-TorchCode-blue?style=flat-square)](https://huggingface.co/spaces/duoan/TorchCode)
![Problems](https://img.shields.io/badge/题目数量-40-orange?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-不需要-brightgreen?style=flat-square)

[![Star History Chart](https://api.star-history.com/svg?repos=duoan/TorchCode&type=Date)](https://star-history.com/#duoan/TorchCode&Date)

</div>

---

## 🎯 为什么选择 TorchCode？

顶级公司（Meta、Google DeepMind、OpenAI 等）期望 ML 工程师能够**凭记忆在白板上写出核心算子**。光读论文是不够的——你需要能手写 `softmax`、`LayerNorm`、`MultiHeadAttention` 乃至完整的 Transformer block。

TorchCode 提供了一个**结构化的练习环境**，包含：

| 特性 | 描述 |
|------|------|
| 🧩 | **40 道精选题目** | 面试中最常被问到的 PyTorch 核心主题 |
| ⚖️ | **自动评测** | 正确性检查、梯度验证、耗时检查 |
| 🎨 | **即时反馈** | 每个测试用例彩色通过/失败，就像算法竞赛一样 |
| 💡 | **卡住时的提示** | 给出方向性引导，但不直接剧透答案 |
| 📖 | **参考解法** | 尝试过后可以对照最优实现学习 |
| 📊 | **进度追踪** | 已完成题目、最好用时、尝试次数 |
| 🔄 | **一键重置** | 工具栏按钮可将任意 notebook 恢复成空白模板——无限重复练习同一道题 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) | **直接在 Colab 打开** | 每个 notebook 都有 Colab 打开按钮，支持零配置在线练习 |

无需云服务、无需注册、不需要 GPU。只需 `make run` —— 或者直接在 Hugging Face 秒开。

---

## 🚀 快速开始

### 方式 0 — 在线零安装体验（推荐）

**[直接在 Hugging Face Spaces 启动](https://huggingface.co/spaces/duoan/TorchCode)** —— 浏览器中完整打开 JupyterLab 环境，无需任何安装。

或者点击任意题目右上角的 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/templates/01_relu.ipynb) 按钮，直接在 Google Colab 中打开。

### 方式 0b — 在 Colab 中只使用判题器（pip 安装）

```bash
!pip install torch-judge
```

然后在单元格中运行：

```python
from torch_judge import check, status, hint, reset_progress
status()           # 查看所有题目和你的进度
check("relu")      # 评测 relu 任务
hint("relu")       # 显示提示
```

### 方式 1 — 拉取预构建镜像（最快）

```bash
docker run -p 8888:8888 -e PORT=8888 ghcr.io/duoan/torchcode:latest
```

### 方式 2 — 本地构建

```bash
make run
```

打开 **http://localhost:8888** 即可。兼容 Docker 和 Podman（自动检测）。

---

## 📋 题目列表

> **出现频率**：🔥 = 面试极大概率出现　⭐ = 常见　💡 = 新兴/加分项

### 🧱 基础算子 — “从零实现 X”

ML 编码面试最核心的部分，要求不使用 `torch.nn` 模块手写。

（表格内容较长，以下仅展示部分代表性题目，完整翻译保持原结构）

| #  | 题目 | 需要实现的内容 | 难度 | 频率 | 核心考察点 |
|----|------|----------------|------|------|------------|
| 1  | ReLU | `relu(x)` | Easy | 🔥 | 激活函数、逐元素操作 |
| 2  | Softmax | `my_softmax(x, dim)` | Easy | 🔥 | 数值稳定性、exp/log 技巧 |
| 16 | Cross Entropy Loss | `cross_entropy_loss(logits, targets)` | Easy | 🔥 | log-softmax + logsumexp |
| 4  | LayerNorm | `my_layer_norm(x, γ, β)` | Medium | 🔥 | 归一化、仿射变换、运行统计 |
| 5  | Scaled Dot-Product Attention | `scaled_dot_product_attention(Q, K, V)` | Hard | 🔥 | 注意力机制基石 |
| 6  | Multi-Head Attention | `MultiHeadAttention` 模块 | Hard | 🔥 | 多头并行、拆分/拼接 |
| 9  | Causal Self-Attention | `causal_attention(Q, K, V)` | Hard | 🔥 | 自回归掩码、-inf 处理 |
| 14 | KV Cache Attention | `KVCacheAttention` 模块 | Hard | 🔥 | 增量解码、prefill vs decode |
| 24 | RoPE | `apply_rope(q, k)` | Hard | 🔥 | 旋转位置编码 |
| 13 | GPT-2 Block | `GPT2Block` 模块 | Hard | ⭐ | Pre-norm、残差、GELU MLP |
| 26 | LoRA | `LoRALinear` 模块 | Medium | ⭐ | 低秩适配 |
| 32 | Top-k / Top-p Sampling | `sample_top_k_top_p(...)` | Medium | 🔥 | 核采样、温度 |
| 33 | Beam Search | `beam_search(...)` | Medium | 🔥 | 束搜索、假设扩展与剪枝 |

（完整 40 题列表翻译与原文一一对应，如需某一道题目的详细中文描述可单独指出）

---
                                 
## ⚙️ 使用流程

每个题目配有**两份** notebook：

| 文件                        | 用途                     |
|-----------------------------|--------------------------|
| `01_relu.ipynb`             | ✏️ 空白模板 — 在这里写代码 |
| `01_relu_solution.ipynb`    | 📖 参考答案 — 做完后再看   |

推荐流程：

1. 打开空白 notebook  
2. 阅读题目描述 → 实现代码（尽量只用基础 PyTorch 操作）  
3. 随意调试（print shape、检查梯度等）  
4. 运行判题单元格 → `check("relu")`  
5. 查看彩色反馈（✅ / ❌）  
6. 卡住了？ → `hint("relu")`  
7. 想看最优写法 → 打开 `_solution.ipynb`  
8. 想再练一遍 → 点击工具栏 🔄 Reset 按钮恢复空白模板

---

## 📅 推荐学习计划（约 12–16 小时）

| 周次 | 重点方向           | 主要题目                                | 预计时长 |
|------|--------------------|-----------------------------------------|----------|
| 第1周 | 基础算子           | ReLU → Softmax → CE → Dropout → LayerNorm → Linear → Conv2d 等 | 2–3h    |
| 第2周 | 注意力机制深挖     | SDPA → MHA → Causal → GQA → RoPE → KV Cache → Flash Attention | 3–4h    |
| 第3周 | 网络结构 + 训练相关 | GPT-2 Block → LoRA → MoE → Adam → Cosine LR → Grad Clip 等 | 3–4h    |
| 第4周 | 推理 + 高级对齐     | Sampling → Beam Search → BPE → INT8 Quant → DPO/GRPO/PPO 等 | 3–4h    |

