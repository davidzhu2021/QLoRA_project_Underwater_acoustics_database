# 基于Qwen3-0.6B的水声领域问答微调系统

> 面向水声专业知识问答场景，通过 SFT + DPO 对齐提升模型领域回答质量与规范性。

---

## 项目背景

实验室新加入的师弟师妹对水声领域知识尚不熟悉，在出海实验或项目开发过程中，面对声呐参数设置、水声传播现象等专业问题时缺乏快速查询的工具。通用大模型（如 ChatGPT）对水声这类细分领域的回答质量参差不齐，存在误导风险。

本项目在开源基座模型 Qwen3-0.6B 上进行领域微调，打造一个专注于水声知识的专业问答助手，供实验室成员以及各位水声科研爱好者在出海等网络受限场景下离线使用。

---

## 技术栈

- **语言 / 框架**：Python / PyTorch
- **微调框架**：LLaMA Factory
- **微调方法**：QLoRA（4bit 量化 + LoRA）/ SFT / DPO
- **基座模型**：Qwen3-0.6B
- **训练硬件**：NVIDIA RTX 3050（4GB 显存）

---

## 项目结构

```
LLaMA-Factory/
├── sft_config.yaml              # SFT 训练配置
├── dpo_config.yaml              # DPO 训练配置
├── eval_model.py                # 评估脚本（BLEU、合规率等）
├── test.py                      # 快速推理测试
├── data/
│   ├── under_acoustic_data/
│   │   └── dataset_underwater.json   # 5000 条 SFT 训练数据
│   ├── dpo_underwater.json           # DPO 偏好对齐数据
│   └── dataset_info.json             # 数据集注册配置
└── output/
    ├── qwen3_sft/               # SFT 训练产出的 LoRA 权重
    └── qwen3_dpo/               # DPO 训练产出的 LoRA 权重
```

---

## 方法说明

### 数据构建

- 构造并清洗 5000 条水声领域 SFT 训练数据，覆盖声呐原理、水声传播、多途效应、混响、多普勒效应、目标探测等主题
- 通过长度过滤、关键词校验、格式规范化等清洗规则，将有效数据占比从 45% 提升至 78%
- 引入 GPT-4 辅助标注，标注一致性达 88%

### SFT 监督微调

采用 QLoRA 方案（4bit 量化 + rank=8），在 RTX 3050 单卡上完成训练：

- 显存占用从全参数微调的 6GB 压缩至 3GB
- 训练轮数：3 epochs，训练时长约 1 小时 40 分钟
- 可训练参数量：1,146,880（占全部参数的 0.19%）

### DPO 偏好对齐

针对 SFT 后模型偶尔出现的口语化、不规范输出，构造 chosen/rejected 偏好数据对进行对齐训练：

- rejected：SFT 模型输出的短而模糊的回答
- chosen：术语准确、逻辑完整的专业回答
- 训练轮数：1 epoch

### 选择微调而非 RAG 的原因

| 维度 | 微调（本项目） | RAG |
|------|--------------|-----|
| 知识更新频率 | 低，水声知识相对稳定 | 适合频繁更新的知识库 |
| 输出风格控制 | 强，可通过 DPO 对齐 | 弱 |
| 推理延迟 | 低，无检索开销 | 高，需额外检索步骤 |
| 离线使用 | 支持 | 依赖网络检索 |

---

## 快速开始

### 环境安装

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install bitsandbytes
```

### SFT 训练

```bash
llamafactory-cli train sft_config.yaml
```

### DPO 训练

```bash
llamafactory-cli train dpo_config.yaml
```

### 交互式对话（推理）

```bash
llamafactory-cli chat \
  --model_name_or_path <基座模型路径> \
  --adapter_name_or_path ./output/qwen3_dpo \
  --template qwen \
  --finetuning_type lora
```

### 评估

```bash
python eval_model.py
```

---

## 核心配置参数

```yaml
# QLoRA 关键参数
quantization_bit: 4        # 4bit 量化，显存减半
lora_rank: 8               # LoRA 秩，平衡效果与显存
lora_alpha: 16             # 缩放系数 = alpha/rank = 2
lora_target: q_proj,v_proj # 在注意力 Q/V 矩阵上插入 LoRA

# 显存优化
gradient_checkpointing: true
optim: adamw_8bit
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # 等效 batch_size = 16
```

---

## 评估指标

| 指标 | 评估方法 |
|------|---------|
| BLEU | nltk sentence_bleu，与标准答案做 n-gram 匹配 |
| 人工准确率 | 随机抽取输出，按专业术语正确性和逻辑完整性打分 |
| 格式合规率 | 规则脚本检测输出是否有完整结句 |
| 非专业输出率 | 检测口语化表述和过短输出 |

---

## 局限性与后续优化方向

- 模型参数量仅 0.6B，回答深度有限；可替换为 7B 基座模型提升效果
- DPO 数据量较少，对齐效果有提升空间；扩充至 200 条以上效果会更显著
- 当前评估为单人人工评估，存在主观偏差；后续可引入多评估者 + Cohen's Kappa 一致性检验
- 可结合 RAG 方案，在知识库扩充时提供更好的可扩展性

---

## 参考资料

- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [Qwen3 技术报告](https://huggingface.co/Qwen)