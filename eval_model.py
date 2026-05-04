from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import json

# 加载模型
base_model_path = "C:/Users/98689/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B"
sft_adapter_path = "./output/qwen3_sft"
dpo_adapter_path = "./output/qwen3_dpo"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

def load_model(adapter_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model

def generate(model, question, max_new_tokens=200):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# 测试集
test_data = [
    {"question": "主动声呐和被动声呐有什么区别？",
     "answer": "主动声呐通过主动发射声脉冲并接收目标回波来探测水下目标，优点是探测距离远、定位精度高，缺点是容易暴露自身位置。被动声呐只接收目标辐射的噪声信号，不主动发射，隐蔽性好，常用于潜艇作战。"},
    {"question": "什么是水声信道的多途效应？",
     "answer": "多途效应是指声波从声源出发后，经海面反射、海底反射、体积散射等多条路径到达接收端的现象，导致接收信号产生时延扩展，引起码间干扰。"},
    {"question": "匹配滤波的原理是什么？",
     "answer": "匹配滤波是一种最优线性滤波器，其冲激响应与期望信号的时间反转共轭相匹配，在输出信噪比最大化的准则下实现信号检测。"},
    {"question": "声速剖面对水声传播有什么影响？",
     "answer": "声速剖面决定了声波在水中的折射规律，影响声线弯曲方向和会聚区的形成，是水声传播预报和声呐性能评估的重要参数。"},
    {"question": "什么是混响？",
     "answer": "混响是指声呐发射信号后，由海面、海底及海水体积散射体返回的干扰回波的总和，是主动声呐的主要干扰来源之一。"},
]

def calc_bleu(pred, ref):
    smoothie = SmoothingFunction().method1
    return sentence_bleu([list(ref)], list(pred), smoothing_function=smoothie)

def calc_nonpro_rate(outputs):
    """非专业输出率：口语化、不确定、过短"""
    bad_patterns = ['我觉得', '可能', '大概', '也许', '不知道', '不清楚', '不太确定']
    count = sum(1 for o in outputs if any(p in o for p in bad_patterns) or len(o) < 20)
    return count / len(outputs)

def calc_format_compliance(outputs):
    """格式合规率：有完整句子结尾"""
    count = sum(1 for o in outputs if o.endswith('。') or o.endswith('.'))
    return count / len(outputs)

# 评估SFT模型
print("加载SFT模型...")
sft_model = load_model(sft_adapter_path)

sft_outputs = []
sft_bleu_scores = []
for item in test_data:
    pred = generate(sft_model, item['question'])
    sft_outputs.append(pred)
    sft_bleu_scores.append(calc_bleu(pred, item['answer']))
    print(f"Q: {item['question']}")
    print(f"A: {pred}\n")

print("=== SFT评估结果 ===")
print(f"BLEU均值: {sum(sft_bleu_scores)/len(sft_bleu_scores):.4f}")
print(f"非专业输出率: {calc_nonpro_rate(sft_outputs):.2%}")
print(f"格式合规率: {calc_format_compliance(sft_outputs):.2%}")

# 释放显存
del sft_model
torch.cuda.empty_cache()

# 评估DPO模型
print("\n加载DPO模型...")
dpo_model = load_model(dpo_adapter_path)

dpo_outputs = []
dpo_bleu_scores = []
for item in test_data:
    pred = generate(dpo_model, item['question'])
    dpo_outputs.append(pred)
    dpo_bleu_scores.append(calc_bleu(pred, item['answer']))

print("=== DPO评估结果 ===")
print(f"BLEU均值: {sum(dpo_bleu_scores)/len(dpo_bleu_scores):.4f}")
print(f"非专业输出率: {calc_nonpro_rate(dpo_outputs):.2%}")
print(f"格式合规率: {calc_format_compliance(dpo_outputs):.2%}")