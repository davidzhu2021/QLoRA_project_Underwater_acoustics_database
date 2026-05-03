import transformers
import torch

# 使用 snapshot_download 获取正确本地路径，避免 ___ 路径问题
from modelscope import snapshot_download
model_id = snapshot_download("Qwen/Qwen3-0.6B")

# 先加载 tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs={"dtype": torch.bfloat16},  # torch_dtype 已废弃，改用 dtype
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Qwen3 的结束符是 <|im_end|>，不是 LLaMA 的 <|eot_id|>
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>")
]
# 过滤掉 None 值，防止 eos_token_id 为 None 时报错
terminators = [t for t in terminators if t is not None]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    pad_token_id=pipeline.tokenizer.eos_token_id,  # 防止 pad_token 为 None
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print(outputs[0]["generated_text"][len(prompt):])