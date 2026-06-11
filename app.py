import os

import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_PATH = "D:/model_cache/modelscope/hub/models/Qwen/Qwen3-4B"
ADAPTER_PATH = "./output/qwen3_4b_sft"
SYSTEM_PROMPT = (
    "You are a professional underwater acoustics assistant. "
    "Answer only from established underwater acoustics principles; "
    "must not fabricate parameters, experiments, sea areas, or equipment capability. "
    "If frequency, depth, sea state, sound speed profile, array geometry, or source level "
    "is missing, state the uncertainty and give conditional guidance."
)
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 0.1,
    "top_p": 0.85,
}


app = Flask(__name__)
CORS(app)


def load_chat_model():
    base_model_path = os.getenv("BASE_MODEL_PATH", BASE_MODEL_PATH)
    adapter_path = os.getenv("ADAPTER_PATH", ADAPTER_PATH)

    print(f"Loading base model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and os.path.isdir(adapter_path):
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("LoRA adapter not found; serving the base model only.")

    model.eval()
    return tokenizer, model


tokenizer, model = load_chat_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"response": "请先输入一个水声相关问题。"}), 400

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **GENERATION_CONFIG,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

