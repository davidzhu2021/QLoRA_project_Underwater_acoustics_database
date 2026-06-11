import argparse
import json
from pathlib import Path


DEFAULT_BASE_MODEL = "D:/model_cache/modelscope/hub/models/Qwen/Qwen3-4B"
DEFAULT_EVAL_SET = "data/under_acoustic_data/eval_underwater_qa.json"
SYSTEM_PROMPT = (
    "You are a professional underwater acoustics assistant. "
    "Answer only from established underwater acoustics principles; must not fabricate "
    "parameters, experiments, sea areas, or equipment capability. If key conditions are "
    "missing, state the uncertainty and answer conditionally."
)
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 0.1,
    "top_p": 0.85,
}
RISK_PATTERNS = [
    "一定",
    "必然",
    "完全准确",
    "任何情况下",
    "固定为",
    "探测距离为",
    "识别概率为",
    "无需考虑",
]
UNCERTAINTY_PATTERNS = [
    "取决于",
    "需要",
    "条件",
    "不确定",
    "不能",
    "缺少",
    "如果",
]


def parse_adapter_args(raw_adapters):
    adapters = []
    for item in raw_adapters:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).name
        adapters.append((name, path))
    return adapters


def load_eval_set(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"id", "question", "reference", "keywords", "risk_flags"}
    for item in data:
        if set(item) != required:
            raise ValueError(f"Bad eval item schema: {item}")
    return data


def load_model(base_model_path, adapter_path=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question):
    import torch

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
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def score_answer(answer, keywords):
    keyword_hits = sum(1 for keyword in keywords if keyword in answer)
    risk_hits = [pattern for pattern in RISK_PATTERNS if pattern in answer]
    uncertainty_hits = [pattern for pattern in UNCERTAINTY_PATTERNS if pattern in answer]
    return {
        "keyword_coverage": keyword_hits / max(len(keywords), 1),
        "risk_hits": risk_hits,
        "mentions_uncertainty": bool(uncertainty_hits),
        "length": len(answer),
    }


def evaluate_model(name, base_model_path, adapter_path, eval_set, limit):
    import torch

    tokenizer, model = load_model(base_model_path, adapter_path)
    rows = []
    selected = eval_set[:limit] if limit else eval_set

    for item in selected:
        answer = generate_answer(tokenizer, model, item["question"])
        score = score_answer(answer, item["keywords"])
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                "reference": item["reference"],
                **score,
            }
        )

    del model
    torch.cuda.empty_cache()
    return summarize(name, rows), rows


def summarize(name, rows):
    if not rows:
        return {"model": name, "count": 0}
    return {
        "model": name,
        "count": len(rows),
        "avg_keyword_coverage": round(
            sum(row["keyword_coverage"] for row in rows) / len(rows), 4
        ),
        "risk_rate": round(sum(bool(row["risk_hits"]) for row in rows) / len(rows), 4),
        "uncertainty_rate": round(
            sum(row["mentions_uncertainty"] for row in rows) / len(rows), 4
        ),
        "avg_answer_chars": round(sum(row["length"] for row in rows) / len(rows), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate underwater acoustics QA quality.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="output/eval_underwater_report.json")
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    runs = [("base", None)] + parse_adapter_args(args.adapter)
    report = {"summaries": [], "samples": {}}

    for name, adapter_path in runs:
        summary, rows = evaluate_model(name, args.base_model, adapter_path, eval_set, args.limit)
        report["summaries"].append(summary)
        report["samples"][name] = rows[:30]
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()

