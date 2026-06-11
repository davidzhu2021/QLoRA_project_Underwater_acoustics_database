# Underwater Acoustics Qwen3 SFT

This project fine-tunes a Qwen3 model for underwater acoustics QA. The current
profile targets an RTX 4090 24GB and prioritizes better domain quality and lower
hallucination risk over minimum VRAM usage.

## Current Training Profile

- Base model: `Qwen/Qwen3-4B`
- Fine-tuning: bf16 LoRA, not 4-bit QLoRA
- LoRA: `rank=32`, `alpha=64`, `target=all`
- Output: `./output/qwen3_4b_sft`
- Dataset: `data/under_acoustic_data/dataset_underwater.json`
- Fixed evaluation set: `data/under_acoustic_data/eval_underwater_qa.json`

The old `Qwen3-0.6B` 4-bit profile is kept only in historical outputs. It is not
recommended for the final professional QA model.

## Setup

Install LLaMA Factory and runtime dependencies:

```bash
pip install llamafactory transformers peft datasets accelerate trl bitsandbytes
pip install flask flask-cors modelscope
```

Download the 4B model to the path used by `sft_config.yaml`:

```bash
python tools/download_qwen3_4b.py
```

Expected model directory:

```text
D:/model_cache/modelscope/hub/models/Qwen/Qwen3-4B
```

## Train

```bash
llamafactory-cli train sft_config.yaml
```

Do not run DPO as the main improvement path yet. The current preference dataset
is too small for reliable alignment. After expanding DPO data to at least a few
hundred high-quality pairs, run:

```bash
llamafactory-cli train dpo_config.yaml
```

## Evaluate

Run a quick SFT comparison:

```bash
python eval_model.py ^
  --base-model D:/model_cache/modelscope/hub/models/Qwen/Qwen3-4B ^
  --adapter sft=./output/qwen3_4b_sft ^
  --limit 100
```

The script reports:

- keyword coverage
- risky over-assertion rate
- uncertainty/condition-awareness rate
- 30 samples for manual review

Manual review should label at least 30 answers as: correct, partially correct,
hallucinated, too vague, or inappropriate refusal.

## Serve

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

`app.py` uses deterministic/low-randomness decoding by default and a system
prompt that asks the model not to fabricate parameters, experiments, sea areas,
or equipment capability.

## Validation

Run the local quality pipeline checks:

```bash
python -m unittest tests.test_quality_pipeline -v
```

These checks ensure:

- SFT config points to the 4090 4B bf16 LoRA profile
- the fixed 100-item evaluation set exists and has the expected schema
- `dpt.json` is valid JSON
- Flask inference defaults use low hallucination settings

