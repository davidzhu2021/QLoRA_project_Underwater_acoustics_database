---
library_name: peft
license: other
base_model: C:/Users/98689/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B
tags:
- base_model:adapter:C:/Users/98689/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3_dpo
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3_dpo

This model is a fine-tuned version of [C:/Users/98689/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B](https://huggingface.co/C:/Users/98689/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B) on the underwater_dpo dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_8BIT with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 1

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2