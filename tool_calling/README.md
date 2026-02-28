# Tool Calling QLoRA Milestone

This folder contains a minimal end-to-end pipeline for tool-calling fine-tuning with GPT-2 + QLoRA.

## 1) Prepare toy dataset (quick milestone run)

```bash
python tool_calling/src/build_dataset.py \
  --input tool_calling/data/raw/toy_tool_calls.jsonl \
  --output tool_calling/data/processed/all.jsonl

python tool_calling/src/split_dataset.py \
  --input tool_calling/data/processed/all.jsonl \
  --train_out tool_calling/data/processed/train.jsonl \
  --dev_out tool_calling/data/processed/dev.jsonl \
  --train_ratio 0.8 \
  --seed 11711

python tool_calling/src/format_prompts.py \
  --input tool_calling/data/processed/train.jsonl \
  --output tool_calling/data/formatted/train.jsonl

python tool_calling/src/format_prompts.py \
  --input tool_calling/data/processed/dev.jsonl \
  --output tool_calling/data/formatted/dev.jsonl
```

## 2) Prompt-only baseline (no fine-tuning)

```bash
python tool_calling/src/infer.py \
  --ckpt gpt2 \
  --input tool_calling/data/formatted/dev.jsonl \
  --out tool_calling/outputs/preds/dev_prompt_only.jsonl \
  --max_new_tokens 64

python tool_calling/src/eval_json.py \
  --pred tool_calling/outputs/preds/dev_prompt_only.jsonl \
  --out tool_calling/outputs/metrics/dev_prompt_only_json.json

python tool_calling/src/eval_schema.py \
  --pred tool_calling/outputs/preds/dev_prompt_only.jsonl \
  --out tool_calling/outputs/metrics/dev_prompt_only_schema.json

python tool_calling/src/eval_em.py \
  --pred tool_calling/outputs/preds/dev_prompt_only.jsonl \
  --out tool_calling/outputs/metrics/dev_prompt_only_em.json
```

## 3) QLoRA fine-tuning

```bash
python tool_calling/src/train_qlora.py \
  --config tool_calling/configs/gpt2_qlora_toy.yaml
```

After training, identify the latest run folder under `tool_calling/outputs/runs/` and use:

`tool_calling/outputs/runs/<RUN_ID>/checkpoints/final`

as checkpoint path for inference.

## 4) Evaluate fine-tuned model

```bash
python tool_calling/src/infer.py \
  --ckpt tool_calling/outputs/runs/<RUN_ID>/checkpoints/final \
  --input tool_calling/data/formatted/dev.jsonl \
  --out tool_calling/outputs/preds/dev_qlora.jsonl \
  --max_new_tokens 64

python tool_calling/src/eval_json.py \
  --pred tool_calling/outputs/preds/dev_qlora.jsonl \
  --out tool_calling/outputs/metrics/dev_qlora_json.json

python tool_calling/src/eval_schema.py \
  --pred tool_calling/outputs/preds/dev_qlora.jsonl \
  --out tool_calling/outputs/metrics/dev_qlora_schema.json

python tool_calling/src/eval_em.py \
  --pred tool_calling/outputs/preds/dev_qlora.jsonl \
  --out tool_calling/outputs/metrics/dev_qlora_em.json
```

## 5) Optional stress tests

```bash
python tool_calling/src/stress_tests.py \
  --input tool_calling/data/processed/dev.jsonl \
  --out_dir tool_calling/data/stress
```

Then run `format_prompts.py` + `infer.py` + eval scripts on each stress file.

## Notes

- `eval_schema.py` now validates against `tool_schema` when available.
- `infer.py` can load either full checkpoints or LoRA adapter checkpoints (`--base_model` required for adapter-only checkpoints).
- Ensure dependencies are installed: `transformers`, `datasets`, `peft`, `bitsandbytes`, `pyyaml`.
