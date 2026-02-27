#!/usr/bin/env python3
import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def load_config(path: Path):
  if path.suffix in {".yaml", ".yml"}:
    try:
      import yaml
    except ImportError as exc:
      raise RuntimeError("pyyaml is required for YAML configs.") from exc
    with path.open("r", encoding="utf-8") as f:
      return yaml.safe_load(f)
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def read_jsonl(path: Path):
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)


@dataclass
class Example:
  text: str


def build_examples(formatted_path: Path):
  rows = list(read_jsonl(formatted_path))
  return [Example(text=row["prompt"] + row["target_json"]) for row in rows]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, type=str)
  args = parser.parse_args()

  cfg = load_config(Path(args.config))
  seed_everything(int(cfg.get("seed", 11711)))

  run_dir = Path(cfg.get("output_root", "tool_calling/outputs/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")
  ckpt_dir = run_dir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  try:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
  except ImportError as exc:
    raise RuntimeError("Install transformers and datasets to run training.") from exc

  model_name = cfg.get("model_name", "gpt2")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  model_kwargs = {}
  use_qlora = bool(cfg.get("use_qlora", True))
  if use_qlora:
    try:
      from peft import LoraConfig, get_peft_model
      import bitsandbytes  # noqa: F401
      model_kwargs["load_in_4bit"] = True
      model_kwargs["device_map"] = "auto"
    except ImportError:
      use_qlora = False

  model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
  if use_qlora:
    lcfg = cfg.get("lora", {})
    peft_cfg = LoraConfig(
      r=int(lcfg.get("r", 8)),
      lora_alpha=int(lcfg.get("alpha", 16)),
      lora_dropout=float(lcfg.get("dropout", 0.05)),
      bias="none",
      task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

  examples = build_examples(Path(cfg["train_path"]))
  raw_ds = Dataset.from_dict({"text": [ex.text for ex in examples]})

  def tokenize_batch(batch):
    toks = tokenizer(
      batch["text"],
      truncation=True,
      padding="max_length",
      max_length=int(cfg.get("max_length", 512)),
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks

  train_ds = raw_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
  train_ds.set_format(type="torch")

  training_args = TrainingArguments(
    output_dir=str(ckpt_dir),
    per_device_train_batch_size=int(cfg.get("batch_size", 2)),
    num_train_epochs=float(cfg.get("epochs", 1)),
    learning_rate=float(cfg.get("learning_rate", 5e-5)),
    save_strategy="epoch",
    logging_steps=10,
    report_to=[],
    fp16=torch.cuda.is_available(),
  )

  trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
  trainer.train()
  trainer.save_model(str(ckpt_dir / "final"))

  with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)
  print(f"Training complete. Artifacts in {run_dir}")


if __name__ == "__main__":
  main()
