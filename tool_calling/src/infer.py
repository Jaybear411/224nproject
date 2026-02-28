#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def read_jsonl(path: Path):
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt", required=True, type=str)
  parser.add_argument("--base_model", required=False, type=str, default=None,
                      help="Optional base model for loading LoRA adapter checkpoints")
  parser.add_argument("--input", required=True, type=str, help="Formatted JSONL (from format_prompts.py)")
  parser.add_argument("--out", required=True, type=str, help="predictions.jsonl path")
  parser.add_argument("--max_new_tokens", default=64, type=int)
  args = parser.parse_args()

  from transformers import AutoModelForCausalLM, AutoTokenizer
  ckpt_path = Path(args.ckpt)
  is_adapter = (ckpt_path / "adapter_config.json").exists()

  tokenizer_source = args.base_model if (is_adapter and args.base_model) else args.ckpt
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  if is_adapter:
    if not args.base_model:
      raise RuntimeError("--base_model is required when --ckpt points to a LoRA adapter.")
    try:
      from peft import PeftModel
    except ImportError as exc:
      raise RuntimeError("Install peft to run inference with LoRA adapters.") from exc
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.ckpt)
  else:
    model = AutoModelForCausalLM.from_pretrained(args.ckpt)

  model.eval()
  if torch.cuda.is_available():
    model = model.to("cuda")

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  with out_path.open("w", encoding="utf-8") as out_f:
    for row in read_jsonl(Path(args.input)):
      prompt = row["prompt"]
      toks = tokenizer(prompt, return_tensors="pt")
      toks = {k: v.to(model.device) for k, v in toks.items()}
      with torch.no_grad():
        gen = model.generate(**toks, max_new_tokens=args.max_new_tokens, do_sample=False)
      prompt_len = toks["input_ids"].shape[1]
      generated_ids = gen[0][prompt_len:]
      output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
      parsed = None
      error = None
      try:
        parsed = json.loads(output)
      except Exception as exc:
        error = str(exc)
      out_f.write(json.dumps({
        "id": row["id"],
        "prompt": prompt,
        "output": output,
        "parsed_output": parsed,
        "error": error,
        "tool_schema": row.get("tool_schema"),
        "target_call": row.get("target_call"),
      }, sort_keys=True) + "\n")

  print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
  main()
