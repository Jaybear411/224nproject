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
  parser.add_argument("--input", required=True, type=str, help="Formatted JSONL (from format_prompts.py)")
  parser.add_argument("--out", required=True, type=str, help="predictions.jsonl path")
  parser.add_argument("--max_new_tokens", default=64, type=int)
  args = parser.parse_args()

  from transformers import AutoModelForCausalLM, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
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
      full = tokenizer.decode(gen[0], skip_special_tokens=True)
      output = full[len(prompt):].strip()
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
        "target_call": row.get("target_call"),
      }, sort_keys=True) + "\n")

  print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
  main()
