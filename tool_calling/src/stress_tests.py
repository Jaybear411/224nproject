#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)


def write_jsonl(path: Path, rows):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")


def perturb_schema(schema, mode, seed):
  random.seed(seed)
  out = json.loads(json.dumps(schema))
  if not isinstance(out, dict):
    return out
  if mode == "reorder":
    return json.loads(json.dumps(out, sort_keys=False))
  if mode == "rename_params":
    params = out.get("parameters", {})
    props = params.get("properties", {})
    renamed = {}
    for k, v in props.items():
      renamed[f"{k}_new"] = v
    if renamed:
      params["properties"] = renamed
      if "required" in params:
        params["required"] = [f"{x}_new" for x in params["required"]]
  if mode == "unseen_tool":
    out["name"] = f"{out.get('name', 'tool')}_unseen"
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, type=str, help="Processed JSONL")
  parser.add_argument("--out_dir", required=True, type=str)
  parser.add_argument("--seed", default=11711, type=int)
  args = parser.parse_args()

  rows = list(read_jsonl(Path(args.input)))
  out_dir = Path(args.out_dir)

  for mode in ["rename_params", "reorder", "unseen_tool"]:
    perturbed = []
    for i, row in enumerate(rows):
      new_row = dict(row)
      new_row["tool_schema"] = perturb_schema(row.get("tool_schema", {}), mode, args.seed + i)
      if mode == "underspecified_instruction":
        new_row["instruction"] = "Do the task."
      perturbed.append(new_row)
    write_jsonl(out_dir / f"{mode}.jsonl", perturbed)

  underspecified = []
  for row in rows:
    new_row = dict(row)
    new_row["instruction"] = "Do the task."
    underspecified.append(new_row)
  write_jsonl(out_dir / "underspecified_instruction.jsonl", underspecified)
  print(f"Wrote stress test files to {out_dir}")


if __name__ == "__main__":
  main()
