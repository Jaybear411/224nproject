#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def canonicalize_json(obj):
  return json.loads(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def read_jsonl(path: Path) -> Iterable[Dict]:
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)


def is_unambiguous(example: Dict) -> bool:
  if example.get("ambiguous", False):
    return False
  if not example.get("instruction"):
    return False
  if not example.get("tool_schema"):
    return False
  call = example.get("target_call", {})
  return isinstance(call, dict) and "name" in call and "arguments" in call


def normalize_example(example: Dict, idx: int) -> Dict:
  ex_id = example.get("id", f"ex-{idx:06d}")
  normalized = {
    "id": ex_id,
    "instruction": example["instruction"],
    "tool_schema": canonicalize_json(example["tool_schema"]),
    "target_call": canonicalize_json(example["target_call"]),
  }
  return normalized


def write_jsonl(path: Path, rows: List[Dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=str, required=True, help="Input JSONL with raw tool-call examples")
  parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
  args = parser.parse_args()

  input_path = Path(args.input)
  output_path = Path(args.output)
  rows = []
  dropped = 0
  for i, example in enumerate(read_jsonl(input_path)):
    if not is_unambiguous(example):
      dropped += 1
      continue
    rows.append(normalize_example(example, i))

  write_jsonl(output_path, rows)
  print(f"Wrote {len(rows)} unambiguous examples to {output_path}. Dropped {dropped}.")


if __name__ == "__main__":
  main()
