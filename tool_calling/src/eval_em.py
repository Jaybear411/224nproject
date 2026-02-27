#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def canonicalize(obj):
  return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path):
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--pred", required=True, type=str)
  parser.add_argument("--out", required=False, type=str)
  args = parser.parse_args()

  rows = list(read_jsonl(Path(args.pred)))
  match = 0
  for row in rows:
    pred = row.get("parsed_output")
    gold = row.get("target_call")
    if not isinstance(pred, dict) or not isinstance(gold, dict):
      continue
    same_name = pred.get("name") == gold.get("name")
    same_args = canonicalize(pred.get("arguments", {})) == canonicalize(gold.get("arguments", {}))
    if same_name and same_args:
      match += 1

  total = len(rows)
  metrics = {"exact_match_rate": match / total if total else 0.0, "matches": match, "total": total}
  print(json.dumps(metrics, indent=2, sort_keys=True))
  if args.out:
    with Path(args.out).open("w", encoding="utf-8") as f:
      json.dump(metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
  main()
