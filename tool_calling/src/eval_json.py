#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


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
  valid = sum(1 for row in rows if row.get("parsed_output") is not None)
  total = len(rows)
  metrics = {
    "json_validity_rate": valid / total if total else 0.0,
    "valid": valid,
    "total": total,
  }
  print(json.dumps(metrics, indent=2, sort_keys=True))
  if args.out:
    with Path(args.out).open("w", encoding="utf-8") as f:
      json.dump(metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
  main()
