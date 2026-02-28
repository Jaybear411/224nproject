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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, type=str)
  parser.add_argument("--train_out", required=True, type=str)
  parser.add_argument("--dev_out", required=True, type=str)
  parser.add_argument("--train_ratio", default=0.8, type=float)
  parser.add_argument("--seed", default=11711, type=int)
  args = parser.parse_args()

  rows = list(read_jsonl(Path(args.input)))
  if len(rows) < 2:
    raise RuntimeError("Need at least 2 rows to create train/dev split.")

  rng = random.Random(args.seed)
  rng.shuffle(rows)

  split_idx = max(1, min(len(rows) - 1, int(len(rows) * args.train_ratio)))
  train_rows = rows[:split_idx]
  dev_rows = rows[split_idx:]

  write_jsonl(Path(args.train_out), train_rows)
  write_jsonl(Path(args.dev_out), dev_rows)
  print(f"Wrote {len(train_rows)} train and {len(dev_rows)} dev examples.")


if __name__ == "__main__":
  main()
