#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

PROMPT_TEMPLATE = """You are given a tool specification and a user instruction.
Return a single JSON object with exactly these keys: "name", "arguments".
Do not output any other text.

TOOL_SPEC:
{tool_spec}

INSTRUCTION:
{instruction}
"""


def read_jsonl(path: Path):
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, type=str)
  parser.add_argument("--output", required=True, type=str)
  args = parser.parse_args()

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)

  with output.open("w", encoding="utf-8") as f:
    for row in read_jsonl(Path(args.input)):
      prompt = PROMPT_TEMPLATE.format(
        tool_spec=json.dumps(row["tool_schema"], sort_keys=True),
        instruction=row["instruction"],
      )
      target = json.dumps(row["target_call"], sort_keys=True)
      f.write(json.dumps({
        "id": row["id"],
        "prompt": prompt,
        "tool_schema": row["tool_schema"],
        "target_call": row["target_call"],
        "target_json": target,
      }, sort_keys=True) + "\n")

  print(f"Wrote formatted prompts to {output}")


if __name__ == "__main__":
  main()
