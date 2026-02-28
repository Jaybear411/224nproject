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


def _json_type_name(value):
  if isinstance(value, bool):
    return "boolean"
  if isinstance(value, int):
    return "integer"
  if isinstance(value, float):
    return "number"
  if isinstance(value, str):
    return "string"
  if isinstance(value, dict):
    return "object"
  if isinstance(value, list):
    return "array"
  return "null"


def _expected_type(type_spec):
  if isinstance(type_spec, list):
    return set(type_spec)
  if isinstance(type_spec, str):
    return {type_spec}
  return set()


def _type_matches(value, expected_types):
  if not expected_types:
    return True
  value_type = _json_type_name(value)
  if value_type == "integer" and "number" in expected_types:
    return True
  return value_type in expected_types


def validate_row(row):
  parsed = row.get("parsed_output")
  target = row.get("target_call")
  schema = row.get("tool_schema")
  if parsed is None:
    return False, "invalid_json"
  if not isinstance(parsed, dict):
    return False, "not_object"
  for k in ["name", "arguments"]:
    if k not in parsed:
      return False, "missing_key"
  extra = set(parsed.keys()) - {"name", "arguments"}
  if extra:
    return False, "extra_key"
  if not isinstance(parsed["arguments"], dict):
    return False, "wrong_type"

  # Prefer schema-based validation when schema is available.
  if isinstance(schema, dict):
    schema_name = schema.get("name")
    if isinstance(schema_name, str) and parsed.get("name") != schema_name:
      return False, "wrong_tool_name"
    params = schema.get("parameters", {})
    if isinstance(params, dict):
      properties = params.get("properties", {})
      required = params.get("required", [])
      if isinstance(required, list):
        for req_key in required:
          if req_key not in parsed["arguments"]:
            return False, "missing_required_arg"
      if isinstance(properties, dict):
        for key, value in parsed["arguments"].items():
          if key not in properties:
            return False, "unknown_argument"
          expected = _expected_type(properties[key].get("type") if isinstance(properties[key], dict) else None)
          if not _type_matches(value, expected):
            return False, "wrong_arg_type"
  elif isinstance(target, dict):
    # Fallback when schema isn't available.
    if parsed.get("name") != target.get("name"):
      return False, "wrong_tool_name"
    target_args = target.get("arguments", {})
    if isinstance(target_args, dict):
      for key, val in target_args.items():
        if key not in parsed["arguments"]:
          return False, "missing_key"
        if _json_type_name(parsed["arguments"][key]) != _json_type_name(val):
          return False, "wrong_type"
  return True, "ok"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--pred", required=True, type=str)
  parser.add_argument("--out", required=False, type=str)
  args = parser.parse_args()

  rows = list(read_jsonl(Path(args.pred)))
  errors = {}
  passed = 0
  for row in rows:
    ok, tag = validate_row(row)
    if ok:
      passed += 1
    else:
      errors[tag] = errors.get(tag, 0) + 1

  total = len(rows)
  metrics = {
    "schema_adherence_rate": passed / total if total else 0.0,
    "passes": passed,
    "total": total,
    "error_taxonomy": errors,
  }
  print(json.dumps(metrics, indent=2, sort_keys=True))
  if args.out:
    with Path(args.out).open("w", encoding="utf-8") as f:
      json.dump(metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
  main()
