"""Exact numeric GSM8K scoring with a fixed last-number extraction rule."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


def number(text):
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return Decimal(matches[-1]) if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1319)
    args = parser.parse_args()
    tests = [json.loads(line) for line in args.tests.read_text().splitlines()]
    requests = [json.loads(line) for line in Path("data/gsm8k.jsonl").read_text().splitlines()]
    assert len(tests) == len(requests) == 1319
    for request, test in zip(requests, tests):
        assert request["messages"][0]["content"].endswith(test["question"])
    results = []
    seen = set()
    for row in map(json.loads, args.responses.read_text().splitlines()):
        index = row["index"]
        assert index not in seen and row["metadata"]["task_id"] == f"gsm8k/{index}"
        seen.add(index)
        expected = number(tests[index]["answer"].split("####")[-1])
        actual = number(row["response"]["choices"][0]["message"]["content"])
        results.append(dict(index=index, passed=actual == expected,
                            expected=str(expected), actual=str(actual),
                            finish_reason=row["response"]["choices"][0]["finish_reason"]))
    passed = sum(r["passed"] for r in results)
    assert len(results) == args.expected_count, "Incomplete run: refusing to publish a final score"
    report = dict(passed=passed, total=len(results), accuracy=passed/len(results),
                  length_limited=sum(r["finish_reason"] == "length" for r in results),
                  extraction="last signed decimal after comma removal; no LLM judge",
                  tests_sha256=hashlib.sha256(args.tests.read_bytes()).hexdigest(), results=results)
    args.responses.with_suffix(".scores.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
