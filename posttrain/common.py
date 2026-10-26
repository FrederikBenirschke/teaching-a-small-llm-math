"""System prompt, answer extraction and grader used by all scripts."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM = (
    "Solve the math problem step by step. "
    "End your response with a final line of the form 'Answer: <number>'."
)

_TAGGED = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def extract_answer(text: str) -> str | None:
    """Last 'Answer: N' if present, else the last number in the text."""
    tagged = _TAGGED.findall(text)
    found = tagged[-1] if tagged else (_NUMBER.findall(text) or [None])[-1]
    if found is None:
        return None
    found = found.replace(",", "").rstrip(".")
    return found or None


def gold_answer(gsm8k_solution: str) -> str:
    """GSM8K solutions end with '#### <number>'."""
    return gsm8k_solution.split("####")[-1].strip().replace(",", "")


def is_correct(completion: str, gold: str) -> bool:
    pred = extract_answer(completion)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def chat_prompt(tokenizer, question: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=True,
    )


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
