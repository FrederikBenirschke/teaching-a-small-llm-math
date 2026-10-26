"""SFT data from the GSM8K reference solutions, for the same train/valid questions as prepare.py.

The <<48/2=24>> calculator annotations are removed and the final '#### N' line becomes
'Answer: N', the format the system prompt asks for.

data/sft_gold/{train,valid}.jsonl   chat format, one reference solution per question

    .venv/bin/python -m posttrain.prep.build_gold
"""

import random
import re

from datasets import load_dataset

from posttrain.common import DATA, SYSTEM, gold_answer, read_jsonl, write_jsonl


def reformat(solution):
    body, _ = solution.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nAnswer: {gold_answer(solution)}"


def main(n_train=1000, n_valid=100, seed=0):
    ds = load_dataset("openai/gsm8k", "main")["train"]
    idx = random.Random(seed).sample(range(len(ds)), n_train + n_valid)  # same draw as prepare.py
    for split, part in (("train", idx[:n_train]), ("valid", idx[n_train:])):
        prompts = read_jsonl(DATA / "prompts" / f"{split}.jsonl")
        assert [ds[i]["question"] for i in part] == [r["question"] for r in prompts], split
        write_jsonl(DATA / "sft_gold" / f"{split}.jsonl", [{"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ds[i]["question"]},
            {"role": "assistant", "content": reformat(ds[i]["answer"])},
        ]} for i in part])
        print(split, len(part))


if __name__ == "__main__":
    main()
