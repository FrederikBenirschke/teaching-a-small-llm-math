"""Download GSM8K and write fixed prompt splits.

train and valid are used to build training data. dev is used to compare runs
and choose settings. test is evaluated once at the end.

data/prompts/{train,valid,dev,test}.jsonl   {"question", "answer"}
data/grpo/{train,valid}.jsonl           mlx-lm-lora GRPO format
"""

import argparse
import random

from datasets import load_dataset

from common import DATA, SYSTEM, gold_answer, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-valid", type=int, default=100)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-dev", type=int, default=300, help="evaluation set for comparing runs, disjoint from train/valid")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    rng = random.Random(args.seed)

    train_idx = rng.sample(range(len(ds["train"])), args.n_train + args.n_valid)
    test_idx = rng.sample(range(len(ds["test"])), args.n_test)
    used = set(train_idx)
    dev_idx = random.Random(args.seed + 1).sample([i for i in range(len(ds["train"])) if i not in used], args.n_dev)

    def rows(split, idx):
        return [
            {"question": split[i]["question"], "answer": gold_answer(split[i]["answer"])}
            for i in idx
        ]

    splits = {
        "train": rows(ds["train"], train_idx[: args.n_train]),
        "valid": rows(ds["train"], train_idx[args.n_train :]),
        "dev": rows(ds["train"], dev_idx),
        "test": rows(ds["test"], test_idx),
    }
    for name, r in splits.items():
        write_jsonl(DATA / "prompts" / f"{name}.jsonl", r)
    for name in ("train", "valid"):
        write_jsonl(
            DATA / "grpo" / f"{name}.jsonl",
            [{"prompt": r["question"], "answer": r["answer"], "system": SYSTEM} for r in splits[name]],
        )
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
