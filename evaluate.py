"""Evaluate base or base+adapter on the fixed GSM8K test prompts.

greedy_acc   accuracy of the temperature-0 answer
mean_acc     average accuracy over the k sampled answers (= pass@1 estimate at temp)
pass@j       unbiased estimator (Chen et al. 2021) from k samples, j in 1,2,4,..,k
mean_tokens  average sampled completion length
"""

import argparse
import json
from math import comb

from common import DATA, RESULTS, is_correct, read_jsonl, write_jsonl
from generation import load_model, sample


def pass_at(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tests = read_jsonl(DATA / "prompts" / "test.jsonl")[: args.limit]
    qs = [t["question"] for t in tests]
    model, tokenizer = load_model(args.adapter)

    greedy, greedy_secs = sample(model, tokenizer, qs, 1, 0.0, args.max_tokens)
    sampled, sample_secs = sample(model, tokenizer, qs, args.k, args.temp, args.max_tokens)

    rows, n_correct = [], []
    for t, g, s in zip(tests, greedy, sampled):
        marks = [is_correct(c, t["answer"]) for c in s]
        n_correct.append(sum(marks))
        rows.append({**t, "greedy": g[0], "greedy_correct": is_correct(g[0], t["answer"]),
                     "samples": s, "correct": marks})

    n = len(tests)
    ks = [j for j in (1, 2, 4, 8, 16, 32, 64) if j <= args.k]
    lengths = [len(tokenizer.encode(c)) for r in rows for c in r["samples"]]
    summary = {
        "name": args.name,
        "adapter": args.adapter,
        "n_prompts": n,
        "k": args.k,
        "temp": args.temp,
        "greedy_acc": sum(r["greedy_correct"] for r in rows) / n,
        "mean_acc": sum(n_correct) / (n * args.k),
        **{f"pass@{j}": sum(pass_at(args.k, c, j) for c in n_correct) / n for j in ks},
        "mean_tokens": sum(lengths) / len(lengths),
        "truncated_frac": sum(L >= args.max_tokens for L in lengths) / len(lengths),
        "eval_secs": greedy_secs + sample_secs,
    }
    RESULTS.mkdir(exist_ok=True)
    write_jsonl(RESULTS / f"{args.name}.samples.jsonl", rows)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
