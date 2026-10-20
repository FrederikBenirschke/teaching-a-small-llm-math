"""Paired comparison of two eval runs on the same problems.

Bootstrap resamples PROBLEMS (the 8 samples of one problem are correlated), and
both runs use the same resampled set, so the CI is on the paired difference.

    .venv/bin/python compare.py base_dev sft_lora_dev
"""

import argparse
from math import comb

import numpy as np

from common import RESULTS, read_jsonl


def per_problem(name):
    rows = read_jsonl(RESULTS / f"{name}.samples.jsonl")
    return {r["question"]: r for r in rows}


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--boot", type=int, default=10_000)
    args = p.parse_args()

    A, B = per_problem(args.a), per_problem(args.b)
    if set(A) != set(B):
        raise SystemExit(f"problem sets differ: {len(A)} vs {len(B)}, {len(set(A) & set(B))} shared")
    qs = sorted(A)

    def metrics(R):
        g = np.array([float(R[q]["greedy_correct"]) for q in qs])
        n = len(R[qs[0]]["correct"])
        c = [sum(R[q]["correct"]) for q in qs]
        out = {"greedy": g, "pass@1": np.array([ci / n for ci in c])}
        for k in (4, n):
            out[f"pass@{k}"] = np.array([pass_at_k(n, ci, k) for ci in c])
        return out

    ma, mb = metrics(A), metrics(B)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(qs), size=(args.boot, len(qs)))
    print(f"{args.b} - {args.a}, N={len(qs)} problems, paired problem bootstrap ({args.boot})")
    for key in ma:
        d = mb[key] - ma[key]
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {key:7s} {ma[key].mean():.3f} -> {mb[key].mean():.3f}   diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
