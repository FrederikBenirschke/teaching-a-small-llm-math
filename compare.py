"""Difference between two evaluation runs on the same questions, with a 95% bootstrap interval.

The bootstrap resamples questions, not samples, because the 8 samples for one
question are correlated. Both runs use the same resampled questions.

    .venv/bin/python compare.py base_dev sft_lora_dev
"""

import argparse
from math import comb

import numpy as np

from common import RESULTS, read_jsonl


def per_question(name):
    rows = read_jsonl(RESULTS / f"{name}.samples.jsonl")
    return {r["question"]: r for r in rows}


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def metrics(rows, qs):
    """Per-question greedy correctness, pass@1, pass@4 and pass@n as arrays."""
    g = np.array([float(rows[q]["greedy_correct"]) for q in qs])
    n = len(rows[qs[0]]["correct"])
    c = [sum(rows[q]["correct"]) for q in qs]
    out = {"greedy": g, "pass@1": np.array([ci / n for ci in c])}
    for k in (4, n):
        out[f"pass@{k}"] = np.array([pass_at_k(n, ci, k) for ci in c])
    return out


def paired_diff(a, b, boot=10_000):
    """{metric: (mean_a, mean_b, diff, lo, hi)} for b - a over the shared questions."""
    ra, rb = per_question(a), per_question(b)
    if set(ra) != set(rb):
        raise SystemExit(f"question sets differ: {len(ra)} vs {len(rb)}, {len(set(ra) & set(rb))} shared")
    qs = sorted(ra)
    ma, mb = metrics(ra, qs), metrics(rb, qs)
    idx = np.random.default_rng(0).integers(0, len(qs), size=(boot, len(qs)))
    out = {}
    for key in ma:
        d = mb[key] - ma[key]
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        out[key] = (ma[key].mean(), mb[key].mean(), d.mean(), lo, hi)
    return out, len(qs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--boot", type=int, default=10_000)
    args = p.parse_args()

    res, n = paired_diff(args.a, args.b, args.boot)
    print(f"{args.b} - {args.a}, n={n} questions, paired bootstrap over questions ({args.boot} draws)")
    for key, (a, b, d, lo, hi) in res.items():
        print(f"  {key:7s} {a:.3f} -> {b:.3f}   diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
