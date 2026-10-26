"""Difference between two evaluation runs on the same questions, with a 95% bootstrap interval.

The bootstrap resamples questions, not samples, because the 8 samples for one
question are correlated. Both runs use the same resampled questions.

    .venv/bin/python -m posttrain.eval.compare base_dev sft_lora_dev
"""

import argparse
import json
from functools import cache
from math import comb

import numpy as np
from transformers import AutoTokenizer

from posttrain.common import _TAGGED, MODEL, RESULTS, read_jsonl

RM = "checkpoints/rm_trunc2/merged"


@cache
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def per_question(name):
    rows = {r["question"]: r for r in read_jsonl(RESULTS / f"{name}.samples.jsonl")}
    bon = RESULTS / f"{name}.bon.json"
    picks = json.loads(bon.read_text())["rms"][RM].get("picked_correct") if bon.exists() else None
    if picks:
        for q, r in rows.items():
            r["bon_correct"] = picks[q]
    return rows


def strict(s, ok, max_tokens=512):
    """Correct, ends with 'Answer: N' and not cut off at the token limit."""
    return ok and bool(_TAGGED.search(s)) and len(tokenizer().encode(s)) < max_tokens


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def metrics(rows, qs):
    """Per-question greedy correctness, pass@1, pass@4, pass@n, strict pass@1 and best-of-n as arrays."""
    g = np.array([float(rows[q]["greedy_correct"]) for q in qs])
    n = len(rows[qs[0]]["correct"])
    c = [sum(rows[q]["correct"]) for q in qs]
    out = {"greedy": g, "pass@1": np.array([ci / n for ci in c])}
    for k in (4, n):
        out[f"pass@{k}"] = np.array([pass_at_k(n, ci, k) for ci in c])
    out["strict@1"] = np.array([np.mean([strict(s, ok) for s, ok in zip(rows[q]["samples"], rows[q]["correct"])])
                                for q in qs])
    if all("bon_correct" in rows[q] for q in qs):
        out[f"bo{n}"] = np.array([float(rows[q]["bon_correct"]) for q in qs])
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
    for key in [k for k in ma if k in mb]:
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
