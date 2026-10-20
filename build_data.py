"""Sample k answers per prompt from the BASE model, grade them, and build the
offline datasets. This is the single source of signal for SFT and DPO; GRPO
uses the same grader online.

data/samples/{split}.jsonl   every graded sample (kept for the write-up)
data/sft/{train,valid}.jsonl correct samples, chat format
data/dpo/{train,valid}.jsonl (correct, incorrect) pairs from the same prompt
"""

import argparse
import random

from common import DATA, SYSTEM, is_correct, read_jsonl, write_jsonl
from generation import load_model, sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-per-prompt", type=int, default=2, help="SFT examples / DPO pairs per prompt")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N prompts")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    model, tokenizer = load_model()

    for split in ("train", "valid"):
        prompts = read_jsonl(DATA / "prompts" / f"{split}.jsonl")[: args.limit]
        outs, secs = sample(model, tokenizer, [r["question"] for r in prompts], args.k, args.temp, args.max_tokens)

        graded, sft, dpo = [], [], []
        for r, completions in zip(prompts, outs):
            marks = [is_correct(c, r["answer"]) for c in completions]
            graded.append({**r, "completions": completions, "correct": marks})
            good = list(dict.fromkeys(c for c, m in zip(completions, marks) if m))
            bad = list(dict.fromkeys(c for c, m in zip(completions, marks) if not m))
            rng.shuffle(good)
            rng.shuffle(bad)
            for c in good[: args.max_per_prompt]:
                sft.append({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": c},
                ]})
            for g, b in list(zip(good, bad))[: args.max_per_prompt]:
                dpo.append({"system": SYSTEM, "prompt": r["question"], "chosen": g, "rejected": b})

        write_jsonl(DATA / "samples" / f"{split}.jsonl", graded)
        write_jsonl(DATA / "sft" / f"{split}.jsonl", sft)
        write_jsonl(DATA / "dpo" / f"{split}.jsonl", dpo)

        acc = sum(sum(g["correct"]) for g in graded) / (len(graded) * args.k)
        mixed = sum(0 < sum(g["correct"]) < args.k for g in graded)
        print(f"[{split}] prompts={len(graded)} sample_acc={acc:.3f} "
              f"mixed_prompts={mixed} sft={len(sft)} dpo_pairs={len(dpo)} secs={secs:.0f}")


if __name__ == "__main__":
    main()
