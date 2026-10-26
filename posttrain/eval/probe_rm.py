"""Check whether a reward model scores unfinished answers above finished ones.

Scores the base model's greedy dev answers in four forms:
  full      complete answer ending at <|im_end|> (the form PPO scores)
  full_nl   complete answer with the chat template's trailing "\n" (the form GRPO scores)
  half      first half of the answer, no EOS
  half_eos  first half of the answer followed by <|im_end|>
It also scores the --policy model's greedy answers. A reward model without the
truncation bias scores full above half and half_eos.

    .venv/bin/python -m posttrain.eval.probe_rm checkpoints/rm/merged checkpoints/rm_trunc/merged
"""

import argparse

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from posttrain.common import MODEL, ROOT, SYSTEM, is_correct, read_jsonl


@torch.no_grad()
def score(model, ids):
    return model(input_ids=torch.tensor([ids])).logits[0, 0].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rms", nargs="+")
    p.add_argument("-n", type=int, default=60)
    p.add_argument("--policy", default="ppo_eos1", help="results/<name>_dev.samples.jsonl to score against base")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = read_jsonl(ROOT / "results" / "base_dev.samples.jsonl")[: args.n]
    pol = read_jsonl(ROOT / "results" / f"{args.policy}_dev.samples.jsonl")[: args.n]
    seqs = []
    for r, q in zip(rows, pol):
        assert q["question"] == r["question"]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}]
        p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        resp = tok(r["greedy"], add_special_tokens=False)["input_ids"]
        half = resp[: len(resp) // 2]
        pol_resp = tok(q["greedy"], add_special_tokens=False)["input_ids"]
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        tmpl = tok.apply_chat_template(msgs + [{"role": "assistant", "content": r["greedy"]}],
                                       tokenize=True, return_dict=True)["input_ids"]  # ends "<|im_end|>\n"
        seqs.append(dict(full=p_ids + resp + [eos], full_nl=tmpl, half=p_ids + half, half_eos=p_ids + half + [eos],
                         pol=p_ids + pol_resp + ([eos] if len(pol_resp) < 512 else []),
                         ok=is_correct(r["greedy"], r["answer"])))

    for path in args.rms:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32).eval()
        s = {k: np.array([score(m, q[k]) for q in seqs]) for k in ("full", "full_nl", "half", "half_eos", "pol")}
        ok = np.array([q["ok"] for q in seqs])
        print(f"{path}  (n={len(seqs)})")
        print(f"  full+EOS (PPO form)  {s['full'].mean():+.2f}  (correct {s['full'][ok].mean():+.2f}, wrong {s['full'][~ok].mean():+.2f})")
        for k in ("full_nl", "half", "half_eos", "pol"):
            print(f"  {(args.policy if k == 'pol' else k):<10} {s[k].mean():+.2f}  scored above full: {(s[k] > s['full']).mean():.0%}")


if __name__ == "__main__":
    main()
