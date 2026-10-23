"""Best-of-N selection with a reward model.

Scores the 8 stored samples per dev question (results/<policy>_<split>.samples.jsonl),
each ending at <|im_end|> as in PPO, and reports the accuracy of the highest-scored
sample. For reference it also prints pass@1 (random choice) and pass@8 (any sample
correct).

    .venv/bin/python bon_rm.py checkpoints/rm/merged checkpoints/rm_trunc2/merged
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import MODEL, ROOT, SYSTEM, read_jsonl


@torch.no_grad()
def score(model, ids, dev):
    return model(input_ids=torch.tensor([ids], device=dev)).logits[0, 0].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rms", nargs="+")
    p.add_argument("--policy", default="base")
    p.add_argument("--split", choices=["dev", "test", "svamp"], default="dev")
    p.add_argument("--max-new", type=int, default=512, help="generation token limit; samples of this length get no EOS")
    args = p.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    rows = read_jsonl(ROOT / "results" / f"{args.policy}_{args.split}.samples.jsonl")
    seqs, ok = [], []
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}]
        p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        per = []
        for s in r["samples"]:
            resp = tok(s, add_special_tokens=False)["input_ids"]
            per.append(p_ids + resp + ([eos] if len(resp) < args.max_new else []))
        seqs.append(per)
        ok.append(r["correct"])
    ok = np.array(ok, dtype=bool)  # (problems, 8)
    print(f"{args.policy} {args.split}  n={len(rows)} x {ok.shape[1]}  pass@1 {ok.mean():.3f}  pass@{ok.shape[1]} {ok.any(1).mean():.3f}")

    rng = np.random.default_rng(0)
    out = {"policy": args.policy, "split": args.split, "n": len(rows), "pass@1": ok.mean(), "pass@8": ok.any(1).mean(), "rms": {}}
    for path in args.rms:
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32).to(dev).eval()
        s = np.array([[score(m, q, dev) for q in per] for per in seqs])
        pick = ok[np.arange(len(ok)), s.argmax(1)]
        boot = [pick[i].mean() - ok[i].mean() for i in (rng.integers(0, len(ok), len(ok)) for _ in range(2000))]
        print(f"  {path}  best-of-{ok.shape[1]} {pick.mean():.3f}  vs pass@1 {pick.mean() - ok.mean():+.3f} "
              f"[{np.quantile(boot, .025):+.3f}, {np.quantile(boot, .975):+.3f}]")
        out["rms"][path] = {"best_of_8": pick.mean(), "gain": pick.mean() - ok.mean(),
                            "gain_ci": [np.quantile(boot, .025), np.quantile(boot, .975)]}
    (ROOT / "results" / f"{args.policy}_{args.split}.bon.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
