"""Tables and plots for the README from results/.

Reads results/<arm>_<split>.json (evaluate.py), <arm>_<split>.bon.json (bon_rm.py)
and the per-problem samples for paired differences against the base model.
Writes results/summary_<split>.md and results/plots/*.png.

    .venv/bin/python summarize.py --split dev
    .venv/bin/python summarize.py --split test
"""

import argparse
import ast
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, ROOT
from compare import paired_diff

ARMS = ["base", "sft_lora", "dpo_lr5e6", "sftdpo_lr5e6", "grpo", "rlhf_grpo", "sftgrpo", "ppo_rmtrunc_small"]
LABEL = {"base": "base", "sft_lora": "SFT", "dpo_lr5e6": "DPO", "sftdpo_lr5e6": "SFT→DPO", "grpo": "GRPO (grader)",
         "rlhf_grpo": "GRPO (RM)", "sftgrpo": "SFT→GRPO", "ppo_rmtrunc_small": "PPO (RM)"}
RM = "checkpoints/rm_trunc2/merged"
PLOTS = RESULTS / "plots"


def load(arm, split):
    r = json.loads((RESULTS / f"{arm}_{split}.json").read_text())
    bon = RESULTS / f"{arm}_{split}.bon.json"
    r["bon"] = json.loads(bon.read_text())["rms"][RM] if bon.exists() else None
    return r


def table(split):
    rows = ["| method | greedy | Δ greedy vs base [95% CI] | pass@1 | Δ pass@1 vs base [95% CI] | pass@8 | best-of-8 (RM) |",
            "|---|---|---|---|---|---|---|"]
    out = {}
    for arm in ARMS:
        if not (RESULTS / f"{arm}_{split}.json").exists():
            rows.append(f"| {LABEL[arm]} | not evaluated on {split} | | | | | |")
            continue
        r = load(arm, split)
        out[arm] = r
        if arm == "base":
            dg = dp = "–"
        else:
            d, _ = paired_diff(f"base_{split}", f"{arm}_{split}")
            dg = "{:+.3f} [{:+.3f}, {:+.3f}]".format(*d["greedy"][2:])
            dp = "{:+.3f} [{:+.3f}, {:+.3f}]".format(*d["pass@1"][2:])
        bon = f"{r['bon']['best_of_8']:.3f}" if r["bon"] else "–"
        rows.append(f"| {LABEL[arm]} | {r['greedy_acc']:.3f} | {dg} | {r['pass@1']:.3f} | {dp} | {r['pass@8']:.3f} | {bon} |")
    return "\n".join(rows), out


def bars(res, split):
    keys = [("greedy_acc", "greedy"), ("pass@1", "pass@1"), ("pass@8", "pass@8"), ("bon", "best-of-8 (RM)")]
    arms = [a for a in ARMS if a in res]
    x = np.arange(len(arms))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (k, name) in enumerate(keys):
        v = [(res[a]["bon"]["best_of_8"] if res[a]["bon"] else np.nan) if k == "bon" else res[a][k] for a in arms]
        ax.bar(x + (i - 1.5) * w, v, w, label=name)
    ax.set_xticks(x, [LABEL[a] for a in arms], rotation=20)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"GSM8K {split} (n={res['base']['n_prompts']}), Qwen2.5-0.5B-Instruct")
    ax.legend(ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"methods_{split}.png", dpi=150)


def ppo_log(path):
    """[(episode, kl, rlhf_reward)] from a TRL PPO training log."""
    pts = []
    for line in (ROOT / path).read_text().splitlines():
        m = re.search(r"\{'.*'objective/kl'.*\}", line)
        if m:
            d = ast.literal_eval(m.group(0))
            pts.append((float(d["episode"]), float(d["objective/kl"]), float(d["objective/rlhf_reward"])))
    return np.array(pts)


def ppo_plot():
    # (label, train log, dev result): the first PPO run (EOS penalty 1) and the final small run on rm_trunc2
    runs = [("PPO, RM v1, EOS penalty 1", "logs/ppo.train.log", "ppo_eos1_dev"),
            ("PPO, rm_trunc2", "logs/ppo_rmtrunc_small.log", "ppo_rmtrunc_small_dev")]
    base = json.loads((RESULTS / "base_dev.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a2.scatter([0], [base["greedy_acc"]], c="k", label="base (KL 0)")
    for name, log, res in runs:
        p = ppo_log(log)
        a1.plot(p[:, 0], p[:, 1], label=name)
        kl_end = p[-10:, 1].mean()
        r = json.loads((RESULTS / f"{res}.json").read_text())
        a2.scatter([kl_end], [r["greedy_acc"]], label=f"{name}: greedy {r['greedy_acc']:.3f}")
    a1.set_xlabel("episode")
    a1.set_ylabel("KL to reference (nats/seq)")
    a1.set_title("PPO KL during training")
    a1.legend()
    a2.set_xlabel("KL, mean of last 10 logged steps")
    a2.set_ylabel("dev greedy accuracy")
    a2.set_title("accuracy vs final KL")
    a2.legend(fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "ppo_kl.png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    args = p.parse_args()
    PLOTS.mkdir(exist_ok=True)
    md, res = table(args.split)
    (RESULTS / f"summary_{args.split}.md").write_text(md + "\n")
    print(md)
    bars(res, args.split)
    if args.split == "dev":
        ppo_plot()


if __name__ == "__main__":
    main()
