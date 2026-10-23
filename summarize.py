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
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import _TAGGED, MODEL, RESULTS, ROOT, read_jsonl
from compare import paired_diff

ARMS = ["base", "sft_lora", "dpo_lr5e6", "sftdpo_lr5e6", "grpo", "rlhf_grpo", "sftgrpo", "ppo_rmtrunc_small"]
LABEL = {"base": "base", "sft_lora": "SFT", "dpo_lr5e6": "DPO", "sftdpo_lr5e6": "SFT→DPO", "grpo": "GRPO (grader)",
         "rlhf_grpo": "GRPO (RM)", "sftgrpo": "SFT→GRPO", "ppo_rmtrunc_small": "PPO (RM)"}
RM = "checkpoints/rm_trunc2/merged"
PLOTS = RESULTS / "plots"
# training log per arm; PPO logs no train_runtime, so its wall-clock comes from the log's first and last write
TRAIN_LOG = {"sft_lora": "sft_lora.train", "dpo_lr5e6": "dpo_lr5e6.train", "sftdpo_lr5e6": "sftdpo_lr5e6.train",
             "grpo": "grpo.train", "rlhf_grpo": "rlhf_grpo.train", "sftgrpo": "sftgrpo.train",
             "ppo_rmtrunc_small": "ppo_rmtrunc_small", "reward model v1": "rm.train", "reward model rm_trunc2": "rm_trunc2",
             "reward model rm_trunc2_q": "rm_trunc2_q"}


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
            d_greedy = d_pass1 = ""
        else:
            d, _ = paired_diff(f"base_{split}", f"{arm}_{split}")
            d_greedy = "{:+.3f} [{:+.3f}, {:+.3f}]".format(*d["greedy"][2:])
            d_pass1 = "{:+.3f} [{:+.3f}, {:+.3f}]".format(*d["pass@1"][2:])
        bon = f"{r['bon']['best_of_8']:.3f}" if r["bon"] else ""
        rows.append(f"| {LABEL[arm]} | {r['greedy_acc']:.3f} | {d_greedy} | {r['pass@1']:.3f} | {d_pass1} | "
                    f"{r['pass@8']:.3f} | {bon} |")
    return "\n".join(rows), out


def format_table(split, max_tokens=512):
    """Share of sampled answers that are truncated, end with 'Answer: N', or lack the tag.

    The grader falls back to the last number when the tag is missing, so pass@1 counts
    untagged answers. strict pass@1 counts only complete, tagged, correct answers.
    """
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = ["| method | truncated | tagged | untagged | acc. tagged | acc. untagged | pass@1 | strict pass@1 |",
            "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        path = RESULTS / f"{arm}_{split}.samples.jsonl"
        if not path.exists():
            continue
        cnt = {"trunc": [0, 0], "tag": [0, 0], "untag": [0, 0]}
        for r in read_jsonl(path):
            for s, ok in zip(r["samples"], r["correct"]):
                k = "trunc" if len(tok.encode(s)) >= max_tokens else ("tag" if _TAGGED.search(s) else "untag")
                cnt[k][0] += 1
                cnt[k][1] += ok
        n = sum(v[0] for v in cnt.values())
        frac = {k: v[0] / n for k, v in cnt.items()}
        acc = {k: (f"{v[1] / v[0]:.3f}" if v[0] else "") for k, v in cnt.items()}
        rows.append(f"| {LABEL[arm]} | {frac['trunc']:.3f} | {frac['tag']:.3f} | {frac['untag']:.3f} | {acc['tag']} | "
                    f"{acc['untag']} | {sum(v[1] for v in cnt.values()) / n:.3f} | {cnt['tag'][1] / n:.3f} |")
    return "\n".join(rows)


def train_secs(log):
    text = (ROOT / "logs" / f"{log}.log").read_text()
    m = re.findall(r"'train_runtime': '([0-9.e+]+)'", text)
    if m:
        return float(m[-1])
    st = (ROOT / "logs" / f"{log}.log").stat()
    return st.st_mtime - st.st_birthtime


def compute_table():
    rows = ["| arm | training wall-clock (h) | dev eval wall-clock (min) |", "|---|---|---|"]
    for arm, log in TRAIN_LOG.items():
        ev = RESULTS / f"{arm}_dev.json"
        ev = f"{json.loads(ev.read_text())['eval_secs'] / 60:.1f}" if ev.exists() else ""
        rows.append(f"| {LABEL.get(arm, arm)} | {train_secs(log) / 3600:.2f} | {ev} |")
    return "\n".join(rows)


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
    # (label, train log, dev result): the first PPO run on reward model v1 and the reported run on rm_trunc2_q
    runs = [("PPO, RM v1, EOS penalty 1", "logs/ppo.train.log", "ppo_eos1_dev"),
            ("PPO, rm_trunc2_q", "logs/ppo_rmtrunc_small.log", "ppo_rmtrunc_small_dev")]
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
    p.add_argument("--split", choices=["dev", "test", "svamp"], default="dev")
    args = p.parse_args()
    PLOTS.mkdir(exist_ok=True)
    md, res = table(args.split)
    fmt = format_table(args.split)
    (RESULTS / f"summary_{args.split}.md").write_text(md + "\n\n" + fmt + "\n")
    print(md + "\n\n" + fmt)
    bars(res, args.split)
    if args.split == "dev":
        ppo_plot()
        comp = compute_table()
        (RESULTS / "summary_compute.md").write_text(comp + "\n")
        print("\n" + comp)


if __name__ == "__main__":
    main()
