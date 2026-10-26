"""PPO with LoRA against a trained reward model (TRL). Uses a policy, a value model
and the frozen base model as reference.

Runs in .venv-ppo: TRL 1.12 is the last release with trl.experimental.ppo.

    .venv-ppo/bin/python -m posttrain.train.ppo --reward-model checkpoints/rm/merged
    .venv-ppo/bin/python -m posttrain.train.ppo --smoke   # untrained reward head, 2 updates
"""

import argparse
import os

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer  # noqa: E402

from posttrain.common import DATA, MODEL, ROOT, SYSTEM, read_jsonl  # noqa: E402


class SnapshotCallback(TrainerCallback):
    """Save the policy adapter to <out>/ep<N> every `every` episodes.

    The snapshots are evaluated afterwards to plot dev accuracy against KL during
    training (reward-model overoptimization, Gao et al. 2022)."""

    def __init__(self, every, out, tok):
        self.every, self.out, self.tok, self.trainer, self.next = every, out, tok, None, every

    def on_step_end(self, args, state, control, **kw):
        if self.every and state.episode >= self.next:
            d = os.path.join(self.out, f"ep{state.episode}")
            self.trainer.model.policy.save_pretrained(d)
            self.tok.save_pretrained(d)
            print(f"snapshot -> {d}", flush=True)
            self.next += self.every
        return control


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-model", default=None, help="trained RM dir; required unless --smoke")
    p.add_argument("--out", default=str(ROOT / "checkpoints" / "ppo"))
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--response-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--kl-coef", type=float, default=0.05)
    # The first reward model was not trained on unfinished answers. On base dev answers cut
    # at half length it gives +0.98 without EOS and -7.43 with EOS (complete answers: +0.78).
    # The penalty for a missing EOS must be larger than this gap of about 8. Otherwise the
    # policy learns not to finish (90% of answers truncated in the first run).
    p.add_argument("--missing-eos-penalty", type=float, default=10.0)
    p.add_argument("--save-every", type=int, default=100, help="episodes between adapter snapshots (0 = off)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.reward_model is None and not args.smoke:
        raise SystemExit("--reward-model is required (or pass --smoke)")

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    rows = read_jsonl(DATA / "prompts" / "train.jsonl")
    if args.smoke:
        rows = rows[: 4 * args.batch_size]
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for r in rows
    ]
    ds = Dataset.from_dict({"input_ids": tok(texts, add_special_tokens=False)["input_ids"]})

    dtype = torch.float32  # some bf16 ops are slow or unsupported on MPS
    policy = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    # TRL takes the rollout log-probs from generate()'s processed scores, but the reference
    # and PPO-update log-probs from the raw logits divided by temperature. transformers
    # applies Qwen's generation_config (repetition_penalty 1.1, top_p 0.8, top_k 20), so the
    # two differ before any training: 8-21 nats KL per 128-token sequence at initialization
    # (about 41 at step 1 of the first run). With these settings disabled the KL is 0.
    policy.generation_config.repetition_penalty = 1.0
    policy.generation_config.top_k = None
    policy.generation_config.top_p = None
    rm_path = MODEL if args.smoke else args.reward_model
    reward_model = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1, dtype=dtype)
    value_model = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1, dtype=dtype)
    for m in (reward_model, value_model):
        m.config.pad_token_id = tok.pad_token_id

    cfg = PPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_ppo_epochs=2,
        num_mini_batches=2,
        total_episodes=2 * args.batch_size if args.smoke else args.episodes,
        response_length=args.response_length,
        kl_coef=args.kl_coef,
        stop_token="eos",
        missing_eos_penalty=args.missing_eos_penalty,
        local_rollout_forward_batch_size=args.batch_size,
        num_sample_generations=0,
        temperature=0.8,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    snap = SnapshotCallback(args.save_every, args.out, tok)
    trainer = PPOTrainer(
        args=cfg,
        processing_class=tok,
        model=policy,
        ref_model=None,  # with peft_config, the frozen base model is the reference
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=ds,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
        callbacks=[snap],
    )
    snap.trainer = trainer
    trainer.train()
    # trainer.save_model fails with TRL 1.12 and transformers 5.20 (missing
    # is_distributed_loading_by_transformers), so the policy adapter is saved directly.
    trainer.model.policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved policy adapter -> {args.out}")


if __name__ == "__main__":
    main()
