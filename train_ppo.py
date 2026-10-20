"""Classic RLHF: PPO (policy + value model + reference) against a learned reward model.

Runs in .venv-ppo (TRL 1.12, the last release with trl.experimental.ppo).

    .venv-ppo/bin/python train_ppo.py --reward-model checkpoints/rm
    .venv-ppo/bin/python train_ppo.py --smoke   # untrained RM head, 2 updates
"""

import argparse
import os

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from trl.experimental.ppo import PPOConfig, PPOTrainer  # noqa: E402

from common import DATA, MODEL, ROOT, SYSTEM, read_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-model", default=None, help="trained RM dir; required unless --smoke")
    p.add_argument("--out", default=str(ROOT / "checkpoints" / "ppo"))
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--response-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--kl-coef", type=float, default=0.05)
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

    dtype = torch.float32  # MPS: bf16 matmuls are slow/unsupported for some ops
    policy = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
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
        missing_eos_penalty=1.0,
        local_rollout_forward_batch_size=args.batch_size,
        num_sample_generations=0,
        temperature=0.8,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = PPOTrainer(
        args=cfg,
        processing_class=tok,
        model=policy,
        ref_model=None,  # with peft_config the frozen base is the reference
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=ds,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.out)


if __name__ == "__main__":
    main()
