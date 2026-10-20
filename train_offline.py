"""Offline post-training on the data from build_data.py (TRL).

    sft   SFTTrainer on correct base samples, loss on the completion only
    dpo   DPOTrainer on (correct, incorrect) pairs from the same prompt
    rm    RewardTrainer on the same pairs -> reward model for RLHF (PPO / GRPO)

LoRA by default; --full trains every weight (the LoRA-vs-full axis).

    .venv/bin/python train_offline.py --method sft --name sft_lora
    .venv/bin/python train_offline.py --method sft --name sft_full --full --lr 1e-5
"""

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer, RewardConfig, RewardTrainer, SFTConfig, SFTTrainer

from common import DATA, MODEL, ROOT, read_jsonl


def _prompt(system, question):
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def sft_rows(split):
    out = []
    for r in read_jsonl(DATA / "sft" / f"{split}.jsonl"):
        sys_msg, user, asst = r["messages"]
        out.append({"prompt": [sys_msg, user], "completion": [asst]})
    return Dataset.from_list(out)


def pair_rows(split):
    return Dataset.from_list([
        {"prompt": _prompt(r["system"], r["prompt"]),
         "chosen": [{"role": "assistant", "content": r["chosen"]}],
         "rejected": [{"role": "assistant", "content": r["rejected"]}]}
        for r in read_jsonl(DATA / "dpo" / f"{split}.jsonl")
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["sft", "dpo", "rm"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    args = p.parse_args()

    out = str(ROOT / "checkpoints" / args.name)
    lora = None if args.full else LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        task_type="SEQ_CLS" if args.method == "rm" else "CAUSAL_LM",
    )
    common = dict(
        output_dir=out,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        bf16=False,
        max_length=1024,
    )

    if args.method == "sft":
        cfg = SFTConfig(learning_rate=args.lr or 1e-4, num_train_epochs=args.epochs or 2,
                        completion_only_loss=True, **common)
        trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=sft_rows("train"),
                             eval_dataset=sft_rows("valid"), peft_config=lora)
    elif args.method == "dpo":
        cfg = DPOConfig(learning_rate=args.lr or 5e-6 * (1 if args.full else 10),
                        num_train_epochs=args.epochs or 1, beta=args.beta, **common)
        trainer = DPOTrainer(model=MODEL, args=cfg, train_dataset=pair_rows("train"),
                             eval_dataset=pair_rows("valid"), peft_config=lora)
    else:
        cfg = RewardConfig(learning_rate=args.lr or 1e-4, num_train_epochs=args.epochs or 2, **common)
        trainer = RewardTrainer(model=MODEL, args=cfg, train_dataset=pair_rows("train"),
                                eval_dataset=pair_rows("valid"), peft_config=lora)

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
