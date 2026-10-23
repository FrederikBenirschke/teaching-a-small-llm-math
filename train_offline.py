"""SFT, DPO and reward-model training on the data from build_data.py (TRL).

    sft   SFTTrainer on correct base-model samples; loss on the answer tokens only
    dpo   DPOTrainer on (correct, incorrect) pairs for the same prompt
    rm    RewardTrainer on the same pairs; the result is used by train_grpo.py and train_ppo.py

Uses LoRA unless --full is given.

    .venv/bin/python train_offline.py --method sft --name sft_lora
    .venv/bin/python train_offline.py --method sft --name sft_full --full --lr 1e-5
"""

import argparse

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig
from transformers import AutoTokenizer
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


def rm_tokenized_rows(split, tok, trunc_neg, seed=0):
    """Tokenized reward-model pairs. Pre-tokenizing stops TRL from appending its own EOS.

    If trunc_neg is set, each pair also yields two pairs in which the complete chosen
    answer is preferred over the chosen answer cut at 20-90% of its tokens, once without
    and once with <|im_end|>. A reward model trained without these pairs scored
    unfinished answers above finished ones, and PPO then stopped emitting EOS.
    """
    rng = np.random.default_rng(seed)
    orig, trunc = [], []
    for r in read_jsonl(DATA / "dpo" / f"{split}.jsonl"):
        prompt = _prompt(r["system"], r["prompt"])
        p_ids = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        eos = tok.convert_tokens_to_ids("<|im_end|>")

        def full(a):  # cut after <|im_end|> (the template adds "\n"), as in PPO's reward-model input
            ids = tok.apply_chat_template(prompt + [{"role": "assistant", "content": a}],
                                          tokenize=True, return_dict=True)["input_ids"]
            return ids[: len(ids) - ids[::-1].index(eos)]
        ch = full(r["chosen"])
        orig.append({"chosen_input_ids": ch, "rejected_input_ids": full(r["rejected"])})
        if trunc_neg:
            resp = tok(r["chosen"], add_special_tokens=False)["input_ids"]
            cut = max(1, int(len(resp) * rng.uniform(0.2, 0.9)))
            trunc.append({"chosen_input_ids": ch, "rejected_input_ids": p_ids + resp[:cut]})
            # with <|im_end|>: without these, EOS alone separates the two answers, and a short
            # run scored early-stopped answers above the full answer 73% of the time
            cut2 = max(1, int(len(resp) * rng.uniform(0.2, 0.9)))
            trunc.append({"chosen_input_ids": ch, "rejected_input_ids": p_ids + resp[:cut2] + [eos]})
    return Dataset.from_list(orig), (Dataset.from_list(trunc) if trunc_neg else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["sft", "dpo", "rm"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    p.add_argument("--loss-type", default="sigmoid", help="DPO loss(es), comma-separated, e.g. sigmoid,sft (RPO)")
    p.add_argument("--loss-weights", default=None, help="comma-separated weights matching --loss-type")
    p.add_argument("--trunc-neg", action="store_true", help="rm: add pairs preferring complete over truncated answers")
    p.add_argument("--init", default=MODEL, help="initial model: hub id or merged checkpoint dir (e.g. an SFT model)")
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
        trainer = SFTTrainer(model=args.init, args=cfg, train_dataset=sft_rows("train"),
                             eval_dataset=sft_rows("valid"), peft_config=lora)
    elif args.method == "dpo":
        cfg = DPOConfig(learning_rate=args.lr or 5e-6 * (1 if args.full else 10),
                        num_train_epochs=args.epochs or 1, beta=args.beta,
                        loss_type=args.loss_type.split(","),
                        loss_weights=[float(w) for w in args.loss_weights.split(",")] if args.loss_weights else None,
                        **common)
        trainer = DPOTrainer(model=args.init, args=cfg, train_dataset=pair_rows("train"),
                             eval_dataset=pair_rows("valid"), peft_config=lora)
    else:
        cfg = RewardConfig(learning_rate=args.lr or 1e-4, num_train_epochs=args.epochs or 2, **common)
        if args.trunc_neg:
            tok = AutoTokenizer.from_pretrained(MODEL)
            tr_o, tr_t = rm_tokenized_rows("train", tok, True)
            va_o, va_t = rm_tokenized_rows("valid", tok, True, seed=1)
            train_ds, eval_ds = concatenate_datasets([tr_o, tr_t]).shuffle(seed=0), {"orig": va_o, "trunc": va_t}
        else:
            train_ds, eval_ds = pair_rows("train"), pair_rows("valid")
        trainer = RewardTrainer(model=args.init, args=cfg, train_dataset=train_ds,
                                eval_dataset=eval_ds, peft_config=lora)

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
