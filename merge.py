"""Merge a LoRA adapter into the base weights and save a Hugging Face checkpoint.

The merged directory is what evaluate.py --model and train_ppo.py --reward-model load.

    .venv/bin/python merge.py checkpoints/sft_lora            # -> checkpoints/sft_lora/merged
    .venv/bin/python merge.py checkpoints/rm --seq-cls
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from common import MODEL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("adapter")
    p.add_argument("--seq-cls", action="store_true", help="reward model (sequence classification head)")
    args = p.parse_args()

    src = Path(args.adapter)
    if not (src / "adapter_config.json").exists():
        raise SystemExit(f"{src} has no adapter_config.json (full fine-tunes do not need merging)")
    # merge onto the model the adapter was trained on (e.g. sft_lora/merged for SFT-then-DPO), not always the hub base
    init = json.loads((src / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"base: {init}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if args.seq_cls:
        base = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1, dtype=torch.float32)
        base.config.pad_token_id = tok.pad_token_id
    else:
        base = AutoModelForCausalLM.from_pretrained(init, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, src).merge_and_unload()
    out = src / "merged"
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
