# LLM post-training ablation

SFT vs DPO vs GRPO (with LoRA) on GSM8K with Qwen2.5-0.5B-Instruct, run locally on Apple Silicon.

All methods learn from one signal: an exact-match grader on the final answer.
SFT trains on correct base-model samples, DPO on (correct, incorrect) pairs
from the same prompt, GRPO on the 0/1 reward online.

    python prepare.py       # fixed train/valid/test prompt splits
    python build_data.py    # sample base model, grade, build SFT/DPO data
    python evaluate.py --name base
