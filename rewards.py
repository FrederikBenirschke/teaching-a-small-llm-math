"""GRPO reward: the same 0/1 grader used to build the SFT/DPO data and to evaluate.

Loaded by mlx-lm-lora via `reward_functions_file`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import is_correct  # noqa: E402
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function  # noqa: E402


@register_reward_function()
def gsm8k_correct(prompts, completions, answer, types=None):
    return [1.0 if is_correct(c, a) else 0.0 for c, a in zip(completions, answer)]
