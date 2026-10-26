"""GRPO reward for mlx-lm-lora (mlx/configs/grpo.yaml): 1 if the answer is correct, else 0."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.common import is_correct  # noqa: E402
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function  # noqa: E402


@register_reward_function()
def gsm8k_correct(prompts, completions, answer, types=None):
    return [1.0 if is_correct(c, a) else 0.0 for c, a in zip(completions, answer)]
