"""Batched sampling with mlx-lm."""

import time

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from common import MODEL, chat_prompt


def load_model(adapter: str | None = None, model: str = MODEL):
    """`model` is a hub id or a merged HF checkpoint dir; `adapter` an MLX LoRA dir."""
    return load(model, adapter_path=adapter)


def sample(model, tokenizer, questions, k, temp, max_tokens, batch_size=64):
    """Return a list (per question) of k completion strings, plus wall seconds."""
    prompts = [chat_prompt(tokenizer, q) for q in questions for _ in range(k)]
    t0 = time.time()
    resp = batch_generate(
        model,
        tokenizer,
        prompts,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temp),
        completion_batch_size=batch_size,
        verbose=True,
    )
    secs = time.time() - t0
    texts = resp.texts
    return [texts[i * k : (i + 1) * k] for i in range(len(questions))], secs
