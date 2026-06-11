#!/usr/bin/env python3
"""Generate the M64.2 Gemma4-MoE forward-parity fixture.

Loads a Gemma4 MoE target (e.g. mlx-community/gemma-4-26b-a4b-it-4bit) with the
upstream mlx-lm reference (the Python oracle — gemma4_text.py carries the full
MoE: Router / Experts / hybrid DecoderLayer), runs a single forward over a fixed
short token sequence on a fresh cache, and writes the token ids + the softcapped
logits to a safetensors fixture that the Swift `Gemma4MoEParityTests` compares
its substrate forward against.

The logits are the end-to-end product of the whole MoE forward (router top-k +
expert gather-matmul + the three extra norms + the hybrid sum + mixed 4/8-bit
quant); if any of those are wrong in the Swift port the logits diverge. So a
high-cosine / matching-argmax assertion on these logits is the MoE-numerics
correctness gate.

Run inside the dflash spike venv, e.g.:
    ~/Source/dflash-mlx/.venv/bin/python deploy/gemma4-moe/gen_parity_fixture.py \
        --model mlx-community/gemma-4-26b-a4b-it-4bit \
        --out Tests/AthenaCoreTests/Fixtures/gemma4_moe_parity.safetensors

The fixture is tied to the specific checkpoint weights; regenerate if the
checkpoint changes.
"""
import argparse
import os

import mlx.core as mx
from mlx_lm import load


# Default real prompt (chat-templated + tokenized below). Using an
# in-distribution prompt keeps the next-token distribution sharp, so a correct
# port agrees tightly — synthetic OOD token ids sit on near-ties everywhere and
# disagree even between two correct ports.
DEFAULT_PROMPT = "What is the capital of France? Answer in one word."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--out", required=True, help="output safetensors path")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="chat-templated prompt")
    args = ap.parse_args()

    model, tok = load(args.model)

    toks = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )
    TOKENS = [int(t) for t in toks]
    ids = mx.array(TOKENS, dtype=mx.int32)[None]  # (1, L)
    cache = None
    try:
        cache = model.make_cache()
    except AttributeError:
        cache = None

    logits = model(ids, cache=cache)  # (1, L, vocab), softcapped
    mx.eval(logits)

    # Keep the last-position logits (next-token distribution) plus the full
    # (1, L, V) tensor for a thorough compare; both saved in bf16 as served.
    last = logits[:, -1, :]
    argmax_last = mx.argmax(last, axis=-1)
    # Per-position greedy argmax (the decisive correctness signal — robust to
    # softcap saturation, which inflates per-logit max-abs diffs).
    argmax_all = mx.argmax(logits, axis=-1).reshape(-1)
    mx.eval(last, argmax_last, argmax_all)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mx.save_safetensors(
        args.out,
        {
            "token_ids": mx.array(TOKENS, dtype=mx.int32),
            "logits_last": last.astype(mx.float32),
            "argmax_last": argmax_last.astype(mx.int32),
            "argmax_all": argmax_all.astype(mx.int32),
        },
    )
    print(
        f"wrote {args.out}: logits {tuple(logits.shape)} "
        f"argmax_last={int(argmax_last.item())}"
    )


if __name__ == "__main__":
    main()
