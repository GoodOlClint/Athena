#!/usr/bin/env python3
"""Generate the M63.1 DFlash-draft parity fixture.

Loads a z-lab DFlash drafter checkpoint with the bstnxbt/dflash-mlx
reference (the Python oracle), runs the no-cache block forward on
deterministic synthetic inputs, and writes inputs + output to a safetensors
fixture the Swift `DFlashDraftParityTests` compares against.

Run inside the dflash spike venv, e.g.:
    ~/Source/dflash-mlx/.venv/bin/python deploy/dflash/gen_parity_fixture.py \
        --draft ~/.cache/huggingface/hub/models--z-lab--gemma-4-31B-it-DFlash/snapshots/<sha> \
        --out Tests/AthenaCoreTests/Fixtures/dflash_draft_parity.safetensors

The fixture is tied to the specific drafter weights; regenerate if the
drafter changes. embed_scale is pinned to 1.0 (the Swift test pins the
same) so the fixture exercises the full forward without the target scale.
"""
import argparse
import glob
import json
import os

import mlx.core as mx
from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="drafter checkpoint dir")
    ap.add_argument("--out", required=True, help="output safetensors path")
    ap.add_argument("--ctx", type=int, default=4, help="context length")
    ap.add_argument("--seed", type=int, default=63)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.draft, "config.json")))
    margs = DFlashDraftModelArgs.from_dict(cfg)
    model = DFlashDraftModel(margs)

    wfile = glob.glob(os.path.join(args.draft, "*.safetensors"))[0]
    weights = mx.load(wfile)
    model.load_weights(list(weights.items()), strict=True)
    model.embed_scale = 1.0  # pinned; Swift test pins the same
    model.eval()

    H = margs.hidden_size
    n_ctx_feat = len(model.target_layer_ids) * H
    block = margs.block_size

    mx.random.seed(args.seed)
    noise = mx.random.normal((1, block, H)).astype(mx.bfloat16)
    target_hidden = mx.random.normal((1, args.ctx, n_ctx_feat)).astype(mx.bfloat16)
    mx.eval(noise, target_hidden)

    out = model(noise_embedding=noise, target_hidden=target_hidden, cache=None)
    mx.eval(out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mx.save_safetensors(
        args.out,
        {
            "noise_embedding": noise,
            "target_hidden": target_hidden,
            "expected_out": out.astype(mx.float32),
        },
        metadata={
            "block_size": str(block),
            "ctx": str(args.ctx),
            "hidden_size": str(H),
            "target_layers": ",".join(str(i) for i in model.target_layer_ids),
            "embed_scale": "1.0",
            "seed": str(args.seed),
        },
    )
    print(f"wrote {args.out}: noise{tuple(noise.shape)} "
          f"target_hidden{tuple(target_hidden.shape)} out{tuple(out.shape)}")


if __name__ == "__main__":
    main()
