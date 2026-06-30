# Speculative decoding (MTP)

Athena accelerates decoding with **Multi-Token Prediction (MTP)** — a small
drafter proposes several tokens, the target model verifies them in one pass, and
the longest correct prefix is committed. Output is **lossless** (the target
verifies every token). Default-off, opt-in per request or per daemon.

Two backends, selected automatically by the resident model — one knob:

| Backend | Drafter | Targets | How it loads |
|---|---|---|---|
| Qwen3.5 fused head | `mtp.*` weights **in the target checkpoint** | vendored Qwen3.5/3.6 | with the model |
| Gemma 4 (ADR 032) | **separate `gemma4_assistant` checkpoint** | Gemma 4 (dense + 26B-A4B MoE) | paired + loaded alongside the target |

## Turning it on

- **Per daemon:** `speculative = true` in config (the `[inference]` knob). This
  also triggers loading the Gemma 4 drafter at model-load time.
- **Per request:** `"speculative": true|false` on `POST /v1/chat/completions`
  (Athena-native field) overrides the daemon default for that request.

Greedy (temperature 0) and sampling (temperature > 0) are both supported and
both lossless. Structured-output / `logprobs` requests use the non-speculative
guided path (the drafter has no schema mask).

## Gemma 4: pair a drafter

Gemma 4 MTP needs a **separate drafter checkpoint** that the target's
`config.json` does not advertise. Athena resolves the pairing as:

1. `mtp_drafter = "<drafter-store-id>"` in config (explicit override), else
2. the seeded default-drafter map
   (`Sources/AthenaCore/Resources/mtp-drafters.toml`; override per-host at
   `<data_dir>/mtp-drafters.toml`), else
3. none → the knob is inert (single-token decode), with a log pointer.

Pull a target **and** its drafter in one step:

```sh
athena pull mlx-community/gemma-4-e4b-it-4bit --with-drafter
athena pull mlx-community/gemma-4-e4b-it-4bit --check   # report the paired drafter, no download
```

Published pairs (all `mlx-community`, drafters **bf16-only**, match the target
family/size):

| Target | Drafter |
|---|---|
| `gemma-4-31b-it-8bit` | `gemma-4-31B-it-assistant-bf16` |
| `gemma-4-26B-A4B-it-bf16` / `-4bit` | `gemma-4-26B-A4B-it-assistant-bf16` |
| `gemma-4-e4b-it-4bit` | `gemma-4-E4B-it-assistant-bf16` |
| `gemma-4-e2b-it-4bit` | `gemma-4-E2B-it-assistant-bf16` |

With `speculative = true` set, the drafter loads automatically when its target
loads; `athena ps` / `healthz` account for both models on the one Metal budget.

## Honesty boundary

- **Lossless** always (target-verify). On the Gemma 4 path, byte-identity to
  non-speculative is bounded to ~the first 64 tokens by an MLX fused-SDPA
  numerical quirk; the speculation-correctness guarantee still holds beyond that.
- **Speedup is measured, not guaranteed.** Dense Gemma 4 is the reliable win
  (31B-8bit measured **~1.5×**). The 26B-A4B **MoE** at batch-1 is **~3% slower
  with MTP on** (measured 114 vs 118 tok/s; the drafter engages at ~0.56 accept,
  but verifying a drafted block routes to extra experts and the paging overhead
  outweighs the accepted drafts) — so **leave `speculative` off for the MoE**.
  Always check the `MTP speculative …` log line (proposed / accepted /
  accept_rate / passthrough) and tok/s for your model before trusting a win.
- Mid-stream KV-cache quantization makes the iterator fall back to single-token
  (`passthrough` set) — correctness kept, speedup dropped.
