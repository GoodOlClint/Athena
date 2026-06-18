# ADR 016 — model-class-aware convert + cause-naming load errors

**Status:** Proposed (M73) — awaiting operator approval.

## Context

The substrate exposes three model-class factories, each with its own
`model_type` registry: `LLMModelFactory` (generative), `VLMModelFactory`
(vision), and `EmbedderModelFactory` (embedding). Athena's serve paths already
use the right one per modality — the embedding module loads through
`EmbedderModelFactory`.

`athena convert`, however, is class-blind: it routes only `VLMModelFactory`
(when `vision_config` is present) or the LLM `loadModelContainer` otherwise. It
never considers the embedder factory. The result is that supported embedding
models fail at convert time:

- `mxbai-embed-large-v1` (`bert`) → `unsupportedModelType("bert")` (the LLM
  registry has no `bert`, though the embedder registry does, and even ships this
  exact model as a default).
- `embeddinggemma-300m` (`gemma3_text`) → `keyNotFound(model.norm.weight)`: the
  LLM factory *does* claim `gemma3_text`, so convert silently builds the
  **causal-LM** Gemma3 architecture and then fails on weights an embedding
  checkpoint does not carry — a wrong-arch pick surfaced as an opaque error.

This is a routing defect, not a missing architecture.

## Decision

1. **`convert` becomes model-class-aware.** A shared, MLX-free detector
   classifies a checkpoint as generative / vision / embedding / unknown from
   on-disk metadata (`config.json` + sentence-transformers artifacts +
   substrate registry membership). Convert dispatches the existing LLM and VLM
   routes by class instead of by an ad-hoc `vision_config` check.

2. **Embedding models are redirected, not converted.** `convert` is a
   *generative*-model quantization pipeline. Embedding models load in source
   precision directly via the serve path and do not need a converted on-disk
   artifact. So convert refuses an embedding model with a cause-naming 4xx
   pointing at `athena pull` + set-as-embedding-model — it does **not** add a
   third (embedder) convert/quantize route.

3. **Errors name the cause.** Convert no longer prints raw substrate
   `unsupportedModelType` / `keyNotFound` dumps. It names the detected class and
   the corrective action, and the detector runs **before** weights download so a
   misrouted model fails fast (config-only fetch), not after a multi-GB pull.

## Alternatives considered

- **Teach convert to quantize embedders via `EmbedderModelFactory` (a third
  route).** Rejected for now: embedding models are small, quantization value is
  marginal, and the serve path already loads them in source precision. Adds a
  parallel quant pipeline (config emission, per-layer rules) for little gain.
  Tripwire: revisit if a consumer needs quantized embedding checkpoints staged
  in the store.
- **Leave convert generative-only, no detection.** Rejected: keeps the opaque
  wrong-arch failure (`keyNotFound`) and the wasteful full download before
  failing.
- **Maintain an Athena-side allowlist of convertible `model_type`s.** Rejected:
  duplicates the substrate's registries (which are the real source of truth and
  grow upstream for free) and reintroduces exactly the "chase every model"
  treadmill this ADR removes. `SupportedModels` stays a *validated label*, not a
  convert gate.

## Consequences

- Both failing embedding models work through the path they should always have
  used (`pull` → serve), with a clear message instead of a stack trace.
- Convert's class routing is bounded by the number of factories (three), not the
  number of models — no per-model maintenance.
- A genuinely novel **architecture** the substrate does not implement still
  needs Swift code upstream (or the deferred signed-arch-plugin workstream);
  this ADR does not change that, but it stops *class*-routing gaps from
  masquerading as unsupported architectures.
- New MLX-free detector + decision-table tests under `swift test` (ADR 009).

## Plan

`docs/m73-model-class-routing-plan.md`.
