# 032 — Gemma 4 MTP speculative decoding (second drafter backend)

**Status:** Accepted — **IMPLEMENTED** (S1 v0.10.227; S2–S5 v0.10.228; see
`docs/gemma4-mtp-plan.md`). Drafter classification + pairing map + config, drafter
load via `MTPDrafterModelFactory`, the `generate(mtpDrafter:)` drive behind the
`speculative` knob, acceptance-rate log, and surface/docs all landed. Pending the
heavy end-to-end DoD (real E4B pair) on the Metal host.
**Date:** 2026-06-30
**Milestone:** M83
**Relates to:** ADR 011 (governor / never compose at inference), ADR 013 (`/v1`
single inference surface), ADR 018/020 (multi-backend precedent), ADR 021
(ModelSupport + the no-hard-coded-repo guidance rule), ADR 023 (admission
footprint truthfulness), ADR 026 (store-is-registry, per-module keys), ADR 028
(MTP path retained; revival belongs upstream), ADR 029 (inference execution
gate).

## Context

Athena's only live speculative path is **MTP, and only on Qwen3.5/3.6
checkpoints** that carry a fused `mtp.*` head, driven by Athena's own
`SpeculativeGeneration.swift` loop. Gemma 4 (and Qwen3-dense) had **no**
speculative decoding since DFlash was removed (ADR 028).

Google ships Gemma 4 MTP — but as a **separate drafter checkpoint**
(`model_type: gemma4_assistant`), not a fused head. The drafter shares the
target's input embeddings and builds on its last-layer activations. The
mlx-swift-lm substrate now implements it end-to-end on the `integration` branch:
the drafter (`Gemma4AssistantDraftModel`), `MTPDrafterModelFactory`, the
`emitDrafterState` capture seam, a ready-made `MTPSpeculativeTokenIterator`, and
a `generate(… mtpDrafter:)` overload — with transparent single-token passthrough
if the target stops emitting drafter state. (E-series centroid embedder included;
the earlier masked-embedder TODO is resolved.)

So this is structurally **different** from the Qwen path — two models, and the
substrate (not Athena) owns the decode loop. Two facts shaped the design:

1. **The target does not advertise its drafter.** Gemma 4 `config.json` carries
   no `mtp_*`/`assistant_*` field (unlike Qwen's `mtp_num_hidden_layers`), and
   the drafter lives in its own HF repo (verified pair:
   `gemma-4-31b-it-8bit` ↔ `gemma-4-31B-it-assistant-bf16`). Pairing cannot be
   sniffed from metadata.
2. **Substrate-first** (ADR 028): the drafter, loop, and capture seam belong
   upstream, not as an Athena-local delta. Athena consumes them.

## Decision

Wire Gemma 4 MTP as a **second speculative-decode backend behind the existing
per-request `speculative` knob**, driven by the substrate's
`generate(… mtpDrafter:)` overload. This is multi-backend-behind-one-knob (ADR
018/020 precedent), **not** a parallel pipeline (the substrate owns Gemma's
loop; Athena's Qwen loop is untouched).

- **Drafter = a separate `gemma4_assistant` store model**, classified by a new
  `ModelModality.mtpDrafter` in the shared `ModelSupport` predicate (ADR 021): a
  paired drafter, never an independently-servable slot (so the ADR 026 ambiguity
  rule never auto-selects it), pulled (bf16) not converted (`convert` redirects
  it like the other non-quantizable classes).
- **Pairing** resolves in order: the explicit `mtp_drafter` config key > a
  **seeded, operator-overridable default-drafter map** > none (knob inert). The
  map ships as **data** (`Sources/AthenaCore/Resources/mtp-drafters.toml`,
  overlaid by `<data_dir>/mtp-drafters.toml`), keyed by target store-id basename
  (case-insensitive) → drafter HF id. `athena pull --with-drafter` and
  `pull --check` use it.
- **Loading + governor:** when a Gemma 4 target loads with a resolved, present
  drafter and `speculative` is on, load the drafter via
  `MTPDrafterModelFactory.shared.load` into a second resident slot; it holds no
  target state, so load once and reuse. Admission counts both footprints (ADR
  023); the iterator runs inside one inference-execution span (ADR 029).
- **Default-off, opt-in, lossless** — same posture as the Qwen MTP path.

**Guidance rule (ADR 021 D5) — surfaced and resolved.** A target→drafter map is
enumerated repo ids; D5 forbids hard-coded repo ids in *code* (and in
error/guidance strings). It is therefore shipped as **operator-overridable config
data, not Swift constants** — the same first-boot-seed posture ADR 026 uses for
default model ids — so a moved/renamed repo is fixed without a recompile.
Error/guidance strings still name the structural requirement ("no drafter
paired"), never a repo id.

## Rejected alternatives

- **Vendor the drafter into Athena** — rejected; substrate-first (ADR 028).
- **Config-metadata auto-detect** — impossible; the Gemma 4 target carries no
  drafter field. A real packaging fact, not a preference.
- **Naming-convention derivation** (`<target>-assistant-*`) — rejected; the
  casing/dtype suffixes don't line up (`31b-…-8bit` vs `31B-…-assistant-bf16`),
  so derivation guesses and mis-pulls.
- **Hard-coded map in Swift** — rejected per ADR 021 D5; ships as data.
- **Unify the two speculative loops** — rejected; different drafter mechanisms,
  and the substrate owns Gemma's loop. One knob, two backends (ADR 018/020).

## Consequences

- The `speculative` knob lights up for Gemma 4 when a paired drafter is present;
  inert (single-token) otherwise — same "inert when no head" semantics it has on
  non-MTP Qwen today. The Qwen3.5 MTP path is byte-unchanged.
- A second resident model on one Metal budget when speculative is engaged for
  Gemma 4 — accounted, not estimated (ADR 023). The drafter is small (E-series
  ~160 MB; 26B/31B ~0.8–0.9 GB).
- **Honesty boundary:** lossless is inherited from the substrate's target-verify;
  byte-identity to non-speculative is bounded ~64 tokens (an MLX fused-SDPA
  quirk), and **speedup is measured, not promised** — at batch-1 the 26B-A4B MoE
  drafter may not help (expert-load overhead). bf16 drafters only; single-stream;
  mid-stream KV-quant onset degrades to passthrough (correctness kept).
- ModelSupport gains a consumer-visible modality; `ModelClass`/audio detectors
  unchanged. Decision logic (classification, pairing, eligibility) MLX-free and
  unit-pinned (ADR 008/009).
