# 028 — Retire DFlash + TurboQuant to track upstream swift-mlx-lm

**Status:** Accepted — Shipped v0.10.217 (DFlash subsystem + TurboQuant KV-quant removed; tree compiles against the consolidated `integration` substrate again). **Amended (publication S0, 2026-07-07):** the "Kept: TriAttention eviction (Athena-vendored)" posture is **no longer true** — the whole vendored `AthenaModels` target (Qwen3.5 base + fused MTP head + GatedDelta + TriAttention) was **de-vendored and deleted** once the substrate `integration` pin (751aaed) exported `Qwen35Model`, the `Qwen35MTPDraftModel` separate-drafter, and `MLXLMCommon.TriAttention*`. TriAttention now rides the substrate `GenerateParameters.kvScheme` hook (the tripwire below fired early, for the stronger reason it anticipated). Serve path verified bit-identical (greedy, speculative on/off) on Qwen3.5-27B-4bit-mtp.
**Date:** 2026-06-22
**Milestone:** substrate-consolidation

## Context

Athena consumes the MLX language-model substrate as a local SwiftPM path
dependency on a fork of `ml-explore/mlx-swift-lm`. Two Athena features depended
on **bespoke substrate deltas** that the fork carried on top of upstream:

- **DFlash** (ADR 001) — lossless block draft/verify speculative decoding for
  non-MTP targets (Gemma4). Athena vendors the drafter model itself
  (`Sources/AthenaModels/DFlash/`), but it binds to a substrate **capture seam**
  (`Gemma4TextModel.callReturningHidden`) that exists only on the fork's
  `parked/dflash` branch.
- **TurboQuant** (M20) — a from-scratch KV-cache quantization codec. The codec
  lives entirely in the substrate (`KVQuantizationScheme.turboQuant`); Athena
  only selected it via the `kv_compression = "turboquant"` knob. That enum
  exists only on the fork's `parked/turboquant` branch.

The fork is being consolidated so it can be kept in sync with upstream until the
clean pieces are merged (see `~/Source/mlx/research/upstreaming-audit.md`, and
the substrate-side ADR 0002). The consolidation drops both bespoke deltas:

- DFlash's capture seam is a verbatim copy-paste of Gemma4's forward — an
  upstreaming liability with no generic design yet.
- TurboQuant is a **layer violation** (a numerical quant kernel reimplemented in
  Swift above the C++/Metal core) and is **superseded upstream by #230's
  `kvScheme: String?`** extensible KV-compression hook, which collides with
  TurboQuant's bespoke `kvQuantizationScheme` enum + `Float? kvBits` API.

The substrate's active `integration` branch therefore has **neither**
`callReturningHidden` **nor** `KVQuantizationScheme`. Athena's source referenced
both unconditionally, so **the daemon no longer compiled against the
consolidated substrate** until the references were removed. This is a forced
removal, not optional cleanup.

Both features were **default-off** and had **no external consumer** (a downstream client
uses neither; DFlash never shipped a non-MTP accelerator anyone depended on), so
removal is behaviour-preserving for every live caller.

## Decision

**Hard-remove both from Athena** (operator chose hard-remove over preserving
dead snapshot branches; recoverability comes from git history + annotated tags +
the substrate's `parked/dflash` / `parked/turboquant` branches + the upstream
`bstnxbt/dflash-mlx` reference for DFlash).

- **DFlash:** delete the whole `Sources/AthenaModels/DFlash/` subsystem,
  `Sources/AthenaLLM/DFlashGeneration.swift`, all DFlash wiring in
  `MLXLLMModule` (drafter box/fields, `ensureDFlashDraft`, the greedy dispatch
  branch, governor byte accounting), the `dflash_enabled` config key +
  `ATHENA_DFLASH` env override, the 4 `DFlash*Tests` + parity fixture +
  `deploy/dflash/`. The per-request `speculative` override and the **MTP**
  speculative path (milestones M2/M40/M47 — note these are milestones, not
  the issue numbers cited below) are **unaffected** — the `speculative` knob
  itself stays. Two of the three things this sentence originally listed do
  not: ~~`greedyEligible`~~ — **gone in `4517fe5b`, 2026-07-31**, before
  issue #47: the eligibility predicates named in-closure decode branches
  publication S0 deleted, and routing is now `DecodeDispatch.route`; and ~~the shared
  `SpeculativeStats` acceptance observer (used by the MTP path too)~~ —
  **deleted by #47**: publication S0 removed its publisher, and the MTP path
  does **not** use it; ADR 032 S4 speculative stats ride the substrate's
  aggregate `GenerateCompletionInfo` counts instead.
- **TurboQuant:** drop the `KVCompression.turboquant` enum case and its
  substrate-typed `generation` accessor (the dead `KVQuantizationScheme` seam).
  The `kv_compression` knob now accepts only `none` (default) and
  `triattention`. TriAttention is **token eviction, not quantization**, and is
  Athena-vendored against the pristine `KVCache` protocol, so it survives
  untouched. Per-token KV accounting is the fp16 geometry for every remaining
  codec.

**Fail-closed for the retired value:** `kv_compression = "turboquant"` is now an
unrecognized value and refuses daemon start (same as any unknown value), rather
than silently falling back. The `e2e-rbac.sh` phase-12 gate asserts this.

## Consequences

- Athena compiles + serves against the consolidated `integration` substrate;
  the fork can resume tracking upstream.
- Lossless speculative decoding remains available **only for MTP checkpoints**
  (the vendored Qwen3.5/3.6 path). Non-MTP targets (Gemma4, Qwen3) decode one
  token per backbone forward again — the pre-M63 behaviour.
- KV-cache compression narrows from 3 options to 2 (`none`, `triattention`).
- **Revival path:** if DFlash or TurboQuant is wanted later, the upstream story
  is the right home — DFlash needs a *generic* substrate capture seam (not the
  copy-paste), and TurboQuant belongs as a core `mlx` quantization mode wired up
  through `kvScheme` (the bottom-up multi-repo stack in the upstreaming audit /
  substrate ADR 0001). ADR 001 is superseded by this ADR.

Decision logic is mechanical deletion; the surviving MLX-free decision seams
(`KVCompression.resolve`, `servesArch`; ~~MTP `SpeculativeStats`~~ — **deleted #47**, no publisher survived publication S0) stay unit-pinned
(ADR 008/009).

## TriAttention retire-tripwire (WP12, 2026-07-01)

TriAttention was deliberately **retained** above, but the 2026-07-01 audit noted
it is now dead weight in practice: default-off, **Qwen3.5-only** (`Load.swift`
warns it is inert on any other target), ~700 vendored lines, zero e2e coverage,
and the fleet serves Gemma4. So it carries the same park-upstream tripwire the
rest of this ADR applies to DFlash/TurboQuant:

> **Tripwire:** if no Qwen3.5 target is served by **~Dec 2026**, park TriAttention
> upstream (its home is a core `mlx` KV-eviction mode) rather than carrying the
> vendored delta. **Superseded if ADR 033 lands** — the Qwen3.5-MTP→substrate
> cutover plan drops TriAttention as part of that collapse, which retires it
> sooner and for a stronger reason (the operator decision recorded in
> `docs/qwen35-mtp-substrate-cutover-plan.md`).

Related standing calendar items (tracked, not actioned now): the three default-on
revert knobs — `cold_load_wait_secs=0` (ADR 015), `governor_admission_mode`
(ADR 023 G2), `inference_gate_enabled` / `ATHENA_INFERENCE_GATE` (ADR 029), plus
the WP2 `metal_fault_degrade` (ADR 030 P2) — are all <6 months old; revisit for
removal ~Sep 2026 if unexercised.
