# 033 — Collapse Qwen3.5 MTP onto the substrate drafter path

**Status:** Proposed (brownfield gate — awaiting operator approval; not yet implemented).
**Date:** 2026-06-30
**Milestone:** substrate-consolidation / MTP unification

## Context

Athena's Qwen3.5 multi-token-prediction (MTP) speculative decoding (M2/M40) is a
**fully Athena-owned implementation**: a vendored fused MTP head
(`AthenaQwen35MTP.swift`), an Athena-owned draft/verify loop
(`SpeculativeGeneration.swift` + `SpeculativeSamplingGenerate.swift`), bespoke
GatedDeltaNet (GDN) recurrent-state rollback (`GDNRollback` + an `nConfirmed`
sub-chunk split inside the GDN forward), and a `mtp.fc`-as-plain-`Module` trick to
dodge the substrate quantizer. The Qwen3.5 backbone itself
(`AthenaQwen35.swift`, ~923 LOC) is **bit-identical to upstream `Qwen35.swift`**
apart from this MTP wiring (see `~/Source/mlx/research/upstreaming-audit.md` and
the 2026-06-29 vendored-MLX audit).

The mlx-swift-lm `integration` fork branch (the substrate Athena consumes) has now
**merged a Qwen3.5 MTP implementation**, structured as a **drafter model**:

- `Qwen35MTPDraftModel: StatefulMTPDrafterModel` (`MLXLLM/Models/Qwen35MTP.swift`)
  — the MTP predictor as a separate model, with `mtp.fc` as a normal `Linear`.
- Substrate-owned decode loop: `MTPSpeculativeTokenIterator` driven by the public
  `generate(…, mtpDrafter:, blockSize:)` / `generateTokens(…)` overloads
  (`MLXLMCommon/Evaluate.swift`).
- GDN rollback handled **in the substrate** via copy-cache + replay-committed-prefix
  for non-trimmable Mamba state (no `GDNRollback`/`nConfirmed` seam).
- `MTPDrafterModelFactory` / `MTPDrafterTypeRegistry` registration for
  `qwen3_5` / `qwen3_5_moe` / `qwen3_5_text`; `qwenMTPSanitizeWeights` keeps + shifts
  the `mtp.*` keys correctly.
- Telemetry via `MTPStatsCollecting`.

**This is the exact mechanism Athena already uses for Gemma4 MTP (ADR 032)** —
substrate owns the loop, drafter is a separate accounted model, one `speculative`
knob. Qwen3.5 MTP can therefore stop being a second, parallel, hand-rolled engine
and **converge onto the ADR-032 drafter pattern.** Two Athena features are coupled
to the loop/model the collapse would delete:

1. **TriAttention** KV-eviction (ADR 028 kept it as the sole `kv_compression`
   codec) injects only through the vendored model's `newCache()`
   (`TriAttentionRequestPolicy` TaskLocal) — it has no other consumer.
2. **Prefix-cache (M59.1)** snapshots GDN recurrent state at 512-token boundaries
   *inside Athena's own decode loop*; warm-start injection survives (the substrate
   `generate` accepts `cache: [KVCache]?`), but the mid-decode snapshot hook does not.

The substrate path is **merged-but-unvalidated** ("merged so we can start testing");
it is a fork delta, **not yet upstream in `ml-explore`** (so the substrate is not
fully pristine until upstreamed — tracked separately).

## Decision

**Collapse Qwen3.5 MTP onto the substrate drafter path, staged, behind an A/B flag
gated on a hard equivalence bar; retire TriAttention; preserve the prefix-cache via
a substrate cache-injection seam.**

1. **Full collapse (staged).** Route `qwen3_5` / `qwen3_5_moe` / `qwen3_5_text`
   through the **stock substrate** Qwen35 model + `Qwen35MTPDraftModel` via
   `generate(mtpDrafter:)`, unifying with the Gemma4 ADR-032 path. On completion this
   **deletes** `AthenaQwen35.swift`, `AthenaQwen35MoE.swift`, `AthenaQwen35MTP.swift`,
   `Qwen3NextHelpers.swift`, `SpeculativeGeneration.swift`,
   `SpeculativeSamplingGenerate.swift`, the `GDNRollback`/`nConfirmed` GDN seam, the
   `mtp.fc` quant dodge, and the custom `AthenaModelRegistration` Qwen3.5 model-types.
   `GatedDelta.swift` is handled by the separate already-upstream reconciliation, not here.

2. **A/B behind a flag + hard equivalence gate (binding).** Both paths remain
   selectable until the substrate path proves, on a real `qwen3_5` MTP checkpoint:
   **(a) bit-identical greedy output** vs the current Athena path, **(b) acceptance
   rate ≥ the current CI floor**, **(c) tok/s ≥ the current path**. The vendored loop
   is deleted **only after** the gate is green and a soak on the substrate default.
   This preserves the M2.2/ADR-009 bit-identical-greedy correctness bar.

3. **Retire TriAttention (amends ADR 028).** TriAttention injects only through the
   vendored model and is experimental/off-by-default; rather than build a new
   substrate eviction seam, remove `Sources/AthenaModels/TriAttention/*`, the
   `TriAttentionRequestPolicy`, and collapse `kv_compression` to **`none`-only**
   (`triattention` becomes an unrecognized value → fail-closed at start, same as the
   `turboquant` retirement). Offering TriAttention upstream as a standalone KVCache
   backend remains possible but is **out of scope** and no longer Athena's concern.

4. **Preserve the M59.1 prefix-cache via cache-injection.** Warm-start by passing the
   pre-warmed `[KVCache]` into the substrate `generate(cache:…)`. The lost mid-decode
   512-boundary snapshot hook is replaced by an end-of-decode snapshot (coarser) or a
   small substrate snapshot seam if the spike shows end-of-decode is insufficient.
   Correctness bar unchanged: **bit-identical warm == cold** on the substrate path.

5. **Governor accounting** reuses the ADR-032 Gemma4 drafter-admission path (target +
   drafter footprints both admitted on one Metal budget; ADR 023). One
   inference-execution span (ADR 029).

Decision logic that is MLX-free (the A/B selector, equivalence gate, acceptance
algebra, prefix-cache key/snapshot policy) stays unit-pinned (ADR 008/009).

## Rejected alternatives

- **Engine-only swap** (keep the vendored backbone, replace only head+loop): leaves
  ~1100 LOC of bit-identical backbone vendored and two parallel model classes — the
  "no parallel pipeline" defect this collapse exists to remove.
- **Preserve TriAttention via a new substrate seam:** cost not justified for an
  off-by-default experimental codec with one consumer.
- **Delete-then-validate / trust upstream:** rejected — the substrate path is merged
  but untested; deleting Athena's working, test-pinned loop before equivalence is
  proven risks the bit-identical/perf guarantees.

## Consequences

- Qwen3.5 + Gemma4 MTP unify onto one substrate-owned engine; Athena sheds its
  largest vendored model (~1.4k LOC across the trio + helpers + two loops).
- **Amends ADR 028:** TriAttention retired; `kv_compression` narrows from
  `{none, triattention}` to `{none}`. The `e2e-rbac.sh` fail-closed gate for the
  retired value extends to `triattention`.
- **Amends ADR 032:** the `speculative` knob now drives the substrate drafter loop
  for **both** Gemma4 and Qwen3.5; the "Qwen3.5 fused-`mtp.*`-head byte-unchanged"
  framing is superseded.
- **Supersedes** the Athena-owned MTP loop of M2/M40 for Qwen3.5 (the shared
  `SpeculativeStats`/CI floor is re-pointed at substrate `MTPStatsCollecting`).
- Substrate is **not yet pristine upstream** — the MTP delta lives on the
  `integration` fork until `ml-explore` merges it (the ADR-028 "revival belongs
  upstream" staging ground); tracked in `~/Source/mlx/research/`.
- **Prerequisite/risk:** a real `qwen3_5` MTP checkpoint (with `mtp.*` weights) must
  be available to run the equivalence gate; the prefix-cache snapshot seam is the
  highest-risk slice and may require a small substrate contribution.

Plan: `docs/qwen35-mtp-substrate-cutover-plan.md`.
