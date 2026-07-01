# Qwen3.5 MTP → substrate drafter cutover plan

**Companion ADR:** `docs/decisions/033-collapse-qwen35-mtp-onto-substrate-drafter.md` (Proposed).
**Operator decisions:** full collapse (staged) · drop TriAttention, keep prefix-cache · A/B behind a flag with a hard equivalence gate.
**Release convention:** each slice = one commit + annotated semantic tag straight to `origin/main`, `Athena.appVersion` bumped in the slice commit (not PRs).

## Goal
Retire Athena's hand-rolled Qwen3.5 MTP engine and route `qwen3_5{,_moe,_text}` through the **stock substrate model + `Qwen35MTPDraftModel`** via `generate(mtpDrafter:)`, unifying with the Gemma4 ADR-032 path. Delete the vendored Qwen3.5 trio + helpers + both Athena draft/verify loops once equivalence is proven.

## Substrate API the cutover targets (integration branch)
- `Qwen35MTPDraftModel: StatefulMTPDrafterModel` — `MLXLLM/Models/Qwen35MTP.swift`
- `generate(input:cache:parameters:context:mtpDrafter:blockSize:)` + `generateTokens(…)` — `MLXLMCommon/Evaluate.swift`
- `MTPSpeculativeTokenIterator` (owns draft/verify + GDN copy/replay rollback) — `MLXLMCommon/`
- `MTPDrafterModelFactory` / `MTPDrafterTypeRegistry` (`qwen3_5`,`qwen3_5_moe`,`qwen3_5_text`), `qwenMTPSanitizeWeights` — `MLXLMCommon/`
- `MTPStatsCollecting` telemetry

## Prerequisite (confirm before S0)
A real `qwen3_5` MTP checkpoint with `mtp.*` weights, loadable both ways (current Athena path + substrate drafter), to drive the equivalence gate. **Blocker if unavailable.**

## Slices (each: discriminating DoD that fails before, passes after — evidenced)

### S0 — A/B selector + equivalence harness *(no deletion)*
Add a per-slot/per-request flag selecting `mtpEngine ∈ {athena, substrate}` (default `athena`). Load the same checkpoint as stock substrate model + `Qwen35MTPDraftModel`; dispatch the substrate path through `generate(mtpDrafter:)`.
**DoD:** harness runs the same `(prompt, seed, params)` through both engines on the real checkpoint and reports: greedy token-id equality, acceptance rate, tok/s. Selector logic unit-pinned (ADR 009). Fails before (no substrate path exists); passes after (both run, numbers captured).

### S1 — Substrate drafter load + governor admission behind the flag
Wire drafter load via `MTPDrafterModelFactory`/registry; admit target+drafter footprints reusing the ADR-032 Gemma4 accounting; dispatch `qwen3_5*` to the substrate loop when `mtpEngine=substrate`.
**DoD:** with the flag on, a `qwen3_5` MTP request serves a correct completion end-to-end; `athena ps`/healthz shows both target+drafter resident bytes; governor admission denies when over budget (no OOM). Athena path unchanged with flag off.

### S2 — Stats bridge
Map substrate `MTPStatsCollecting` → Athena `SpeculativeStats` observer; re-point the CI acceptance-floor test at the substrate telemetry.
**DoD:** `SpeculativeStatsTests` (acceptance ≥ floor) passes against the substrate path; the bit-identical-greedy algebra test (`SpeculativeAcceptanceTests`) still green.

### S3 — Prefix-cache (M59.1) via substrate cache-injection *(highest risk — spike first)*
Half-day spike: confirm warm-start by passing pre-warmed `[KVCache]` into `generate(cache:…, mtpDrafter:)`; decide snapshot strategy (end-of-decode snapshot vs. a small substrate snapshot seam if end-of-decode proves insufficient).
**DoD:** `e2e` prefix-cache warm == cold **bit-identical** on the substrate path (mirrors the current M59.1 gate). If a substrate seam is required, it's a separate tracked contribution; this slice does not regress cold-path correctness.

### S4 — Equivalence gate review → flip default
With S0–S3 green, review the captured equivalence evidence (bit-identical greedy, acceptance ≥ floor, tok/s ≥ current). On pass, flip `mtpEngine` default to `substrate`; soak.
**DoD:** evidence doc attached; default flipped; e2e suite green on the substrate default. **Hard gate — no later slice proceeds if this fails.**

### S5 — Retire TriAttention *(amends ADR 028; can land in parallel after S1)*
Remove `Sources/AthenaModels/TriAttention/*`, `TriAttentionRequestPolicy`, `AthenaCore/KVCompression.swift` `triattention` case; `kv_compression` → `{none}` only (fail-closed on `triattention`). Update `DefaultConfig`/`ConfigEditor`/`AthenaConfig`/`SupportedModels`/docs.
**DoD:** `kv_compression="triattention"` refuses daemon start (e2e phase asserts, mirroring the `turboquant` retirement); `none` path unchanged; no dangling refs.

### S6 — Delete the vendored Qwen3.5 engine *(gated on S4 green + soak)*
Delete `AthenaQwen35.swift`, `AthenaQwen35MoE.swift`, `AthenaQwen35MTP.swift`, `Qwen3NextHelpers.swift`, `SpeculativeGeneration.swift`, `SpeculativeSamplingGenerate.swift`, the custom Qwen3.5 model-type registrations, and the `mtp.fc` dodge. Route `qwen3_5*` through the stock substrate model + drafter unconditionally. Remove the A/B flag (or leave `athena` as a dead-revert no-op for one release).
**DoD:** tree builds against `integration` with the trio gone; full e2e + speculative tests green; `git grep AthenaQwen35` returns only history. ~1.4k LOC removed.

### S7 — Docs / ADR reconciliation *(same edit window as the change)*
Flip ADR 033 → Accepted with shipped versions; amend ADR 028 (TriAttention retired), ADR 032 (unified knob), update the CLAUDE.md ADR list + `SupportedModels` descriptions + `~/Source/mlx/research/athena-vendored-upstreaming-handoff.md` (MTP B2 + TriAttention items resolved).

## Risk register
| Risk | Slice | Mitigation |
|---|---|---|
| No real qwen3_5 MTP checkpoint to test | prereq | confirm availability before S0; blocker if absent |
| Prefix-cache snapshot hook lost with the loop | S3 | spike; end-of-decode snapshot or small substrate seam; bit-identical gate holds the line |
| Substrate path not bit-identical / slower | S4 | hard A/B gate — do not delete (S6) until green; `athena` engine stays as revert |
| GDN copy+replay rollback cost vs Athena sub-chunk | S4 | tok/s is part of the gate |
| Governor double-count target+drafter | S1 | reuse ADR-032 accounting; healthz assertion |
| Substrate MTP only on fork, not upstream | — | accepted; tracked in research/ until ml-explore merges |

## Out of scope
- `GatedDelta.swift` already-upstream reconciliation (separate handoff item).
- Contributing TriAttention upstream (no longer Athena's concern once retired).
- Gemma4 MTP path (already on this pattern; unaffected except shared knob/stats).
