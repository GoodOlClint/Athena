# Governor memory-accounting truthfulness — change plan (ADR 023)

**Status:** Plan — awaiting operator review (no code yet).
**Decision of record:** `docs/decisions/023-governor-memory-accounting-truthfulness.md`
(+ its 2026-06-19 amendment: probe is `phys_footprint`, not `activeMemory`).
**Milestone:** operator-assigned tag (ADR 011's long-deferred "next milestone =
governor accounting truthfulness").

## Principle

Smallest-blast-radius first. The decode/inference numerics are byte-unchanged; this
changes only the governor's *accounting* and a process-wide *cache bound*. All
decision logic stays MLX-free and unit-pinned (ADR 008/009); only the probe reads the
MLX/OS counters. Each slice: `./deploy/build.sh Release` → unit + `deploy/e2e-rbac.sh`
→ appVersion bump in the slice commit → annotated tag → `graphify update .`.

## Current state (verified)

- Serve path sets **no** `MLX.Memory.cacheLimit`; only `convert` caps it (256 MB,
  `ModelConvert.swift`). Decode loops call `clearCache()` every 256 tokens
  ([WhisperDecode.swift:188](../Sources/AthenaTranscription/WhisperDecode.swift#L188),
  GuidedGreedy/GuidedSubstrate) — bounds growth *within* a decode only.
- `MemoryGovernor` tracks `residentBytes` (sum of per-model estimates, reconciled by an
  RSS-delta at load) and a `memoryProbe` = process RSS
  ([Load.swift:473](../Sources/athena/Commands/Load.swift#L473),
  `processResidentBytes`). `reconcile` does `residentBytes += observed − estimate`
  ([MemoryGovernor.swift:638](../Sources/AthenaCore/MemoryGovernor.swift#L638)). The MLX
  **cache** grows during decode (not at load), so it never enters `residentBytes` — the
  accounting gap.
- `ProcessMemory.sample()` already returns `(resident, physFootprint)`
  ([ProcessMemory.swift:21](../Sources/AthenaCore/ProcessMemory.swift#L21));
  `relievePromptCachePressureIfNeeded` already uses `physFootprint` (M60.6,
  [MemoryGovernor.swift:490](../Sources/AthenaCore/MemoryGovernor.swift#L490)).
- healthz already surfaces `mlxActiveBytes`/`mlxCacheBytes`
  ([AthenaServer.swift:6214](../Sources/athena/Server/AthenaServer.swift#L6214)).

## Slices

### G1 — bound the serve-path MLX cache (`mlx_cache_limit_bytes`) — ✅ SHIPPED v0.10.191
The load-bearing fix and fully independent of G2/G3. Config key `mlx_cache_limit_bytes`
(CLI `--mlx-cache-limit-bytes` + TOML, full ConfigEditor/LaunchdPlist round-trip); the
MLX-free `GovernorMemory.resolveCacheLimit` (default budget/3, `0`=unbounded) is
unit-pinned (`GovernorMemoryTests`, 657/0); `Load.run` sets `MLX.Memory.cacheLimit`;
`/healthz` exposes `mlxCacheLimitBytes` (e2e 574/0). RUNBOOK soak (plateau ≤ limit) pending.
- New config key `mlx_cache_limit_bytes` (TOML + `AthenaConfig`), default a sensible
  fraction of the budget (~⅓; NOT convert's 256 MB, which would hurt decode throughput).
  `0` ⇒ unbounded (today's behavior, explicit opt-out). MLX-free parse + default
  resolution unit-pinned (range, fraction-of-budget) — mirror the ADR-017
  `max_audio_upload_bytes` pattern.
- In the serve bring-up (`Load.run`, where the governor is built) set
  `MLX.GPU.set(cacheLimit:)` (the same seam `convert` uses) once.
- Surface the resolved limit in `/healthz` next to `mlxCacheBytes`.
- **Test:** unit — config parse + default/fraction/`0`-opt-out resolution. RUNBOOK soak —
  `mlxCacheBytes` plateaus at ~limit instead of climbing to budget.
- **Risk:** low (one process-wide setting; a too-low limit only costs throughput, which
  the default avoids). Effort S–M.

### G2 — admission against `committed = phys_footprint − mlxCacheBytes` (truth)
- Give the governor a footprint probe returning `(physFootprint, cacheBytes)` (injected,
  like `memoryProbe`, so the core stays MLX-free; the closure reads `ProcessMemory.sample`
  + `MLX.Memory.cacheMemory` at the call site in `Load.run`).
- Replace the admission denominator: `committed = physFootprint − cacheBytes`;
  `freeBytes = budget − committed`. On pressure (`committed + estimate > budget`):
  `clearCache()` → re-probe → admit if it now fits, else evict/reject as today.
- Extract the **decision algebra** (committed computation, the reclaim-then-admit
  ladder, what counts toward the ceiling) into a pure MLX-free function unit-pinned per
  ADR 009 (the heart of the change is testable without MLX).
- Keep `residentBytes`/estimates as the per-tenant *attribution* (and a fallback when the
  probe is absent, e.g. tests); `committed` is the *admission* number.
- **Test:** unit — admission algebra (fits/over-budget, cache-as-headroom reclaim path,
  probe-absent fallback). RUNBOOK — mixed load does not overcommit; peak `phys_footprint`
  stays ≤ budget.
- **Risk:** med (touches admission; a bug over-evicts or over-admits). Mitigations: the
  reclaim-then-admit ladder; estimate fallback when probe nil; a config switch to revert
  to estimate-based admission. Effort M.

### G3 — measured per-tenant footprint surfaced (visibility) — ✅ SHIPPED v0.10.192
**Verified during build:** the per-module displayed number was *already* the reconciled
measurement, not the estimate — `reconcile` (M5.1/M5.4) replaces the reservation with the
load-time probe delta (`observed`) and records `learnedFootprint[id]`, and the snapshot
surfaces `reservation?.bytes`. So ADR-023 #3's "numbers are estimates" premise was largely
already addressed. The residual gap was *legibility*: a consumer couldn't tell a reconciled
measurement from a not-yet-reconciled estimate. G3 adds that — a `measured` flag
(`learnedFootprint[id] != nil`) on `ModuleSnapshot` → healthz `ModuleHealth.measured` →
`athena ps` (`~` marks an estimated RESIDENT value) — unit-pinned
(`testMeasuredFlagReflectsReconcile`, 658/0; e2e 574/0). The `phys_footprint`-delta probe
upgrade (vs today's RSS) rides with **G2** (it touches the admission/reconcile probe), so it
is deliberately NOT in G3 — G3 stays additive/visibility-only. Warmup caveat documented:
mmap'd weights fault in lazily, so an at-load delta is a lower bound the running reconcile
tightens.

#### Original G3 sketch (for reference)
- Switch the reconcile/measurement probe from RSS to `phys_footprint` (consistency with
  G2); surface the reconciled per-tenant bytes in `athena ps`/healthz (replacing the
  pre-load estimate in the display).
- Warmup caveat (ADR 023 amendment): mmap'd weights fault in lazily, so the at-load delta
  is a lower bound until the model runs — document it; the running reconcile tightens it.
- **Test:** unit — display picks measured over estimate when present. RUNBOOK —
  per-model measured bytes within tolerance of on-disk weight size.
- **Risk:** low (additive visibility; doesn't change admission). Effort S. Largely
  subsumed by the existing reconcile — smallest slice.

## Sequencing & gates

G1 → G2 → G3. **G1 alone** delivers most of the value (caps the runaway cache, so
`phys_footprint` stops ballooning over budget) and is low-risk — ship it first and observe
before the admission rework. G2 is the riskier admission change; land it behind the revert
switch. G3 is cosmetic-truthful and can land any time after G1.

## Honesty boundary (from ADR 023, binding)

No per-tenant attribution of the reclaimable cache (MLX global pool, no tenant tags). We
**count + bound** the cache (one global number) and **measure active** per-tenant bytes —
never attribute the cache. CI proves the decision algebra + config resolution; the MLX
numerics and the plateaus-below-budget soak are RUNBOOK/gated (ADR 009).
