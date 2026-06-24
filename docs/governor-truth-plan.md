# Governor memory-accounting truthfulness — change plan (ADR 023)

**Status:** G1 SHIPPED (v0.10.191) · G3 SHIPPED (v0.10.192) · **G2 SHIPPED
(v0.10.209–211, M82)** — admission now meters `max(committed, reserved)` against
budget; `/healthz` surfaces `admissionMode` + honest `freeBytes`; default-ON
behind the `governor_admission_mode` revert knob. **ADR 023 fully shipped.**
RUNBOOK soak (mixed load, peak `phys_footprint` ≤ budget) pending.
**Decision of record:** `docs/decisions/023-governor-memory-accounting-truthfulness.md`
(+ its 2026-06-19 amendment: probe is `phys_footprint`, not `activeMemory`).
**Milestone:** operator-assigned tag (ADR 011's long-deferred "next milestone =
governor accounting truthfulness").

## Principle

Smallest-blast-radius first. The decode/inference numerics are byte-unchanged; this
changes only the governor's *accounting* and a process-wide *cache bound*. All
decision logic stays MLX-free and unit-pinned (ADR 008/009); only the probe reads the
MLX/OS counters. Each slice: `./deploy/build.sh Release` → unit + `deploy/e2e-rbac.sh`
→ appVersion bump in the slice commit → annotated tag.

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

### G2 — admission against `committed = phys_footprint − mlxCacheBytes` (truth) — ✅ SHIPPED v0.10.209–211
**As built:** G2.1 (v0.10.209) landed the MLX-free `GovernorMemory` seam
(`committedBytes`/`admissionDenominator`/`freeBytes`/`fits` + `AdmissionMode`,
11 unit tests). G2.2 (v0.10.210) wired the injected `footprintProbe`
(`phys_footprint` + `MLX.Memory.cacheMemory`) + `reclaimCache` hook into
`MemoryGovernor.makeRoom` (front-door `max(committed, reserved)` gate → rung-1
prompt-pool/cache reclaim → rung-2 reserved-metered LRU eviction → reject on the
committed ceiling only when nothing was evictable, avoiding the async-unload lag),
made `snapshot().freeBytes` honest, and plumbed `governor_admission_mode`
(CLI > TOML > default `footprint`) through `AthenaConfig`/`ConfigEditor`/
`LaunchdPlist` with set-time enum validation; 3 actor tests. G2.3 (v0.10.211)
surfaced `admissionMode` on `GovernorSnapshot` → `/healthz` + the self-describing
OpenAPI healthz description. Honesty boundary held: `residentBytes` stays the
per-tenant reservation sum (+ G3 `measured` flag); the reclaimable cache is one
global number, never attributed.

**Design notes (as decided):**

**Operator decisions (2026-06-21 interview):**
- **Denominator = `max(committed, reserved)`** (warmup-safe). `committed = physFootprint
  − reclaimableCache` is the live ceiling (catches the ungoverned cache + real
  footprint); `reserved` (today's `residentBytes` reservation sum) is the floor during
  the lazy-mmap fault-in window, so a just-loaded-but-not-yet-warm model can't be
  transiently double-admitted. Using the larger of the two never overcommits and never
  regresses pre-G2 behavior.
- **Default ON + revert knob.** New config `governor_admission_mode`
  (`"footprint"` default / `"estimate"` = pre-G2 reservation-only admission). It's a
  correctness fix (mirrors M5.5 RSS-reconcile shipping on); the knob is the escape hatch,
  not an opt-in gate.

**The pure seam (MLX-free, unit-pinned — `GovernorMemory`, ADR 008/009):**
```swift
enum AdmissionMode { case footprint, estimate }
// committed = physFootprint − reclaimableCache, clamped ≥ 0
static func committedBytes(physFootprint: Int, reclaimableCache: Int) -> Int
// denominator = footprint ? max(committed, reserved) : reserved
static func admissionDenominator(mode:, committed:, reserved:) -> Int
static func freeBytes(budget:, denominator:) -> Int        // budget − denom, ≥ 0
static func fits(request:, denominator:, budget:) -> Bool  // denom + request ≤ budget
```
All four are total functions over plain Ints — the entire admission decision is testable
with no MLX/Mach. The actor only supplies the probe values.

**Wiring (smallest blast radius):**
- Add an injected `footprintProbe: @Sendable () -> (physFootprint: Int, cacheBytes: Int)?`
  (nil ⇒ unavailable → fall back to `reserved`, i.e. today's behavior — the tests' path).
  Backed in `Load.run` by `ProcessMemory.sample().physFootprint` + `MLX.Memory.cacheMemory`.
  Keep the existing RSS `memoryProbe` for `relievePressure`/reconcile unchanged.
- Add an injected `reclaimCache: @Sendable () -> Void` (backed by `MLX.Memory.clearCache()`)
  so the admission ladder can reclaim the cache directly without going through an unload.
- `makeRoom(for:requestedBy:)` becomes: compute `denominator` via the seam; if
  `fits` → admit. Else the **reclaim-then-evict ladder**: `reclaimCache()` → re-probe →
  `fits`? → else evict LRU (today's loop) → re-check → shed prompt pool → throw
  `memoryBudgetExceeded` (unchanged terminal). When `mode == .estimate` or the probe is
  nil, `denominator == reserved` and `makeRoom` is byte-identical to today.
- `snapshot().freeBytes` reports the same `max(committed, reserved)`-derived honest free
  (one cheap Mach call per healthz); `residentBytes` (reservation sum) stays for
  per-tenant attribution + the G3 `measured` flag — never re-derived from the cache
  (honesty boundary).

**Test:** unit — `committedBytes` clamp, `max(committed,reserved)` selection, `fits`
fits/over-budget, `.estimate`-mode + probe-nil fall back to reserved (byte-identical),
reclaim-then-evict ordering via a fake probe sequence. RUNBOOK (gated, real model) —
mixed LLM+ASR+diarization load: peak `phys_footprint` stays ≤ budget; admission refuses
the load that today would overcommit.
- **Risk:** med (touches admission; a bug over-evicts or over-admits). Mitigations: the
  `max(committed,reserved)` floor (never under-reserves), the reclaim-then-evict ladder,
  the probe-nil/`.estimate` fallback (exact pre-G2 path), and the `governor_admission_mode`
  revert knob. Effort M.

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

G1 → G3 → **G2** (current). G1+G3 shipped (v0.10.191/192). **G1 alone** delivered most of
the value (caps the runaway cache, so `phys_footprint` stops ballooning over budget) and
was low-risk. **G2 is the remaining slice** — the riskier admission change, landing behind
the `governor_admission_mode` revert switch (default `footprint`). Suggested commit
breakdown (Tests → Security → Quality → Refactor pipeline, each bumping `appVersion` from
v0.10.209):

1. **G2.1 (Tests-first seam)** — `GovernorMemory` admission algebra
   (`committedBytes`/`admissionDenominator`/`freeBytes`/`fits`) + `AdmissionMode`, with
   its unit tests; no wiring yet. Pure, MLX-free, no behavior change.
2. **G2.2 (wire admission)** — `footprintProbe` + `reclaimCache` injection, `makeRoom`
   reclaim-then-evict ladder, `snapshot().freeBytes` honesty; `governor_admission_mode`
   config (CLI + TOML + ConfigEditor/LaunchdPlist round-trip, default `footprint`).
3. **G2.3 (surface + docs)** — healthz `admissionMode` + honest `freeBytes`, ADR 023 →
   G2 SHIPPED, RUNBOOK soak note.

## Honesty boundary (from ADR 023, binding)

No per-tenant attribution of the reclaimable cache (MLX global pool, no tenant tags). We
**count + bound** the cache (one global number) and **measure active** per-tenant bytes —
never attribute the cache. CI proves the decision algebra + config resolution; the MLX
numerics and the plateaus-below-budget soak are RUNBOOK/gated (ADR 009).
