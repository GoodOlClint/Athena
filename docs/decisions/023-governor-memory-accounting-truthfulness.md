# ADR 023 — governor memory-accounting truthfulness + serve-path cache bound

**Status:** Accepted (M79) — **ALL SLICES SHIPPED.** **G1 (bound the serve cache) v0.10.191** (commit `0d2338f`); **G3 (measured per-tenant footprint) v0.10.192** (commit `193fca1`); **G2 (reconcile admission against the real footprint) v0.10.209–211** (M82): the MLX-free admission seam (`committed = phys_footprint − reclaimable cache`, denominator `max(committed, reserved)`) is unit-pinned in `GovernorMemory`; `MemoryGovernor.makeRoom` gates on it (reclaim-then-evict ladder), `snapshot().freeBytes` reports `budget − max(committed, reserved)`, and `/healthz` surfaces `admissionMode`. Default-ON behind the `governor_admission_mode` (`footprint`|`estimate`) revert knob. See `docs/governor-truth-plan.md`. Realizes the long-deferred "**next milestone = governor accounting truthfulness**" from ADR 011 and the standing *heartbeat-RSS-undercounts-GPU* finding. Motivated by a field observation: after heavy overnight use the daemon's `phys_footprint` sat at **96.2 GiB** (= the full governor budget; Activity Monitor confirmed) while the governor believed only **~40 GiB** was resident.

## Context

The daemon's real footprint and the governor's view of it diverge badly. From a live `/healthz` (128 GB machine, 96 GiB budget):

| Number | Value | What it is |
|---|---|---|
| `phys_footprint` | **96.2 GiB** | what the OS / Activity Monitor sees (peaked 100 GB) |
| `mlxActiveBytes` | 16.9 GiB | MLX live working set ≈ the resident model weights |
| `mlxCacheBytes` | **79.1 GiB** | MLX buffer cache — freed buffers retained, **reclaimable** |
| governor `residentBytes` | 39.6 GiB | sum of per-model **estimates** |

Three distinct problems, one root cause — **the governor accounts by per-model estimates and is blind to the actual MLX allocator state:**

1. **The cache is ungoverned and grows to the whole budget.** `mlxActiveBytes + mlxCacheBytes == 96 GiB` exactly: MLX's allocator has filled the entire budget, 79 GiB of it reclaimable cache. The serve path sets **no** `MLX.Memory.cacheLimit` (the `convert` path caps it at 256 MB; serving runs with MLX's high default), so the cache retains the peak working set indefinitely. The per-256-token `clearCache()` in the decode loops bounds growth *during* a decode but nothing bounds it across mixed operations.
2. **Admission overcommits.** The governor's `freeBytes` is `budget − residentBytes` (≈ 56 GiB free by its math) while the true footprint is 96 GiB and **peaked at 100 GiB — over the budget**. The cache is invisible to admission, so the governor can admit work the box can't actually fit — the mechanism behind tight-memory episodes and a plausible contributor to the prior suspected GPU/Metal wedge.
3. **Per-model numbers are estimates, not measurements.** `residentBytes` is a conservative pre-load guess (e.g. the LLM's 37.7 GiB estimate vs ~15 GiB actual on-disk weights), so `athena ps`/healthz misreport per-tenant cost.

This is not a leak — the 79 GiB is reclaimable cache, bounded by the budget. It is an **accounting + control** gap: the governor neither *sees* nor *bounds* the real allocator footprint.

## Decision

Make the governor's view of memory **true**, and **bound** the serve-path cache. Three changes, smallest-blast-radius first:

1. **Bound the serve-path MLX buffer cache (control).** Set a configurable `MLX.Memory.cacheLimit` for the serve process (new `mlx_cache_limit_bytes`; default a sensible fraction of the budget, e.g. ~⅓, not the convert path's aggressive 256 MB which would hurt decode throughput). The cache stays large enough to avoid alloc/free churn but can no longer grow to fill the entire budget. Reuses the existing `MLX.Memory.cacheLimit` seam `convert` already uses — no new mechanism.

2. **Reconcile admission against the real allocator (truth).** The governor's live-footprint probe counts **`phys_footprint` minus the reclaimable MLX buffer cache** — i.e. `committed = phys_footprint − mlxCacheBytes`, the genuinely-pinned memory (MLX active + mmap'd weight pages), with the cache treated as headroom-available-on-pressure (reclaim it before evicting) — instead of trusting per-model estimates alone, so `freeBytes` reflects what the box can actually fit and admission stops overcommitting. The decision logic (what counts toward the ceiling, how the cache's reclaimability is treated) stays MLX-free and unit-pinned (ADR 008/009); only the probe reads the MLX/OS counters. **(Corrected: the denominator is `phys_footprint`, NOT `MLX.Memory.activeMemory` — `activeMemory` misses the file-backed mmap'd weights and would re-introduce the M5.5 order-of-magnitude undercount. See the 2026-06-19 amendment below for the full rationale.)**

3. **Measure per-model footprint at load (visibility).** Snapshot `MLX.Memory.activeMemory` before/after each module load; the delta is that tenant's **measured** resident bytes, replacing the pre-load estimate in `athena ps`/healthz/the governor. Cheap, additive, and immediately makes the reported per-tenant numbers honest.

### Honesty boundary (binding)

**Per-tenant attribution of the reclaimable cache is out of scope — it is gated by MLX, not by us.** MLX's buffer cache is a single global pool with no per-allocation tenant tags; nothing can attribute "these 79 GiB of freed buffers belong to the LLM vs. transcription." This ADR makes the governor **count** and **bound** the cache (a single global number), and measures **active** per-tenant bytes — it does **not** claim per-tenant cache attribution. The cache is fungible and reclaimable, so that attribution is also low-value.

### Rejected / deferred

- **Lower the budget as the only fix** — rejected; treats the symptom. A smaller budget still fills with ungoverned cache; the accounting stays untrue. (Budget tuning remains available to the operator, orthogonally.)
- **A custom Metal allocator / per-tenant memory pools** — rejected; a large re-architecture for marginal gain over bounding + counting the existing cache.
- **Per-tenant cache tagging** — not feasible (MLX global pool); see the honesty boundary.
- **Auto-`clearCache()` on idle** — deferred; a periodic idle flush is a possible follow-up, but bounding `cacheLimit` is the cleaner primary lever.

## Consequences

- `phys_footprint` tracks the active working set far more closely; the cache can no longer silently grow to the whole budget, and peak stops exceeding budget.
- Admission becomes truthful — the governor stops admitting work that overcommits physical memory, reducing the tight-memory / wedge risk.
- `athena ps`/healthz report **measured** per-tenant footprint, not estimates; the active/cache split is already surfaced (M60.1) and stays.
- The thesis (ADR 011: the unified governor is the moat) gets the truthful accounting it was always supposed to have — the governor finally governs the *real* Metal footprint, not an estimate of it.

### Validation

- MLX-free decision logic (admission against measured footprint, cache-as- headroom treatment) → unit tests (ADR 008/009).
- A gated soak: drive mixed LLM + transcription + diarization + video load and assert `phys_footprint` plateaus **below budget** with the cache bounded, and that `mlxCacheBytes` is not monotonically climbing (the existing leak signal noted in `AthenaServer.swift`).
- Regression: per-model measured bytes are within a tolerance of on-disk weight size for the resident models.

## Amendment (2026-06-19) — the probe is `phys_footprint`, NOT `activeMemory`

Review surfaced a conflict between fixes #2/#3 as first drafted and the standing **M5.5** decision ([Load.swift:459-471](../../Sources/athena/Commands/Load.swift#L459)). M5.5 deliberately reconciles against **process RSS, not `MLX.Memory.activeMemory`**, because `activeMemory` sees only the MLX allocator pool and **misses file-backed mmaps from the HF cache** (the embedder / whisper / diarizer / speaker weights load that way): on a 4B embedder the `activeMemory` delta is ~120 MB while the process holds ~8 GB. Metering admission against `activeMemory` would therefore *re-introduce* the order-of-magnitude undercount M5.5 fixed. Fixes #2 and #3 are corrected:

- **The truth probe is `phys_footprint`** (`ProcessMemory.sample().physFootprint`, already used by `relievePromptCachePressureIfNeeded`, M60.6), **not** `activeMemory`. `phys_footprint` is the OS-level number the budget was derived from (the 96.2 GiB) and captures **all three** components — MLX active + MLX cache + resident mmap'd weight pages. The MLX counters `activeMemory`/`cacheMemory` are read **only** to split the footprint for reporting and to size the reclaimable headroom (below), never as the admission denominator.

- **Admission excludes the reclaimable cache** (the concrete rule for #2's "bounded view of the cache as headroom"): define `committed = phys_footprint − mlxCacheBytes` (≈ MLX active + mmap'd weights = the genuinely pinned memory); `freeBytes = budget − committed`. On admission pressure (`committed + estimate > budget`) the governor **`clearCache()` first to actually reclaim**, re-probes, and only then evicts/rejects. This avoids both failure modes — over-commit (ignoring the cache) and over-conservatism (counting reclaimable cache as pinned). Fix #1's `mlx_cache_limit_bytes` keeps `mlxCacheBytes` small so this correction term stays bounded.

- **#3 measures via the `phys_footprint` (RSS) delta at load, not the `activeMemory` delta** — same M5.5 reason. This is largely what `reconcile` already does (`memoryProbe` before/after = an RSS delta); the #3 increment is (a) switching that probe to `phys_footprint` for consistency, (b) surfacing the reconciled per-tenant number in `athena ps`/healthz, and (c) a **warmup caveat**: mmap'd weights fault in lazily, so a pre-first-decode delta under-reads until the model runs — take the measured number after a warmup forward, or accept it as a lower bound the running reconcile tightens.

Net: the moat-defining change is **#1 (bound the cache)** plus **#2 (meter `committed = phys_footprint − reclaimable cache`)**; **#3 is mostly already present** (reconcile) plus per-tenant surfacing. M5.5's RSS-over-`activeMemory` instinct was right; this ADR upgrades RSS → `phys_footprint` (its strictly-more-complete sibling), it does **not** revert to `activeMemory`. The honesty boundary and rejected alternatives are unchanged.

Plan + slices to follow on approval (separate `docs/governor-truth-plan.md`).
