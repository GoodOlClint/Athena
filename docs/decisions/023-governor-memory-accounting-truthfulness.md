# ADR 023 — governor memory-accounting truthfulness + serve-path cache bound

**Status:** Proposed (M79) — awaiting operator review. No production code yet.
Realizes the long-deferred "**next milestone = governor accounting truthfulness**"
from ADR 011 and the standing *heartbeat-RSS-undercounts-GPU* finding. Motivated
by a field observation: after heavy overnight use the daemon's
`phys_footprint` sat at **96.2 GiB** (= the full governor budget; Activity
Monitor confirmed) while the governor believed only **~40 GiB** was resident.

## Context

The daemon's real footprint and the governor's view of it diverge badly. From a
live `/healthz` (128 GB machine, 96 GiB budget):

| Number | Value | What it is |
|---|---|---|
| `phys_footprint` | **96.2 GiB** | what the OS / Activity Monitor sees (peaked 100 GB) |
| `mlxActiveBytes` | 16.9 GiB | MLX live working set ≈ the resident model weights |
| `mlxCacheBytes` | **79.1 GiB** | MLX buffer cache — freed buffers retained, **reclaimable** |
| governor `residentBytes` | 39.6 GiB | sum of per-model **estimates** |

Three distinct problems, one root cause — **the governor accounts by per-model
estimates and is blind to the actual MLX allocator state:**

1. **The cache is ungoverned and grows to the whole budget.** `mlxActiveBytes +
   mlxCacheBytes == 96 GiB` exactly: MLX's allocator has filled the entire
   budget, 79 GiB of it reclaimable cache. The serve path sets **no**
   `MLX.Memory.cacheLimit` (the `convert` path caps it at 256 MB; serving runs
   with MLX's high default), so the cache retains the peak working set
   indefinitely. The per-256-token `clearCache()` in the decode loops bounds
   growth *during* a decode but nothing bounds it across mixed operations.
2. **Admission overcommits.** The governor's `freeBytes` is `budget −
   residentBytes` (≈ 56 GiB free by its math) while the true footprint is 96 GiB
   and **peaked at 100 GiB — over the budget**. The cache is invisible to
   admission, so the governor can admit work the box can't actually fit — the
   mechanism behind tight-memory episodes and a plausible contributor to the
   prior suspected GPU/Metal wedge.
3. **Per-model numbers are estimates, not measurements.** `residentBytes` is a
   conservative pre-load guess (e.g. the LLM's 37.7 GiB estimate vs ~15 GiB
   actual on-disk weights), so `athena ps`/healthz misreport per-tenant cost.

This is not a leak — the 79 GiB is reclaimable cache, bounded by the budget. It
is an **accounting + control** gap: the governor neither *sees* nor *bounds* the
real allocator footprint.

## Decision

Make the governor's view of memory **true**, and **bound** the serve-path cache.
Three changes, smallest-blast-radius first:

1. **Bound the serve-path MLX buffer cache (control).** Set a configurable
   `MLX.Memory.cacheLimit` for the serve process (new `mlx_cache_limit_bytes`;
   default a sensible fraction of the budget, e.g. ~⅓, not the convert path's
   aggressive 256 MB which would hurt decode throughput). The cache stays large
   enough to avoid alloc/free churn but can no longer grow to fill the entire
   budget. Reuses the existing `MLX.Memory.cacheLimit` seam `convert` already
   uses — no new mechanism.

2. **Reconcile admission against the real allocator (truth).** The governor's
   live-footprint probe counts `MLX.Memory.activeMemory` (+ a bounded view of
   the cache as headroom-available-on-pressure) instead of trusting per-model
   estimates alone, so `freeBytes` reflects what the box can actually fit and
   admission stops overcommitting. The decision logic (what counts toward the
   ceiling, how the cache's reclaimability is treated) stays MLX-free and
   unit-pinned (ADR 008/009); only the probe reads the MLX counters.

3. **Measure per-model footprint at load (visibility).** Snapshot
   `MLX.Memory.activeMemory` before/after each module load; the delta is that
   tenant's **measured** resident bytes, replacing the pre-load estimate in
   `athena ps`/healthz/the governor. Cheap, additive, and immediately makes the
   reported per-tenant numbers honest.

### Honesty boundary (binding)

**Per-tenant attribution of the reclaimable cache is out of scope — it is gated
by MLX, not by us.** MLX's buffer cache is a single global pool with no
per-allocation tenant tags; nothing can attribute "these 79 GiB of freed buffers
belong to the LLM vs. transcription." This ADR makes the governor **count** and
**bound** the cache (a single global number), and measures **active** per-tenant
bytes — it does **not** claim per-tenant cache attribution. The cache is fungible
and reclaimable, so that attribution is also low-value.

### Rejected / deferred

- **Lower the budget as the only fix** — rejected; treats the symptom. A smaller
  budget still fills with ungoverned cache; the accounting stays untrue. (Budget
  tuning remains available to the operator, orthogonally.)
- **A custom Metal allocator / per-tenant memory pools** — rejected; a large
  re-architecture for marginal gain over bounding + counting the existing cache.
- **Per-tenant cache tagging** — not feasible (MLX global pool); see the honesty
  boundary.
- **Auto-`clearCache()` on idle** — deferred; a periodic idle flush is a possible
  follow-up, but bounding `cacheLimit` is the cleaner primary lever.

## Consequences

- `phys_footprint` tracks the active working set far more closely; the cache can
  no longer silently grow to the whole budget, and peak stops exceeding budget.
- Admission becomes truthful — the governor stops admitting work that overcommits
  physical memory, reducing the tight-memory / wedge risk.
- `athena ps`/healthz report **measured** per-tenant footprint, not estimates;
  the active/cache split is already surfaced (M60.1) and stays.
- The thesis (ADR 011: the unified governor is the moat) gets the truthful
  accounting it was always supposed to have — the governor finally governs the
  *real* Metal footprint, not an estimate of it.

### Validation

- MLX-free decision logic (admission against measured footprint, cache-as-
  headroom treatment) → unit tests (ADR 008/009).
- A gated soak: drive mixed LLM + transcription + diarization + video load and
  assert `phys_footprint` plateaus **below budget** with the cache bounded, and
  that `mlxCacheBytes` is not monotonically climbing (the existing leak signal
  noted in `AthenaServer.swift`).
- Regression: per-model measured bytes are within a tolerance of on-disk weight
  size for the resident models.

Plan + slices to follow on approval (separate `docs/governor-truth-plan.md`).
