# 007 — Native `/api` metering + token-budget quotas (in-program)

**Status:** Accepted — **SUPERSEDED IN PART by ADR 041 (2026-07-25), which carries #9 forward and closes #8.** Native `/api` metering (#8) + NA8 single-principal resolution shipped (v0.10.140, `AthenaServer.swift` `meter()` chokepoint). Token-budget quotas (#9) NOT implemented — milestone never scheduled; now designed in ADR 041.

**AMENDED 2026-07-25 — #8 is closed by obsolescence, not by implementation.** ADR 013 made `/v1` the sole inference surface and `/api/*` the control plane; ADR 025 S2 removed the queue and ADR 031 removed `/api/chat`, and the `/api/embed` alias went with it (`AthenaServer.swift:239`). No native inference route survives, so `/api/*` carries **zero model tokens** and there is nothing left to meter. The caveat below names `handleNativeEmbed`, which no longer exists. #8 requires no further work; metering is complete for every surface that has tokens. NA8 remains in force (`bearerPrincipal` / `usagePrincipal`). #9 is the only live item and is designed in [ADR 041](041-per-principal-token-budgets.md).
**Date:** 2026-06-12
**Milestone:** M69+ (audit-remediation; resolves standing DECISION #5; backlog #8/#9 + audit NA8)

## Context

Usage metering covers the `/v1/*` surface only; native `/api/*` traffic is **not
metered** (pre-existing backlog #8). There are no **token-budget quotas** (#9, which
needs #8 first). Separately, **NA8** notes the streaming metering path re-resolves the
bearer a second time instead of reusing the principal AuthMiddleware already resolved.
These were carried as "M27 follow-ups," outside the audit program's milestones.

## Decision

**Bring #8 and #9 into the audit-remediation program** as their own milestone (rather
than leaving them as open-ended backlog):

- **#8 — native `/api` metering:** meter `/api/*` through the same per-principal SQLite
  counters as `/v1/*`, so `GET /api/usage` and `athena usage` reflect the whole data
  plane, not half of it.
- **#9 — token-budget quotas:** a per-principal token budget (config/SQLite), enforced
  at admission, returning the standard quota error envelope when exceeded. Depends on
  #8's accounting.
- **NA8** folds in: thread the AuthMiddleware-resolved principal through to metering so
  there is a single auth resolution per request. (NA8's cheap half can still land
  early in M69.2; the quota enforcement rides #8/#9.)

The new milestone slots after the core security/correctness milestones (it is
operability + product surface, not a vulnerability), with its exact position recorded
in the live tracker.

## Consequences

- `/api` and `/v1` metering reach parity; quotas become enforceable per principal.
- New config keys (per-principal budget, window) following the 5-touchpoint config
  pattern; new error code in the envelope family.
- The program grows one milestone; the tracker's milestone list and the
  commercial-readiness ledger are updated to reflect #8/#9 moving from backlog to
  planned.
- NA8's single-resolution change is a small perf/correctness win independent of the
  rest.
