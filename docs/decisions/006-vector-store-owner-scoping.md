# 006 — Vector-store owner-scoping

**Status:** Implemented
**Date:** 2026-06-12
**Milestone:** M66 (audit-remediation; resolves standing DECISION #3; audit H5)

**Implemented:** M66.6 (v0.10.128). Additive `owner TEXT` column + `vectors_owner`
index; `putVector`/`getVector`/`deleteVector`/`allVectors` gained the owner; the
`VectorStore` actor carries a `Caller` and owner-filters `query`/`stats`/`delete`
(cache) with the SQLite layer enforcing too (defense-in-depth). Upsert stamps the
principal; a cross-owner upsert is 409, a cross-owner delete is 404; legacy
NULL-owner rows are admin-only; auth-off shares one space. OpenAPI vectors
descriptions document it.

## Context

The built-in vector DB (`/v1/vectors`) has **no owner column** — any caller holding the
vectors permission can read, query, and delete every principal's vectors. Queue jobs
and usage counters are already owner-scoped (per-principal); vectors are the outlier,
so one tenant's embeddings are visible to another. Either the store is intended to be
shared single-tenant (then document and close H5), or it should be owner-scoped like
the rest of the data plane.

## Decision

**Add owner-scoping.** Schema migration + enforcement:

- add a **nullable `owner` column** to the vectors table, set to the authenticated
  principal on upsert;
- `query` / `get` / `delete` (and `stats`) filter to the caller's `owner`; an **admin**
  principal sees across owners;
- **legacy `NULL`-owner rows** (written before the migration) are treated as
  **admin-only** — they are not silently exposed to, or claimable by, the next caller;
- auth-off loopback (single trusted operator) sees everything, mirroring queue/usage.

The migration is additive and NULL-safe (matches the M36 token-expiry column pattern):
`ALTER TABLE … ADD COLUMN owner TEXT` with the filter logic treating `NULL` as
admin-only.

## Consequences

- Cross-principal vector leakage closed; vectors join queue/usage as owner-scoped.
- Migration touches `VectorStore` + the SQLite schema version; existing rows become
  admin-only (operators relying on shared visibility must re-own or use an admin token).
- Query paths gain an `owner` predicate — keep it indexed so the M69.3 resident-matrix
  caching (H3) still pays off.
- Document the new semantics in the vectors section of the OpenAPI spec / docs.
