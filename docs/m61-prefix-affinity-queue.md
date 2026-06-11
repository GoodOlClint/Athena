# M61 — prefix-affinity queue scheduling (SPEC, proposed)

**Status:** spec for review. Not built. Milestone number `M61` is a proposal —
versioning is the maintainer's call.

**One-line:** denormalize the prompt-cache key onto the `jobs` row and let the
serial queue worker drain same-prefix jobs consecutively (bounded, fair), so a
fanned-out multi-pass job set reuses one warm KV prefix instead of re-prefilling
each pass under queue contention.

---

## 1. Motivation

The cross-request prompt-prefix KV cache (M59) already reuses a document's prefix
across separate queue submissions ([queue-prompt-cache-contract.md](queue-prompt-cache-contract.md)).
Its weak point is **eviction under contention**: the pool is global and bounded
(default 4 entries), and the worker drains FIFO behind one resident model. If
other tenants' jobs interleave between two passes of the same document, the LRU
or pressure-shed can drop that document's prefix, forcing a cold prefill on the
next pass (`cached_tokens: 0`). Correct, but the multi-pass economics are lost
exactly when the box is busy.

**Enabling insight (client-side):** in the motivating multi-pass extraction
workload, passes 2–N are *mutually independent* — each depends only on pass 1's
backbone output, not on each other. So once pass 1 completes, the client can
**fan them out concurrently**, putting several same-prefix jobs in the queue at
once. M61 makes the worker exploit that: drain the co-queued same-prefix jobs
back-to-back so the prefix is touched (and `lastUsed`-refreshed) repeatedly and
never goes cold mid-set.

**Why this needs a column:** today `prompt_cache_key` lives *inside* the request
BLOB ([AthenaStore.swift jobs table](../Sources/AthenaStore/AthenaStore.swift#L187-L192)).
The scheduler can't group or query on it without deserializing every pending row
on every pick. Denormalizing it (plus `model`) onto the row makes grouping a
cheap indexed lookup and unlocks queue introspection.

**Worker reality:** one resident model, single serial drain
([RequestQueue.swift:137-179](../Sources/athena/Server/RequestQueue.swift#L137-L179)).
"Knock them out at once" means **consecutively, without an intervening cold
prefill** — not in parallel.

---

## 2. Non-goals

- **Not parallel execution.** The worker stays serial; one governed inference at
  a time. This is purely pick-order.
- **Not a correctness mechanism.** The M59 warm-vs-cold path is already
  bit-identical (the M59.1 gate). M61 changes *order*, never output bytes.
- **Not a conversation/continuation engine.** Each job stays a stateless
  one-shot; the client still resends the full prefix per pass (self-healing).
- **Not a client-controlled batch-submit endpoint.** That's a separate, heavier
  API option deliberately deferred (see §9). M61 schedules over independent
  submits.
- **Not a priority queue / SLA tiering.** Affinity is a locality hint bounded by
  fairness, not a priority system.

---

## 3. Design overview

Three independently shippable slices. M61.1 is useful alone (introspection +
retention-correct columns); M61.2 is the actual scheduler; M61.3 is legibility.

| Slice | What | Risk |
|---|---|---|
| **M61.1** | Add `cache_key`, `model` columns to `jobs`; populate on submit; clear under `dropRequestContent`; backfill NULL-safe | low, additive |
| **M61.2** | Bounded prefix-affinity scheduler in `drain()` (opt-in), with streak fairness cap and scope-correct grouping | medium (touches pick-order) |
| **M61.3** | Per-group queue depth on `/healthz` + `athena queue` legibility | low |

Default behavior with M61 present but `queue_prefix_affinity_enabled = false`
(the default) is **byte-identical to today**: strict FIFO. Affinity is opt-in.

---

## 4. M61.1 — denormalized columns

### Schema migration
Follow the established additive pattern ([AthenaStore.swift:222-236](../Sources/AthenaStore/AthenaStore.swift#L222-L236)):

```sql
ALTER TABLE jobs ADD COLUMN cache_key TEXT;   -- NULL ⇒ no affinity grouping
ALTER TABLE jobs ADD COLUMN model     TEXT;   -- resident-model the job targets (post-rebind intent)
```

`try?`-wrapped; the dup-column error on already-migrated DBs is expected and
ignored, same as `owner`/`expires`. Pre-M61 rows get NULL for both — a NULL
`cache_key` is **never** affinity-grouped, so legacy jobs schedule exactly as
they do today (fail-safe).

Optional covering index for the grouping lookup (only if profiling shows the
linear scan matters at realistic queue depths — likely unnecessary for hundreds
of rows, so default to *not* adding it):

```sql
CREATE INDEX IF NOT EXISTS jobs_affinity ON jobs(status, owner, model, cache_key);
```

### Population
At submit ([`handleQueueSubmit`](../Sources/athena/Server/AthenaServer.swift#L2517) →
`insertJob`), parse `prompt_cache_key` and `model` from the conversation body and
pass them to an extended `insertJob(... cacheKey:model:)`. For `kind != "conversation"`
(e.g. embeddings) both are NULL. Parsing failure ⇒ NULL (never block a submit on
a missing hint).

> Decision: store the **raw `cache_key` + `model`**, not a frozen
> `PrefixKVCache.scopeKey` string. The effective scope depends on
> `prompt_cache_scope` (runtime config) and `owner` (already a column), so the
> scheduler computes the grouping at pick time from current config. Freezing a
> scope string would rot if the operator changes scope mode.

### Retention / privacy
`cache_key` may be a document id — sensitive metadata. Parity with M34:
- Under `dropRequestContent`, when the worker clears the request blob on
  completion ([RequestQueue.swift:169-171](../Sources/athena/Server/RequestQueue.swift#L169-L171)),
  also `NULL` out `cache_key` and `model`. Safe because a terminal job is never
  re-scheduled — the scheduler only reads these on `queued`/`running` rows.
- At-rest encryption (M34.3 SQLCipher) covers the whole DB file; the new columns
  inherit it with no extra work.
- `cache_key`/`model` are **never** added to any audit-log `detail` or surfaced
  to a non-owner (the `/healthz` aggregate in M61.3 reports *counts only*, never
  the key values).

---

## 5. M61.2 — bounded affinity scheduler

### Grouping key
A job is affinity-eligible iff `cache_key != NULL`. Two eligible jobs share a
group iff `(owner, model, cache_key)` match. Rationale:
- `cache_key` is the client's explicit "these share a prefix" signal — it's the
  locality hint regardless of `prompt_cache_scope` (which governs actual KV
  reuse, not pick-order).
- `owner` and `model` must match or the jobs can't share KV anyway (principal
  scope + a rebind to a different model both break reuse). Batching across them
  would reorder for no benefit.
- A NULL `model` (no rebind, daemon default) groups with other NULL-model jobs of
  the same owner+key — they all hit the resident default, so they do share.

### Algorithm (replaces the pick in [`drain()`](../Sources/athena/Server/RequestQueue.swift#L137-L179))

```
state across the drain loop: lastGroup: Group? = nil, streak: Int = 0

each iteration:
  pending = listJobs(status in {queued, running}) ordered by created   // unchanged source

  # 1. Safety: an interrupted `running` leftover is already mid-flight — always
  #    finish it first, ignoring affinity (matches today's re-pick behavior).
  if let r = pending.first(where: status == running): job = r; resetAffinity(); run(job); continue

  if pending.isEmpty: sweep(); return

  # 2. Affinity pick (opt-in), bounded by the streak cap.
  if affinityEnabled, let g = lastGroup, streak < maxStreak,
     let same = pending.first(where: group(of:) == g):
       job = same; streak += 1
  else:
       job = pending.first            // oldest — strict FIFO fallback
       let g = group(of: job)         // nil if cache_key NULL
       lastGroup = g
       streak = (g == nil) ? 0 : 1

  run(job)   // update→running, executor, update→done/error, dropRequestContent
```

Properties:
- **Bounded:** at most `maxStreak` consecutive same-group picks, then a forced
  FIFO pick. One tenant's same-key stream can't monopolize the worker.
- **Non-blocking:** only ever picks among *already-queued* jobs. It never holds
  the worker idle waiting for a future same-key arrival — if no co-queued sibling
  exists, it falls straight to FIFO. So a serial (non-fanned-out) pipeline
  behaves exactly like FIFO (groups of size 1).
- **Self-limiting benefit:** if the client doesn't fan out, there's nothing to
  batch and the scheduler is inert — which is why M61 pairs with the client-side
  fan-out, not a substitute for it.

### Fairness analysis
The streak cap converts unbounded affinity into a bounded "run of ≤ maxStreak,
then yield." Worst-case extra latency for a FIFO-waiting job behind an active
group ≈ `maxStreak × per-job-time`. Default `maxStreak = 8` (tunable). Setting it
to `1` ⇒ strict FIFO (affinity off in all but name). There is no per-owner
accounting beyond the streak cap in v1; if multi-tenant fairness needs more, a
follow-up can weight the cap by queue pressure — explicitly deferred.

### Ordering-expectation contract
This reorders execution among co-queued jobs. It is safe because queued jobs are
**independent one-shots** — Athena has never promised cross-job ordering, and the
multi-pass dependency is enforced *client-side* (the reconcile cron doesn't
enqueue pass N+1 until pass N is `done`), never by queue position. The spec makes
this explicit so a future client that *does* rely on FIFO can keep
`queue_prefix_affinity_enabled = false`. `depth()`, `cancel()` (queued-only), and
the M34.1 result sweep are all order-independent and unaffected.

---

## 6. M61.3 — legibility

- `/healthz`: add a `queue` block reporting total depth and the top-K group depths
  as `{count}` **without** key values (privacy) — e.g.
  `"queue":{"depth":12,"groups":[{"n":4},{"n":3},{"n":1,...}]}`. Lets an operator
  see "a 4-deep prefix batch is queued" at a glance.
- `athena queue` (existing CLI): show per-job `cache_key` (owner-scoped /
  admin-only, honoring the same access check as poll) and a "grouped" view.
- One `notice` log when an affinity streak engages/breaks, tagged with the
  existing component/function metadata (per [logging-merge-sortability]).

---

## 7. Config (5-touchpoint, [config-surface])

New `[queue]` keys (TOML, per [config-format-toml]):

```toml
[queue]
# queue_prefix_affinity_enabled    = false   # opt-in; false ⇒ strict FIFO (unchanged)
# queue_prefix_affinity_max_streak = 8        # max consecutive same-prefix picks before a forced FIFO pick; 1 ⇒ FIFO
```

Touchpoints: default in `DefaultConfig.swift`, field in `AthenaConfig.swift`,
wire in `Load.swift` (into `RequestQueue` init), commented block in
`deploy/athena.toml`, consumed in `RequestQueue.drain()`. `appVersion` bump rides
in the slice commit that first exposes a key ([appversion-bump]).

No new mythology-derived sentinel needed — keys are descriptive, matching the
existing `prompt_cache_*` style.

---

## 8. Edge cases & failure modes

| Case | Behavior |
|---|---|
| Affinity off (default) | Strict FIFO; columns still populated for introspection |
| Client doesn't fan out (serial chain) | Groups of size 1; scheduler inert; identical to FIFO |
| Prefix evicted despite batching (pressure-shed mid-streak) | Next pass cold-prefills; `cached_tokens 0`; still correct (self-healing) |
| `cache_key` NULL (embeddings, legacy rows, no hint) | Never grouped; FIFO |
| Job rebinds `model` mid-batch | Different `model` ⇒ different group ⇒ not batched (correct: a rebind breaks KV reuse anyway) |
| `dropRequestContent` on | `cache_key`/`model` cleared on completion alongside blob |
| Interrupted `running` leftover after restart | Picked first, affinity reset (safety rule) |
| Streak cap hit | Forced FIFO pick; oldest waiting job runs |
| `max_streak = 1` | Affinity effectively off |
| Two owners, same `cache_key` value | Different `owner` ⇒ different group (no cross-tenant batching; matches KV scope isolation) |

---

## 9. Alternatives considered

- **Client-controlled batch-submit endpoint** (`POST /v1/queue/batch` with N
  prefix-sharing bodies, drained consecutively). More explicit, but adds API +
  result-envelope surface and an OpenAPI drift-guard entry, and is less general
  than affinity over independent reconcile-driven submits. Deferred; can layer on
  top of M61.1's columns later if explicit control is wanted.
- **Freeze the scope string as a column.** Rejected — rots when
  `prompt_cache_scope` changes; compute at pick time instead (§4).
- **Hold the worker for a future same-key arrival.** Rejected — would idle the
  worker and risk deadlock/starvation; pick only among already-queued jobs.
- **Do nothing, just raise `max_entries`/`idle_ttl`.** The recommended *first*
  step (Stage 0). Sufficient under light load; M61 is the contention answer.

---

## 10. Test plan

Manual host-bound integration tier ([integration-test-topology]) + the e2e gate
(`e2e-rbac.sh`-style, expect the running total to advance, e.g. prior gate + new
asserts / 0):

1. **Migration idempotence** — open a pre-M61 DB twice; assert no error, columns
   present, legacy jobs schedule FIFO.
2. **Column population** — submit a conversation job with `prompt_cache_key` +
   `model`; assert the row carries both; submit embeddings; assert both NULL.
3. **Affinity off = FIFO** — interleave A,B,A,B submits (same key A, key B);
   assert strict created-order drain.
4. **Affinity on, fanned out** — submit pass1(K); on done, fan out
   pass2/3/4(K) + one other-tenant job interleaved; assert 2/3/4 drain
   consecutively and each reports `cached_tokens > 0` (the contention win), while
   the other-tenant job still runs within `max_streak`.
5. **Streak cap fairness** — queue `max_streak + 3` same-key jobs plus one
   other-key job at the tail; assert the other-key job runs no later than
   `max_streak + 1` picks in.
6. **Retention** — with `dropRequestContent`, complete a job; assert `cache_key`
   and `model` NULLed.
7. **Bit-identical** — same document, warm-batched vs cold; assert identical
   response bytes (reaffirms M59 gate holds through the scheduler).
8. **Privacy** — `/healthz` queue block exposes counts only, no key values; CLI
   `cache_key` view honors owner/admin access.

---

## 11. Open questions

1. **Default `max_streak`** — 8 is a guess; want a value tied to a fairness SLO
   (e.g. "no job waits > T seconds behind a batch")? That argues for a *time*
   budget rather than a *count* cap. Count is simpler; flagging the trade.
2. **Per-owner fairness beyond the streak cap** — needed for v1, or is one global
   streak cap enough given the single-tenant-dominant deployment? Leaning enough;
   deferred otherwise.
3. **Index or not** — skip `jobs_affinity` until profiling at realistic depth
   says otherwise? Default: skip.
4. **`/healthz` group exposure** — counts-only is the privacy-safe default;
   confirm no per-key depth is wanted even for admins.
```
