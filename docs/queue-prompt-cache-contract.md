# Multi-pass prefix reuse over the durable queue

**Audience:** clients driving a chained, multi-pass extraction job set over
`POST /v1/queue/conversation` + `GET /v1/queue/{id}`, where several passes share
one byte-identical document prefix and each pass emits a different JSON schema.

**TL;DR — Option A already works, unchanged.** The prompt-prefix KV cache (M59)
is a server-side pool keyed by `(resident-model, scope)` plus a longest-common
**token**-prefix scan. It has no notion of "request" or "job" — it lives in the
model's serial access domain and is consulted on every generation, sync or
queued. Two separate queue submissions by the same principal whose leading
messages render to a byte-identical token prefix reuse the first pass's KV
automatically. The queue worker already forwards `prompt_cache_key` and the
principal into the same `generateMetered` path the sync endpoint uses, and the
polled job result is the full `ChatCompletionResponse` (with `usage` and
`finish_reason`). **No new request fields are required; no Athena code change is
required.** What follows is the contract and the few knobs that make it robust
under a busy, multi-tenant queue.

---

## 1. Mechanism: A, not B

**Use Option A (cross-job prefix cache by key + identical-prefix resend).**

Option B (server-side conversation continuation via `parent_id`, sending only
the new turn) is the *wrong* trade for this architecture, and I'm declining to
build it. Reasons:

- **The KV pool is in-memory and pressure-shed by design.** It is bounded by
  count, bytes, and idle-TTL, and the memory governor sheds the whole pool when
  the process crosses 90 % of its budget — both on model-load admission and
  after every generation (M59.2 / M60.6). So even a `parent_id` could *not*
  guarantee the parent's KV is still resident when pass N+1 runs. Option B would
  therefore *still* need a full-prefix fallback path — it buys no reliability
  over A.
- **Option B requires the server to become stateful** (durably store and
  reconstruct conversation context, reconcile drift, version a conversation
  graph). The queue is deliberately "one job = one stateless request→response."
  Option B duplicates, statefully, what the prompt cache already does
  statelessly.
- **Option A is self-healing.** Because the client resends the full prefix every
  pass, the cache is a *pure optimization*: a miss (cold, evicted, or disabled)
  costs a full prefill and returns `cached_tokens: 0` — never an error, never a
  wrong answer. There is no eviction failure mode to handle.

The only thing Option B saves is wire bytes (the 6–40 KB prefix, ×3 passes), on
what is typically a LAN or loopback hop. That is not worth a stateful
conversation engine and a second failure mode. If wire size ever becomes the
real bottleneck we can revisit a *stateless* `prefix_from: <job_id>` hint (server
re-reads the prior job's stored request blob and re-runs the same common-prefix
scan) — but that is a wire optimization, not a KV-reuse mechanism, and it is not
needed today.

---

## 2. Preconditions (operator-side, one-time)

The cache is **off by default**. Confirm the daemon serving these jobs has it on
and tuned for the workload. All keys live under `[prompt_cache]` in
`athena.toml` (5-touchpoint config; see [config-surface]):

```toml
[prompt_cache]
prompt_cache_enabled       = true     # default false — MUST be on
prompt_cache_scope         = "both"   # see §4; "principal" also works
prompt_cache_max_entries   = 16       # default 4 — raise for concurrency, see §6
prompt_cache_idle_ttl_secs = 600      # default 600 (10 min) — see §5
prompt_cache_max_bytes     = 0        # 0 ⇒ governor-derived (25 % of budget)
```

Two more standing constraints:

- **Resident model must be the MTP path** (`MLXLLMModule` / the Qwen3.5 MTP
  model). The cross-request KV cache only exists on that path; non-MTP substrate
  models do not cache. A queued job that rebinds (`"model"` field) to a non-MTP
  model silently gets no reuse (`cached_tokens: 0`), not an error.
- **The shared prefix must be ≥ 512 tokens.** Reuse is granted only when the
  shared prefix spans at least one full 512-token chunk
  ([`PrefixKVCache.chunkSize`](../Sources/AthenaLLM/PrefixKVCache.swift#L61)), and
  an entry is only stored when at least one 512-boundary checkpoint was captured.
  Your 6–40 KB document prefixes clear this comfortably; very short documents
  will simply not cache.

---

## 3. What you send per pass (request shape)

Submit body for **every** pass — same shape you already use, with two fields
added/used:

```jsonc
POST /v1/queue/conversation
Authorization: Bearer <same token for all 4 passes>
{
  "model": "...",                       // optional; if set, MUST be identical across passes
  "messages": [
    {"role": "system", "content": "<SYS>"},          // ── byte-identical leading prefix ──
    {"role": "user",   "content": "<DOCUMENT_PROMPT>"}, //   (same content, same order, same roles)
    ...pass-specific TAIL messages...                  // ── varies per pass (see §7) ──
  ],
  "temperature": 0.1,                   // honored per pass (inert under a schema; see §below)
  "max_tokens": 16384,                  // honored per pass
  "max_completion_tokens": 16384,       // honored per pass; wins over max_tokens
  "response_format": {                  // honored per pass — different schema each pass is fine
    "type": "json_schema",
    "json_schema": {"name": "...", "schema": { ... }, "strict": false}
  },
  "prompt_cache_key": "<DOCUMENT_ID>"   // ← add this: stable per document, same across its passes
}
```

Returns `{"id": "...", "status": "queued"}`.

**The one rule that makes reuse fire:** the *leading* messages
(`[system, user(document)]`) must render to a byte-identical **token** prefix
across all passes — same content, same roles, same order, same `model`,
same `chat_template_kwargs`. Put **all** per-pass variation — the pass
instruction *and* the backbone result that passes 2–4 reference — in the
**trailing** messages only (§7). The common-prefix scan is over the *rendered*
token stream, so anything that perturbs how the leading messages tokenize
(a different model, a different template kwarg, reordered messages) breaks the
prefix and you fall back to a cold prefill.

> Chat-template note: the Qwen-family template renders each message
> independently in sequence, so prepending identical messages produces an
> identical leading token run regardless of what trailing messages follow. Keep
> the leading two messages fixed and you are safe.

---

## 4. Keying & scope

The cache key is `resident-model-id` + a **scope string**, never a hash you
control. Within a scope, lookup is a longest-common-token-prefix scan over stored
entries; the longest match that spans ≥ 512 tokens and has a stored 512-boundary
checkpoint wins ([PrefixKVCache.swift:201-252](../Sources/AthenaLLM/PrefixKVCache.swift#L201-L252)).

Scope is operator-configured (`prompt_cache_scope`), not per-request:

| `prompt_cache_scope` | scope string | `prompt_cache_key` role |
|---|---|---|
| `principal` (default) | `<model>·p:<principal>` | **ignored for keying** — all your jobs share one scope; documents are disambiguated purely by the prefix scan |
| `cache_key` | `<model>·k:<cache_key or principal>` | groups by key |
| `both` (recommended) | `<model>·p:<principal>·k:<cache_key>` | narrowest — each `prompt_cache_key` is its own sub-scope |

**Recommendation: `scope = "both"`, `prompt_cache_key = <document_id>`.** This
makes a document's reuse *deterministic and isolated*: pass 1 cold-prefills and
stores under that document's scope; passes 2–4 match only that document's entry;
two different documents never even compare prefixes. Under the `principal`
default it *also* works (the scan disambiguates by content, and never returns a
wrong answer — at worst it matches only a shared system-prompt run and resumes
from a lower boundary), but `both` + per-doc key is the predictable, legible
choice and is what I'd build the client against.

`prompt_cache_key` is a hint, never an auth boundary — see §8.

---

## 5. TTL / warmth across the gap between passes

- **Idle-TTL, default 600 s (10 min)**, configurable via
  `prompt_cache_idle_ttl_secs`. The clock is **idle time since last use**, and
  `lastUsed` is refreshed on **every hit and every store**
  ([PrefixKVCache.swift:248](../Sources/AthenaLLM/PrefixKVCache.swift#L248)). So
  it's the **gap between consecutive passes** that must stay under the TTL, not
  the document's total wall-clock. A document whose passes are each < 10 min
  apart stays warm indefinitely.
- Each successful pass makes the document's entry the **most-recently-used**
  entry, which also makes it the *last* LRU eviction candidate — active
  documents naturally resist eviction.
- **Pinning / extending:** there is no explicit pin API, and I'm intentionally
  not adding one — a pin would fight the memory governor (the pool is shed under
  pressure precisely to keep the daemon alive). The supported levers are
  `idle_ttl_secs` (raise it if your reconcile cadence + queue depth can space
  passes further apart) and `max_entries` (§6). Because the client always
  resends the full prefix, a lost entry is a silent re-prefill, not a failure —
  so warmth is a cost lever, never a correctness one.

---

## 6. Eviction semantics (and why you never need a fallback branch)

The pool evicts on three axes, all refcount-protected (an entry held by an
in-flight generation is never evicted):

- **Count** — `prompt_cache_max_entries` (default **4**). Note this is a
  **global** pool, not per-scope. Every `store()` from *any* principal's job
  appends an entry and may evict the LRU one. On a busy multi-tenant queue
  draining serially behind one model, ~`max_entries` intervening jobs between
  two of your passes can evict your document's prefix. **Raise
  `max_entries`** to comfortably exceed your expected concurrent in-flight
  document count (e.g. 16–32), bounded by the byte cap below.
- **Bytes** — `prompt_cache_max_bytes` (0 ⇒ governor-derived, 25 % of the memory
  budget). Large prefixes × many entries hit this ceiling; it trades off against
  `max_entries`.
- **Idle-TTL** — §5.
- **Pressure-shed** — at > 90 % of the memory budget the governor flushes the
  whole pool (all refcount-0 entries) to admit a load or after a generation
  (M60.6).

**What a submit/poll returns if the prefix was evicted before pass N+1:** exactly
the same thing as a normal job — `status: queued → running → done`, a correct
result, and `usage.prompt_tokens_details.cached_tokens: 0` (i.e. the field is
absent or zero). There is **no eviction error, no special status, no degraded
mode**. Your "fallback to a full-prefix resubmit" *is the steady-state request* —
you already send the full prefix every pass — so detection is optional. If you
want to *observe* reuse health, read `cached_tokens` per pass (§below) and/or the
pool stats on `GET /healthz`.

Status vocabulary is unchanged: `queued | running | done | error | canceled`.

---

## 7. The per-pass tail (where backbone output goes)

Passes 2–4 reference pass 1's events (`e0, e1, …`), so their input must *carry*
the backbone result — and that must live in the **trailing** messages so the
leading `[system, user(document)]` prefix stays byte-identical. Either shape
works; pick one and keep it stable:

**Shape A — assistant turn + new instruction (most cache-friendly):**
```jsonc
"messages": [
  {"role": "system",    "content": "<SYS>"},              // shared prefix
  {"role": "user",      "content": "<DOCUMENT_PROMPT>"},   // shared prefix
  {"role": "assistant", "content": "<BACKBONE_JSON>"},     // tail: pass-1 output
  {"role": "user",      "content": "<QUOTES_INSTRUCTION>"} // tail: this pass's ask
]
```

**Shape B — single user turn embedding backbone + instruction:**
```jsonc
"messages": [
  {"role": "system", "content": "<SYS>"},               // shared prefix
  {"role": "user",   "content": "<DOCUMENT_PROMPT>"},    // shared prefix
  {"role": "user",   "content": "Given these events: <BACKBONE_SUMMARY>\n<QUOTES_INSTRUCTION>"}
]
```

In both, reuse resumes from the largest 512-boundary `B ≤ shared-prefix length`.
You re-prefill only `(prefix_len − B)` leftover tokens of the last partial shared
chunk **plus** your tail — i.e. `cached_tokens ≈ floor(prefix_len / 512) × 512`.
For an 8 000-token document prefix, passes 2–4 report `cached_tokens ≈ 7680` and
prefill only a few hundred shared-prefix tokens + the tail.

**Per-pass independence — all confirmed honored per submission:**

- **`response_format` / `json_schema`** is compiled fresh per request from the
  schema JSON (M53 llguidance; the only cross-request shared state is the
  per-*model* vocabulary factory, never anything tying a schema to a prefix).
  Each pass may carry a **completely different schema** (backbone vs quotes vs
  analytical vs grounded-refs) with full strict-shape enforcement, and the Guide
  is applied on top of a reused KV prefix with no interaction. ✅
- **`max_tokens` / `max_completion_tokens`** — resolved per pass as
  `max_completion_tokens ?? max_tokens`. ✅
- **`temperature`** — honored per pass, but **inert when a schema is present**
  (the Guide mask collapses the distribution; structured decode is effectively
  greedy regardless of temperature — M48.3). Your `0.1` is fine; don't expect it
  to do anything under `response_format`. ✅

---

## 8. Auth / ownership

No new scope. Submit all passes with the **same bearer token**; the queue records
that principal as the job `owner` and meters usage to it, and `GET /v1/queue/{id}`
404s for any other principal (admins exempt). The KV scope is keyed by that same
principal, so continuation by the same bearer is exactly what shares the cache.
`prompt_cache_key` is **not** an auth boundary — a different principal sending the
same key gets a different scope and cannot read your KV (and under the security
default `principal` scope, key is ignored for keying entirely). ✅

---

## 9. Reading reuse + truncation off the poll

The polled result for a `done` conversation job is the **full**
`ChatCompletionResponse` — you are already receiving these fields, just read
past `choices[0].message.content`:

```jsonc
GET /v1/queue/{id}  →
{
  "id": "...", "kind": "conversation", "status": "done",
  "result": {
    "id": "chatcmpl-...",
    "model": "...",                       // the model actually served (post-rebind)
    "choices": [{
      "index": 0,
      "message": {"role": "assistant", "content": "<schema-valid JSON>"},
      "finish_reason": "stop"             // ← "length" ⇒ hit max_tokens (truncated); also "tool_calls"
    }],
    "usage": {
      "prompt_tokens": 8200,
      "completion_tokens": 640,
      "total_tokens": 8840,
      "prompt_tokens_details": {"cached_tokens": 7680}   // ← reuse fired; ABSENT/0 ⇒ cold or evicted
    }
  },
  "error": null
}
```

- **`finish_reason: "length"`** is your truncation signal — the decode hit the
  per-pass token cap. Re-submit that pass with a higher
  `max_completion_tokens`.
- **`usage.prompt_tokens_details.cached_tokens > 0`** confirms the prefix cache
  fired and tells you how many prefix tokens were reused. It is **omitted when
  zero** (OpenAI-compatible), so treat absent as `0`.

---

## 10. Worked 4-pass chain (exact bytes)

One document, `prompt_cache_key = "doc-7e3a"`, same bearer throughout. `<SYS>`
and `<DOC>` are byte-identical in every submit.

**Pass 1 — backbone (cold prefill, primes the prefix):**
```jsonc
POST /v1/queue/conversation
{ "prompt_cache_key": "doc-7e3a", "temperature": 0.1, "max_completion_tokens": 16384,
  "response_format": {"type":"json_schema","json_schema":{"name":"backbone","schema":{/*events skeleton*/},"strict":false}},
  "messages": [
    {"role":"system","content":"<SYS>"},
    {"role":"user","content":"<DOC>"},
    {"role":"user","content":"Emit the timeline-event skeleton. No quotes, no refs."}
  ] }
→ {"id":"job-A","status":"queued"}
```
Poll `job-A` → `done`. `usage.prompt_tokens_details` **absent** (`cached_tokens 0`
— first time; this pass paid full prefill and **stored** the `[<SYS>,<DOC>]`
prefix under scope `<model>·p:<you>·k:doc-7e3a`). Read `result.choices[0].message.content`
→ `<BACKBONE_JSON>` (events `e0…eN`). Reconcile cron enqueues pass 2.

**Pass 2 — quotes (warm):**
```jsonc
POST /v1/queue/conversation
{ "prompt_cache_key": "doc-7e3a", "temperature": 0.1, "max_completion_tokens": 16384,
  "response_format": {"type":"json_schema","json_schema":{"name":"quotes","schema":{/*per-event verbatim quotes*/},"strict":false}},
  "messages": [
    {"role":"system","content":"<SYS>"},            // identical
    {"role":"user","content":"<DOC>"},              // identical → prefix reused
    {"role":"assistant","content":"<BACKBONE_JSON>"},
    {"role":"user","content":"Backfill a verbatim quote for each event id e0..eN."}
  ] }
→ {"id":"job-B","status":"queued"}
```
Poll `job-B` → `done`, `usage.prompt_tokens_details.cached_tokens ≈ 7680` (reuse
fired). Different schema than pass 1, strictly enforced.

**Pass 3 — analytical (warm):** identical leading two messages; tail =
`<BACKBONE_JSON>` + analytical instruction; `response_format` = analytical schema.
→ `cached_tokens ≈ 7680`.

**Pass 4 — grounded-refs (warm):** identical leading two messages; tail =
`<BACKBONE_JSON>` + refs instruction; `response_format` = grounded-refs schema.
→ `cached_tokens ≈ 7680`.

Every pass independently carries its own schema, token cap, and temperature;
every pass after the first reads ~`floor(len(<SYS>+<DOC>)/512)×512` prefix tokens
from cache and prefills only the tail. If a pass ever comes back with
`cached_tokens 0` (busy queue evicted the entry, or pressure-shed), the result is
still correct — it just paid a full prefill, and you do nothing different.

---

## 11. Your specific questions, answered

1. **Mechanism / fields?** Option A. No new fields required. Use `prompt_cache_key`
   (already parsed and forwarded on the queue) + identical leading prefix. (§1, §3)
2. **Does reuse survive the gap? TTL? Pin/extend?** Yes, while the inter-pass gap
   stays under `idle_ttl_secs` (default 600 s, reset on each use) and the entry
   isn't evicted by the count/byte/pressure caps. No pin API by design; tune
   `idle_ttl_secs` + `max_entries`; rely on self-healing resend. (§5, §6)
3. **Different schema per turn? Strict enforcement?** Yes — `response_format` is
   per-request, compiled fresh, fully enforced, independent of the reused prefix.
   (§7)
4. **Per-turn `max_tokens` / `temperature`?** Yes, both honored per pass
   (temperature inert under a schema). (§7)
5. **Surface `finish_reason` + `cached_tokens`?** Already present in the polled
   `result` — `result.choices[0].finish_reason` and
   `result.usage.prompt_tokens_details.cached_tokens`. (§9)
6. **Auth for continuation?** Same bearer principal; no new scope. (§8)
7. **Eviction failure semantics / fallback?** No failure surfaced — an evicted
   prefix yields a normal `done` job with `cached_tokens 0`. Your full-prefix
   resend *is* the fallback, sent every time. (§6)
