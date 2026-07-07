# ADR 027 — disk-backed KV snapshots (versioned, encrypted, resumable)

**Status:** **Accepted — S1–S4 SHIPPED v0.10.212–216** (M59.5 revived). Disk tier
proven on real Qwen3.5-27B-4bit-mtp: **bit-identical across a clean restart**
(`e2e-m59-disk-restart.sh`) **and survives a `SIGKILL`** (`e2e-m59-disk-crash.sh`).
**S5 (explicit named snapshots) DROPPED** — Athena's KV is prefix-keyed and
prompt-matched, so there is no single mutable "session" to load by name (a restore
needs the matching prompt); the transparent L2 delivers the value, and a disk-tier
*management* surface (list/stat/clear) is left as optional future work. **S6
(SEP-bound KEK) DEFERRED**, gated on the headless-launchd SEP operability spike (the
blob `kek_type` reserves the slot for a drop-in swap). Companion: the **ADR 024
amendment** (SEP envelope key). Research: internal SSD-streaming / KV-snapshot notes;
usage: `docs/kv-cache-disk-snapshots.md`; plan: `docs/kv-cache-disk-snapshots-plan.md`.
Operator decisions: **disk-first with a swappable KEK**, **auto + frontier (eager)**
triggers, transparent L2 (named-snapshot surface dropped on review).

## Context

The M59 prompt-prefix cache (`PrefixKVCache`) lives **in RAM only** and is lost on
restart ([PrefixKVCache.swift](../../Sources/AthenaLLM/PrefixKVCache.swift)). M59.5
(disk persistence) was **deferred** for three reasons
(`docs/m59-prompt-prefix-cache.md` (internal repo) §M59.5):

1. **No serializer existed** — *now obsolete:* ADR 024 T3 shipped `KVFrame`
   (MLX-free wire format) + `KVByteCodec` (`[MLXArray] ↔ Data`, bit-preserving)
   + `IdleKVCipher` (AES-256-GCM), plus a **passed bit-identical gate**.
2. **Blobs are huge** (~1.5 GB fp16 / ~400 MB 4-bit for a 6k prefix) — a *knob*
   (quant / suffix-trim / disk-cap), not a wall.
3. **Warm-daemon back-to-back access buys nothing from disk** — true for that
   access pattern; **false** for the one this ADR targets.

**A downstream client supplies the missing motivation and a concrete format.** It
persists KV as **versioned snapshots** (fixed header, `save-reason`,
SHA1-of-rendered-prefix filename) so "a two-hour coding session is a file you can
come back to tomorrow" — **resume across restart with zero re-prefill**. That is
exactly the access pattern reason (3) excluded, and it is what consumers asked
for. (that client's *mechanism* is a from-scratch C/Metal engine; we borrow its
**format and policy**, not its engine — see the research note §3a.3.)

**Constraint stack.** ADR 025 (minimize request data-at-rest; loopback writes
nothing), ADR 024 (in-memory protection — and now an at-rest surface), ADR 011/023
(the governor), and the passive-oracle rule (local file I/O only — untouched).

## Decision

Persist KV as **versioned, encrypted, disk-backed snapshots** — a disk **L2 tier**
beneath the in-RAM **L1** prefix cache. Seven points:

1. **Flat files in the data dir, keyed by rendered-prefix hash** (the recorded
   M59.5 home), **not SQLite.** Large opaque blobs belong in files, not rows.
   (SQLCipher stays the at-rest path for the auth/audit/usage DB, ADR 034 — not
   for KV.)

2. **Off by default; encryption mandatory when on.** New `[prompt_cache]`
   `persist_to_disk` (default **off**). When off — including the ADR 025
   stateless-loopback default — **nothing is written** (ADR 025 invariant
   preserved). When on, **every blob is encrypted**; no plaintext KV ever reaches
   disk.

3. **Envelope (DEK/KEK) key with a swappable KEK** (per the ADR 024 amendment). A
   random per-blob **DEK** (AES-256-GCM — the existing `IdleKVCipher` cipher)
   encrypts the KV bytes; the **DEK is wrapped by a KEK** and the wrapped-DEK rides
   in the header. **KEK type is a header field** — `passphrase`/`keyfile` **now**
   (cross-restart works immediately), `sep`-bound **later** (drop-in, no format
   change), per-peer ECIES wraps **reachable** for future multi-node. This encodes
   the chosen **disk-first** sequencing: ship with a keyfile/passphrase KEK, swap
   to SEP as a follow-up slice — the envelope makes the swap a header-type change,
   not a reformat.

4. **Versioned, self-describing header (decoupled from `appVersion`).** Fixed
   leading header: magic, **`format_version`**, **`kek_type`** + wrapped-DEK (+
   salt / ephemeral pubkey as the KEK needs), GCM nonce/tag, **model-id + quant/
   dtype bits**, **`save_reason`** (`cold`/`continued`/`evict`/`shutdown`),
   `token_count`, `context_size`, created/last-used stamps, prefix-hash, scope key.
   A binary on a different `format_version`, or a **model/quant mismatch**,
   **skips-on-skew** (cold path) — it never mis-restores.

5. **Save triggers = auto (idle-evict + shutdown) + frontier-interval.**
   (a) when the governor would **drop** an idle L1 entry under pressure,
   **demote it to disk** (spill) instead of discard
   ([MemoryGovernor.swift:556-561](../../Sources/AthenaCore/MemoryGovernor.swift#L556),
   [PrefixKVCache.flushIdle](../../Sources/AthenaLLM/PrefixKVCache.swift#L515));
   (b) **flush idle entries on graceful SIGTERM drain**; (c) **continued frontier
   saves** at the existing **512-token snapshot grid**
   ([PrefixKVCache.swift:162-183](../../Sources/AthenaLLM/PrefixKVCache.swift#L162))
   during long sessions, so an in-progress session survives a **crash**, not only
   clean eviction/shutdown.

6. **Transparent prefix-cache L2** — a request whose rendered-prefix hash matches a
   disk blob restores transparently (decrypt → `KVByteCodec.decode` → rehydrate L1 →
   tokenize only the new suffix), **zero new API**, reusing existing principal /
   `prompt_cache_key` scoping
   ([PrefixKVCache.swift:82-94](../../Sources/AthenaLLM/PrefixKVCache.swift#L82)).
   **Amended on review (S5 dropped):** the originally-planned *explicit named
   snapshot* surface (`athena cache save/load <name>`, `/api/cache/*`) is **not
   built** — Athena's KV is prefix-keyed and prompt-matched, so "load snapshot X"
   has no prompt to attach to and cannot generate; the name would be a bare label
   over the prefix-hash key, adding surface without capability. The transparent L2 is
   the resume mechanism; a disk-tier *management* surface (list/stat/clear) is
   optional future work. No new `/api/*` routes ⇒ the OpenAPI surface is unchanged.

7. **Governor owns L1↔L2.** Disk blobs consume **disk, not the Metal budget** (no
   change to ADR 023 accounting). **Retention** caps by count / bytes / age
   (mirroring the in-RAM caps + ADR 034 retention) with **LRU eviction** of disk
   blobs. The post-decode relief hook gains a **demote-to-disk** path (spill then
   drop) in place of pure drop. Spill/restore I/O sits on latency-sensitive paths —
   **measured and budgeted**, not assumed free.

### Honesty boundary (binding)

- **This is a new data-at-rest surface; it does not change ADR 024's in-RAM
  boundary.** The active working set stays plaintext in DRAM; ADR 027 governs only
  bytes written to **disk**.
- **Encryption strength = the KEK, and the header says which.** A
  `passphrase`/`keyfile` KEK gives **encrypted-at-rest** but the key lives on the
  host (operator-managed) — it defends disk theft / backup leakage / another local
  user reading the file, **not** an adversary who also holds the key material. A
  `sep` KEK (follow-up) is **hardware-bound / non-extractable**. `kek_type` makes
  the posture self-describing; we never claim more than the KEK delivers.
- **Bit-identical-across-restart is the correctness bar.** serialize → kill →
  reload → **byte-identical** output, with **model/quant match enforced by the
  header** (mismatch ⇒ skip-restore, never wrong bytes). AES-GCM is lossless; a
  decrypt that cannot reproduce exact bytes is a **correctness bug**, not a
  degraded mode. The existing gate uses two live daemons and never restarts — this
  ADR adds the restart variant.
- **The ADR 025 trade is explicit.** Turning this on **reintroduces request-derived
  data-at-rest** that ADR 025 worked to remove. It is an **operator opt-in** that
  trades minimization for resume; **default-off keeps ADR 025's loopback
  zero-write guarantee intact**.
- **Passive-oracle preserved** — local file I/O only, no outbound.

### Supersedes / amends

- **M59.5 (deferred)** — revived and superseded by this ADR;
  `docs/m59-prompt-prefix-cache.md` status updated on acceptance.
- **ADR 024** — amended (companion section) for the **SEP envelope key** + the new
  at-rest secret class; satisfies ADR 024's explicit re-open condition (*"Re-open
  SEP only if a future M59.5 persists the key"*).
- **ADR 025** — this is the **named opt-in exception** to "minimize request
  data-at-rest"; default-off preserves the loopback guarantee.
- **ADR 011 / 023** — the governor gains an **L2 disk tier**; **no change** to
  Metal-budget accounting (disk ≠ Metal bytes).
- **CLAUDE.md canonical pipeline** — new `/api/cache/*` routes land in
  `OpenAPISpec.swift` + `NativeAPIDTO.swift` in the same edit.

### Rejected / deferred

- **SQLite/SQLCipher home for blobs** — rejected; flat files keyed by prefix hash
  is the recorded M59.5 plan and the right shape for large opaque blobs.
- **Plaintext disk blobs as an option** — rejected; encryption mandatory when on.
- **SEP-first sequencing** — deferred; the envelope makes SEP a drop-in KEK swap,
  so we ship disk-first with passphrase/keyfile and swap later, **not blocking on
  the headless-SEP operability spike** (research note §5.3).
- **KV quant-on-write** (the client's 2-bit) — deferred blob-size optimization; the
  first slice writes at the cache's existing dtype under a disk-bytes cap.
- **Per-peer ECIES multi-node wraps** — deferred (YAGNI); the envelope keeps them
  reachable without building them now.

## Consequences

- New **opt-in** capability: resume a long session across restart with **zero
  re-prefill**; warm-restore survives a daemon bounce.
- New config surface, `/api/cache/*` + CLI verbs, an RBAC permission, retention
  knobs, and a new **at-rest secret class** (encrypted blob + KEK; threat model in
  the ADR 024 amendment).
- **All decision logic** (header parse / version-skew, save-policy, retention,
  L1↔L2 demotion, envelope wrap/unwrap algebra, `kek_type` dispatch) stays
  **MLX-free + unit-pinned** (ADR 008/009); `KVByteCodec` / MLX numerics unchanged.

### Validation (on implementation)

- **MLX-free unit:** header encode/decode + skip-on-skew; `save_reason` policy;
  retention/LRU; envelope DEK/KEK wrap/unwrap round-trip + tamper/mismatch reject;
  `kek_type` dispatch.
- **e2e (the new bar):** extend `deploy/integration/e2e-m59-prefix-cache.sh` —
  prime + persist on daemon #1 (`persist_to_disk` + encryption, keyfile KEK) →
  **kill** → start daemon #2 on the **same data dir** → request B → assert
  **byte-identical to cold** AND **restored-from-disk** (L2 hit); assert disk blobs
  are ciphertext; assert a **loopback default** run writes nothing; assert
  model/quant mismatch ⇒ **skip-restore** (cold).
- **Named-snapshot e2e:** `save` → bounce → `load` → resumes byte-identically.

Plan + slices: `docs/kv-cache-disk-snapshots-plan.md` (on approval).
