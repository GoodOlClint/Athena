# ADR 024 Tier 3 — Confidential idle prompt-cache KV (encrypt-at-rest-in-RAM)

Status: **change plan, pending operator approval** (brownfield change gate). Milestone M81.
ADR: amends **ADR 024** T3 (deferred → accepted); no new ADR — the key model is a
*narrowing* of what 024 already framed (see §10).

Next version: shipped slices start at **v0.10.206** (current = v0.10.205).

---

## 0. One-paragraph summary

Make the M59 prompt-prefix cache keep its **idle KV entries encrypted in RAM**
(AES-256-GCM, CryptoKit), so that the long-dwell, high-value reused prefixes
(patient documents, cardholder data) are ciphertext whenever they are not the
single entry currently decoding. Opt-in, off by default, mirroring
`encrypt_store`. The net-new engineering is a KV **serialize/deserialize seam**
(`MLXArray` KV state ↔ bytes); built once, T3 uses it now and a future M59.5
(disk-backed cache) reuses it. The bit-identical gate is sacred and re-run:
AES-GCM is lossless, so warm output stays byte-identical to cold.

---

## 1. What does T3 buy over T1? (the honest justification — lead with this)

T1 (process lockdown, v0.10.198) already blocks a **non-root** co-resident
`task_for_pid`, and substantially raises the bar against SIP-intact root. So T3
must justify itself *beyond* T1. In priority order:

1. **Swap / compressed-memory leakage — the strongest, T1-independent reason and
   the cleanest compliance story.** macOS compresses idle pages and can swap them
   under memory pressure. The M59 idle pool is *precisely* long-dwell idle pages
   (idle-TTL default 600 s) holding reused PHI/PAN prefixes. If swap
   encryption / FileVault is off on the host, plaintext KV could be written to
   disk by the OS — entirely independent of any process-memory adversary, and
   entirely outside T1's scope. T3 guarantees **only ciphertext is ever
   swappable**. This maps directly onto the compliance requirement ("sensitive
   data encrypted at rest, including in volatile in-memory caches and anything
   that can reach swap"). Without T3 the only swap answer is an *operator-config
   assertion* ("require FileVault + encrypted swap"); T3 lets the **daemon
   itself** make the guarantee.
   - **Why T3 and not `mlock` here:** T2 deferred `mlock` because GB-scale KV
     pinned against the unified-memory budget fights the governor (ADR 011/023)
     — pinned pages can't be reclaimed under pressure, which breaks the whole
     point of an *evictable* idle pool. Encrypted pages stay swappable /
     reclaimable (good for the governor) while never exposing plaintext to swap.
     For the idle pool, T3 is strictly the better tool than `mlock`.

2. **Residency-window collapse.** A momentary read — a core dump that slips past
   `RLIMIT_CORE=0`, a debugger that attaches before `PT_DENY_ATTACH`, a future
   `task_for_pid` regression — catches at most the **one entry currently
   decoding**, not the whole pool of reused PHI/PAN prefixes sitting idle.

3. **Defense-in-depth + attestation.** Survives a T1 misconfiguration (a dev
   build shipped without the hardened runtime, an entitlement regression) and
   lets an auditor be told the idle cache is **AES-256-GCM encrypted at rest**,
   regardless of the process-lockdown posture.

**Verdict: build it.** Reason #1 alone justifies T3 — it is a real, T1-orthogonal
leak vector that the daemon can close itself and that maps one-to-one onto the
stated HIPAA/PCI requirement. #2 and #3 are genuine but secondary. (If the gate
had been a vaguer "confidential cache" ask with FileVault already mandated,
the marginal value would be thinner; it is not — the swap vector is concrete.)

---

## 2. Honesty boundary (binding — carried verbatim in spirit from ADR 024)

T3 **shrinks the plaintext window; it does not close it.** Unchanged residuals:

- **The active working set is irreducibly plaintext.** Resident weights and the
  KV slice currently being decoded must be plaintext `MLXArray`s — there is no
  encrypted-compute path on Metal. T3's guarantee is scoped to **idle cache
  entries**, i.e. an entry that is not the one currently being restored/decoded.
- **The key lives in our RAM during every seal/open.** Bulk AES runs in-process,
  so a kernel / SIP-off-root adversary reads both the symmetric key and the
  plaintext working buffer — same exposure as the keychain's in-use secrets. T3
  does **not** defend against that adversary; T1 is what blocks the non-root one.
- **The transient plaintext wipe is best-effort** (ADR 023 / T2 boundary): we
  `secureZero` the *intermediate serialized byte buffer* we own; the source
  `MLXArray`s are released to MLX, whose pool may hand the page back before we
  touch it. The **at-rest** guarantee does not depend on that wipe — see §4.

What T3 may claim, exactly: *"idle prompt-cache KV is AES-256-GCM encrypted in
RAM; only the single entry currently decoding, plus the resident weights, remain
plaintext; the encryption key is process-ephemeral and never persisted."* It may
**not** claim confidential computing, protection from a kernel/root adversary, or
that the active working set is encrypted.

---

## 3. Subsystem map (what T3 hooks into)

`Sources/AthenaLLM/PrefixKVCache.swift` — lock-guarded `@unchecked Sendable`
class; `entries: [Entry]`. Each `Entry` holds the at-rest KV:

- `attn: [KVCache?]` — full prompt-length attention-cache clones (nil at
  recurrent/Mamba layers).
- `checkpoints: [Int: [Int: [MLXArray]]]` — 512-boundary → layerIdx →
  `[conv, ssm]` recurrent state.
- `byteEstimate`, `lastUsed`, `refcount`.

Three seams (all already exist; T3 changes their bodies, not their signatures):

| Seam | Today | T3 |
|---|---|---|
| `store(scope:promptTokens:backbone:recorder:)` (cold-prefill seam, `SpeculativeGeneration.generate:178`) | clones attention `KVCache`s + keeps recurrent checkpoint MLXArrays as plaintext in the `Entry` | **serialize → seal → store ciphertext**; release source plaintext + `secureZero` the intermediate byte buffer |
| `acquire(scope:promptTokens:model:)` (hit, `SpeculativeGeneration.generate:85`) | clones the entry's plaintext tensors into a fresh `working` cache, bumps refcount | **open (decrypt) → deserialize directly into the `working` cache → `secureZero` the transient buffer**; bump refcount unchanged |
| `release(_:)` / `flushIdle()` | refcount-/policy-driven eviction; frees plaintext | frees ciphertext; on **full flush** also rotates+zeroes the key (§5) |

**Key architectural fact that makes T3 clean:** clone-on-hit means the entry's
own stored tensors are *never* the working buffer — `acquire` reads them under
the lock, clones into a separate `working` cache, and the caller decodes the
clone. So the entry's stored representation is genuinely idle except for the
brief decrypt inside `acquire` (held under the pool lock). This lets T3 keep the
entry **ciphertext-at-rest at all times** rather than literally "encrypt when
refcount hits 0" — see §4.

---

## 4. Design: ciphertext-at-rest-always (a refinement of "encrypt-on-evict")

ADR 024 framed T3 as "encrypt-on-evict-to-idle, decrypt-on-restore." Given the
clone-on-hit architecture (§3), an entry's stored tensors are idle the *entire*
time except the transient decrypt under lock — so the simplest, strongest
invariant is: **an `Entry` holds only ciphertext, from birth (`store`) to death
(evict).** This strictly dominates literal refcount-gated encryption (which would
keep the entry's tensors plaintext while `refcount > 0`, even though the decode
never touches them — widening the plaintext window for zero benefit). Same
guarantee ADR 024 described ("keep idle entries encrypted; decrypt only the slice
being restored"), simpler invariant.

`Entry` becomes (T3-enabled):

```
attn:        [KVCache?]                       → sealedAttn: [Data?]   (+ per-slot shape/dtype meta)
checkpoints: [Int: [Int: [MLXArray]]]         → sealedCheckpoints: [Int: [Int: Data]] (+ meta)
```

(plaintext fields retained when T3 is **off**, so the default path is byte-for-byte
the M59 code that exists today.)

**Store (cold path).** The plaintext attention clones + recurrent checkpoints are
built exactly as today; T3 then, per tensor: `KVByteCodec.encode(MLXArray) ->
(bytes, shape, dtype)` → `IdleKVCipher.seal(bytes) -> Data` → store the sealed
`Data`; `secureZero` the intermediate `bytes` buffer; drop the source `MLXArray`
references (released to MLX). The long-lived at-rest representation is therefore
ciphertext; the transient plaintext exists only during `store` and is not parked
in a swappable idle page for the TTL window — **so the swap guarantee (§1.1)
holds even though the MLXArray release is best-effort.**

**Acquire (hit path).** Under the existing pool lock: for each sealed slot,
`IdleKVCipher.open(Data) -> bytes` → `KVByteCodec.decode(bytes, shape, dtype) ->
MLXArray` → install into the fresh `working` cache (attention: build a
`KVCacheSimple`, set `.state`, `trim` to `len - B` exactly as today; recurrent:
set `MambaCache.state` + `offset = B`). `secureZero` the transient `bytes`. The
`working` cache is the plaintext active entry (irreducibly so). Bit-identicality:
the decoded `MLXArray` must equal the stored one byte-for-byte (§7).

**Latency / fall-back-to-cold (operator chose "must beat cold").** No fixed SLA;
invariant: a restore must cost less than a cold re-prefill. Hardware AES at
~GB/s decrypts a ~760 MB entry in well under a second vs ~190 s cold prefill, so
it wins comfortably; the plan measures it on a realistically-large entry (§8) and
puts the numbers in the ADR. We add a guard so a hit whose decrypt+deserialize is
estimated to exceed re-prefill prefers cold (in practice unreachable, but it
encodes the constraint and is unit-pinnable on the estimate).

---

## 5. Crypto + key model (operator chose RAM-only key, no SEP)

- **Cipher:** CryptoKit `AES.GCM`, 256-bit. Each tensor sealed as an independent
  `AES.GCM.SealedBox` (random per-seal nonce; 12-byte nonce + 16-byte tag
  overhead per slot — negligible against hundreds of MB of KV). AAD binds the
  slot's shape/dtype/layer-index so a slot can't be swapped for another.
- **Key:** a random per-process `SymmetricKey(.bits256)`, generated lazily on the
  first seal. **No SEP wrapping** — recorded rationale: the key is
  process-ephemeral and is necessarily in our RAM during every bulk seal/open, so
  against the co-resident/kernel threat model SEP-wrapping changes nothing (a
  kernel adversary reads the unwrapped key regardless; a non-root one is already
  blocked by T1). SEP's only value here would be an auditor "keys in Secure
  Enclave" checkbox, which would be partly theater for a RAM-only key; the
  operator declined it. (Re-open only if a future M59.5 *persists* the key across
  restarts — then SEP wrapping becomes substantive.)
- **Key lifecycle / zeroization:** `SymmetricKey` self-zeroes its backing on
  dealloc (CryptoKit guarantee). Key **rotation = replace the instance** (old one
  dealloc'd → zeroed), done on a **full pool flush** (`flushIdle` that empties the
  pool, operator `DELETE /api/cache/prompt`, governor pressure-shed to empty).
  After rotation, any not-yet-freed sealed blobs are unopenable — but flush frees
  them anyway, so this is consistent. Our own transient plaintext byte buffers are
  wiped with the existing `ProcessHardening.secureZero(_ data:)`.

---

## 6. MLX-free vs MLX-gated split (ADR 008/009)

- **MLX-free + unit-pinned** (pure Swift, runs under `swift test`):
  - `IdleKVCipher` (new, `Sources/AthenaCore/`): `seal(Data) -> Data`,
    `open(Data) -> Data?`, key rotation/zeroize, AAD binding. Operates on byte
    buffers only — pure CryptoKit. Tests: seal→open→byte-equal; tampered
    ciphertext/tag → nil; wrong-AAD → nil; post-rotation open → nil; key zeroized
    after rotation.
  - **Mode selection**: "is an entry eligible to be encrypted?" = is T3 enabled.
    Trivial predicate, unit-pinned alongside the config resolver.
  - **Restore-vs-cold cost guard** (§4): the estimate comparison is pure
    arithmetic, unit-pinned.
- **MLX-gated** (stays in `AthenaLLM`, validated by the heavy/bit-identical tier):
  - `KVByteCodec` (new, `Sources/AthenaLLM/`): `MLXArray` KV state ↔ `(bytes,
    shape, dtype)`. Uses `.asArray(T.self)` per dtype to flatten and
    `MLXArray(_, shape)` to rebuild; captures shape + dtype (fp16/bf16/int for
    TurboQuant-packed state) so the round-trip is exact. This is the
    serialize/deserialize seam a future M59.5 reuses.

---

## 7. The bit-identical gate is sacred (re-run, pin)

AES-GCM is lossless, so warm-with-T3 output **must** stay byte-identical to a cold
prefill. A decrypt that can't reproduce the exact KV bytes is a **correctness
bug**, not a degraded mode. Validation:

- Reuse `deploy/integration/e2e-m59-prefix-cache.sh` (the existing cold-vs-warm
  byte-for-byte gate on real Qwen3.5-27B-4bit-mtp), run with
  `prompt_cache_encrypt_idle = true` (env `ATHENA_PROMPT_CACHE_ENCRYPT_IDLE=1`).
  Pass = cold == warm token-for-token *with encryption on*.
- A `KVByteCodec` round-trip unit (gated heavy tier): encode→decode an `MLXArray`
  and assert element-wise equality across the KV dtypes.
- `IdleKVCipher` round-trip + the codec round-trip composed: seal(encode(x)) then
  decode(open(·)) == x.

---

## 8. Governor accounting + performance (both unchanged in spirit)

- **Accounting:** ciphertext entries count toward the pool byte budget exactly as
  plaintext did; `byteEstimate` becomes the sealed size (≈ plaintext +
  nonce+tag/slot, negligible). `poolBytesAndEntries` / `GovernorSnapshot` /
  `/healthz` semantics unchanged. Pressure-shed frees ciphertext via the existing
  `flushIdle` path. No new governor field.
- **Performance:** measure decrypt-on-restore on a realistically-large entry
  (~760 MB, a ~2 K-token 27B-4bit prefix) and record ms + implied GB/s in the
  ADR. Expectation: sub-second, ≫ cheaper than the ~190 s cold prefill. The store
  path adds one seal of the same magnitude on the cold (miss) path, where it is
  dwarfed by the prefill it accompanies. If any measured restore ever exceeds
  re-prefill, the §4 guard prefers cold.

---

## 9. Config — 5 touchpoints + flag + env (mirrors `encrypt_store` / `kv_compression`)

New key **`prompt_cache_encrypt_idle`** (Bool, default `false`):

1. `Sources/AthenaDeploy/AthenaConfig.swift` — field `promptCacheEncryptIdle`,
   init param, assign, TOML bool parse (next to `prompt_cache_*`).
2. `Sources/AthenaDeploy/DefaultConfig.swift` — commented default in the
   `[prompt_cache]` block, noting the ADR-024-T3 honesty boundary.
3. `Sources/AthenaDeploy/ConfigEditor.swift` — add to `knownKeys` / `rawKeys` /
   bool-coercion set.
4. `Sources/athena/Commands/ConfigEditor.swift` — `get` case.
5. `Sources/AthenaDeploy/LaunchdPlist.swift` — TOML-only re-read list comment +
   any plist passthrough consistent with the other `prompt_cache_*` keys.

Plus: a `--prompt-cache-encrypt-idle` load flag and env override
`ATHENA_PROMPT_CACHE_ENCRYPT_IDLE` (1/true/0/false), precedence **env > TOML >
false**, mirroring `ATHENA_PROMPT_CACHE`. Wired in
`Sources/athena/Commands/Load.swift` next to `prefixCacheEnabled`, passed into the
`PrefixKVCache(...)` initializer (new `encryptIdle: Bool = false` param, default
off so every existing call site is unchanged). `deploy/athena.toml` documents it.

**Doctor:** extend the existing prompt-cache posture check
(`Sources/athena/Commands/Doctor.swift:391`) to report whether idle-cache
encryption is on, and — when the cache is enabled with sensitive workloads but
encryption is **off** — emit an informational finding pointing at the flag. No
fail-closed (it's opt-in hardening, like the other T-tier toggles).

---

## 10. ADR / decision records

- **Amend `docs/decisions/024-...md`:** flip the T3 section from *deferred* →
  **accepted**; record (a) the swap-leakage-led justification (§1), (b) the
  RAM-only-key model and the recorded SEP rejection (§5) — a *narrowing* of 024's
  original "key wrapped by the Secure Enclave" sketch, (c) the ciphertext-at-rest
  -always refinement (§4), (d) the unchanged honesty boundary (§2), (e) the
  measured restore-vs-prefill numbers (§8) on completion. Update the ADR Status
  line (T1+T2 shipped, **T3 accepted/shipped**).
- **No ADR 027.** The key model introduces no decision beyond what ADR 024
  framed; it removes the SEP option 024 floated. Folding it into 024 keeps the
  threat-model and the defense as one document.
- Update CLAUDE.md ADR-024 index line + the project memory
  `project_hardening-program-024-025-026.md` on completion.

---

## 11. Slice plan (small, test-pinned, stacked; each bumps `appVersion`)

> Pre-commit pipeline per slice: **Tests → Security → Quality → Refactor**.
> Build with full Xcode (`./deploy/build.sh Release`); run tests as
> `./deploy/test.sh > /tmp/t3-test.log 2>&1; echo EXIT=$?` (the `tee|tail` idiom
> masks the exit code) and grep for `error:`.
> Direct-to-main, annotated semantic tags, messages framed by the
> Athena-internal security reason (never name the consumer repo).

- **T3.1 — `v0.10.206`: serialize/deserialize seam + MLX-free cipher, dormant.**
  Add `IdleKVCipher` (AthenaCore, CryptoKit, MLX-free) + `KVByteCodec`
  (AthenaLLM). Unit tests: cipher round-trip / tamper / rotate / zeroize / AAD;
  codec round-trip (gated heavy tier). No `PrefixKVCache` behavior change yet —
  both are unused. *Acceptance:* `swift test` green; new units cover the crypto
  algebra; default inference path byte-unchanged.

- **T3.2 — `v0.10.207`: wire ciphertext-at-rest into `PrefixKVCache` + config.**
  `Entry` gains sealed fields; `store`/`acquire`/`flushIdle` updated (§4); key
  held + rotated-on-flush; `byteEstimate` = sealed size. Add the
  `prompt_cache_encrypt_idle` config (5 touchpoints) + `--` flag + env + the
  `PrefixKVCache(encryptIdle:)` param + `Load.swift` wiring. Default **off** ⇒
  existing M59 path untouched. *Acceptance:* `swift test` green incl. a pure
  `PrefixCachePolicy`-level test that an encrypted entry's metadata
  (byte-accounting/eviction order) matches plaintext; e2e-rbac stub tier green.

- **T3.3 — `v0.10.208`: bit-identical gate (encryption on) + doctor + perf + docs.**
  Re-run `e2e-m59-prefix-cache.sh` with encryption on → cold == warm byte-for-byte
  (the sacred gate). Doctor posture string. Measure restore latency on a ~760 MB
  entry. Write `docs/confidential-kv-cache.md` (operator usage + the honesty
  boundary verbatim). Amend ADR 024 T3 → accepted with the measured numbers;
  update CLAUDE.md index + memory. *Acceptance:* gate byte-identical with
  encryption on; perf numbers recorded; spec/docs consistent.

---

## 12. Out of scope / explicitly not done

- Encrypting the **active working set** or weights — impossible on Metal (§2).
- **SEP key-wrapping** — rejected for a RAM-only key (§5); re-open only if a key
  is ever persisted.
- **Disk-backed cache (M59.5)** — still deferred; T3 *builds the serialize seam it
  will reuse* but adds no disk path.
- **Non-MTP substrate path** — M59 prefix reuse is MTP-only; T3 inherits that
  scope.
- Defending a **kernel / SIP-off-root** adversary — out of scope, same as all
  prior tiers (§2).

---

## 13. Open risks

1. **TurboQuant-packed KV dtype round-trip** — stored KV may be 4-bit-packed
   (`kv_compression`). `KVByteCodec` must capture the exact dtype/packing and
   round-trip it; the bit-identical gate is the proof. *Mitigation:* the gate runs
   on the 4-bit-mtp model (KV path is fp16 at default `kv_compression=none`); add
   an explicit codec unit over the packed dtype, and run the gate a second time
   with `kv_compression` on before claiming TurboQuant support.
2. **`MLXArray` plaintext release is best-effort** — the at-rest guarantee is
   designed *not* to depend on it (§4), but document the residual.
3. **Per-hit decrypt on a rapidly-reused entry** — the M59 pattern is 2–4
   back-to-back passes then idle; the sub-second decrypt is dwarfed by
   suffix-prefill+generation. Accepted; measured in T3.3.

---

**This plan is the approval gate. No implementation begins until it is approved.**
