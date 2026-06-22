# KV-cache disk snapshots — change plan (ADR 027)

**Status:** Proposed (design gate). Implementation begins **only after operator
approval** of ADR 027 + the ADR 024 amendment. House style: small, test-pinned,
stacked slices; each slice = one commit + annotated semantic tag pushed direct to
`origin/main`, with the `Athena.appVersion` bump **in the slice commit**. Pre-commit
pipeline per slice: **Tests → Security → Quality → Refactor**.

**ADRs:** `docs/decisions/027-disk-kv-snapshots.md`,
`docs/decisions/024-…` (amendment). **Research:**
`docs/the downstream client-ssd-streaming-and-kv-snapshot-research.md`.

**Operator decisions (gate):** disk-first / swappable KEK · auto + frontier-interval
triggers · both surfaces (transparent L2 + explicit named snapshots).

## Scope

In: persist M59 prefix-cache KV to encrypted, versioned disk blobs; transparent L2
restore; explicit named snapshots; auto + frontier save triggers; retention; the
DEK/KEK envelope with `keyfile`/`passphrase` KEK now and a `sep` follow-up. Out
(deferred): SEP KEK lands as **S6** gated on the headless-SEP spike; KV quant-on-write;
per-peer multi-node wraps.

## Binding invariants (carried into every slice)

- **Off by default**; loopback default writes nothing (ADR 025).
- **Encryption mandatory** when `persist_to_disk` is on — no plaintext KV on disk.
- **Decision logic MLX-free + unit-pinned** (ADR 008/009); `KVByteCodec`/MLX
  numerics byte-unchanged.
- **Bit-identical-across-restart** is the correctness bar (S3 gate); model/quant
  mismatch ⇒ skip-restore, never wrong bytes.
- **Canonical pipeline:** any `/api/cache/*` route change lands in
  `OpenAPISpec.swift` + `NativeAPIDTO.swift` in the same edit (S5).

## Slices

### S1 — disk format + envelope (MLX-free, `AthenaCore`)
- `KVSnapshotHeader` (versioned, self-describing): magic, `format_version`,
  `kek_type`, wrapped-DEK (+ salt/eph-pubkey slot), GCM nonce/tag, model-id +
  quant/dtype bits, `save_reason`, token/context counts, timestamps, prefix-hash,
  scope key. Encode/decode + **skip-on-skew** (version / model / quant mismatch).
- `KEKProvider` seam + `KeyfileKEK` / `PassphraseKEK` (HKDF/KDF wrap of the DEK).
  `SepKEK` is a stub conforming to the seam (filled in S6).
- **No wiring.** Pure types + algebra.
- **Tests:** header round-trip, skew-skip matrix, envelope wrap→unwrap, tamper +
  wrong-key + `kek_type`-dispatch rejection. **Ship gate:** `./deploy/test.sh`.

### S2 — disk store + serializer wiring (`AthenaLLM`)
- `KVSnapshotStore`: write/read flat files in the data dir keyed by prefix hash;
  body = `KVByteCodec.encode(...)` sealed via DEK; header per S1. Retention: count
  / bytes / age caps + LRU evict (mirror in-RAM caps + ADR 034).
- Config: `[prompt_cache] persist_to_disk` (off), `persist_dir`, `persist_max_bytes`
  / `_entries` / `_age_secs`, `persist_kek` (`keyfile:<path>` | `passphrase:env`).
- **Off-by-default; no read/restore path yet** (write+enumerate+evict only).
- **Tests:** store write→read→evict; retention LRU; ciphertext-on-disk assertion;
  loopback-default writes nothing.

### S3 — transparent L2 restore + auto triggers (`AthenaLLM` + governor) — **the gate**
- On L1 miss, look up L2 by prefix hash → decrypt → `KVByteCodec.decode` → rehydrate
  L1 → tokenize only the new suffix.
- Governor relief hook gains **demote-to-disk** (spill idle entry, then drop) in
  place of pure drop; **SIGTERM drain** flushes idle entries to disk.
- **e2e gate (new bar):** extend `deploy/integration/e2e-m59-prefix-cache.sh` —
  prime+persist on daemon #1 (encryption + `persist_to_disk`, keyfile KEK) → **kill**
  → daemon #2 on same data dir → request B → **byte-identical to cold** AND
  **L2-restored**; model/quant mismatch ⇒ cold; loopback ⇒ no writes.

### S4 — frontier-interval continued saves (`AthenaLLM`)
- During long decode, write `save_reason=continued` snapshots at the existing
  512-token snapshot grid (interval configurable) so an in-progress session survives
  a **crash**.
- **Tests:** frontier-save cadence (unit, MLX-free policy); e2e crash-mid-session →
  resume from latest frontier.

### S5 — explicit named snapshots (surface)
- `athena cache save/load/list/rm <name>` + `/api/cache/snapshots*`
  (`OpenAPISpec.swift` + `NativeAPIDTO.swift` same edit), new RBAC permission,
  audit actions, `/ui` mirror if applicable.
- **Tests:** e2e save → bounce → load → byte-identical; RBAC deny path; OpenAPI
  drift-guard.

### S6 — SEP KEK swap (follow-up; gated on the headless-SEP spike)
- Fill `SepKEK`: SecureEnclave P-256, ECIES wrap of the DEK; add `kek_type=sep`.
  **Drop-in** — no blob-format change (S1 already reserves the header slots).
- Precondition: the headless-launchd SEP operability spike (research §5.3) resolved
  (device-bound non-biometric access; reboot/restore-to-different-Mac behavior).
- **Tests:** unwrap-requires-enclave; non-extractable assertion; headless daemon path.

### S7 — docs + reconciliation
- `docs/kv-cache-disk-snapshots.md` (usage); update `docs/m59-prompt-prefix-cache.md`
  (M59.5 → shipped via ADR 027); flip ADR 027 + the ADR 024 amendment Status to
  Accepted/Shipped; CLAUDE.md ADR index + `/v1`↔`/api` notes if surface changed.

## Sequencing & rollout

S1 → S2 → **S3 (gate)** delivers the core capability (transparent resume across
restart, encrypted, keyfile KEK). S4 and S5 are independent adds after S3. **S6
(SEP)** lands when its spike clears — the envelope means it never blocks S1–S5.
Each slice independently shippable, off-by-default until the operator opts in.

## Open items folded from the research note (§5)

- Real blob-size budget after trim/cap (sizing spike) → informs S2 defaults.
- Post-decode demote latency (measure) → informs S3 relief-hook budget.
- Headless-SEP operability spike → gates S6.
- KV quant-on-write (the downstream client 2-bit) → future blob-size optimization, post-S5.
