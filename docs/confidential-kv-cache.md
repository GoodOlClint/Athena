# Confidential prompt-cache KV (encrypt idle entries at rest in RAM)

**ADR 024 Tier 3.** When the M59 prompt-prefix cache is enabled, Athena can hold
its **idle** KV entries as AES-256-GCM ciphertext in RAM, so that sensitive
reused prefixes (PHI, cardholder data / PAN) are never plaintext-at-rest and only
ciphertext can be written to swap. Opt-in, off by default.

## When to use it

Turn it on if the prompt cache will hold **sensitive reused prefixes** — e.g. a
static system prompt + a verbatim patient document or cardholder record, reused
across several extraction passes — and you need to assert *"sensitive data is
encrypted at rest, including in volatile in-memory caches and anything that can
reach swap."* It is the cleanest way to make that assertion for the idle cache
without depending on host swap-encryption / FileVault being configured.

If the prompt cache is off, or holds nothing sensitive, you do not need it.

## What it protects (and what it does not)

**Protects — the idle pool.** Every cache entry that is not the one currently
being decoded is held as AES-256-GCM ciphertext. Concretely this buys, over and
above the Tier-1 process lockdown:

1. **Swap / compressed-memory leakage (the main reason).** macOS can compress and
   swap idle pages under memory pressure. The cache's idle entries are long-dwell
   (idle-TTL, default 600 s) — exactly the pages most likely to be swapped. With
   encryption on, **only ciphertext is ever swappable.** This is independent of
   any process-memory adversary and of whether host swap-encryption is on.
2. **Residency-window collapse.** A momentary read — a core dump that slips past
   `RLIMIT_CORE=0`, a debugger that attaches before `PT_DENY_ATTACH`, a future
   task-port regression — catches at most the **one entry currently decoding**,
   not the whole pool of reused sensitive prefixes.
3. **Defense-in-depth.** Survives a Tier-1 misconfiguration (a build shipped
   without the Hardened Runtime, an entitlement regression), and lets an auditor
   be told the idle cache is AES-256-GCM encrypted at rest.

**Does NOT protect — the active working set (binding honesty boundary).**

- The resident model **weights** and the KV slice **currently being decoded** are
  irreducibly plaintext in DRAM — Metal kernels operate on plaintext arrays;
  there is no encrypted-compute path. T3 shrinks the plaintext window to the
  active entry; it does not close it.
- The encryption **key lives in process RAM** during every seal/open (bulk AES
  runs in-process). A **kernel-level / SIP-disabled-root adversary reads both the
  key and the plaintext working set** — the same exposure the keychain's in-use
  secrets carry. T3 does not defend against that adversary; the Tier-1 lockdown
  is what blocks the non-root co-resident scraper.
- The key is **process-ephemeral and never persisted** (a fresh random key per
  daemon run, rotated when the pool drains to empty). It is deliberately *not*
  wrapped by the Secure Enclave: a RAM-only key gains nothing from SEP against
  the threat above, and SEP cannot do the GB/s bulk crypto.

In one line: **idle prompt-cache KV is AES-256-GCM encrypted in RAM; only the
single entry currently decoding, plus the resident weights, remain plaintext; the
key is process-ephemeral and never written to disk.** It is not confidential
computing and not a defense against a kernel/root adversary.

## Correctness

Encryption is **lossless** — AES-GCM round-trips the KV bytes exactly, so warm
prefix-reuse with encryption on produces **byte-identical** output to a cold
prefill. This is pinned by the host-bound bit-identical gate
(`deploy/integration/e2e-m59-prefix-cache.sh` with `ENCRYPT_IDLE=1`). If an entry
ever fails to decrypt, the request silently falls back to a cold prefill rather
than serving a corrupt cache.

## Performance

A cache hit pays one decrypt + deserialize of the reused entry. Hardware AES runs
at ~GB/s, so even a large (~hundreds-of-MB) entry decrypts in well under a second
— far cheaper than the cold prefill it replaces (tens to ~hundreds of seconds on
a large model). The store path adds one seal of the same magnitude on the cold
(miss) path, where it is dwarfed by the prefill it accompanies. Ciphertext counts
toward the governor's pool byte budget exactly as plaintext did (nonce+tag
overhead is negligible).

## Enabling it

It requires the prompt cache itself to be enabled (`prompt_cache_enabled`).

TOML (`athena.toml`):

```toml
prompt_cache_enabled = true
prompt_cache_encrypt_idle = true
```

CLI:

```sh
athena config set prompt_cache_encrypt_idle true
```

Environment override (precedence: env > TOML > default-false):

```sh
ATHENA_PROMPT_CACHE_ENCRYPT_IDLE=1
```

Verify the posture:

```sh
athena doctor   # → prompt cache: ON (… idle_encryption=on) / idle entries encrypted at rest …
```

The daemon also logs at startup:
`hardening: prompt-cache idle entries encrypted at rest (AES-256-GCM, ADR 024 T3)`.

## See also

- `docs/m59-prompt-prefix-cache.md` — the cache this protects.
- `docs/decisions/024-in-memory-data-protection-coresident-threat.md` — the full
  threat model and defense ladder (T1 process lockdown, T2 side-channel hygiene,
  T3 here).
- `docs/confidential-kv-cache-plan.md` — the T3 change plan and rationale.
