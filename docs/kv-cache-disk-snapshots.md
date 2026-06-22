# Disk-backed KV snapshots (ADR 027)

The prompt-prefix cache (M59) can persist its KV to **encrypted disk blobs** so a
session resumes **across a daemon restart** with zero re-prefill — a returning
request whose prompt shares a leading token run with an earlier one restores from
disk instead of cold-prefilling. This is a disk **L2 tier** beneath the in-RAM
**L1** pool; it is **off by default** and, when on, **always encrypted**.

Status: shipped v0.10.212–216 (S1–S4). SEP-bound keys (S6) and an explicit
management surface are follow-ups; see *Limitations*.

## Enabling it

Disk persistence is **opt-in** and requires a **keyfile** (encryption is
mandatory — there is no plaintext-on-disk mode). A loopback daemon with the flag
off writes nothing (ADR 025 preserved).

```sh
# 1) Provision a keyfile (≥32 bytes of high entropy), readable only by the daemon user.
head -c 64 /dev/urandom > /etc/athena/prompt-cache.key && chmod 600 /etc/athena/prompt-cache.key

# 2a) Via TOML (athena.toml):
#     prompt_cache_enabled        = true
#     prompt_cache_persist_to_disk = true
#     prompt_cache_persist_kek     = "keyfile:/etc/athena/prompt-cache.key"
#
# 2b) Or via env (overrides TOML), e.g. for a foreground run:
ATHENA_PROMPT_CACHE=1 \
ATHENA_PROMPT_CACHE_PERSIST=1 \
ATHENA_PROMPT_CACHE_PERSIST_KEYFILE=/etc/athena/prompt-cache.key \
  athena load --model Qwen3.5-27B-4bit-mtp
```

If persistence is on but no readable ≥32-byte keyfile is found, the daemon logs a
cause-naming error and **disables disk persistence** (fail-closed) — it never
writes plaintext.

### Configuration keys

| TOML key | env override | default | meaning |
|---|---|---|---|
| `prompt_cache_persist_to_disk` | `ATHENA_PROMPT_CACHE_PERSIST` | `false` | master switch for the disk tier |
| `prompt_cache_persist_kek` | `ATHENA_PROMPT_CACHE_PERSIST_KEYFILE` | — | `keyfile:<path>` (env is the bare path); **required when on** |
| `prompt_cache_persist_dir` | `ATHENA_PROMPT_CACHE_PERSIST_DIR` | `<data_dir>/prompt-cache` | where blobs live |
| `prompt_cache_persist_max_entries` | `…_MAX_ENTRIES` | `0` (unbounded) | retention: max blob count |
| `prompt_cache_persist_max_bytes` | `…_MAX_BYTES` | `0` (unbounded) | retention: max total bytes |
| `prompt_cache_persist_max_age_secs` | `…_MAX_AGE_SECS` | `0` (no expiry) | retention: max blob age |
| `prompt_cache_persist_eager` | `…_EAGER` | `false` | spill a new entry at the store seam (crash survival) — see *Triggers* |

## How it works

- **Content-addressed, no tokens on disk.** Each blob is keyed by
  `SHA-256(scope ‖ tokens[0..<B])` at a 512-token boundary `B`. A returning prompt
  probes its own descending boundaries; a digest hit means the first `B` tokens
  match (so the restored state at `B` is exactly what a cold prefill would hold),
  and the suffix `[B:N]` is prefilled normally. **Prompt tokens are never written
  to disk** — neither in the (plaintext) header nor the (encrypted) body. The
  `scope` binds the principal / `prompt_cache_key`, so one caller's blob is never
  restored for another.
- **Encryption (DEK/KEK envelope).** Each blob's body is AES-256-GCM-sealed under
  a random per-blob data key (DEK); the DEK is wrapped by the keyfile-derived KEK
  (HKDF-SHA256), and the wrapped DEK rides in the blob header. The body is AAD-bound
  to its `(model, quant, prefix-hash)` identity. Blobs are written `0600`.
- **Versioned, skip-on-skew.** A self-describing header (magic, format version
  decoupled from `appVersion`, model+quant identity, save reason, timestamps) means
  a blob from a different binary version, a different model, or a different quant is
  **skipped** (cold path), never mis-restored.

### Triggers (when blobs are written)

- **Idle-drop** — when an idle entry would be evicted (idle-TTL sweep) or flushed
  under memory pressure, it is **demoted to disk** instead of discarded.
- **Graceful shutdown** — idle entries are spilled on `SIGTERM` drain so a planned
  restart resumes them.
- **Eager / frontier** (`prompt_cache_persist_eager`, off by default) — a new entry
  is written to disk **at the store seam**, so a hard crash (`SIGKILL`/panic) before
  the next idle-drop/shutdown doesn't lose it. This adds synchronous I/O on the
  post-prefill path (a TTFT cost), so it is opt-in.

## Honesty boundary (binding)

- **This is a data-at-rest surface; it does not change the in-RAM boundary** (ADR
  024). The active working set stays plaintext in RAM during a restore.
- **Encryption strength is the KEK.** A keyfile KEK gives encrypted-at-rest, but
  the key lives on the host — it defends disk theft / backups / another local user
  reading the file, **not** an adversary who also holds the keyfile. SEP-bound keys
  (hardware-bound, non-extractable) are the follow-up (ADR 024 amendment).
- **It re-introduces request-derived data-at-rest** that ADR 025's stateless-loopback
  mode removes. That's the operator's opt-in trade (resume vs. minimization);
  default-off keeps the loopback zero-write guarantee.
- **Local file I/O only** — passive-oracle preserved.

## Verification (host-bound gates)

- `deploy/integration/e2e-m59-disk-restart.sh` — prime → flush to disk → SIGTERM →
  **fresh process, same data-dir** → restore → **byte-identical to a cold prefill**
  (proven on Qwen3.5-27B-4bit-mtp). Also asserts ciphertext-at-rest (no plaintext
  prompt on disk).
- `deploy/integration/e2e-m59-disk-crash.sh` — eager spill survives a `kill -9`.

The MLX-free format/envelope/store/probe logic is unit-pinned (ADR 008/009).

## Limitations

- **SEP keys are a follow-up (S6).** Today the KEK is a keyfile (or passphrase);
  SEP-bound wrapping is gated on the headless-launchd SEP operability spike. The
  blob format already reserves the `kek_type`, so SEP is a drop-in swap with no
  reformat.
- **Encrypt-idle + persist don't compose yet.** When `prompt_cache_encrypt_idle`
  (RAM cipher) is on, idle entries are sealed in RAM and are **not** spilled to
  disk. Use one at-rest strategy at a time.
- **No named/explicit save-load surface.** Restore is transparent and prefix-keyed;
  Athena has no single mutable "session" to load by name (a restore needs the
  matching prompt). A disk-tier *management* surface (list/stat/clear) is possible
  future work.
