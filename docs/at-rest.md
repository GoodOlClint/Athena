# Data retention & at-rest protection

Athena keeps everything on one box: an embedded SQLite store under the
data dir (`~/.athena/athena.sqlite` by default) holds RBAC users/tokens,
usage counters, the audit log, the built-in vector DB, and the async
request queue. This document covers **how long that data lives** and
**how it's protected on disk**.

## What's stored, and how it's protected

| Data | At rest |
| --- | --- |
| User passwords | PBKDF2 salted hashes — never reversible |
| Bearer tokens | SHA-256 hashes — the raw token is shown once, never stored |
| Outbound secrets (HF token, proxy creds, store key) | macOS Keychain, not the DB |
| **Vector blobs** (embeddings + metadata) | the SQLite file |
| **Queue blobs** (request prompts + result completions) | the SQLite file |

Passwords, token hashes, and Keychain secrets are safe regardless of disk
posture. The **vector and queue blobs hold inference inputs/outputs**, so
they are the focus of both retention and at-rest encryption.

## Bounded retention (opt-in)

Left unbounded, queue results and vectors accumulate forever. Three knobs
(all in `deploy/athena.toml`, all off by default) bound them:

- `queue_result_ttl_secs` — prune terminal (done/error/canceled) queue
  results older than the window. Swept on the worker's idle path.
- `queue_max_rows` — cap total queue rows, trimming the oldest terminal
  results first. Pending (queued/running) jobs are never dropped.
- `vector_ttl_secs` — prune vectors written longer ago than the window,
  swept opportunistically on each upsert. Vectors written before this
  feature (no timestamp) are never auto-pruned.

And a content opt-out:

- `drop_request_content` — clear a queued job's prompt blob the moment it
  finishes, so inference **inputs** don't sit on disk past completion. The
  result the client polls for is retained (bounded by
  `queue_result_ttl_secs`). Inputs to the synchronous `/v1` + `/api`
  paths are never persisted regardless — this only affects the async
  queue, which must store the request to run it later.

## At-rest encryption

The store engine is **SQLCipher** (AES-256, on the Apple CommonCrypto
provider — no OpenSSL). It is opt-in:

### Option A — `encrypt_store` (recommended for sensitive content)

Set `encrypt_store = true`. The daemon resolves an encryption key, opens
the store keyed (so the whole file — vector + queue blobs included — is
ciphertext at rest), and **migrates an existing plaintext store in place**
on the first encrypted start (data and credentials are preserved).

Key resolution precedence:

1. `ATHENA_STORE_KEY` environment variable, else
2. the login Keychain (service `athena`, account `store:key`).

If neither exists, the daemon mints a random 256-bit key and stores it in
the Keychain on first start. It is **fail-closed**: if encryption is
requested but no key can be obtained or stored, the daemon refuses to
start rather than serving an unencrypted store it claimed to protect.

> **Back up the key.** The key is never written to `athena.toml` or to
> disk in plaintext. If you lose both `ATHENA_STORE_KEY` and the Keychain
> item, an encrypted store is unrecoverable. Every same-user CLI verb
> (`athena auth`, `audit`, `usage`, `doctor`) resolves the key the same
> way, so an encrypted store stays administrable from the box.

### Option B — FileVault (the default fallback)

With `encrypt_store` off, the store is a standard SQLite file and at-rest
protection relies on **macOS FileVault** (full-disk encryption). For any
deployment handling sensitive prompts, keep FileVault **on** — or set
`encrypt_store` for defense-in-depth (so a copied/backed-up store file is
useless without the key, independent of disk encryption).

## Checking posture

`athena doctor` reports the data-at-rest posture (check #13):

- whether the store is encrypted (and where the key comes from),
- if not, whether FileVault is on,
- and the configured retention bounds.

It warns when a plaintext store sits on a box with FileVault **off**, and
when `encrypt_store` is set but the store hasn't been migrated yet.

## Notes

- This is all local — encrypting/retaining the on-box store initiates no
  outbound connections, so the passive-oracle thesis is intact.
- `SQLITE_TEMP_STORE=2` keeps SQLite's temp/sorter spill in memory, so an
  encrypted store never leaks plaintext to a temp file on disk.
