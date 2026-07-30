# Data retention & at-rest protection

Athena keeps everything on one box: an embedded SQLite store under the
data dir (`~/.athena/athena.sqlite` by default) holds RBAC users/tokens,
usage counters, and the audit log. This document covers **how long that
data lives** and **how it's protected on disk**.

> ADR 025: the built-in vector DB (v0.10.201) and the async request queue
> (v0.10.203) were removed — there is no longer a `vectors` or `jobs` table,
> nor a `/v1/vectors*` / `/v1/queue*` surface. The store no longer persists
> **any** request inputs/outputs.

## What's stored, and how it's protected

| Data | At rest |
| --- | --- |
| User passwords | PBKDF2 salted hashes — never reversible |
| Bearer tokens | SHA-256 hashes — the raw token is shown once, never stored |
| Outbound secrets (HF token, proxy creds, store key) | macOS Keychain, not the DB |
| Audit log + usage counters | the SQLite file (principal/action/target/result + token counts — **no request content**) |

Passwords, token hashes, and Keychain secrets are safe regardless of disk
posture. With the queue and vector tenants gone, the store carries **no
inference inputs/outputs** — only credentials and audit/usage metadata.

## Upload bytes are never written to disk (ADR 025 S5)

Audio and video uploads (`/v1/audio/*`, `/v1/video/*`) are decoded **in
memory** — Apple's `AVAssetReader` is fed from the in-memory request `Data`
through an `AVAssetResourceLoaderDelegate` (`InMemoryAsset.swift`), so the raw
upload bytes are **never staged to `NSTemporaryDirectory()`**. There is no
on-disk upload residue to leak after a crash, and a boot-time sweep clears any
legacy `athena-*` temp file left by the pre-S5 path. The same Apple codecs and
hardware decode are used (no format-coverage or PCM-parity change). The upload
`Data` and decoded PCM still live in process RAM during the request — that
in-memory exposure is the ADR-024 Tier-1 concern (process lockdown), not a
data-at-rest one.

## Request-content retention

There is none. After ADR 025 the daemon persists **no** request inputs or
outputs: `/v1` + `/api` inference is synchronous and keeps nothing on disk,
the queue (the only tenant that ever stored request blobs) is gone, and
audio/video uploads are decoded in memory (S5, above). Only the audit log
(retention bounded by `audit_retention_days`) and usage counters persist,
and neither holds request content. (The former `queue_result_ttl_secs` /
`queue_max_rows` / `drop_request_content` knobs are gone with the queue.)

## At-rest encryption

The store engine is **SQLCipher** (AES-256), consumed as the
[`skiptools/swift-sqlcipher`](https://github.com/skiptools/swift-sqlcipher)
package — source-built amalgamation on the embedded LibTomCrypt provider,
cross-platform, exact-version pinned (ADR 043). It is opt-in:

### Option A — `encrypt_store` (recommended for sensitive content)

Set `encrypt_store = true`. The daemon resolves an encryption key, opens
the store keyed (so the whole file — credentials + audit/usage metadata — is
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
