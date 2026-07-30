# ADR 043 — Adopt swift-sqlcipher package over vendored SQLCipher amalgamation

- **Status:** Accepted (operator, 2026-07-29)
- **Date:** 2026-07-29
- **Deciders:** operator + agent
- **Context source:** 2026-07-29 session (vendored-upstreams review during the Linux/CUDA spike); supersedes the M34.3a vendoring decision (commit `834470b6`)

## Context

`AthenaStore`'s engine is the in-tree `Sources/CSQLCipher` target: the SQLCipher **4.6.1** amalgamation (9.3 MB `sqlite3.c`) compiled on the CommonCrypto provider (M34.3a). It is now **11 releases stale** (upstream: 4.17.0) — exactly the "out-dated security component" failure mode Zetetic warns third-party packagings about — updates are manual, and the Linux port (ADR 040 gate; spike verdict GO-conditional 2026-07-29) would require building and maintaining a second, OpenSSL-provider variant.

The packaging landscape changed since M34: [`skiptools/swift-sqlcipher`](https://github.com/skiptools/swift-sqlcipher) is an actively-maintained, source-built SPM packaging of the amalgamation (SQLCipher 4.17.x) with **bot-automated upstream tracking** (a PR per SQLCipher/SQLite release), cross-platform targets (macOS/iOS/Linux/Android/Windows), a `SQLCipher` product exposing the **raw C module** (the `sqlite3_*` API `AthenaStore` already calls — their Swift wrapper is not used), `SQLITE_TEMP_STORE=2` as an unconditional default, and the same `strchrnul`/deployment-target fix our README documents. Its crypto provider is **vendored LibTomCrypt** (the only SQLCipher provider embeddable with zero platform dependencies), not CommonCrypto/OpenSSL.

Facts established before deciding: SQLCipher's **on-disk format is crypto-provider-independent** (same page format/KDF/HMAC parameters), and Athena uses stock cipher settings only (`sqlite3_key` + `sqlcipher_export`; no custom cipher PRAGMAs) — so existing stores open unchanged across the swap. Both production installs are operator-controlled. Key management is untouched: Athena derives and passes raw key bytes (RAM-only, zeroized after `sqlite3_key`); no SEP/TPM/keychain involvement exists on either side of the swap.

## Decision

Replace `Sources/CSQLCipher` with a dependency on `skiptools/swift-sqlcipher`'s `SQLCipher` product (raw C module), **pinned to an exact version and bumped deliberately** — a crypto engine must not change under `swift package update`; version bumps are reviewed commits. Delete `Sources/CSQLCipher` outright (delete-not-toggle; git history is the revert path). Land **before v0.11.0** so the first public release ships a current engine and no vendored amalgamation.

Accepted posture change (the price, on the record): at-rest crypto is now **LibTomCrypt** (software AES, community-maintained, vendored+bot-updated in the package) instead of Apple CommonCrypto (hardware AES, OS-maintained). Assessed acceptable because: the store is sparse control-plane rows (auth/audit/usage — ADR 025), `encrypt_store` is opt-in, PBKDF2-at-open slows by ~tens of ms once per daemon start, and the software-AES cache-timing side-channel class is out of scope under ADR 024's threat model (a co-resident adversary able to mount it is past the honesty boundary already; store crypto traffic is sparse besides).

Verification bar for the swap (all required):
1. `NDEBUG` presence on their C target — if absent, one-line PR upstream (they merge external PRs), not a fork.
2. **Bidirectional encrypted-store compat**: a store created by the current binary opens under the new engine, and vice versa.
3. `deploy/e2e-rbac.sh` §27 (`encrypt_store` open/migrate/plaintext paths) green; full unit tier green.

## Rejected alternatives

- **Keep the in-tree vendoring** (status quo): 11 releases stale with manual maintenance, and the Linux port doubles it (OpenSSL variant). Fails the reason the review happened.
- **Zetetic's official `SQLCipher.swift`**: prebuilt binary `xcframework` — no source build, no flag control, Apple-platforms-only forever (no Linux), and a binary supply-chain link in a hardened, notarized appliance. Currency and official support are real but don't outweigh source + cross-platform.
- **Build and publish our own package** (auto-regenerating CC/OpenSSL amalgamation): duplicates two existing maintained packagings; permanent maintenance for a differentiator (platform crypto) the threat model doesn't require. Killed on the reuse rung.
- **Contribute CC/OpenSSL provider traits to skiptools**: viable (their trait system invites it; ~⅔ acceptance odds judged from repo history) but unnecessary once LibTomCrypt was assessed acceptable — it would preserve a posture we no longer need to preserve, at the cost of an upstream-PR dependency (cf. the stalled swift-huggingface #50).

## Consequences

- **ADR 024's at-rest narrative** (store leg) now names LibTomCrypt; the swap paragraph above is the re-argument. Working-set honesty boundary unchanged.
- **Linux port**: the CUDA audit's "SQLCipher CC → OPENSSL" work item is **deleted** — the same dependency serves macOS and Linux. One fewer platform seam.
- **Upkeep ritual**: SQLCipher currency arrives as a reviewed pin-bump (their bot PRs upstream releases; we bump on our cadence). Athena no longer regenerates amalgamations; `Sources/CSQLCipher/README.md`'s recipe retires with the directory.
- **Migration**: none — both installs open existing stores unchanged (format is provider-independent; verified stock PRAGMAs). No config, CLI, or API surface changes.
- **Docs**: `docs/at-rest.md` re-points its engine/provider description; CLAUDE.md ADR list gains this entry.
- Execution is a single slice: Package.swift dep+target swap, `AthenaStore.swift` import change, delete `Sources/CSQLCipher/`, docs touch-ups, verification bar above.
