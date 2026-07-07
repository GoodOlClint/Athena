# ADR 000 — Pre-ADR foundational decisions (retrospective)

- **Status:** Accepted (retrospective record)
- **Date:** 2026-07-07 (reconstructed); decisions themselves date 2026-05-16 → 2026-06-11
- **Deciders:** operator + agent
- **Context source:** reconstructed from the git tag ledger (v0.1.0 → v0.10.107) and milestone commit messages; see per-entry evidence pointers

## Why this record exists

Athena's numbered ADRs begin at ADR 001 (`9b58bc3`, 2026-06-11), but the codebase begins at `65f2666` (2026-05-16, v0.1.0) — the first commit already ships the memory governor, the module protocol, the launchd daemon, the `/v1` surface, and the passive-oracle posture. The foundational architectural decisions of that first three-week window were made without contemporaneous decision records.

This document reconstructs them so the public repository's decision trail is complete from the first commit. **Every entry here is reconstructed, not contemporaneous** — the reasoning is inferred from the code as it landed and from the milestone commit messages, not transcribed from a decision written at the time. Where a later ADR formalized or revised one of these decisions, the entry points forward to it.

Evidence pointers are commit short-hashes (stable across the publication history rewrite only in relative order — after the rewrite, re-anchor to the milestone message text if a hash no longer resolves) and the tag ledger.

## The decisions

### 1. Swift + MLX, one native daemon on Apple Silicon

- **Date / evidence:** 2026-05-16 — `65f2666` (m0, governed serve path), `d17e43b` (m1, real MLX-backed generation); v0.1.0.
- **Context:** the goal was local LLM chat, text embeddings, and audio transcription on Apple Silicon, in one process, with no cloud dependency. MLX is the Metal-native inference substrate with a Swift surface; nothing else gives Metal-backed inference and a single-language daemon.
- **Decision:** one Swift package building one binary, targeting macOS on Apple Silicon, with MLX as the inference substrate for every modality. The build requires full Xcode (the Metal shaders cannot compile under Command-Line Tools alone).
- **Later:** the single-binary, single-substrate shape is reaffirmed by ADR 040 (monorepo, no OS split; Linux would enter as platform seams, not a sibling implementation).

### 2. Passive-oracle rule (no outbound network except model fetches)

- **Date / evidence:** 2026-05-16 — `65f2666` (m0); later pinned by `PassiveOracleContractTests`.
- **Context:** a local daemon holding a user's prompts and audio must not exfiltrate. The simplest defensible security posture is to forbid outbound network entirely.
- **Decision:** Athena answers inbound requests only. Outbound network is forbidden except Hugging Face model-weight fetches (and, added later, an opt-in remote-syslog sink). No result webhooks, no billing callbacks, no telemetry.
- **Later:** binding rule in `CLAUDE.md`; the opt-in remote-syslog carve-out is the single sanctioned exception.

### 3. Unified Metal memory governor (the thesis)

- **Date / evidence:** 2026-05-16 — `65f2666` (m0, memory governor + module protocol).
- **Context:** multiple inference modalities share one Metal memory pool. Two uncoordinated allocators on that pool thrash and OOM. The differentiator is coordinating them, not any one modality.
- **Decision:** one `MemoryGovernor` arbitrates a single Metal budget; every inference module reserves and releases against it through a shared module protocol. Inference never composes at the allocation layer.
- **Later:** formalized as the product thesis in ADR 011; accounting truthfulness in ADR 023; execution-exclusive slot in ADR 029; per-sequence KV accounting in ADR 039.

### 4. Two-dialect HTTP API (`/v1` + `/api`)

- **Date / evidence:** 2026-05-16 `65f2666` (`/v1/chat/completions`); 2026-05-17 `89bf701` (m6.1, native/compat shim) and `336ef22` (m7.1) grow the native surface.
- **Context:** OpenAI-compatible drop-in clients need `/v1`; the daemon also needs a native control plane (model store, lifecycle) with no OpenAI equivalent.
- **Decision:** `/v1/*` is the inference + data surface (OpenAI-compatible drop-ins plus native extensions under the same namespace); `/api/*` is the Athena-native control plane.
- **Later:** ADR 013 makes `/v1` the single inference surface and `/api` control-only; ADR 031 removes the deprecated native inference `/api/chat`.

### 5. Loopback default on port 7447

- **Date / evidence:** 2026-05-16 — `65f2666` (m0).
- **Context:** the daemon needs a stable default endpoint that does not collide with common local inference servers (e.g. 11434). The project's identifiers are drawn from Greek-Minoan myth with explainable etymology.
- **Decision:** bind `127.0.0.1:7447` by default; auth is disabled in loopback dev mode.
- **Later:** unchanged; the port is part of the public wire contract.

### 6. Bearer-token RBAC

- **Date / evidence:** 2026-05-17 — `57a6fdc` (m12.1b, hash-only auth keys at rest + `athena auth`).
- **Context:** non-loopback deployments need authenticated, role-scoped access; tokens must not sit in plaintext at rest.
- **Decision:** `Authorization: Bearer <token>`; each token resolves to a user with roles; each route requires a single RBAC permission. Tokens are stored hashed. Loopback with no seeded users opens every route (dev mode).
- **Later:** hardened across M12–M17 (constant-time compare, expiry, login limiter keyed on peer IP — ADR 004); the auth/audit/usage store is the sole reason a non-loopback daemon keeps SQLite (ADR 025).

### 7. One embedded SQLite store, SQLCipher-capable

- **Date / evidence:** 2026-05-17 `336ef22` (m7.1, `AthenaStore` — one embedded SQLite store); 2026-05-22 `9181795` (m34.3a, vendored SQLCipher on CommonCrypto, inert swap).
- **Context:** the daemon needs embedded persistence (originally vectors and a job queue; later auth, audit, usage) with an at-rest encryption option and no external database process.
- **Decision:** a single embedded SQLite store behind the `AthenaStore` actor; the engine is the vendored SQLCipher amalgamation on Apple CommonCrypto (no OpenSSL), which vends as standard SQLite until a key is set, making at-rest encryption opt-in.
- **Later:** ADR 025/026 remove the content-bearing tenants (vector DB, job queue) and make loopback mode create no store at all; the container survives only for auth/audit/usage.

### 8. Substrate-fork strategy (consume mlx-swift-lm, keep it pristine)

- **Date / evidence:** 2026-05-16 — `d17e43b` (m1); `Package.swift`.
- **Context:** Athena reuses the substrate's Qwen3.5/Gemma4/TokenIterator code but needs model deltas (a custom model type) the upstream does not carry. Editing the substrate in place would fork it unmanageably.
- **Decision:** consume `mlx-swift-lm` as a dependency (a path dependency during cross-repo development, an SCM pin on the operator's public fork for reproducible releases); vendor Athena-owned model code into `AthenaModels` and register it into the substrate's public model-type registry; keep the substrate itself an unmodified upstream.
- **Later:** ADR 028 retires the bespoke substrate deltas (DFlash, TurboQuant) so the fork can track upstream; the 2026-07-02 SCM-pin change ends silent path-dep branch drift.

### 9. Vendored Qwen3.5 + MTP approach

- **Date / evidence:** 2026-05-16 — `65f2666` (scaffolding), `6b71cb9` (m2, fork RMSNorm convention — MTP checkpoints coherent).
- **Context:** multi-token-prediction (MTP) speculative decoding needs a model type not present upstream, and the substrate must stay pristine (decision 8).
- **Decision:** vendor Qwen3.5 with a custom MTP head into `AthenaModels`, registered into the substrate's model-type registry; greedy-only to start, with bit-identical-greedy as the correctness bar.
- **Later:** ADR 032 adds Gemma 4 MTP as a second drafter backend behind the same per-request `speculative` knob; the vendored-MLX upstreaming work folds several deltas back into the substrate.
