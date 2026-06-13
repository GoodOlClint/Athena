# Audit Remediation Program — M65–M71

Remediates the combined findings ledger: `AUDIT_FINDINGS.md` (2026-06-08 baseline, IDs `A1`…`L11`) and `AUDIT_FINDINGS_V2.md` (2026-06-09 re-audit, IDs `NA1`…`NL4`). Both docs stay frozen as the findings record; **this file is the live tracker** — check items off here, with the closing version noted.

Scope counts at plan time: 1 Critical + ~49 High + ~55 Medium confirmed across both audits, plus a Low tail (~100). Already closed before this plan: C8 (v0.10.107). Refuted, no action: B9 (behavioral half — help-text drift folded into M71.2), B14, E8.

## Principles

- **Order = blast radius**: remote attack surface → data loss/lockout → silent inference corruption → concurrency → perf/operability → CI blindness → polish tail.
- Each slice is independently shippable in the usual style: commit + annotated tag to main, `Athena.appVersion` bump in the slice commit, e2e gate green.
- Every fix that *can* carry a unit test does — but the structural "make Server/Commands testable at all" work is its own milestone (M70), so earlier slices test what's testable today and M70 backfills.
- When a slice touches a module, it also **re-verifies that module's `not-refound` baseline items** (listed per milestone below) and closes or re-confirms them in this tracker — they're cheap to adjudicate with the code already open.
- Items marked **DECISION** change a contract or policy; they need explicit sign-off before implementation and a CLAUDE.md / ADR edit in the same window.

---

## M65 — Remote attack surface (security)

The only Critical lives here (G1), plus everything an unprivileged or low-privileged remote caller can trigger.

### M65.1 — FFI hardening (rust-shim) ✅ v0.10.117
- [x] G1 (Critical) per-entry `catch_unwind` via `ffi_guard`; crate stays `panic="unwind"` (abort path is parked DECISION #4, not taken) (v0.10.117)
- [x] G2 reject oversized token ids before `max_id+1` dense alloc (`MAX_VOCAB` cap) (v0.10.117)
- [x] G3 cap schema bytes/depth/repetition before compile (`validate_schema`: 1 MiB / depth 64 / 100k reps) (v0.10.117)
- [x] G6 cap caller-supplied `n`/`l` in `build_words` (`MAX_VOCAB`/`MAX_TOKEN_BYTES`) (v0.10.117)
- [x] G10 `id < size` guard; mask short-write fails loud; `set_err` interior-NUL + reentrancy safe (v0.10.117)

> Note: the 7 `StructuredGuideTests` regex cases + 1 `StructuredShimTests` ordering artifact are pre-existing reds on the dead regex path (engine is schema-only since M53) — that's **NG1**, scheduled for **M70.2**. Untouched by M65.1; new hostile-schema Swift test passes.

### M65.2 — WebUI & login surface — 5/6 ✅ v0.10.118 (A3 deferred)
- [x] NA1 (High) escape `t.label`/`t.username`/`t.scope` (+ users/roles fields) in /ui users page — shared JS `esc()` + `data-u` delete button (v0.10.118)
- [x] A11 HTML-escape config values via shared `esc()` (replaces `"`-only escape; also path + error sinks) (v0.10.118)
- [x] A3 rate-limit `POST /ui/login` by IP + backoff — shipped in **M65.6** (v0.10.122) once the `AppRequestContext` peer-IP plumbing landed (ADR 004: peer addr, no XFF trust)
- [x] A10 generic login failure + always-run dummy PBKDF2 (`Passwords.dummyVerify`) — closes enumeration timing oracle (v0.10.118)
- [x] A12 `Secure` cookie attribute when the daemon serves TLS (`tlsEnabled`) (v0.10.118)
- [x] NA7 percent-decode fallback keeps `+`→space normalization (v0.10.118)

> Server/ login/session/XSS logic isn't unit-testable until M70 (NA2); validated via build + e2e-rbac 494/0 (login/cookie path) + binary inspection.

### M65.3 — Untrusted input caps (DoS) ✅ v0.10.119
- [x] ND1 (High) finite-check + Double-domain clamp before `Int()` in speaker embeddings → 400 audio_segment_invalid (v0.10.119)
- [x] D4 `AudioDecode` decode-sample ceiling (~4h) + clamped/guarded header reserve (v0.10.119)
- [x] A9 multipart `firstRange(of:)` split + `maxParts=256` cap (v0.10.119)
- [x] NI4 per-input embedding token ceiling (32768) → 400 input_too_long (new AthenaError) (v0.10.119)
- [x] G4 + NC2 fail closed: `structuredRequestError()` 400 on missing/unserializable schema (sync+queued); MLXLLMModule throws structured_output_unavailable on unresolvable vocab / uncompilable schema (v0.10.119)

### M65.4 — Path confinement (file ops on caller-influenced paths) ✅ v0.10.120
- [x] A1 (High) store export confined under `exports/` dir; bare filename only; no overwrite (409) (v0.10.120)
- [x] C5 ModelPull `isValidName` before `removeItem` (v0.10.120)
- [x] C6 ModelConvert `isValidName(outName)` before removeItem/createDirectory (v0.10.120)
- [x] C7 `copy` confines src too (`isValidName(src)`) (v0.10.120)
- [x] C20/C24 prune child-name assert; C20 size/cp confined at entry via `isValidName` (HF-cache symlink target follow is by-design); C24 aux-copy = `isAux` exclusion-allowlist over trusted snapshot, bare `lastPathComponent` (v0.10.120)
- [x] D6 confined at the shared resolver `ModelStoreLayout.localDirectory` (covers whisper + all modules) (v0.10.120)
- [x] NC12 same shared-resolver guard validates the allowlist id at load-path resolution (v0.10.120)

### M65.5 — AuthZ gaps & info leaks — contained items ✅ v0.10.121 (A5/A3/H6 → M65.6)
- [x] NA6 ownerless queue jobs: `owner==nil` → admin-only under enforced auth (v0.10.121)
- [x] A21 + NE7 substrate detail (paths/repo ids/state) scrubbed from client `message`; `serverDetail` → os_log only; remaining A21 sites are benign decode/validation errors (v0.10.121)
- [x] A22 fail closed on group/other-accessible auth-keys file (skip its keys, error not warning) (v0.10.121)
- [x] **A5 (High) → M65.6**: single caller/permission resolution in AuthMiddleware (4 drifted copies) — shipped M65.6 v0.10.122 (custom `AppRequestContext` + `ResolvedCaller` task-local)
- [x] **A3 → M65.6**: peer-IP login limiter (ADR 004) — shipped M65.6 v0.10.122
- [x] **H6 → M65.6**: store-layer owner filter for jobs/usage — shipped M65.6 v0.10.122

### M65.6 — Auth-context refactor (structural) ✅ v0.10.122
- [x] A5 single resolution in `AuthMiddleware`: a new `AppRequestContext` (RequestContext + RemoteAddressRequestContext) replaces the default `Router()` context; the resolved caller is published once via a request-scoped `ResolvedCaller` task-local (bound beside the existing `LogScope`), and the 4 former copies (`callerPermissions`, `uiCaller`, `auditPrincipal`, `queuePrincipal`) now read it instead of re-deriving from headers — no more bearer-vs-cookie/sentinel/loopback drift (v0.10.122)
- [x] A3 peer-IP `POST /ui/login` limiter: `loginLimiter` (token bucket, burst 5 / 0.2 per sec) keyed on `context.remoteAddress` (TCP peer only, **no XFF trust** per ADR 004); 429 + Retry-After; behind-a-proxy limitation documented in code + ADR. e2e phase 33 asserts the 429 (v0.10.122)
- [x] H6 optional owner filter at the store layer (`getJob`/`listJobs`/`allUsage`); nil = unfiltered (admin/worker/auth-off), a scoped principal confines the rows (ownerless NULL never matches) — defense-in-depth beneath the handlers' `canAccess`, wired via `ownerScope(who)`. New `AthenaStoreTests.testOwnerScopedQueriesH6` (v0.10.122)

> Server/ resolution/login logic isn't unit-testable until M70 (NA2); A5/A3 validated via build + e2e-rbac (497/0, incl. the new login-throttle phase 33) + reasoning. The H6 store helpers ARE unit-testable and now are.

**DECISION items parked at M65 (need sign-off, then ADR):**
- A2/K1: require TLS (or explicit `--insecure`) for non-loopback auth-on daemons; add https to the portable client. Contract change for remote deployments.
- H5: per-owner `owner` column on vectors (schema migration + cross-principal semantics).
- B2/K7/K13: stop accepting secrets via argv (breaking CLI change; `--password-stdin` pattern).

Re-verify while here (not-refound): A2, A9*, A10*, A12*, A13, A14, A16–A22, A26, A27, G-module (none), D12, D13. (*already being fixed above — confirm the rest.)

---

## M66 — Data integrity & credential lifecycle

Local data loss, lockouts, and the config parser silently lying.

### M66.1 — Store integrity ✅ v0.10.123
- [x] NH1 (High) recoverable `migrateToEncrypted` swap: plaintext → `.migrate-bak`, move `enc-migrate` in, delete `.bak` only on success; rolls back on a failed move. New static `recoverInterruptedMigration(at:)` run at startup (Load.swift) completes/rolls back an interrupted swap (3 crash states). `testRecoverInterruptedMigrationNH1` (v0.10.123)
- [x] H2 transactional allowlist default clear-then-set via new `withTransaction` (`BEGIN IMMEDIATE…COMMIT`/`ROLLBACK`) — exactly-one-default invariant survives a mid-swap throw. `testAllowlistDefaultSingleSwapH2` (v0.10.123)
- [x] H11 `deleteUser` cascade (roles+tokens+user) in one `withTransaction`; now `throws -> Bool` (errors surfaced, not swallowed); callers (server→500, CLI→die) updated (v0.10.123)
- [x] NH2 `VectorStore.delete` mutates the resident cache only when the persisted delete succeeded (no cache/store desync on a SQLite failure) (v0.10.123)
- [x] H13 `trimJobs` excess computed against the TERMINAL-row count (retained results), not total rows — a large pending backlog no longer evicts finished results; `queueMaxRows` doc + the two trim tests updated (v0.10.123)
- [x] H14 failed audit write → `.error` log + `metrics.recordAuditWriteFailure()` surfaced as `athena_audit_write_failures_total` on `/metrics`; e2e phase 31 asserts the counter (v0.10.123)
- [x] H15 zero the at-rest key byte buffer after `sqlite3_key` (defer-zero on every exit path) (v0.10.123)

> Store helpers ARE unit-testable: `AthenaStoreTests` now 18 cases (NH1/H2 added, H11/H13 updated), all green. Server wiring (H11 500 path, H14 metric) validated via build + e2e-rbac 498/0.

### M66.2 — Token/user lifecycle ✅ v0.10.124
- [x] NA4 server `handleTokenRotate` now `putToken` (new) BEFORE `deleteToken` (old) — a putToken failure leaves the holder's existing token working instead of a lockout-on-partial-failure (v0.10.124)
- [x] B4 CLI `auth token rotate`: same put-before-delete ordering (v0.10.124)
- [x] B5 `auth user add` refuses to overwrite an existing account without `--force` (new flag); points at `auth user passwd` for a roles-preserving reset. e2e phase 1 asserts refuse/force (v0.10.124)
- [x] NB11 `usersWithRole` now THROWS on a query error (was `[]`, indistinguishable from "no admins"); the three last-admin guards (CLI `guardLastAdmin`, server user-delete + role-revoke) fail CLOSED on the throw; the two `.count` display sites use `try?` (v0.10.124)
- [x] H12 `listTokens` returns a 12-hex `hashPrefix` only (no full SHA-256 digest); a new `tokensMatchingHashPrefix` confines the full hash to the rm/rotate paths (SQL `lower(hex(hash)) LIKE`). `testTokenHashPrefixMatchingH12` + e2e `/api/tokens` length assert (v0.10.124)
- [x] B17 install-seeded admin token gets a default 90-day TTL (was never-expiring); post-install message notes the expiry + `auth token rotate` (the seeded password recovers access if it lapses) (v0.10.124)

> Store helpers (`usersWithRole` throws, `tokensMatchingHashPrefix`) unit-tested (`AthenaStoreTests` 19/0). Server/CLI fail-closed wiring + rotate ordering validated via build + e2e-rbac 502/0 (added B5 + H12 asserts).

### M66.3 — Install & client store safety ✅ v0.10.125
- [x] B3 (High) install seed-DB chown goes through `FsOwn.chown` (open `O_NOFOLLOW` → `fchown` the fd), so a symlink swapped in after the root-seeded DB can't redirect the chown onto an arbitrary file (v0.10.125)
- [x] B7 `ensureDir`/`makeTraversable` chmod+chown via the same `O_NOFOLLOW`-fd helpers (`FsOwn.chmod`/`chown`) instead of by-path `setAttributes` (v0.10.125)
- [x] B6 new `isSafeToOwn` rejects a config `data_dir`/`model_store`/`log_dir` that resolves to the fs root, a bare top-level dir, or a system subtree (e.g. `data_dir="/etc"`) before any chown; install prefix + service home + external volumes stay allowed (v0.10.125)
- [x] NB1 `Start.run` validates `isValidLabel` UP FRONT (dies) before euid branching, so a malformed `--label` no longer skips the M43.1 root-daemon hard-fail and spawns the MLX daemon as root; e2e phase 1 asserts the rejection (v0.10.125)
- [x] NK1 client `store import`: tri-state liveness probe (reachable/absent/**unknown**) — a probe timeout no longer reads as "absent"; refuses on unknown unless `--force`. Copy-to-temp-then-atomic-swap (with restore-on-failure) so the existing DB is never destroyed before the copy succeeds; old WAL/SHM removed only after (v0.10.125)
- [x] B8 `StoreKey.resolve(trustEnv:)`/`source(trustEnv:)` — privileged (root) `install`/`doctor` pass `trustEnv: false` so the sudo-inherited environment is never trusted as the key source. A fresh install seeds plaintext; the daemon encrypts on first boot (crash-safe per NH1) as the service user with its own Keychain key (v0.10.125)

> Filesystem/privilege helpers live in the executable + client targets (not unit-testable until M70/NA2); validated via xcodebuild Release + e2e-rbac 545/0 (NB1 assertion) + a clean client-target compile (NK1).

### M66.4 — Config parsing truth (AthenaDeploy + ConfigEditor) ✅ v0.10.126
- [x] J1 (High) new strict `parseBool` (true/false, 1/0, yes/no, on/off — case-insensitive); all 6 bool keys route through it; an unrecognized value is `ParseError.invalidBool`, not silent false. `testBoolKeysTruthyAndStrict` + `testBoolKeyInvalidValueThrows` (v0.10.126)
- [x] NJ1 (High) `scalar()` normalizes CRLF/CR→LF up front, so a Windows-saved config no longer leaves a trailing `\r` on every value (broke int/bool/quote parsing). `testCRLFLineEndingsParse` (v0.10.126)
- [x] J2 `scalar()` treats a quoted value as literal — inline `#` inside quotes is part of the value; only an unquoted value starts a comment. `testQuotedHashIsLiteralUnquotedIsComment` (v0.10.126)
- [x] NB2 `setScalarThrowing` rejects a string value containing a quote, backslash, or control char (incl. CR/LF — the newline-injection vector via `/ui/api/config`); AND rolls back to the pre-edit contents when THIS edit makes a previously-valid config unparseable (pre-existing breakage still keeps+warns). e2e NB2 asserts (v0.10.126)
- [x] NB8 `setScalarThrowing` validates `engine`∈`Engine.allCases` and `kv_compression`∈`KVCompression.allCases` at set-time (reject the typo here, not at the next boot). e2e NB8 asserts (v0.10.126)
- [x] B15 a NEW bare key is inserted before the first `[section]` header (stays top-level) instead of landing inside the last table. e2e B15 assert (v0.10.126)
- [x] NJ2 + NB9 `ConfigEditor.resolvePath(nil)` now honors `$ATHENA_CONFIG`; the launchd plist exports `ATHENA_CONFIG=<prefix>/etc/athena/athena.toml` (`EnvironmentVariables`), so the daemon's TOML-only re-reads (kv_compression/prompt_cache_*/proxy) resolve to the prefix-correct file on a non-default `--prefix` install — and the CLI gets the same escape hatch. `testConfigPathExportsAthenaConfigEnv` (v0.10.126)

> Parser (AthenaDeploy) + plist changes are unit-tested (`AthenaConfigTests` 31/0, `LaunchdPlistTests` 4/0). ConfigEditor lives in the executable target (not unit-testable until M70) — validated via e2e-rbac 551/0 (offline `config set` NB8/NB2/B15 asserts).

### M66.5 — Remove secrets from argv (ADR 005, breaking) ✅ v0.10.127
- [x] B2 hard-removed `--password` from `auth user add`, `auth user passwd`, `proxy login` (macOS) + the portable `auth user add` (client). Secrets resolve via a shared `PasswordInput.resolve` (macOS) / `RemoteAuth.resolvePassword(stdin:)` (client): `--password-stdin` (one line) > `$ATHENA_PASSWORD` > interactive no-echo prompt (confirmed for new). e2e migrated to `ATHENA_PASSWORD`; new B2 assert that `--password` is rejected. ADR 005 → Implemented (v0.10.127)
- K7 (`Credentials` Keychain `security -w <value>` argv leak) + K13 (`--key` in argv) — the remaining argv-secret items; audited here, **deferred to M71** (need a Keychain-API / `--key`-steering change beyond the `--password` removal).

### M66.6 — Vector-store owner-scoping (ADR 006 / audit H5) ✅ v0.10.128
- [x] H5 nullable `owner` column on vectors (additive migration + `vectors_owner` index); upsert stamps the authenticated principal; `query`/`stats`/`delete` owner-filter at the `VectorStore` cache AND the SQLite layer (defense-in-depth); admin/auth-off see across owners; legacy NULL-owner rows are admin-only; a cross-owner upsert is rejected 409 (`vector_owner_conflict`), a cross-owner delete is 404. New `VectorStore.Caller`. ADR 006 → Implemented. Tests: `testVectorOwnerColumnH5` (store) + `testOwnerScopingCI`/`testLegacyNullOwnerIsAdminOnlyCI` (CI) + `testQueryOwnerScopedGated` (MLX-gated); e2e phase 25.2b cross-tenant (query/stats/delete/overwrite on the real binary). OpenAPI vectors descriptions updated (v0.10.128)

**M66 complete (M66.1–.6).**

Re-verify while here: B6*, B7*, B8*, B10–B13, B15*, B18–B23, H6*, H14*, J5, K3, K6.

---

## M67 — Inference correctness (silent wrong output)

### M67.1 — TriAttention / models ✅ v0.10.129
- [x] NF1 (High) `TriAttentionKVCache` now tracks a monotonic `absolutePosition` (total tokens ever appended, never decremented by `compress()`, decremented by `trim()`) and overrides `ropeOffset` to drive RoPE off it instead of the eviction-compacted `offset`. Keys are stored post-RoPE, so retained keys carry their original absolute rotations; pre-fix, `compress()` regressing `offset` rotated new q/k at a too-small position and corrupted attention the moment eviction fired. `offset` is now purely stored-array slice/mask extent. Persisted as a 9th `metaState` field (legacy 8-field meta defaults it to `offset`). New cache-level assertion in `TriAttentionE2ETests.testEvictionBoundsCacheAndPopulatedRoundTrip` (ropeOffset == absolute, > compacted offset) (v0.10.129)
- [x] NF2 request-scoped eviction policy is now a `@TaskLocal TriAttentionRequestPolicy.current` bound around `container.generate` in `beginGeneration`, read EAGERLY by the substrate's same-Task `newCache` (TokenIterator init). Removed the shared `var triAttentionEviction` from both model classes — the former set-in-one-perform / consume-in-a-separate-hop split let a concurrent structured/speculative request clobber it across the await gap (last-writer-wins → standard request silently built `KVCacheSimple`, KV grew unbounded). Default `nil` ⇒ `KVCacheSimple`, so the speculative/guided paths (which never bind it) are eviction-inert with no explicit clear. Subsumes baseline C2 (v0.10.129)
- [x] F1 masked gated-delta Metal kernel now writes a defined `0` on the masked branch (`else { y[dv_idx] = 0 }`) instead of leaving `y` uninitialized for padded/cross-seq positions (garbage / NaN-poison risk); state carries forward unchanged, matching the ops fallback. Dead code in the unmasked kernel variant (v0.10.129)
- [x] F4 `AthenaQwen35MoEModel.sanitize` extracted a `splitFusedExperts(prefix:)` helper and now applies it to the MTP-MoE layers too: a fused `mtp.layers.L.mlp.experts.gate_up_proj` checkpoint is split into `switch_mlp.{gate,up,down}_proj.weight` (was: only per-expert MTP handled → a fused-MTP checkpoint was unloadable). Per-expert layout still falls through unchanged (v0.10.129)
- [x] NF9 `fromState` now validates restored geometry loudly (throws `CacheError`): `divideLength >= 1`, `0 <= prefixLength <= offset`, `absolutePosition >= offset`, `offset <= keys.dim(2)`. `shouldCompress()` also guards `divideLength > 0` against a `% 0` trap on the live setter path. New CI-safe rejection tests in `TriAttentionCacheTests` (zero-divide, prefix>offset, absolute<offset) (v0.10.129)
- [x] NF7 `gatedDeltaUpdate` routes a non-multiple-of-32 `Dk` (linear_key_head_dim) to the `gatedDeltaOps` fallback (correct for any Dk) instead of the kernel, whose `n_per_t = Dk/32` over 32 simd lanes would silently truncate the trailing `Dk % 32` state dims. Stock Qwen3.5 Dk=192 (÷32) keeps the kernel path. (Finding said ÷32, not ÷8 — kernel reality.) (v0.10.129)
- Re-verified while here: **F6** (createSSMMask per-call alloc incl. n==1) — confirmed still valid; it's a substrate-side perf item (not an AthenaModels function), correctly deferred to M69 (perf), not a correctness fix.

### M67.2 — Embeddings ✅ v0.10.130
- [x] NI1 (High) + I1 + I4: the per-bucket mask is now built from each row's REAL token length (floored to ≥1 so an empty input can't make mean-pool divide by zero → NaN), not pad-equality — and the SAME mask is passed to `ctx.pooling(mask:)`, not just the forward. Fixes: pooling over pad positions on any mixed-length batch (mean) / selecting the final PADDED token (last-token), and a real token whose id==pad (id 0 when eos nil) no longer being masked (v0.10.130)
- [x] NI3 added a `desiredName` staging field to `MLXEmbeddingModule` (+ stub parity), set by rebind/embed and honored by `load(reservation:)` as `desiredName ?? residentId ?? defaultId` (survives `unload()`); a governor evict→reload no longer silently reverts the slot to the default model. Mirrors the LLM module's M62 cold-load seam (v0.10.130)
- [x] NI2 `StubEmbeddingModule` now resolves the request `model` via `canonicalByStoreIdentity` (bare basename OR full HF id), identical to `MLXEmbeddingModule` — a bare store-dir name no longer 400s under `--engine stub` while succeeding under `--engine mlx`. New CI-safe `testBareStoreNameResolvesForParityWithMLX` (v0.10.130)
- [x] I5 stopped hardcoding `applyLayerNorm: true` (now `false`): the substrate's parameterless `MLXFast.layerNorm` is not part of canonical sentence-transformers pooling for the configured models (bge-small, Qwen3-Embedding), so Athena was producing non-canonical vectors. `normalize: true` kept (L2 contract). **Output change**: every produced vector differs vs ≤v0.10.129 → persisted `/v1/vectors` indexes must be re-embedded (user-approved) (v0.10.130)
- [x] I6 produced-vector width is validated against `ctx.pooling.dimension` (when configured) and rejected if zero-length, before storing — a wrong/mis-sanitized model fails loudly instead of silently storing wrong-width vectors (v0.10.130)
- Re-verified while here: NI4 per-input token ceiling already shipped (M65.3, the `maxInputTokens` guard); I7 (canonicalization-before-resident-fast-path, perf) and I8 (stub empty-string all-zero vector) remain open Low items, deferred (I8 is stub-only cosmetic; I7 is perf → M69-class).

### M67.3 — Sampling & structured determinism — 5/6 ✅ v0.10.131 (C11 deferred)
- [x] C1 (High) `SpeculativeSampling.distribution` top_k AND top_p sorts now use a total-order comparator with an `$0 < $1` index tie-break. Swift's `sorted` is not stable, so equal-probability ties reordered run-to-run and changed which ids survived truncation — breaking same-seed reproducibility. Inert at temp=0 (the greedy one-hot returns before truncation). New CI-safe `testTopKTieBreakKeepsLowestIndices`/`testTopPTieBreakKeepsLowestIndices` (v0.10.131)
- [x] C12 `StructuredVocab.tokens` no longer fabricates `vocabSize-1` (a real token) as the eos when the tokenizer has none — it would `continue` past it, dropping that token from the guide's allowed set. Uses a sentinel one past the real range (`vocabSize`): the loop never hits it (drops no real token) and the shim's `build_words` adds a single never-emitted control slot. Real-model behavior (tokenizers WITH eos) byte-unchanged (v0.10.131)
- [x] C13 `StopStreamFilter` measures the hold-back in UNICODE SCALARS, not graphemes. A multi-scalar grapheme stop (ZWJ emoji, combining sequence) counted 1 grapheme → hold-back 0 → the stop could split across a chunk boundary and slip through. New CI-safe `testMultiScalarStopSplitAcrossChunks`; ASCII stops unchanged (v0.10.131)
- [x] G7 added `JSONValue.integer(Int64)`, decoded BEFORE Double, threaded through encode/foundationValue + the two `OpenAIDTO` switches. A schema integer constant >2^53 (`const`/`enum`/`minimum`/`maximum`) was decoded as Double and silently corrupted to its even neighbor. New CI-safe `testIntegerConstantAbove2to53RoundTripsExactly` (v0.10.131)
- [x] G9 rust-shim `advance` now `set_err`s the out-of-range case (a caller contract violation) so it is distinguishable from a legitimate schema rejection (a clean errorless `false`). New rust `advance_distinguishes_out_of_range_from_disallowed_g9` (cargo 10/0). Staticlib rebuilt (v0.10.131)
- [x] C11 **DONE in M67.6 v0.10.134** — see below. (The substrate's `TopPSampler`/`CategoricalSampler` time-seeded their private `RandomState` and ignored both the global seed and the task-local; the fix added a seedable seam to the substrate.)

### M67.4 — Whisper / transcription ✅ v0.10.132
- [x] D5 + ND3 `WhisperDecode.langOrder` is now the full 100-entry canonical Whisper ordering (16→100, large-v3 `yue` at index 99). Fixes BOTH directions: `languageToken` (forced `language:` past index 15 silently forced English) and `languageCode` (auto-detected language past 15 reported as "en" in `verbose_json`). New CI-safe `testLanguageTableCoversBeyondSixteen` (v0.10.132)
- [x] ND2 `MLXTranscriptionModule.loadModel` rejects any whisper checkpoint whose `config.n_vocab != 51866` (the large-v3 family the special-token ids are pinned to) — a v1/v2/medium model (vocab 51865/51864) shifts every special id from `transcribe` on and silently mis-decodes. Chosen the "fail loud at load" option over per-checkpoint id derivation (the family Athena ships) (v0.10.132)
- [x] D8 `transcribeResult` early-returns an empty result for empty PCM (was `LogMel.logMel([])` crash/NaN) (v0.10.132)
- [x] D9 `parseSegments` flush uses `start: segStart ?? (lastTs ?? 0)` — content after a closing timestamp now starts at the last timestamp, not 0 (out-of-order span). New CI-safe `testContentAfterClosingTimestampStartsAtLastTs` (v0.10.132)
- [x] D12 `TranscriptionFormat.srt`/`vtt` clamp the cue end to `max(start, end)` — an inverted span no longer emits a backwards cue. New CI-safe `testInvertedSegmentClampsEnd` (v0.10.132)
- [x] ND5 all three real audio modules (transcription/diarization/speaker-embedding) now evict in `setAllowedModelIds` via `ids.canonicalByStoreIdentity(r) == nil` (the same resolver as load/rebind) instead of a stricter `!ids.contains(r)` — an M42 live refresh re-supplying the resident model under an equivalent spelling no longer forces a needless multi-GB reload (v0.10.132)
- [x] D10 `AgglomerativeClustering.cluster` clamps `threshold` to the valid cosine-distance range [0, 2] and adds a `minClusters` floor (default 1 = prior behavior) so a permissive threshold can't over-merge distinct speakers into one. New CI-safe `testMinClustersFloorPreventsOverMerge` (v0.10.132)
- Re-verified while here: ND6 (`num_speakers` silently ignored on the default Sortformer path) is a contract/threading change (needs `diarize(...)` signature + OpenAPI reconcile) — left for M71 API-surface; ND4 (per-frame GPU→host `.item()` syncs) is perf → M69.4.

### M67.5 — LLM module odds ✅ v0.10.133
- [x] NC1 (High) `AthenaModelRegistration.currentModelDirectory` is now a `@TaskLocal`, bound per-load via `$currentModelDirectory.withValue(url)` around `loadModelContainer` (in both `MLXLLMModule.loadModel` and the free `ModelConvert.convert`). The substrate's `loadModelContainer → load{ loadContainer }` is a plain `await` chain (no detached hop before the registry creator), so the creator reads THIS load's directory. A queued `model_convert` and a serve cold-load can no longer clobber a shared global → no wrong-checkpoint MTP-suppression / keyNotFound (v0.10.133)
- [x] NC3 `preflightPromptCache(messages:tools:chatTemplateKwargs:)` now renders the SAME prompt generation will (tools + chat-template kwargs), so the governed prompt-cache cap check matches the real prefill size — a tool/kwargs-bearing request no longer passes preflight and then exceeds the cap mid-prefill. Threaded at the sync `/v1/chat` + queued-chat callsites; native `/api/chat` carries none (nil) (v0.10.133)
- [x] NC11 `residentDirectory` returns `URL?` via `.first?.url`, never a literal `[0]` subscript — `setAllowedModelIds([])` (empty DB allowlist) no longer leaves a property that would trap if wired up (v0.10.133)
- [x] NC13 `StubLLMModule.load` refuses to bind against an empty allowlist (`guard !modelIds.isEmpty`), so `residentBytes` stays 0 and `refreshAllowlist` releases the governor slot instead of holding a reservation for no models. New CI-safe `testEmptyAllowlistLoadReservesNothing` (v0.10.133)
- [x] NE5 the LLM rebind / `selectColdLoadModel` / `/api/models/load` lookups (+ the stub) now use `canonicalByStoreIdentity` (bare name OR full HF org/name id), uniform with the embedding/audio modules — `model:"Qwen/Qwen3-4B"` to `/v1/chat/completions` against a bare-name allowlist no longer 400s while the same form works on `/v1/embeddings`. New CI-safe `testFullHFIdResolvesToBareAllowlistRow` (v0.10.133)

### M67.6 — C11 seeded sampling (substrate seam) ✅ v0.10.134
- [x] C11 the substrate `TopPSampler`/`CategoricalSampler` time-seeded a private `RandomState` (`DispatchTime.now()`) and consulted neither the global seed nor the task-local — so a seeded `temp>0` request on a NON-MTP model (via `container.generate`) was non-reproducible AND the old global `MLXRandom.seed` was both ineffective for them and a cross-request race. **Fix:** added an additive `seed: UInt64?` to `GenerateParameters` (substrate), threaded through `sampler()` into `RandomState(seed:)` on both samplers (default nil = prior entropy-seeded behavior; inert at temp==0). Athena's `beginGeneration` now passes the per-request seed via `GenerateParameters.seed` and **dropped the racy global `MLXRandom.seed`**. Per-request, concurrency-safe by construction; the MTP speculative path was already per-request via `SamplingRNG`. **SUBSTRATE DELTA** `mlx-swift-lm@7eb154c` (additive, upstream-PR-shaped — candidate to roll up to `ml-explore/mlx-swift-lm` and re-pin to stock). Validated: Release build + e2e 561/0 + a real-model smoke (Qwen3.5-9B-mlx, temp 0.8: same seed byte-identical across runs, different seed diverges); gated `SeededSamplingReproducibilityTests` written for the M70.2 harness.

**M67 COMPLETE (M67.1–.6, v0.10.129–134).** No items deferred.

Re-verified across M67: F6 (createSSMMask alloc) → M69 perf; I7 (embed canonicalize-before-resident) → M69 perf; I8 (stub empty-string vector) deferred (stub cosmetic); C16–C19/C21/C23 (dead-code/dedup Lows) → M71 tail; NE-uncertains adjudicated in their slices (NE5 done; NE6 deadline-cancel callback → M68.4). NC2 (structured silent-unconstrained on unresolvable vocab) already closed in M65.3 (G4/NC2).

---

## M68 — Concurrency & lifecycle

### M68.1 — MemoryGovernor ✅ v0.10.135
- [x] NE1 (High) `.unloading` teardown now tracked in a per-id `teardown` Task handle (mirror of `inFlight`); `performLoad` awaits it before re-loading so `load()` can't race the still-pending detached `unload()` on the module actor. Both the eviction path (`evictSync`) and the explicit `unload(_:)` route through it; `unload(_:)`'s final `.unloaded` write is now state-guarded in `markUnloaded` (`if state == .unloading`) so a reload that already moved the slot to `.loaded` isn't clobbered back. `testReloadDuringTeardownWaitsForUnloadNE1` + `testExplicitUnloadThenReloadEndsLoadedNE1` (v0.10.135)
- [x] E1 `ensureLoaded`/`beginLoadIfNeeded` decide on `entries[id]?.state` read FRESH (not a value-copy of the `Entry` struct taken before the state-mutating `relievePressure`); `performLoad` re-reads `.loaded` after each suspension (teardown drain + `memoryEstimate()`) and early-returns (v0.10.135)
- [x] E2 in-flight cleanup is generation-token-keyed (`inFlightToken`/`loadSeq`): the detached `clearInFlight(_:token:)` and the `ensureLoaded` defer wipe the slot ONLY if it still holds the same generation, so a newer load that replaced it survives (`Task` is a non-identity value type — a token, not `===`) (v0.10.135)
- [x] E6 the load-failure path returns bytes ONLY if our reservation is still on the books (subtracts `reservation.bytes`, then nils it) — a concurrent `releaseSlot` during the load `await` can't trigger a double-subtract that corrupts `residentBytes`; the `.unloaded` fallback is guarded to the `.loading` transition we own (v0.10.135)
- [x] NE2 admission (`makeRoom`) failure now records `lastLoadError[id]` exactly like a `module.load()` throw, so the non-blocking `beginLoadIfNeeded` path surfaces the real `memory_budget_exceeded` 503 to the next caller instead of kicking another doomed load and 503-ing `module_loading` forever. `testAdmissionFailureSurfacedNotPerpetualLoadingNE2` (v0.10.135)
- [x] E5 both governor inits clamp a non-positive `totalBudgetBytes` (and the derived prompt-cache cap) to the physical-memory-derived default via `safeBudget`, so a misconfigured budget degrades to the standard budget instead of a daemon that 503s every load. `testNonPositiveBudgetClampsToDefaultE5` (v0.10.135)
- [x] E12 when the process-global probe delta lands below the static estimate (a concurrent teardown deflated it, possibly to ≤0 → reconcile skipped), the reconcile falls back to the module's own `residentBytes` self-report so `learnedFootprint` is still recorded; `max(...)` keeps the normal delta-≥-estimate case byte-unchanged. `testReconcileFallsBackToSelfReportE12` (v0.10.135)

> AthenaCore governor tests ARE CI-runnable (Stub modules, no MLX kernels): `MemoryGovernorTests` 22/0 incl. the 5 new M68.1 cases. Binding gate: Release build + e2e-rbac **561/0** (no route/spec change). Real-model concurrency smoke (Qwen3.5-9B-mlx): 4-way concurrent cold-load reserves the right bytes + reconciles to `loaded`, 4-way hot `ensureLoaded` fast-path all 200, and greedy temp=0 stays byte-identical across runs (bit-identical contract intact — the changes are admission/lifecycle accounting only, no decode/sampling/cache touch).
> Re-verified while here: **E17** (resident-id one-off downcast in the load-event log path) — confirmed a Low arch nicety (cache on `Entry`), NOT a concurrency defect; the downcast is on the post-load log path only, deferred to the M71 polish tail. **E15** (RBAC rawString golden tests) is RBAC.swift, not governor — left for M70 test-debt.

### M68.2 — LLM & structured guide ownership ✅ v0.10.136
- [x] C3 (High) the per-model structured-vocab build is now memoized as a `vocabBuild: Task<VocabBundle?, Error>?` (`structuredVocab()`). The pre-fix actor-reentrant check-then-act (`cachedVocabTokens == nil` → `await container.perform` → write) let two requests both run the tens-of-seconds 150k-decode build AND let a rebind landing in the `await` window clobber the cache with a stale-tokenizer result (a silently-wrong guide). Memoizing makes the nil-check + Task-assignment actor-atomic (no `await` between), so exactly one build runs and concurrent callers await the same Task; the Task captures THIS model's `container`, so a rebind mid-build still yields the correct vocab for the request that started it; a throwing build is dropped (not cached) unless a rebind already replaced it. `cachedStructuredVocabulary` factory build was already synchronous (actor-atomic) — left as-is, invalidated alongside `vocabBuild` (v0.10.136)
- [x] C10 single `dropResidentModel()` + `resetStructuredCaches()` replace the 5 copy-pasted invalidation blocks (unload / loadModel-catch / loadModel-success / rebind / setAllowedModelIds) — the M53 `cachedStructuredVocabulary` line had drifted to inconsistent indentation across them; now one method, the C3 prereq (the build Task is invalidated with the rest). Validated by an unload→reload→fresh-vocab real-model structured smoke (v0.10.136)
- [x] G5 `StructuredGuide` single-owner invariant locked in: forceful type doc ("**must never** add `Sendable`/`@unchecked Sendable`"; it wraps a mutable non-reentrant Rust parser + 32-entry rollback ring), the deliberately-NOT-`Sendable` type stays the primary compile-time guard, plus a DEBUG-only `os_unfair_lock` `trylock` backstop across `allowedMask`/`advance`/`rollback` that asserts on concurrent use (a raw-pointer escape). Release carries no sentinel → bit-identical decode path byte-unchanged (v0.10.136)
- [x] G8 rust-shim `oc_last_error` now COPY-AND-CLEARS ("take") on a buffered read, so a recorded error belongs to exactly one read (the failing call's own caller) and can't bleed into a later call on a reused pool thread that returned a failure sentinel without `set_err`, nor be misread cross-thread post-await with the wrong attribution; the length-sizing probe (`buf==null`/`len==0`) doesn't clear, so the two-call read protocol still works. New `oc_last_error_takes_and_clears_g8` (cargo 11/0); staticlib rebuilt (v0.10.136)

> AthenaLLM/AthenaStructured guided-decode is MLX-gated (and the dead-regex `StructuredGuideTests` are the pre-existing NG1 reds), so validated via Release build + e2e-rbac **561/0** + a real-model structured smoke (Qwen3.5-9B-mlx): structured request returns schema-valid JSON, **4-way concurrent structured requests each spin their own guide with no cross-request corruption (G5)**, greedy temp=0 under the guide is **byte-identical across runs** (bit-identical structured-output contract intact), and an unload→reload→structured request rebuilds the vocab and still validates (C10). rust-shim `cargo test` 11/0 (G8).

### M68.3 — Transcription & embedding modules ✅ v0.10.137
- [x] D1 (High) `diarize` now serializes its generate through a per-module FIFO `generateInFlight` Task chain — two diarize calls can no longer run concurrent forwards on the shared `SortformerModel` (whose `generate`/`generateStream` each run on a `Task.detached` that races the global `MLX.Memory.clearCache`). The detached escape inside the Sortformer model is kept (it's the off-actor compute seam — ND9/liveness), but only ONE is ever in flight at a time, which is what the "serialize" fix requires (v0.10.137)
- [x] D2 `rebind`/`unload` await `generateInFlight` before dropping the model, so a model swap never lands mid-generate (composes with the M68.1 governor teardown drain: `evict`/`unload` now waits out the in-flight diarization) (v0.10.137)
- [x] ND8 `SortformerModel.generateStream` captures its detached producer and registers `continuation.onTermination { producer.cancel() }` — the consumer's teardown (client disconnect / deadline / `break`) now actually cancels the producer (its in-loop `try Task.checkCancellation()` finally has a flag that gets set), instead of decoding the whole clip on the GPU unobserved under the governor reservation (v0.10.137)
- [x] I2 (High) `embed` split into a FIFO-chaining wrapper + `embedSerialized` worker: concurrent embeds for different models can no longer interleave their rebind+forward (pre-fix call A resumed after its load `await`, re-read `self.container` now holding call B's model, and embedded A's texts against the wrong model while reporting A's id). Serialized so each load→capture→forward is atomic; the worker also captures `liveContainer` locally instead of re-reading `self.container`. Real-model validated: 4-way concurrent alternating bge(384)/Qwen-4B(2560) embeds each returned their OWN correct dim/model (v0.10.137)
- [x] I3 the per-bucket `MLX.Memory.clearCache()` is now gated on `maxLength * bucket.count >= 16_384` (a large working set), so a common length-1 short query no longer thrashes the warm allocator pool (or races concurrent global-pool users) every call; the tens-of-GiB-drift big-batch case still clears (v0.10.137)
- [x] H4 (High) `VectorStore.ensureLoaded` memoized as a `loadTask` (concurrent first-touch callers coalesce on one `store.allVectors()` instead of double-loading + clobbering); `upsert` reserves the new-row slot synchronously via `pendingNew` BEFORE the `putVector` await, so two concurrent new upserts can't both pass a cap check that straddled the await and overrun `capBytes`. `testConcurrentUpsertsRespectCapH4` (8 concurrent → exactly 2 admitted under a 2-row cap) (v0.10.137)
- [x] H10 `upsert` rejects a zero-length vector (`VectorError.zeroLengthVector` → 400 `zero_length_vector`) before the empty-store path could adopt it as the authoritative `dim` (0) and make every later dim-check vacuous. `testZeroLengthVectorRejectedH10` (v0.10.137)

> Embedding/diarization paths are MLX-gated. Validated via Release build + e2e-rbac **561/0** + CI unit tests (`VectorStoreTests` H4/H10 — 8/0, 3 query tests gated) + a real-model embedding smoke (Qwen3.5-9B + bge-small + Qwen3-Embedding-4B): concurrent cross-model embeds each return their own correct dim (I2), embed output byte-identical across runs, a length-1 short query still returns a valid vector (I3). Diarization (D1/D2/ND8) shares the exact serialization pattern proven live for I2; no audio fixture for a live diarization smoke, validated by reasoning + the shared pattern + e2e.
> Re-verified while here: **D14** — its concurrency premise (the `Send` box's "the actor already serialises access" claim) is now actually TRUE via the D1/D2 FIFO serialization; its other half (unvalidated Whisper Hub download) is already covered by M54.3 (inference loads local-store-only, never auto-downloads) + M65.4 (load-path id confinement). No separate action.

### M68.4 — Cancellation bridge ✅ v0.10.138
- [x] A8 + E3 the streamed SSE (`/v1/chat`) and NDJSON (`/api/chat`) paths don't go through `collectMetered`, so pre-fix they bound NO `DecodeProgress.counter` and never called `cancelGeneration()` — a client disconnect or a deadline truncation closed the wire but the synchronous decode loop (which polls the counter, NOT `Task.isCancelled`) ran on to `maxTokens`. Now each streaming branch creates a `HeartbeatCounter`, binds it via `DecodeProgress.$counter.withValue` around the stream construction (so `generateMetered`'s non-detached Task — created synchronously inside the scope — inherits the TaskLocal), and flips `cancelGeneration()` on BOTH a downstream disconnect (`streamSSE`/`streamNDJSON` `onConsumerCancel` in `continuation.onTermination`, A8) AND a deadline truncation (`deadlineBounded`'s `onTimerFired`, E3). The non-streaming path's E3 already worked via M60.5's `withTaskCancellationHandler`. **Verified end-to-end:** disconnecting a `max_tokens:2000` stream after 3 s frees the model so a follow-up returns in 1.6 s (a runaway would hold `container.perform` ~100 s) (v0.10.138)
- [x] NE6 `deadlineBoundedNanos`'s timer task now returns early on `Task.isCancelled` (the consumer terminated the stream → `onTermination` → `timer.cancel()`) BEFORE `markTimerFiredIfFirst`, so the "deadline truncated the stream" callback fires only on a genuine expiry, not on a disconnect. `testDownstreamCancelDoesNotFireCallbackNE6` (v0.10.138)
- [x] E13 the streaming decode loop's task now sees the `@TaskLocal` counter — the missing binding was the streaming path (A8); the non-streaming structured path already had it (M48.1 proved `incrementToken` reaches the loop inside `container.perform`). `generateMetered` uses a non-detached `Task {}` that inherits the binding established at its synchronous creation inside the `withValue` scope (v0.10.138)
- [x] A23 `RequestQueue.cancel` is now one atomic `AthenaStore.cancelQueuedJob` (`UPDATE jobs SET status='canceled' … WHERE id=? AND status='queued'`, returns `sqlite3_changes > 0`) instead of a `getJob`-check-then-`updateJob` across two awaits — the worker can no longer pick a job up (queued→running) between the check and the write and have a running/done job clobbered back to `canceled`. `testCancelQueuedJobIsConditionalA23` (v0.10.138)

> AthenaCore (`InferenceDeadlineTests` NE6) + AthenaStore (`AthenaStoreTests` A23) CI tests pass (29/0 across the two suites). Binding gate: Release build + e2e-rbac **561/0**. A8 verified live (disconnect-frees-model follow-up 1.6 s); normal streaming stays byte-identical across runs (the counter binding doesn't affect generation — cancellation only triggers on disconnect/deadline).
> Re-verified: **A26** (`lastDecodeTokensPerSec` last-writer-wins) — confirmed a coarse "recent decode rate" indicator by design; documentation-only, → M71 polish (the streaming path adds no heartbeat writer, so no new race). **K3** (client `RemoteLogs.swift` SSE URLSession-task leak) — a client-target fix distinct from the server cancellation bridge; left for **M71** per plan (needs a cancellable task + `invalidateAndCancel` defer + a client build). **E15/E17** adjudicated in M68.1.

**M68 COMPLETE (M68.1–.4, v0.10.135–138).** No items deferred within M68's scope; A26/K3 routed to M71 as planned.

---

## M69 — Performance & operability

> **Follow-up discovered during M67.6 (not from the audits) — dangling HF-cache store symlinks.** `athena pull` lands a model as a symlink into the HF cache (`~/.cache/huggingface/hub/.../snapshots/<rev>/`). If the cache is pruned independently, the store symlink dangles: the entry still shows in `athena ls`, but every inference 500s with `module_load_failed — "missing config.json"` at REQUEST time — not caught at startup, by `doctor`, or by `ls`/`show`. **Add a doctor/startup validation** that resolves each allowlisted store entry and confirms `config.json` (+ a `*.safetensors`) exists, reporting dangling entries. (Real-dir/converted models are unaffected; only `pull`-created symlinks.)

### M69.1 — Queue/store access patterns
- [ ] A4 (High) `nextPending()` minimal-column query (drain stops loading all blobs)
- [ ] H9 `INDEX jobs(status, created)`
- [ ] A15 `depth()` = `SELECT COUNT(*)`
- [ ] NA5 /ui state uses a no-blob recent-N summary query
- [ ] A24 explicit `ORDER BY created ASC`
- [ ] A16 `/v1/models/:id` returns mtime without full-store walk

### M69.2 — Admission-control & metrics truth on streams
- [ ] NA3 concurrency slot + inflight gauge + latency timer live until the streamed body terminates (M29.2 currently doesn't bound streams; /healthz lies during decode)
- [ ] NA8 reuse resolved principal for metering (drop second auth resolution)
- [ ] A25 nearest-rank percentile

### M69.3 — Vector store
- [ ] H3 (High) cache resident normalized matrix (no per-query rebuild)
- [ ] H7 tie-break top-k on id
- [ ] H8 min-heap top-k; binary-insert; id→index map

### M69.4 — Diarization & whisper hot paths
- [ ] D3 (High) O(n²) clustering (NN-chain/PQ UPGMA)
- [ ] D7 vDSP gram-matrix fill; union-find relabel
- [ ] ND4 batch GPU→host syncs in predsToSegments/trimSilence
- [ ] ND10 O(k²) subwordSplit re-decode
- [ ] ND11 skip logprob syncs when avg_logprob unused
- [ ] ND9 move heavy sync decode off the cooperative pool (DECISION-lite: executor strategy)

### M69.5 — Decode-path costs (benchmark-gated; skip any that don't pay)
- [ ] C4 (High) `guidedArgmax` reusable buffer / O(allowed) mask / on-device
- [ ] F2 on-device eviction scoring (no per-layer host sort)
- [ ] F3 hoist `clearCache` out of per-layer eviction
- [ ] F5 chunked KV pre-alloc; C22 prefix-hit tie-break; C14 partial-select sampling; NF8 prefix gather skip

---

## M70 — Test debt (CI blindness)

### M70.1 — Make the executable target testable
- [ ] NA2 (High) + NB4: extract Server/ + Commands/ pure logic into a testable library target (or `@testable import` the executable target); then table-driven tests for `constantTimeEqual`, `resolve()` expiry, `AuthPolicy.required`, `RateLimiter.take`, `ConcurrencyLimiter`, `Session.validate`/CSRF, `MultipartForm`, `AthenaMetrics.prometheus`
- [ ] L-cross-cutting: stub-model CI tier so the MLX/inference/vector surface isn't only validated by env-gated manual xcodebuild

### M70.2 — Re-arm the env-gated load-bearing invariants
- [ ] L1 bit-identical-greedy speculative parity on a CI stub model
- [ ] L2 acceptance-rate floor observer test
- [ ] L3 VectorStore ranking on CI fixtures; L4 dim-mismatch `[]` test
- [ ] NG1 StructuredGuide tests rebuilt off the dead regex path onto real schema compilation
- [ ] NC4 PrefixKVCache bit-identity/scope-isolation/eviction tests
- [ ] NC5 cancellation early-break tests for all four decode loops
- [ ] NC6 StructuredVocab + GuidedDecoder unit tests

### M70.3 — Remaining coverage gaps
- [ ] L5–L11 (schema-mask seam, stop-filter cases, multi-seed sampling, embed-vector asserts, governor coalesce count, basename-collision, tautological asserts)
- [ ] NL1 governor load-transition/coalescing; NL4 Counter race in test harness
- [ ] NF3 TriAttention compress + gatedDeltaOps parity CI coverage
- [ ] ND14/ND15 whisper language round-trip; RMS gate; clustering branches
- [ ] NI6 bucketing pack/reassembly; NJ4 config-key parse tests; NE8 governor hooks + proxy redaction

---

## M71 — Conformance & polish tail (batch, low individual risk)

- [ ] API-surface drift: A17/A18 OpenAPI tags + cached_tokens/usage reconcile; A20 `top_logprobs:0`; NA9 queued-vs-sync param parity; NA10 transcription response_format enum; NA11 `encoding_format` honored-or-rejected; A14 403-vs-redirect
- [ ] Client robustness: K2 (High) SSE follows check HTTP status; K4/K9 optional DTO fields; K5/K11 terminal-status exit codes; K10 Run.swift status/error handling; K12; NK2 portable rotate/--ttl parity; K3 SSE teardown; K6 idempotency key on queue submit
- [ ] CLI consistency: B9 help-text default fix; B12 user arg style; B13 `--json` + footers to stderr; B16 status exit code; B18 `\r` trim; B19 prune error detail; B21/B22 exit-code & verb collisions; NB3 `default` vs installed daemon; NB10 `--revision` honored; NB12 pid 0/negative guard; NB13 status honors TLS
- [ ] Dedup/dead code: NA12, NB7, NC9, NC10, ND12, ND13, NK6, NK7, NG4, NH3, NF5, C15–C18, E11, E16, B24
- [ ] Larger refactors held to last (each its own slice if taken): A6 server god-object split; C9 `runSpeculative` extraction; E10 protocol unification; A19/A27 AuthPolicy single source

---

## Standing decisions queue (blockers for their items, not for the program)

All five decisions are now **ruled and recorded as ADRs** (003–007); see `docs/decisions/`.
Implementation lands in the milestones below.

| # | Decision | Affects | Ruling (ADR) |
|---|---|---|---|
| 1 | Non-loopback auth-on TLS posture + login-IP source | A2, K1, K8, A12, A3 | **Warn-only** — loud startup warning + doctor finding, no `--insecure` gate, client https deferred; A3 keys on TCP **peer IP** (no XFF trust), now unblocked. [ADR 004] |
| 2 | Secrets-in-argv | B2, K7, K13 | **Hard-remove `--password`** — prompt / `--password-stdin` / env only (breaking). [ADR 005] |
| 3 | Vector-store `owner` column | H5 | **Add owner-scoping** — nullable `owner`, owner-filtered query/get/delete, admin sees all, legacy NULL = admin-only (additive migration). [ADR 006] — **DONE M66.6 v0.10.128** |
| 4 | rust-shim panic strategy | G1 | **Per-entry `catch_unwind`**, crate stays `panic="unwind"` — DONE M65.1 v0.10.117. [ADR 003] |
| 5 | `/api` metering + token-budget quotas | NA8, backlog #8/#9 | **Bring #8/#9 into the program** as a new milestone; NA8 single-resolution folds in. [ADR 007] |

**Resulting plan deltas:**
- M65.5 — A2/K1 downgraded to a warn-only startup-warning + doctor check (not fail-closed); **A3 moves here** (peer-IP login limiter via a new `RemoteAddressRequestContext`).
- M66 — `--password` removal (ADR 005) + vector `owner`-scoping migration (ADR 006) become first-class slices.
- New milestone **M69.6 (or M72)** — native `/api` metering (#8) + token-budget quotas (#9) per ADR 007; NA8's cheap half still lands in M69.2.

## Tracking

- Closing a checkbox: note the version, e.g. `[x] NF1 … (v0.10.118)`.
- If a slice's re-verification shows a not-refound baseline item is already fixed or invalid, mark it here with a one-line reason rather than fixing.
