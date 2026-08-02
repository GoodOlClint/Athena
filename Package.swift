// swift-tools-version: 6.1
import PackageDescription

// Athena — a self-hosted AI inference substrate.
// One native binary hosting LLM + transcription + embeddings under a single
// Metal/MLX memory governor. M1 wires the real MLX-backed LLM module on top
// of the local mlx-swift-lm substrate clone (a SwiftPM path dependency); the
// Rust structured-output shim is still introduced later (M3).

// Usability audit 2026-07-02 §7 — substrate pinning vs cross-repo dev.
// The path deps below build whatever branch the neighbor working trees happen
// to have checked out, and nothing records it (v0.10.251–256 silently shipped
// against `pr/pin-swift-format`, not the intended `integration`). Mirror the
// substrate's own MLX_LOCAL_DEV precedent: DEFAULT to reproducible SCM pins on
// the GoodOlClint forks (Package.resolved-recorded, deterministic releases);
// `ATHENA_LOCAL_DEV=1` swaps back to the sibling path deps for cross-repo dev
// loops (WP-D edits swift-huggingface in place). The SwiftPM package identity
// is the last URL/path component either way ("mlx-swift-lm" / "swift-huggingface"),
// so every `.product(package:)` reference below is unchanged.
// deploy/build.sh Release asserts the local-dev branches + records both HEADs.
let athenaLocalDev = Context.environment["ATHENA_LOCAL_DEV"] == "1"
let substrateDep: Package.Dependency =
    athenaLocalDev
    ? .package(path: "../mlx/mlx-swift-lm")
    : .package(
        url: "https://github.com/GoodOlClint/mlx-swift-lm.git",
        // Pinned to an immutable `integration-YYYY-MM-DD` TAG, never a bare
        // hash and never the `integration` branch. This is the substrate
        // fork's documented consumer contract (`~/Source/mlx/CLAUDE.md`
        // "Discipline"), and it exists because of THIS incident: `integration`
        // is force-pushed on every rebuild, which orphaned Athena's `751aaed`
        // pin on 2026-07-31. An unreferenced commit is fetchable only by
        // explicit SHA while SPM fetches refs, so every COLD build failed
        // while CI stayed green on cache warmth alone (#86).
        //
        // This names the SAME COMMIT main already pinned (751aaede) via the
        // tag that rescued it — an availability fix with zero semantic
        // change, nothing to re-verify. Moving FORWARD to a newer dated tag
        // is deliberately separate work; see #91.
        //
        // The tags are date-stamped precisely so they are never moved — a
        // rebuild publishes a NEW tag (`rebuild-integration.sh -p` tags the
        // outgoing head before overwriting, then tags the new one). So "don't
        // move a tag" is the fork's invariant, not a hope on our side.
        //
        // Belt-and-braces regardless: Package.resolved records the resolved
        // SHA, so even a moved tag forces 751aaede or fails loudly under
        // `swift build` / `swift test` — it never silently follows. Verified
        // by reproducing #86 in a scratch repo (force-move + GC ⇒ hard error,
        // never silent drift).
        //
        // The exception, since it would otherwise read wider than it is:
        // `swift package update` DOES re-resolve a non-SHA `revision:` to the
        // tag's new target. A bare-SHA pin could not move at all. So the
        // guarantee is "no accidental drift", not "immovable".
        //
        // Note SPM writes this as `"branch": "integration-2026-07-31"` in
        // Package.resolved — its pin state for any non-SHA `revision:`. It is
        // a tag, not a branch; the reproducibility guarantee comes from the
        // sibling `"revision"` field.
        revision: "integration-2026-07-07")  // 751aaed — the SAME commit as before, now named by its tag; TriAttention (MLXLMCommon) + Qwen3.5 MTP separate-drafter (publication S0 de-vendor)
let hubDep: Package.Dependency =
    athenaLocalDev
    ? .package(path: "../swift-huggingface")
    : .package(
        url: "https://github.com/GoodOlClint/swift-huggingface.git",
        revision: "727df097cfeceab6378b0ff6b3ce6791a50b5a05")  // athena/pr-50-download-progress (+per-file progress, audit §2)

let package = Package(
    name: "athena",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        // The unified macOS CLI (full surface). The portable
        // Linux/Windows `athena` is the same command built from the
        // AthenaClient subset via a standalone package (M14.4).
        .executable(name: "athena", targets: ["athena"]),
        // (M14.2d's `athenad` launcher was removed in M43.3 — the
        // bare argv[0] it execv'd broke MLX's metallib-bundle lookup
        // under hardened-runtime spawn; `athena start` and the
        // LaunchDaemon now invoke `athena load` directly with the
        // resolved executable path.)
        .library(name: "AthenaCore", targets: ["AthenaCore"]),
        .library(name: "AthenaClient", targets: ["AthenaClient"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/hummingbird-project/hummingbird",
            from: "2.5.0"),
        // swift-nio / swift-nio-extras / swift-http-types are already in
        // the graph transitively (Hummingbird). Declared direct (ADR 017)
        // so AthenaServerKit can import NIOCore + NIOHTTPTypes + HTTPTypes
        // for the `Expect: 100-continue` channel handler — HummingbirdCore
        // `package import`s NIOHTTPTypes, so it is NOT re-exported to us.
        // Loose lower bounds; SwiftPM resolves to Hummingbird's pins.
        .package(
            url: "https://github.com/apple/swift-nio",
            from: "2.65.0"),
        .package(
            url: "https://github.com/apple/swift-nio-extras",
            from: "1.20.0"),
        .package(
            url: "https://github.com/apple/swift-http-types",
            from: "1.0.0"),
        .package(
            url: "https://github.com/apple/swift-argument-parser",
            from: "1.4.0"),
        // swift-log is already in the graph transitively (Hummingbird/
        // NIO). Declared directly so the centralized-logging bridge
        // (M10) can import Logging + bootstrap a custom handler.
        .package(
            url: "https://github.com/apple/swift-log",
            from: "1.5.0"),
        // swift-crypto (already transitive via NIO) — declared direct
        // for the auth middleware: SHA-256, HMAC, constant-time.
        .package(
            url: "https://github.com/apple/swift-crypto",
            from: "4.0.0"),
        // swift-service-lifecycle (already transitive via Hummingbird) —
        // declared direct so the daemon can register the queue worker as
        // a managed Service and drain in-flight work on graceful shutdown
        // (M33.2).
        .package(
            url: "https://github.com/swift-server/swift-service-lifecycle",
            from: "2.11.0"),
        // ADR 043: SQLCipher engine (raw C module product; the SQLiteDB
        // Swift wrapper is unused). EXACT pin — a crypto engine must not
        // change under `swift package update`; bumps are reviewed commits.
        .package(
            url: "https://github.com/skiptools/swift-sqlcipher",
            exact: "1.11.0"),
        // The MLX substrate (reusable Qwen3.5/Gemma4/TokenIterator code).
        // SCM pin on GoodOlClint/mlx-swift-lm @ integration by default;
        // ATHENA_LOCAL_DEV=1 swaps to ../mlx/mlx-swift-lm — see `substrateDep`.
        substrateDep,
        // Direct mlx-swift dep: the inference modules need the MLX/MLXNN
        // modules (products of mlx-swift, not mlx-swift-lm).
        // Same range mlx-swift-lm pins, so SwiftPM unifies the version.
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.31.3")),
        // swift-transformers (Jinja chat templates + tokenizers) and
        // swift-huggingface (Hub client) are NOT transitive deps of
        // mlx-swift-lm — the MLXHuggingFace macros expand into code that
        // references them, so Athena must declare them directly.
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.3.0"),
        // swift-huggingface (Hub client) — fork carrying the unmerged
        // upstream fix from PR #50 (tracking issue #48): the stock async
        // `session.download(for:delegate:)` never delivered `didWriteData`,
        // so model-pull progress sat at 0% then jumped, and a stalled
        // transfer couldn't resume (restarted multi-GB shards from zero).
        // The fork's continuation-based download bridge restores real
        // per-byte progress + resume for EVERY download site. SCM pin on
        // GoodOlClint/swift-huggingface @ athena/pr-50-download-progress by
        // default; ATHENA_LOCAL_DEV=1 swaps to ../swift-huggingface — see
        // `hubDep`. Revert to the upstream `url:` dep + a version bump once
        // PR #50 merges.
        hubDep,
        // M60.3 — sudoless Apple Silicon GPU clock + die temperature via
        // IOReport/SMC (the in-process replacement for a root `powermetrics`
        // subprocess). Renamed from AppleSiliconMetrics; product `SoCMetrics`.
        .package(
            url: "https://github.com/GoodOlClint/swift-soc-metrics.git",
            from: "0.4.0"),
    ],
    targets: [
        // The memory governor + module protocol. This is the thesis
        // subsystem: every inference module shares one global budget.
        .target(
            name: "AthenaCore",
            dependencies: [
                // ADR 024 T3 — AES-256-GCM for the idle prompt-cache KV
                // cipher (`IdleKVCipher`). swift-crypto (cross-platform, the
                // same product AthenaServerKit links) over Apple CryptoKit so
                // AthenaCore stays buildable off Darwin.
                .product(name: "Crypto", package: "swift-crypto")
            ],
            path: "Sources/AthenaCore",
            // ADR 032 — the seeded MTP target↔drafter default-pairing map ships
            // as DATA (operator-overridable at <data_dir>/mtp-drafters.toml), not
            // Swift constants, so a moved/renamed HF repo is fixed without a
            // recompile (the spirit of ADR 021 D5).
            resources: [.copy("Resources/mtp-drafters.toml")]),

        // C ABI of the Rust outlines-core structured-output staticlib.
        // The module map links `athena_structured_shim`; the search path
        // (-L rust-shim/target/release) is supplied by AthenaStructured's
        // linkerSettings. Build the lib first: rust-shim/build.sh.
        .systemLibrary(
            name: "CAthenaStructured",
            path: "Sources/CAthenaStructured"),

        // Safe Swift surface over the structured-output shim (M3.2+);
        // M3.1 is the FFI bring-up only.
        .target(
            name: "AthenaStructured",
            dependencies: ["CAthenaStructured"],
            path: "Sources/AthenaStructured",
            linkerSettings: [
                .unsafeFlags([
                    "-L\(Context.packageDirectory)/rust-shim/target/release"
                ])
            ]),

        // Deploy logic, pure and testable: flat-TOML config parsing,
        // launchd plist generation, and install planning. The `athena
        // install` command is a thin imperative shell over this.
        .target(
            name: "AthenaDeploy",
            dependencies: ["AthenaCore"],
            path: "Sources/AthenaDeploy"),

        // Inference modules. M0 ships governed stubs that conform to the
        // module protocol and reserve/release real budget; the MLX-backed
        // implementations land in M1 (llm), M4 (transcription/embedding).
        .target(
            name: "AthenaLLM",
            dependencies: [
                "AthenaCore",
                "AthenaStructured",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                // M71.2 — the substrate's vision-language models (Gemma4 VLM
                // tower + Gemma4Processor). Daemon-graph only; the portable
                // `clients/` package never links it (Apple/Metal-bound).
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Logging", package: "swift-log"),
            ],
            path: "Sources/AthenaLLM"),
        .target(
            name: "AthenaTranscription",
            dependencies: [
                "AthenaCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/AthenaTranscription"),
        .target(
            name: "AthenaEmbedding",
            dependencies: [
                "AthenaCore",
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/AthenaEmbedding"),

        // M70.1 (audit NA2) — the daemon's HTTP server PRIMITIVES, split
        // out of the `athena` executable target so they are unit-testable
        // under `swift test` (the executable target is unreachable by the
        // test bundle, and importing it would drag the whole MLX/Metal
        // graph into CI). MLX-FREE by construction: only the pure,
        // security-critical seams live here — bearer/RBAC auth
        // (constant-time compare, token resolve/expiry, the route→permission
        // map), the rate-limit + concurrency token buckets, the WebUI
        // session/CSRF HMAC, the multipart reader, the Prometheus metrics +
        // percentile math, the request context, and the logging bootstrap.
        // The `AthenaServer` god-object and every MLX-linked handler stay in
        // the executable; this target is its testable substrate. See
        // docs/decisions/008-testable-server-seam.md.
        .target(
            name: "AthenaServerKit",
            dependencies: [
                "AthenaCore",
                "AthenaStore",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "Crypto", package: "swift-crypto"),
                .product(name: "Logging", package: "swift-log"),
                // ADR 017 — the `Expect: 100-continue` channel handler.
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOHTTPTypes", package: "swift-nio-extras"),
                .product(name: "HTTPTypes", package: "swift-http-types"),
            ],
            path: "Sources/AthenaServerKit"),

        // M7: one embedded SQLite store (auth/audit/usage since ADR 025).
        // Engine = SQLCipher via skiptools/swift-sqlcipher (ADR 043;
        // raw C module, LibTomCrypt provider, SQLITE_TEMP_STORE=2 is the
        // package default) so the store can be encrypted at rest (M34.3);
        // MLX for the governed cosine working set.
        .target(
            name: "AthenaStore",
            dependencies: [
                "AthenaCore",
                .product(name: "SQLCipher", package: "swift-sqlcipher"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/AthenaStore",
            // sqlite3_key/_v2 are behind `#ifdef SQLITE_HAS_CODEC` in the
            // package's sqlite3.h; the define must reach the Clang importer
            // of THIS target (same idiom as the package's own SQLiteDB).
            cSettings: [.define("SQLITE_HAS_CODEC")]),

        // Portable client surface (M14.1): Keychain `Secrets`,
        // `Credentials`/`HFAuth`/`ProxyAuth`, `DaemonOptions`, the
        // `HTTPClient`. Foundation + ArgumentParser only; the
        // AthenaCore dep (for `GovernorConfig.defaultPort`) is
        // Foundation + ArgumentParser ONLY — no AthenaCore/Darwin/MLX
        // graph, so the portable `athena` client is Linux-clean
        // (M14.3). Platform bits are `#if canImport`-guarded. Sources
        // live under `clients/` and are ALSO built by the standalone
        // cross-platform `clients/Package.swift` (M14.4) — one copy.
        .target(
            name: "AthenaClient",
            dependencies: [
                .product(
                    name: "ArgumentParser",
                    package: "swift-argument-parser")
            ],
            path: "clients/Sources/AthenaClient"),

        // The unified macOS `athena` CLI: full surface — local daemon
        // lifecycle + Apple-host operator ops + the HTTP client verbs
        // (reused from AthenaClient). The governed HTTP surface +
        // MLX/Metal modules (macOS-only). The portable Linux/Windows
        // `athena` is the AthenaClient subset, packaged separately
        // (M14.4). The `athenad` daemon binary lands in M14.2d.
        .executableTarget(
            name: "athena",
            dependencies: [
                "AthenaCore",
                "AthenaClient",
                "AthenaDeploy",
                "AthenaServerKit",
                "AthenaLLM",
                "AthenaStructured",
                "AthenaTranscription",
                "AthenaEmbedding",
                "AthenaStore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hummingbird", package: "hummingbird"),
                // In-daemon TLS (M28). Pulls swift-nio-ssl, but ONLY into
                // the macOS `athena`/daemon graph — the portable
                // AthenaClient target stays Foundation+ArgumentParser, so
                // the cross-platform client package is unaffected.
                .product(name: "HummingbirdTLS", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Crypto", package: "swift-crypto"),
                .product(
                    name: "ServiceLifecycle",
                    package: "swift-service-lifecycle"),
                .product(
                    name: "SoCMetrics",
                    package: "swift-soc-metrics"),
            ],
            path: "Sources/athena",
            linkerSettings: [
                .unsafeFlags([
                    "-L\(Context.packageDirectory)/rust-shim/target/release"
                ])
            ]),

        .testTarget(
            name: "AthenaCoreTests",
            dependencies: [
                "AthenaCore", "AthenaClient", "AthenaDeploy",
                "AthenaServerKit",
                "AthenaLLM", "AthenaStructured",
                "AthenaTranscription", "AthenaEmbedding",
                "AthenaStore",
                // ADR 017 — NIOEmbeddedChannel pins ExpectContinueHandler.
                .product(name: "NIOEmbedded", package: "swift-nio"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOHTTPTypes", package: "swift-nio-extras"),
                .product(name: "HTTPTypes", package: "swift-http-types"),
            ],
            path: "Tests/AthenaCoreTests",
            exclude: ["Fixtures"],
            linkerSettings: [
                .unsafeFlags([
                    "-L\(Context.packageDirectory)/rust-shim/target/release"
                ])
            ]),
    ]
)
