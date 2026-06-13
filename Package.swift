// swift-tools-version: 6.1
import PackageDescription

// Athena — the AI inference substrate of Project the platform.
// One native binary hosting LLM + transcription + embeddings under a single
// Metal/MLX memory governor. M1 wires the real MLX-backed LLM module on top
// of the local mlx-swift-lm substrate clone (a SwiftPM path dependency); the
// Rust structured-output shim is still introduced later (M3).

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
        // The MLX substrate. Local clone per the the platform environment
        // (~/Source/mlx-swift-lm); a path dependency keeps M1 buildable
        // against the exact reusable Qwen3.5/TokenIterator code.
        .package(path: "../mlx-swift-lm"),
        // Direct mlx-swift dep: the vendored AthenaModels Qwen3.5 needs the
        // MLX/MLXNN modules (products of mlx-swift, not mlx-swift-lm).
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
        // swift-huggingface (Hub client) — TEMPORARY local fork (sibling
        // clone, like mlx-swift-lm above). Pinned at 0.9.0 (b721959) plus
        // the unmerged upstream fix from PR #50 (tracking issue #48): the
        // stock async `session.download(for:delegate:)` never delivered
        // `didWriteData`, so model-pull progress sat at 0% then jumped,
        // and a stalled transfer (e.g. the box napping) couldn't resume —
        // it restarted multi-GB shards from zero. The fork's
        // continuation-based download bridge restores real per-byte
        // progress + resume for EVERY download site (pull/init/convert and
        // the on-demand whisper/embedding/diarization fetches). Revert to
        // the upstream `url:` dep + a version bump once PR #50 merges.
        .package(path: "../swift-huggingface"),
        // M60.3 — sudoless Apple Silicon GPU clock via IOReport (the
        // in-process replacement for a root `powermetrics` subprocess).
        .package(
            url: "https://github.com/GoodOlClint/AppleSiliconMetrics.git",
            from: "0.1.0"),
    ],
    targets: [
        // The memory governor + module protocol. This is the thesis
        // subsystem: every inference module shares one global budget.
        .target(
            name: "AthenaCore",
            path: "Sources/AthenaCore"),

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

        // Athena-owned Qwen3.5 model, vendored from the pristine
        // mlx-swift-lm clone (the 3 internal helpers it needs are not
        // importable). Registered into the substrate's public model-type
        // registry; M2 grows the MTP head here. The substrate stays an
        // unmodified upstream path dependency.
        .target(
            name: "AthenaModels",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/AthenaModels",
            exclude: ["DFlash/NOTICE"]),

        // Inference modules. M0 ships governed stubs that conform to the
        // module protocol and reserve/release real budget; the MLX-backed
        // implementations land in M1 (llm), M4 (transcription/embedding).
        .target(
            name: "AthenaLLM",
            dependencies: [
                "AthenaCore",
                "AthenaModels",
                "AthenaStructured",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
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

        // Vendored SQLCipher amalgamation (v4.6.1): API-compatible SQLite
        // with transparent AES-256 page encryption (sqlite3_key /
        // PRAGMA key) on the CommonCrypto backend — no OpenSSL, single
        // self-contained binary on Apple crypto. Inert without a key
        // (standard SQLite on-disk format), so it vends in safe and the
        // at-rest encryption is opt-in (M34.3 `encrypt_store`).
        // SQLITE_TEMP_STORE=2 keeps sorter/temp spill in memory so an
        // encrypted store never leaks plaintext to a temp file.
        .target(
            name: "CSQLCipher",
            path: "Sources/CSQLCipher",
            exclude: ["README.md"],
            cSettings: [
                // Enable the SQLCipher codec on the Apple CommonCrypto
                // backend (no OpenSSL).
                .define("SQLITE_HAS_CODEC"),
                .define("SQLCIPHER_CRYPTO_CC"),
                // The amalgamation's internal asserts reference
                // SQLITE_DEBUG-only helpers; NDEBUG (the production
                // setting) compiles them out. SwiftPM C targets don't
                // define it even under -Os, so set it explicitly.
                .define("NDEBUG"),
                // Keep sorter/temp spill in memory so an encrypted store
                // never leaks plaintext to a temp file on disk.
                .define("SQLITE_TEMP_STORE", to: "2"),
                .define("SQLITE_THREADSAFE", to: "1"),
                // usleep() is always present on macOS — pick the precise
                // busy-sleep without pulling in the autoconf-generated
                // sqlite_cfg.h (which is keyed to the BUILD host's OS and
                // would force newer-than-deployment APIs like strchrnul).
                .define("HAVE_USLEEP", to: "1"),
            ],
            linkerSettings: [
                .linkedFramework("Security"),
                .linkedFramework("Foundation"),
            ]),

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
            ],
            path: "Sources/AthenaServerKit"),

        // M7: one embedded SQLite store backing the built-in vector DB
        // and the async request queue. Engine = vendored SQLCipher
        // (CSQLCipher) so the store can be encrypted at rest (M34.3);
        // MLX for the governed cosine working set.
        .target(
            name: "AthenaStore",
            dependencies: [
                "AthenaCore",
                "CSQLCipher",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/AthenaStore"),

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
                    package: "swift-argument-parser"),
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
                    name: "AppleSiliconMetrics",
                    package: "AppleSiliconMetrics"),
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
                "AthenaLLM", "AthenaModels", "AthenaStructured",
                "AthenaTranscription", "AthenaEmbedding",
                "AthenaStore",
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
