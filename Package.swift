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
        .executable(name: "athena", targets: ["athena"]),
        .library(name: "AthenaCore", targets: ["AthenaCore"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/hummingbird-project/hummingbird",
            from: "2.5.0"),
        .package(
            url: "https://github.com/apple/swift-argument-parser",
            from: "1.4.0"),
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
        .package(
            url: "https://github.com/huggingface/swift-huggingface",
            from: "0.9.0"),
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
            path: "Sources/AthenaModels"),

        // Inference modules. M0 ships governed stubs that conform to the
        // module protocol and reserve/release real budget; the MLX-backed
        // implementations land in M1 (llm), M4 (transcription/embedding).
        .target(
            name: "AthenaLLM",
            dependencies: [
                "AthenaCore",
                "AthenaModels",
                "AthenaStructured",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/AthenaLLM"),
        .target(
            name: "AthenaTranscription",
            dependencies: [
                "AthenaCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
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

        // The `athena` executable: CLI (serve/run/pull/list/ps/...) and the
        // governed HTTP surface.
        .executableTarget(
            name: "athena",
            dependencies: [
                "AthenaCore",
                "AthenaDeploy",
                "AthenaLLM",
                "AthenaStructured",
                "AthenaTranscription",
                "AthenaEmbedding",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
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
                "AthenaCore", "AthenaDeploy", "AthenaLLM", "AthenaModels",
                "AthenaStructured", "AthenaTranscription", "AthenaEmbedding",
            ],
            path: "Tests/AthenaCoreTests",
            linkerSettings: [
                .unsafeFlags([
                    "-L\(Context.packageDirectory)/rust-shim/target/release"
                ])
            ]),
    ]
)
