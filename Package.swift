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

        // Deploy logic, pure and testable: flat-TOML config parsing,
        // launchd plist generation, and install planning. The `athena
        // install` command is a thin imperative shell over this.
        .target(
            name: "AthenaDeploy",
            path: "Sources/AthenaDeploy"),

        // Inference modules. M0 ships governed stubs that conform to the
        // module protocol and reserve/release real budget; the MLX-backed
        // implementations land in M1 (llm), M4 (transcription/embedding).
        .target(
            name: "AthenaLLM",
            dependencies: [
                "AthenaCore",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/AthenaLLM"),
        .target(
            name: "AthenaTranscription",
            dependencies: ["AthenaCore"],
            path: "Sources/AthenaTranscription"),
        .target(
            name: "AthenaEmbedding",
            dependencies: ["AthenaCore"],
            path: "Sources/AthenaEmbedding"),

        // The `athena` executable: CLI (serve/run/pull/list/ps/...) and the
        // governed HTTP surface.
        .executableTarget(
            name: "athena",
            dependencies: [
                "AthenaCore",
                "AthenaDeploy",
                "AthenaLLM",
                "AthenaTranscription",
                "AthenaEmbedding",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/athena"),

        .testTarget(
            name: "AthenaCoreTests",
            dependencies: [
                "AthenaCore", "AthenaDeploy", "AthenaLLM",
                "AthenaTranscription", "AthenaEmbedding",
            ],
            path: "Tests/AthenaCoreTests"),
    ]
)
