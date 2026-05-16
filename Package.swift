// swift-tools-version: 6.1
import PackageDescription

// Athena — the AI inference substrate of Project the platform.
// One native binary hosting LLM + transcription + embeddings under a single
// Metal/MLX memory governor. M0 scaffold: governor + module protocol + the
// `athena serve` HTTP surface with a governed stub module. Heavy substrates
// (mlx-swift-lm, swift-transformers, the Rust structured-output shim) are
// introduced from M1 onward so each milestone stays independently shippable.

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
    ],
    targets: [
        // The memory governor + module protocol. This is the thesis
        // subsystem: every inference module shares one global budget.
        .target(
            name: "AthenaCore",
            path: "Sources/AthenaCore"),

        // Inference modules. M0 ships governed stubs that conform to the
        // module protocol and reserve/release real budget; the MLX-backed
        // implementations land in M1 (llm), M4 (transcription/embedding).
        .target(
            name: "AthenaLLM",
            dependencies: ["AthenaCore"],
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
                "AthenaCore", "AthenaLLM", "AthenaTranscription",
                "AthenaEmbedding",
            ],
            path: "Tests/AthenaCoreTests"),
    ]
)
