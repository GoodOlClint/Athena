// swift-tools-version: 6.1
import PackageDescription

// Standalone, cross-platform package for the portable `athena`
// client (M14.4). It vends ONLY the Foundation + ArgumentParser
// graph — none of the monorepo's Apple-only MLX/mlx-swift-lm path
// dependencies — so `swift build` works on Linux/Windows. The
// AthenaClient sources live under this package's directory and are
// ALSO compiled by the monorepo (its Package points its AthenaClient
// target at `clients/Sources/AthenaClient`), so there is exactly one
// copy of the client code.
let package = Package(
    name: "athena-client",
    products: [
        .executable(name: "athena", targets: ["athena"]),
        .library(name: "AthenaClient", targets: ["AthenaClient"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-argument-parser",
            from: "1.4.0"),
    ],
    targets: [
        .target(
            name: "AthenaClient",
            dependencies: [
                .product(
                    name: "ArgumentParser",
                    package: "swift-argument-parser"),
            ],
            path: "Sources/AthenaClient"),
        .executableTarget(
            name: "athena",
            dependencies: [
                "AthenaClient",
                .product(
                    name: "ArgumentParser",
                    package: "swift-argument-parser"),
            ],
            path: "Sources/athena"),
    ])
