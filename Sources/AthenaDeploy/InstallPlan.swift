import Foundation

/// Pure description of where the install lands, given the build output
/// directory and an install prefix. No filesystem mutation here — the
/// `athena install` command executes this plan; tests assert on it.
///
/// The binary is statically linked but its MLX/crypto/NIO resource bundles
/// (incl. `mlx-swift_Cmlx.bundle/.../default.metallib`) MUST sit next to it,
/// so everything installs together under `<prefix>/libexec/athena` with a
/// thin `<prefix>/bin/athena` symlink for PATH.
public struct InstallPlan: Sendable, Equatable {
    public let sourceDir: URL
    public let prefix: URL
    public let label: String

    public init(sourceDir: URL, prefix: URL, label: String) {
        self.sourceDir = sourceDir
        self.prefix = prefix
        self.label = label
    }

    public var libexecDir: URL {
        prefix.appendingPathComponent("libexec/athena", isDirectory: true)
    }
    public var installedBinary: URL {
        libexecDir.appendingPathComponent("athena")
    }
    /// The daemon launcher installed beside `athena` (launchd points
    /// here; it execs the sibling `athena load`). M14.2d.
    public var installedDaemon: URL {
        libexecDir.appendingPathComponent("athenad")
    }
    public var binSymlink: URL {
        prefix.appendingPathComponent("bin/athena")
    }
    public var configDir: URL {
        prefix.appendingPathComponent("etc/athena", isDirectory: true)
    }
    public var installedConfig: URL {
        configDir.appendingPathComponent("athena.toml")
    }
    public var workingDir: URL {
        prefix.appendingPathComponent("var/athena", isDirectory: true)
    }
    public var plistPath: URL {
        URL(fileURLWithPath: "/Library/LaunchDaemons/\(label).plist")
    }
    /// Guard: present only in an xcodebuild-produced binary. Absent in a
    /// `swift build` binary, which cannot run MLX.
    public var sourceMetallib: URL {
        sourceDir.appendingPathComponent(
            "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
    }

    /// Names to copy from `sourceDir`: the `athena` binary, the
    /// `athenad` daemon launcher, plus every `*.bundle` resource
    /// directory beside them.
    public func artifactNames(fileManager: FileManager = .default)
        -> [String]
    {
        let bundles =
            (try? fileManager.contentsOfDirectory(
                atPath: sourceDir.path)) ?? []
        return ["athena", "athenad"]
            + bundles.filter { $0.hasSuffix(".bundle") }.sorted()
    }
}
