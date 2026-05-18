import Foundation

/// Client-side bearer-credential resolution for the thin CLI commands
/// that talk to a (local or remote) daemon. Single point of use:
/// precedence is `--key` > `ATHENA_KEY` env > the platform secret
/// store > none. macOS backend = the login Keychain (one entry per
/// daemon endpoint, so local + remote daemons keep distinct keys).
///
/// Portable by design: the Keychain calls are `#if os(macOS)`-gated;
/// other platforms resolve via flag/env only (a future portable
/// client target compiles unchanged). M12.3.
enum Credentials {
    private static let service = "athena"
    private static func account(_ host: String, _ port: Int) -> String {
        "\(host):\(port)"
    }

    /// Resolve the bearer key for a daemon endpoint, or nil.
    static func resolve(
        explicit: String?, host: String, port: Int
    ) -> String? {
        if let explicit, !explicit.isEmpty { return explicit }
        if let env = ProcessInfo.processInfo
            .environment["ATHENA_KEY"], !env.isEmpty
        {
            return env
        }
        return keychainRead(host: host, port: port)
    }

    #if os(macOS)
        @discardableResult
        private static func security(_ args: [String]) -> (
            Int32, String
        ) {
            let p = Process()
            p.executableURL = URL(
                fileURLWithPath: "/usr/bin/security")
            p.arguments = args
            let out = Pipe()
            p.standardOutput = out
            p.standardError = FileHandle.nullDevice
            try? p.run()
            p.waitUntilExit()
            let data = out.fileHandleForReading.readDataToEndOfFile()
            return (
                p.terminationStatus,
                String(data: data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    ?? ""
            )
        }

        static func keychainRead(host: String, port: Int) -> String? {
            let (rc, val) = security([
                "find-generic-password", "-s", service, "-a",
                account(host, port), "-w",
            ])
            return rc == 0 && !val.isEmpty ? val : nil
        }

        static func store(
            _ key: String, host: String, port: Int
        ) throws {
            // -U updates an existing item rather than erroring.
            let (rc, _) = security([
                "add-generic-password", "-U", "-s", service, "-a",
                account(host, port), "-w", key,
            ])
            guard rc == 0 else {
                throw CredentialError.keychain(
                    "security add-generic-password failed (\(rc))")
            }
        }

        @discardableResult
        static func remove(host: String, port: Int) -> Bool {
            security([
                "delete-generic-password", "-s", service, "-a",
                account(host, port),
            ]).0 == 0
        }
    #else
        static func keychainRead(host: String, port: Int) -> String? {
            nil
        }
        static func store(
            _ key: String, host: String, port: Int
        ) throws {
            throw CredentialError.unsupported
        }
        @discardableResult
        static func remove(host: String, port: Int) -> Bool { false }
    #endif
}

enum CredentialError: Error, CustomStringConvertible {
    case keychain(String)
    case unsupported
    var description: String {
        switch self {
        case .keychain(let m): return m
        case .unsupported:
            return
                "secret store unavailable on this platform — use "
                + "ATHENA_KEY or --key"
        }
    }
}
