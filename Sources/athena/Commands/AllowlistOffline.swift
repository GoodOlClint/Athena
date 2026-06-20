import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaStore
import Foundation

// #11 / M43.5 — offline `--data-dir` allowlist editing.
//
// The portable `AllowlistCmd` (AthenaClient) is HTTP-only: every verb
// routes to the running daemon's `/api/models/allow`. That leaves an
// operator with a wedged/stopped daemon unable to fix the allowlist —
// the very situation where a bad default needs correcting before the
// daemon will come back up cleanly.
//
// This macOS-native group overloads each verb: when `--data-dir` is
// given (offline intent — a local SQLite path), it opens `AthenaStore`
// and mutates the `model_allowlist` table directly, mirroring exactly
// what the server handlers do (same `ModuleID` validation, same store
// methods). Without `--data-dir` it delegates to `RemoteAllowlist`, so
// the existing local-loopback / off-box HTTP behaviour is byte-for-byte
// unchanged. The daemon re-resolves the table on next boot (M42.1
// seed), so an offline edit takes effect on restart — no live refresh
// needed because the daemon is, by construction, down.
//
// Named `AllowlistCommand` to avoid colliding with the portable
// `AllowlistCmd`; the macOS executable wires THIS one.

private func validModulesHelp() -> String {
    ModuleID.allCases.map(\.rawValue).joined(separator: ", ")
}

private func openOfflineStore(_ dataDir: String) throws -> AthenaStore {
    try AthenaStore(
        path: storeDBPath(dataDir), key: StoreKey.resolve())
}

private func requireModule(_ raw: String) throws -> ModuleID {
    guard let m = ModuleID(rawValue: raw) else {
        FileHandle.standardError.write(
            Data(
                ("error: unknown module '\(raw)' — expected one of: "
                    + "\(validModulesHelp())\n").utf8))
        throw ExitCode.failure
    }
    return m
}

public struct AllowlistCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "allowlist",
        abstract:
            "Manage the persistent per-module model allowlist.",
        discussion:
            "Talks to the running daemon over HTTP by default. Pass "
            + "--data-dir to edit the on-disk allowlist directly while "
            + "the daemon is stopped.",
        subcommands: [
            OfflineAllowList.self, OfflineAllowAdd.self,
            OfflineAllowRm.self, OfflineAllowDefault.self,
        ])
    public init() {}
}

public struct OfflineAllowList: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List allowlist entries (optionally one module).")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Narrow to one module (llm, textEmbedding, …).")
    public var module: String?
    @Option(help: "Edit the on-disk store directly while the daemon is down (offline).")
    public var dataDir: String?
    public init() {}
    public func run() async throws {
        guard let dd = dataDir else {
            try await RemoteAllowlist.list(daemon, module: module)
            return
        }
        if let m = module { _ = try requireModule(m) }
        let store = try openOfflineStore(dd)
        let rows = await store.listModelAllowlist(module: module)
        RemoteAllowlist.renderList(
            rows.map { ($0.module, $0.id, $0.isDefault) })
    }
}

public struct OfflineAllowAdd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Add a model id to a module's allowlist.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Module: llm, textEmbedding, transcription, diarization, speakerEmbedding.")
    public var module: String
    @Option(help: "Model id (HF id for aux modules, store name for llm).")
    public var id: String
    @Flag(
        name: .customLong("default"),
        help: "Mark this row as the module's default.")
    public var asDefault: Bool = false
    @Option(help: "Edit the on-disk store directly while the daemon is down (offline).")
    public var dataDir: String?
    public init() {}
    public func run() async throws {
        guard let dd = dataDir else {
            try await RemoteAllowlist.add(
                daemon, module: module, id: id, asDefault: asDefault)
            return
        }
        let m = try requireModule(module)
        guard !id.isEmpty else {
            FileHandle.standardError.write(
                Data("error: id must be non-empty\n".utf8))
            throw ExitCode.failure
        }
        let store = try openOfflineStore(dd)
        try await store.addModelAllowlist(
            module: m.rawValue, id: id, isDefault: asDefault)
        print("added \(m.rawValue):\(id)")
    }
}

public struct OfflineAllowRm: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove a model id from a module's allowlist.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Module class.")
    public var module: String
    @Option(help: "Model id to remove.")
    public var id: String
    @Option(help: "Edit the on-disk store directly while the daemon is down (offline).")
    public var dataDir: String?
    public init() {}
    public func run() async throws {
        guard let dd = dataDir else {
            try await RemoteAllowlist.remove(
                daemon, module: module, id: id)
            return
        }
        let m = try requireModule(module)
        let store = try openOfflineStore(dd)
        let removed = await store.removeModelAllowlist(
            module: m.rawValue, id: id)
        if removed {
            print("removed \(m.rawValue):\(id)")
        } else {
            FileHandle.standardError.write(
                Data(
                    ("error: \(m.rawValue):\(id) not in allowlist\n")
                        .utf8))
            throw ExitCode.failure
        }
    }
}

public struct OfflineAllowDefault: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "default",
        abstract:
            "Mark an existing allowlist row as the module's default.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Module class.")
    public var module: String
    @Option(help: "Model id to mark as default.")
    public var id: String
    @Option(help: "Edit the on-disk store directly while the daemon is down (offline).")
    public var dataDir: String?
    public init() {}
    public func run() async throws {
        guard let dd = dataDir else {
            try await RemoteAllowlist.setDefault(
                daemon, module: module, id: id)
            return
        }
        let m = try requireModule(module)
        let store = try openOfflineStore(dd)
        do {
            try await store.setModelAllowlistDefault(
                module: m.rawValue, id: id)
        } catch {
            FileHandle.standardError.write(
                Data("error: \(error)\n".utf8))
            throw ExitCode.failure
        }
        print("default \(m.rawValue):\(id)")
    }
}
