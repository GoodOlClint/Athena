import ArgumentParser
import AthenaCore
import AthenaDeploy
import AthenaLLM
import Foundation

/// `athena doctor` — read-only environment preflight. Surfaces the
/// failure modes this appliance keeps hitting: SSD model store not
/// mounted, a `swift build` binary with no Metal lib (MLX aborts at
/// first inference), unparseable config, unresolvable default model,
/// unwritable data/log dirs. Exit non-zero if any check FAILs. M9.6a.
struct Doctor: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "doctor",
        abstract: "Diagnose the runtime environment (read-only).")

    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    private enum Level: String { case ok = "ok  ", warn = "warn", fail = "FAIL" }

    func run() async throws {
        var fails = 0
        func say(_ l: Level, _ msg: String) {
            if l == .fail { fails += 1 }
            print("\(l.rawValue)  \(msg)")
        }
        let fm = FileManager.default

        // 1. Metal library next to THIS binary — absent ⇒ a `swift
        //    build` binary that aborts at first MLX inference.
        let binDir = Bundle.main.executableURL!
            .resolvingSymlinksInPath().deletingLastPathComponent()
        let metallib = InstallPlan(
            sourceDir: binDir,
            prefix: URL(fileURLWithPath: "/usr/local"),
            label: "x"
        ).sourceMetallib
        if fm.fileExists(atPath: metallib.path) {
            say(.ok, "Metal library present (MLX can run)")
        } else {
            say(
                .fail,
                "no Metal library at \(metallib.path) — this binary "
                    + "cannot run MLX (build via deploy/build.sh)")
        }

        // 2. xcodebuild — only needed to build/convert-from-source,
        //    not to run an installed binary.
        if fm.isExecutableFile(atPath: "/usr/bin/xcodebuild") {
            say(.ok, "xcodebuild present")
        } else {
            say(.warn, "xcodebuild not found (needed only to rebuild)")
        }

        // 3. Model store root — the recurring SSD-detach failure.
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: root.path, isDirectory: &isDir),
            isDir.boolValue
        {
            say(.ok, "model store present: \(root.path)")
        } else {
            say(
                .fail,
                "model store missing: \(root.path) — pull/convert a "
                    + "model, set model_store, or mount the volume "
                    + "if it lives on one")
        }

        // 4. Config resolve + parse.
        let cfgURL = ConfigEditor.resolvePath(config)
        var parsed: AthenaConfig?
        if !fm.fileExists(atPath: cfgURL.path) {
            say(.warn, "no config at \(cfgURL.path) (using defaults)")
        } else {
            do {
                parsed = try AthenaConfig.parse(file: cfgURL)
                say(.ok, "config parses: \(cfgURL.path)")
            } catch {
                say(.fail, "config unparseable (\(cfgURL.path)): \(error)")
            }
        }

        // 5. Default model resolvable + healthy.
        let store = ModelStore(rootDirectory: root)
        let modelURL = store.resolve(parsed?.model)
        if !fm.fileExists(atPath: modelURL.path) {
            say(
                .warn,
                "default model not present: \(modelURL.path) "
                    + "(pull/convert before serving)")
        } else {
            let problems = ModelHealth.check(modelURL)
            if problems.isEmpty {
                say(.ok, "default model ok: \(modelURL.lastPathComponent)")
            } else {
                say(
                    .fail,
                    "default model unhealthy "
                        + "(\(modelURL.lastPathComponent)): "
                        + problems.joined(separator: "; "))
            }
        }

        // 6. Data dir writable (store + queue live here).
        let dataDir =
            parsed?.dataDir.map {
                URL(
                    fileURLWithPath:
                        ($0 as NSString).expandingTildeInPath,
                    isDirectory: true)
            }
            ?? AthenaEnv.userHome()
                .appendingPathComponent(".athena", isDirectory: true)
        if writable(dataDir) {
            say(.ok, "data dir writable: \(dataDir.path)")
        } else {
            say(.fail, "data dir not writable: \(dataDir.path)")
        }

        // 7. Log dir (config-driven; advisory).
        if let log = parsed?.logDir {
            if fm.fileExists(atPath: log) {
                say(.ok, "log dir present: \(log)")
            } else {
                say(.warn, "log dir missing: \(log) (created on install)")
            }
        }

        // 8. Daemon reachability (informational).
        let host = parsed?.listenHost ?? "127.0.0.1"
        let port = parsed?.listenPort ?? GovernorConfig.defaultPort
        var req = URLRequest(
            url: URL(string: "http://\(host):\(port)/healthz")!)
        req.timeoutInterval = 2
        if (try? await URLSession.shared.data(for: req)) != nil {
            say(.ok, "daemon responding at \(host):\(port)")
        } else {
            say(.warn, "daemon not responding at \(host):\(port)")
        }

        if fails > 0 {
            print("\n\(fails) critical issue(s).")
            throw ExitCode.failure
        }
        print("\nall critical checks passed.")
    }

    /// Writable = exists & we can create a probe file, or the parent
    /// exists & is writable (dir will be created on first use).
    private func writable(_ dir: URL) -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: dir.path) {
            let probe = dir.appendingPathComponent(".athena-doctor")
            if (try? Data().write(to: probe)) != nil {
                try? fm.removeItem(at: probe)
                return true
            }
            return false
        }
        return fm.isWritableFile(
            atPath: dir.deletingLastPathComponent().path)
    }
}
