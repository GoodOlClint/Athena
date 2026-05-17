import ArgumentParser
import AthenaLLM
import Foundation

/// `athena prune` — delete dangling/broken store entries (a dead
/// `pull` symlink whose HF-cache/SSD target vanished, a half-converted
/// dir, a model missing config/safetensors). `--dry-run` previews.
/// Only entries that clearly look like models are ever touched. M9.5c.
struct Prune: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "prune",
        abstract: "Remove broken/dangling models from the store.")

    @Flag(help: "Show what would be removed; change nothing.")
    var dryRun = false

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    /// Only prune things that are plausibly a model: a symlink (how
    /// `pull` lands), or a directory holding model-shaped files. A
    /// random unrelated dir is left strictly alone.
    private func looksLikeModel(_ url: URL, isSymlink: Bool) -> Bool {
        if isSymlink { return true }
        let fm = FileManager.default
        if fm.fileExists(
            atPath: url.appendingPathComponent("config.json").path)
        {
            return true
        }
        let kids =
            (try? fm.contentsOfDirectory(atPath: url.path)) ?? []
        return kids.contains {
            $0.hasSuffix(".safetensors")
                || $0.hasPrefix("tokenizer")
        }
    }

    func run() async throws {
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isSymbolicLinkKey])
        else {
            FailableExit.die("error: no store at \(root.path)")
        }

        var victims: [(name: String, problems: [String])] = []
        for e in entries.sorted(by: { $0.path < $1.path }) {
            let isSymlink =
                (try? e.resourceValues(forKeys: [.isSymbolicLinkKey])
                    .isSymbolicLink) ?? false
            guard looksLikeModel(e, isSymlink: isSymlink) else {
                continue
            }
            let problems = ModelHealth.check(e)
            if !problems.isEmpty {
                victims.append((e.lastPathComponent, problems))
            }
        }

        if victims.isEmpty {
            print("nothing to prune in \(root.path)")
            return
        }
        for v in victims {
            print(
                "\(dryRun ? "would remove" : "removing") \(v.name)")
            for p in v.problems { print("    - \(p)") }
        }
        if dryRun {
            print("(dry-run) \(victims.count) entr"
                + (victims.count == 1 ? "y" : "ies") + " would go")
            return
        }
        var removed = 0
        for v in victims {
            do {
                try fm.removeItem(
                    at: root.appendingPathComponent(v.name))
                removed += 1
            } catch {
                FileHandle.standardError.write(
                    Data("error: \(v.name): \(error)\n".utf8))
            }
        }
        print("pruned \(removed)/\(victims.count)")
        if removed != victims.count { throw ExitCode.failure }
    }
}
