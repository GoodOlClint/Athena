import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaLLM
import Foundation

/// `athena pull MODEL` — download an HF repo into the model store
/// (linked, no multi-GB copy), ollama-style.
struct Pull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a model (HF id) into the local store."
    )

    @Argument(help: "HF repo id, e.g. mlx-community/whisper-large-v3-turbo")
    var model: String

    @Option(help: "Git revision / branch / tag (remote only).")
    var revision: String?

    @Flag(help: "Stream job progress (remote only).")
    var follow = false

    @Option(help: "Long-poll N seconds for completion (remote only).")
    var wait: Int?

    @Flag(
        help: ArgumentHelp(
            "Dry run: fetch only the model's config and report whether Athena "
            + "can load it (modality + loadability). Downloads no weights, "
            + "touches no daemon. Exits non-zero if unsupported."))
    var check = false

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        // ADR 021 S4 — `--check` is a host-independent dry run: it classifies
        // from a config-only HF metadata fetch (the sanctioned passive-oracle
        // egress) and downloads nothing, so it behaves identically in local and
        // remote mode and never contacts the daemon.
        if check {
            ProxyEnv.applyConfigAndAuth()  // egress proxy (M13.2)
            HFAuth.exportToEnv()  // gated/private repos (M13)
            try await runCheck()
            return
        }
        if daemon.isRemote {
            var b: [String: Any] = ["id": model]
            if let revision { b["revision"] = revision }
            try await RemoteModels.job(
                daemon, op: "pull", body: b, follow: follow,
                wait: wait)
            return
        }
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        ProxyEnv.applyConfigAndAuth()  // egress proxy (M13.2)
        HFAuth.exportToEnv()  // gated/private repos (M13)
        // ADR 021 S4 — config-only preflight gate BEFORE the multi-GB pull:
        // refuse-early on a known-unsupported packaging (no wasted download),
        // warn-and-proceed on `.unknown` (best-effort generative/vision), and
        // proceed silently on `.loadable`. The gate is best-effort: a preflight
        // fetch error (offline/404/gated) is NOT fatal here — the normal pull
        // below has its own retry + error reporting and will surface it.
        if let support = try? await ModelPreflight.check(
            id: model, revision: revision)
        {
            switch ModelPreflight.gate(support, id: model) {
            case .proceed:
                break
            case .warn(let msg):
                FileHandle.standardError.write(Data(("warning: " + msg + "\n").utf8))
            case .refuse(let reason):
                print("error: Athena cannot load '\(model)' as packaged.")
                print("  \(reason)")
                print(
                    "  Run `athena pull \(model) --check` for the full verdict. "
                        + "Nothing was downloaded.")
                throw ExitCode.failure
            }
        }
        print("pulling \(model) …")
        let bar = ProgressBar("  \(model)")
        do {
            let dest = try await ModelPull.pull(
                id: model, into: root,
                progress: { f in bar.update(f) },
                onRetry: { attempt, maxAttempts, err in
                    bar.finish()
                    FileHandle.standardError.write(
                        Data(
                            ("  \(ModelPull.friendlyError(err)) — "
                                + "retrying (\(attempt)/\(maxAttempts - 1))…\n")
                                .utf8))
                })
            bar.finish()
            print("pulled \(model) → \(dest.path)")
        } catch {
            bar.finish()
            print("error: pull failed — \(ModelPull.friendlyError(error))")
            print(
                "  re-run `athena pull \(model)` to resume "
                    + "(already-downloaded files are kept).")
            throw ExitCode.failure
        }
    }

    /// `athena pull <id> --check` — config-only dry run. Fetches the model's
    /// metadata, classifies it with the shared `ModelSupport` predicate, prints
    /// the verdict, and downloads no weights. Exits non-zero when the packaging
    /// is unsupported so a script can gate on it.
    private func runCheck() async throws {
        let support: ModelSupport
        do {
            support = try await ModelPreflight.check(
                id: model, revision: revision)
        } catch {
            // Couldn't even fetch the config (offline / 404 / gated repo) — that
            // IS the answer for a dry run.
            print("error: could not preflight '\(model)' — "
                + "\(ModelPull.friendlyError(error))")
            print(
                "  The repo may not exist, be gated (set `HF_TOKEN`), or the "
                    + "network is unavailable.")
            throw ExitCode.failure
        }
        print(model)
        print("  modality:    \(support.modality.label)")
        switch support.loadability {
        case .loadable:
            print("  loadability: loadable")
            // Honesty boundary (ADR 021 D4): routing + packaging, NOT numeric
            // correctness — say "can load," never "is correct."
            print(
                "verdict: Athena can load this model "
                    + "(routing + required fields confirmed from config; this "
                    + "is not a correctness check).")
        case .unknown:
            print("  loadability: unknown (best-effort)")
            print(
                "verdict: Athena will attempt a best-effort load — config alone "
                    + "can't confirm the substrate implements this "
                    + "architecture; that's decided when the model loads.")
        case let .unsupported(reason, guidance):
            print("  loadability: unsupported")
            print("  reason:      \(reason)")
            print("  fix:         \(guidance)")
            print("verdict: Athena cannot load this model as packaged.")
            throw ExitCode.failure
        }
    }
}
