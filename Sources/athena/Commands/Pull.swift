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

    @Flag(help: "Print download progress as it streams (remote only).")
    var follow = false

    @Flag(
        help: ArgumentHelp(
            "Dry run: fetch only the model's config and report whether Athena "
            + "can load it (modality + loadability). Downloads no weights, "
            + "touches no daemon. Exits non-zero if unsupported."))
    var check = false

    @Flag(
        name: .long,
        help: ArgumentHelp(
            "Also pull the model's paired MTP speculative drafter, if one is "
            + "known (ADR 032). Resolved from the seeded default-drafter map; "
            + "no-op when the target has no mapped drafter."))
    var withDrafter = false

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
                daemon, op: "pull", body: b, progress: follow)
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
        try await pullOne(id: model, into: root)

        // ADR 032 — `--with-drafter`: also fetch the target's paired MTP
        // speculative drafter if one is mapped. The seed map alone (no data_dir)
        // covers the published pairs; a miss is a no-op, not an error.
        if withDrafter {
            if let drafter = MTPDrafterPairing.resolve(
                targetID: model, explicit: nil,
                defaults: MTPDrafterPairing.defaultMap(dataDir: nil))
            {
                print("pulling paired MTP drafter \(drafter) …")
                try await pullOne(id: drafter, into: root)
            } else {
                FileHandle.standardError.write(
                    Data(
                        ("note: no MTP drafter is mapped for '\(model)'; "
                            + "nothing more to pull.\n").utf8))
            }
        }
    }

    /// Pull one HF id into `root` with a progress bar + retry reporting. Throws
    /// `ExitCode.failure` (already messaged) on failure.
    private func pullOne(id: String, into root: URL) async throws {
        print("pulling \(id) …")
        let bar = ProgressBar("  \(id)")
        do {
            let dest = try await ModelPull.pull(
                id: id, into: root,
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
            print("pulled \(id) → \(dest.path)")
        } catch {
            bar.finish()
            print("error: pull failed — \(ModelPull.friendlyError(error))")
            print(
                "  re-run `athena pull \(id)` to resume "
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
        // ADR 032 — for a generative/vision target, note a mapped MTP drafter so
        // the operator knows `--with-drafter` (and the `speculative` knob) has
        // something to pair. Seed map only (no data_dir) in this host-independent
        // dry run.
        if support.modality == .llm || support.modality == .vision,
            let drafter = MTPDrafterPairing.resolve(
                targetID: model, explicit: nil,
                defaults: MTPDrafterPairing.defaultMap(dataDir: nil))
        {
            print("  mtp_drafter: \(drafter) (pull with --with-drafter)")
        }
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
