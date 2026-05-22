import ArgumentParser
import AthenaClient
import AthenaLLM
import Foundation

/// `athena init` — pull the default auxiliary modules (embeddings,
/// transcription, diarization) into the model store so a fresh appliance
/// can serve every aux endpoint without per-request first-use downloads.
/// The LLM is deliberately NOT fetched — there is no hard LLM default;
/// the operator picks one with `athena pull` / `athena default`.
///
/// Local model-store op (Apple host only) — reuses ModelPull + the
/// download progress bar + the M13 egress-proxy/HF-auth apply, exactly
/// like `pull`. Idempotent: an already-present model is skipped.
struct Init: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "init",
        abstract:
            "Pull the default auxiliary models (embeddings, "
            + "transcription, diarization, speaker-embeddings) into "
            + "the model store."
    )

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    func run() async throws {
        let root =
            modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot

        // Source the ids from `load`'s defaults — one source of truth.
        let aux: [(role: String, id: String)] = [
            ("embeddings", Load.defaultEmbeddingModel),
            ("transcription", Load.defaultTranscriptionModel),
            ("diarization", Load.defaultDiarizationModel),
            ("speaker-embeddings", Load.defaultSpeakerEmbeddingModel),
        ]

        print(
            "athena init — pulling \(aux.count) default auxiliary models "
                + "into \(root.path)")
        print(
            "note: these are multi-GB downloads. The LLM is NOT included "
                + "— choose one with `athena pull` / `athena default`.")

        // Same outbound setup as `pull`: egress proxy (M13.2) then the
        // Keychain HF token (M13) for gated/private repos. This is the
        // sanctioned passive-oracle egress (model fetch only).
        ProxyEnv.applyConfigAndAuth()
        HFAuth.exportToEnv()

        var pulled = 0
        var skipped = 0
        var failed = 0
        for (role, id) in aux {
            let name =
                id.split(separator: "/").last.map(String.init) ?? id
            let dest = root.appendingPathComponent(
                name, isDirectory: true)
            // Idempotent: a present (non-dangling) model dir is skipped.
            if FileManager.default.fileExists(atPath: dest.path) {
                print("• \(role): \(id) — already present, skipping")
                skipped += 1
                continue
            }
            print("• \(role): pulling \(id) …")
            let bar = ProgressBar("  \(name)")
            do {
                let d = try await ModelPull.pull(
                    id: id, into: root,
                    progress: { f in bar.update(f) })
                bar.finish()
                print("  pulled → \(d.path)")
                pulled += 1
            } catch {
                bar.finish()
                print("  error: pull failed — \(error)")
                failed += 1
            }
        }

        print(
            "init complete: \(pulled) pulled, \(skipped) skipped, "
                + "\(failed) failed.")
        if failed > 0 { throw ExitCode.failure }
    }
}
