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

    @Option(
        name: .customLong("embedding-model"),
        help: """
            Embedding model HF id(s) to pull. Repeatable: pass it once per \
            model you intend to make selectable at `load` (matching its \
            --embedding-model set), so per-request model selection works \
            without a first-use download. Default BAAI/bge-small-en-v1.5.
            """
    )
    var embeddingModels: [String] = [Load.defaultEmbeddingModel]

    func run() async throws {
        let root =
            modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot

        // Source the ids from `load`'s defaults — one source of truth.
        // Every declared embedding model is pulled (M39) so the whole
        // selectable set is present before any request.
        var aux: [(role: String, id: String)] =
            embeddingModels.map { ("embeddings", $0) }
        aux.append(("transcription", Load.defaultTranscriptionModel))
        aux.append(("diarization", Load.defaultDiarizationModel))
        aux.append(("speaker-embeddings", Load.defaultSpeakerEmbeddingModel))

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
                print("  pulled → \(d.path)")
                pulled += 1
            } catch {
                bar.finish()
                print("  error: pull failed — \(ModelPull.friendlyError(error))")
                failed += 1
            }
        }

        print(
            "init complete: \(pulled) pulled, \(skipped) skipped, "
                + "\(failed) failed.")
        if failed > 0 { throw ExitCode.failure }
    }
}
