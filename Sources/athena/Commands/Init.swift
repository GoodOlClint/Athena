import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
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

    @Flag(
        name: .customLong("from-config"),
        help: """
            Read the per-module default model ids from the daemon's TOML \
            config (`model`, `embedding_model`, `transcription_model`, \
            `diarization_model`, `speaker_embedding_model`) and pull those \
            instead of the compiled-in defaults (ADR 026). After setting a \
            default with `athena default --module M <id>`, this re-runs init \
            to prefetch it.
            """
    )
    var fromConfig: Bool = false

    @Option(help: "Config file (overrides auto-resolution). Used with --from-config.")
    var config: String?

    func run() async throws {
        let root =
            modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot

        let aux: [(role: String, id: String)]
        if fromConfig {
            aux = readConfigAsAux()
            guard !aux.isEmpty else {
                print(
                    "athena init --from-config: no per-module default model "
                        + "keys set in the config. Set one with "
                        + "`athena default --module <m> <id>` first.")
                return
            }
        } else {
            // Source the ids from `load`'s defaults — one source of truth.
            // Every declared embedding model is pulled (M39) so the whole
            // selectable set is present before any request.
            var def: [(role: String, id: String)] =
                embeddingModels.map { ("embeddings", $0) }
            def.append(("transcription", Load.defaultTranscriptionModel))
            // ADR 020 — Parakeet-TDT transcription backend, selectable via the
            // `model` form field (higher multilingual quality). ~2.4 GB; pulled
            // alongside Whisper so the selectable set is present pre-request.
            def.append(
                ("transcription", Load.defaultParakeetTranscriptionModel))
            def.append(("diarization", Load.defaultDiarizationModel))
            // ADR 018 — pyannote segmentation backend for method=pyannote
            // (>4 / overlapping speakers). Tiny (~6 MB); pulled alongside.
            def.append(("diarization", Load.defaultPyannoteSegmentationModel))
            def.append(("speaker-embeddings", Load.defaultSpeakerEmbeddingModel))
            aux = def
        }

        print(
            "athena init — pulling \(aux.count) "
                + (fromConfig ? "configured " : "default auxiliary ")
                + "models into \(root.path)")
        if !fromConfig {
            print(
                "note: these are multi-GB downloads. The LLM is NOT included "
                    + "— choose one with `athena pull` / `athena default`.")
        }

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

    /// ADR 026 — read the per-module default model ids from the TOML config
    /// and map them into the `(role, id)` shape the pull loop expects, so an
    /// operator who set `athena default --module M <id>` can prefetch every
    /// configured default with one command. Modules with no configured default
    /// are skipped (nothing to pull).
    private func readConfigAsAux() -> [(role: String, id: String)] {
        guard
            let cfg = try? AthenaConfig.parse(
                file: ConfigEditor.resolvePath(config))
        else { return [] }
        var out: [(role: String, id: String)] = []
        if let id = cfg.model { out.append(("llm", id)) }
        if let id = cfg.embeddingModel { out.append(("embeddings", id)) }
        if let id = cfg.transcriptionModel {
            out.append(("transcription", id))
        }
        if let id = cfg.diarizationModel { out.append(("diarization", id)) }
        if let id = cfg.speakerEmbeddingModel {
            out.append(("speaker-embeddings", id))
        }
        return out
    }
}
