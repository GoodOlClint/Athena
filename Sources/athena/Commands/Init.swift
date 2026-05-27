import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaLLM
import AthenaStore
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
        name: .customLong("from-allowlist"),
        help: """
            Read the persistent allowlist and pull every allowed id \
            instead of the compiled-in defaults. After an operator adds \
            a new model via `athena allowlist add`, this re-runs init to \
            prefetch it. The data-dir hosts the SQLite store; default \
            ~/.athena. LLM ids are included.
            """
    )
    var fromAllowlist: Bool = false

    @Option(help: "Runtime/data dir (default ~/.athena). Used with --from-allowlist.")
    var dataDir: String?

    func run() async throws {
        let root =
            modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot

        let aux: [(role: String, id: String)]
        if fromAllowlist {
            aux = try await readAllowlistAsAux()
            guard !aux.isEmpty else {
                print(
                    "athena init --from-allowlist: no allowlist rows in "
                        + "\(dataDir ?? "~/.athena")/athena.sqlite. Run "
                        + "`athena load` once (seeds CLI flags) or "
                        + "`athena allowlist add` first.")
                return
            }
        } else {
            // Source the ids from `load`'s defaults — one source of truth.
            // Every declared embedding model is pulled (M39) so the whole
            // selectable set is present before any request.
            var def: [(role: String, id: String)] =
                embeddingModels.map { ("embeddings", $0) }
            def.append(("transcription", Load.defaultTranscriptionModel))
            def.append(("diarization", Load.defaultDiarizationModel))
            def.append(("speaker-embeddings", Load.defaultSpeakerEmbeddingModel))
            aux = def
        }

        print(
            "athena init — pulling \(aux.count) "
                + (fromAllowlist ? "allowlist " : "default auxiliary ")
                + "models into \(root.path)")
        if !fromAllowlist {
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

    /// M43.2 — load every (module, id) row from the SQLite allowlist and
    /// map it into the `(role, id)` shape the existing pull loop expects.
    /// Mirrors the module→role labels init has always used; `llm` rows
    /// are included so an operator who added an LLM via `athena allowlist
    /// add` can prefetch the weights with one command.
    private func readAllowlistAsAux() async throws
        -> [(role: String, id: String)]
    {
        let dir = dataDir.map {
            URL(
                fileURLWithPath: ($0 as NSString).expandingTildeInPath,
                isDirectory: true)
        } ?? AthenaEnv.userHome()
            .appendingPathComponent(".athena", isDirectory: true)
        let dbURL = dir.appendingPathComponent("athena.sqlite")
        let store = try AthenaStore(
            path: dbURL, key: StoreKey.resolve())
        let rows = await store.listModelAllowlist()
        return rows.map { row -> (role: String, id: String) in
            let role: String
            switch row.module {
            case "llm": role = "llm"
            case "textEmbedding": role = "embeddings"
            case "transcription": role = "transcription"
            case "diarization": role = "diarization"
            case "speakerEmbedding": role = "speaker-embeddings"
            default: role = row.module
            }
            return (role: role, id: row.id)
        }
    }
}
