import AthenaModels
import Foundation
import MLXLLM
import MLXLMCommon

/// Registers Athena's vendored Qwen3.5 model into the substrate's public,
/// last-write-wins model-type registry, overriding the upstream
/// `qwen3_5*` mappings. This keeps `mlx-swift-lm` a pristine path
/// dependency: the local-dir loader resolves `model_type` through this
/// same registry, so a Qwen3.5 directory loads Athena's class.
///
/// MTP gate: `mtp_num_hidden_layers` is an architectural constant present
/// even in stock checkpoints that ship NO `mtp.*` weights, and mlx-swift
/// forbids removing a `@ModuleInfo` submodule after construction. So the
/// head must be suppressed at construction when the checkpoint lacks the
/// tensors — done here by inspecting the model directory the creator is
/// about to load (the substrate's creator only gets config Data, so the
/// directory is passed via `currentModelDirectory`).
enum AthenaModelRegistration {

    /// Set by `MLXLLMModule.load()` immediately before
    /// `loadModelContainer`. Safe as `nonisolated(unsafe)`: LLM loads are
    /// serialized (one governed `ensureLoaded(.llm)` at a time, driven
    /// from the `MLXLLMModule` actor), so there is never a concurrent
    /// creator invocation racing this.
    nonisolated(unsafe) static var currentModelDirectory: URL?

    /// True iff the checkpoint at `dir` actually contains `mtp.*` weights.
    /// nil dir ⇒ true (defer to config; Athena always sets the dir).
    static func checkpointHasMTP(_ dir: URL?) -> Bool {
        guard let dir else { return true }
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let weightMap = obj["weight_map"] as? [String: Any]
        {
            return weightMap.keys.contains { $0.contains("mtp.") }
        }
        // No index (single-file checkpoint): scan safetensors headers.
        let entries =
            (try? FileManager.default.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: nil)) ?? []
        for url in entries where url.pathExtension == "safetensors" {
            if Self.safetensorsHeaderHasMTP(url) { return true }
        }
        return entries.contains { $0.pathExtension == "safetensors" }
            ? false : true
    }

    private static func safetensorsHeaderHasMTP(_ url: URL) -> Bool {
        guard let fh = try? FileHandle(forReadingFrom: url) else {
            return false
        }
        defer { try? fh.close() }
        guard let lenData = try? fh.read(upToCount: 8), lenData.count == 8
        else { return false }
        let headerLen = lenData.withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        guard headerLen > 0, headerLen < 50_000_000,
            let header = try? fh.read(upToCount: Int(headerLen)),
            let json = String(data: header, encoding: .utf8)
        else { return false }
        return json.contains("mtp.")
    }

    private static func decode<C: Codable>(_ type: C.Type, _ data: Data) throws
        -> C
    {
        if let json5 = try? JSONDecoder.json5().decode(C.self, from: data) {
            return json5
        }
        return try JSONDecoder().decode(C.self, from: data)
    }

    /// Idempotent (registry is last-write-wins); safe before every load.
    static func install() async {
        let registry = LLMModelFactory.shared.typeRegistry

        await registry.registerModelType("qwen3_5") {
            @Sendable data in
            let c = try decode(AthenaQwen35Configuration.self, data)
            return AthenaQwen35Model(
                checkpointHasMTP(currentModelDirectory)
                    ? c : c.withMTPDisabled())
        }
        await registry.registerModelType("qwen3_5_moe") {
            @Sendable data in
            let c = try decode(AthenaQwen35Configuration.self, data)
            return AthenaQwen35MoEModel(
                checkpointHasMTP(currentModelDirectory)
                    ? c : c.withMTPDisabled())
        }
        await registry.registerModelType("qwen3_5_text") {
            @Sendable data in
            let c = try decode(AthenaQwen35TextConfiguration.self, data)
            return AthenaQwen35TextModel(
                checkpointHasMTP(currentModelDirectory)
                    ? c : c.withMTPDisabled())
        }
    }
}
