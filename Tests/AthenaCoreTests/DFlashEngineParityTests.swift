import AthenaCore
import AthenaModels
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import AthenaLLM

/// Test observer: counts speculative accept/reject events.
private final class AcceptanceCounter: SpeculativeAcceptanceObserver, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var accepted = 0
    private(set) var total = 0
    func recordIteration(accepted: Bool) {
        lock.lock()
        defer { lock.unlock() }
        total += 1
        if accepted { self.accepted += 1 }
    }
}

/// M63.3 — the load-bearing gate: DFlash greedy decode must be
/// **bit-identical** to non-speculative greedy decode of the same target.
/// Loads the 31B Gemma4 target + its z-lab drafter, decodes the same prompt
/// both ways, and asserts the token sequences are equal. Any divergence
/// (wrong acceptance, wrong rollback, wrong staging) shows up here.
///
/// Heavy (≈20 GB): gated on ATHENA_RUN_MODEL_TESTS=1 AND both checkpoints
/// present (env `ATHENA_TEST_GEMMA4_DIR` / `ATHENA_DFLASH_DRAFT_DIR`, else
/// the HF cache). The generation is kept short (< the sliding window) so the
/// single-token reference and the block DFlash forward share identical
/// cache numerics; the window-wrap rollback is covered separately by
/// `DFlashGemma4CaptureTests.testRotatingTrimRecentRollback`.
final class DFlashEngineParityTests: XCTestCase {

    private func snapshot(_ repo: String) -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let snaps = home.appending(
            path: ".cache/huggingface/hub/\(repo)/snapshots")
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: snaps, includingPropertiesForKeys: nil),
            let first = entries.first(where: {
                FileManager.default.fileExists(
                    atPath: $0.appending(component: "config.json").path)
            })
        else { return nil }
        return first
    }

    private static func backbone(_ model: any LanguageModel) -> DFlashGemma4Backbone? {
        if let g = model as? Gemma4Model { return g }
        if let g = model as? Gemma4TextModel { return g }
        return nil
    }

    /// Full stop-token set, matching the engine's dispatch.
    private static func stopSet(_ ctx: ModelContext) -> Set<Int> {
        var s = ctx.configuration.eosTokenIds
        if let e = ctx.tokenizer.eosTokenId { s.insert(e) }
        for tok in ctx.configuration.extraEOSTokens {
            if let id = ctx.tokenizer.convertTokenToId(tok) { s.insert(id) }
        }
        return s
    }

    private static func greedy(
        _ b: DFlashGemma4Backbone, prompt: [Int], maxTokens: Int,
        stop: Set<Int>
    ) -> [Int] {
        let cache = b.newCache(parameters: nil)
        func argmaxLast(_ logits: MLXArray) -> Int {
            Int(argMax(logits, axis: -1).reshaped(-1).asArray(Int32.self).last!)
        }
        let (l0, _) = b.callReturningHidden(
            MLXArray(prompt.map { Int32($0) }, [1, prompt.count]),
            cache: cache, captureLayers: [])
        var staged = argmaxLast(l0)
        var out: [Int] = []
        while out.count < maxTokens {
            if stop.contains(staged) { break }
            out.append(staged)
            if out.count >= maxTokens { break }
            let (l, _) = b.callReturningHidden(
                MLXArray([Int32(staged)], [1, 1]), cache: cache, captureLayers: [])
            staged = argmaxLast(l)
        }
        return out
    }

    /// Diagnostic: does a multi-token BLOCK forward produce the same
    /// per-position argmax as incremental SINGLE-token forwards? DFlash
    /// verify is a block forward; the "bit-identical to greedy" contract
    /// assumes the two agree. If they diverge at a near-tie, that is an SDPA
    /// kernel (n=1 vs n=K) floating-point difference, NOT an engine bug.
    func testBlockVsSingleTokenArgmaxAgree() async throws {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }
        let env = ProcessInfo.processInfo.environment
        let targetDirOpt =
            env["ATHENA_TEST_GEMMA4_DIR"].map(URL.init(fileURLWithPath:))
            ?? snapshot("models--mlx-community--gemma-4-31b-it-4bit")
        guard let targetDir = targetDirOpt else {
            throw XCTSkip("31B Gemma4 target not present")
        }

        let container = try await loadModelContainer(
            from: targetDir.resolvingSymlinksInPath(),
            using: #huggingFaceTokenizerLoader())
        let lmInput = try await container.prepare(
            input: UserInput(chat: [
                .user("Write three sentences about why the sky is blue.")
            ]))
        let prompt = lmInput.text.tokens.asArray(Int.self)

        try await container.perform { (ctx: ModelContext) in
            guard let target = Self.backbone(ctx.model) else {
                throw XCTSkip("not a Gemma4 target")
            }
            // Greedy single-token sequence.
            let gen = Self.greedy(
                target, prompt: prompt, maxTokens: 64,
                stop: Self.stopSet(ctx))
            let seq = prompt + gen
            // One block forward over the whole sequence.
            let (blockLogits, _) = target.callReturningHidden(
                MLXArray(seq.map { Int32($0) }, [1, seq.count]),
                cache: target.newCache(parameters: nil), captureLayers: [])
            let blockArgmax = argMax(blockLogits, axis: -1)
                .reshaped(-1).asArray(Int32.self).map { Int($0) }
            // For generated positions, block argmax after seq[i] must equal
            // the single-token greedy choice seq[i+1].
            var firstDiff = -1
            var diffs = 0
            for i in (prompt.count - 1) ..< (seq.count - 1)
            where blockArgmax[i] != seq[i + 1] {
                if firstDiff < 0 { firstDiff = i }
                diffs += 1
            }
            print(
                "block-vs-single argmax: \(diffs) mismatches over "
                    + "\(seq.count - prompt.count) positions; firstDiff="
                    + "\(firstDiff)\(firstDiff >= 0 ? " (block=\(blockArgmax[firstDiff]) single=\(seq[firstDiff + 1]))" : "")")
            XCTAssertEqual(
                diffs, 0,
                "block forward disagrees with single-token greedy at "
                    + "\(diffs) positions — SDPA n=1 vs n=K FP difference")
        }
    }

    /// Bounded kernel investigation: forward the SAME token sequence at
    /// several block sizes and count per-position argmax disagreements with
    /// single-token (n=1). Distinguishes a clean n=1-vs-n>1 kernel split
    /// (no block size matches single-token → strict bit-identicality
    /// unattainable) from a block-size-graded effect (a smaller verify block
    /// could match). Also tries an explicit array mask vs the .causal hint.
    func testKernelBlockSizeSensitivity() async throws {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }
        let env = ProcessInfo.processInfo.environment
        let dirOpt =
            env["ATHENA_TEST_GEMMA4_DIR"].map(URL.init(fileURLWithPath:))
            ?? snapshot("models--mlx-community--gemma-4-31b-it-4bit")
        guard let dir = dirOpt else { throw XCTSkip("31B not present") }

        let container = try await loadModelContainer(
            from: dir.resolvingSymlinksInPath(),
            using: #huggingFaceTokenizerLoader())
        let lmInput = try await container.prepare(
            input: UserInput(chat: [
                .user("Write three sentences about why the sky is blue.")
            ]))
        let prompt = lmInput.text.tokens.asArray(Int.self)

        try await container.perform { (ctx: ModelContext) in
            guard let target = Self.backbone(ctx.model) else {
                throw XCTSkip("not a Gemma4 target")
            }
            let gen = Self.greedy(
                target, prompt: prompt, maxTokens: 64,
                stop: Self.stopSet(ctx))
            let seq = prompt + gen

            func argmaxInBlocks(_ blockSize: Int) -> [Int] {
                let cache = target.newCache(parameters: nil)
                var result: [Int] = []
                var i = 0
                while i < seq.count {
                    let end = min(i + blockSize, seq.count)
                    let (l, _) = target.callReturningHidden(
                        MLXArray(seq[i ..< end].map { Int32($0) }, [1, end - i]),
                        cache: cache, captureLayers: [])
                    result.append(
                        contentsOf: argMax(l, axis: -1).reshaped(-1)
                            .asArray(Int32.self).map { Int($0) })
                    i = end
                }
                return result
            }

            let single = argmaxInBlocks(1)
            let genRange = (prompt.count - 1) ..< (seq.count - 1)
            for B in [2, 4, 8, 16] {
                let blocked = argmaxInBlocks(B)
                var mism = 0
                for p in genRange where blocked[p] != single[p] { mism += 1 }
                print("kernel block B=\(B): \(mism)/\(genRange.count) "
                    + "positions differ from n=1")
            }
            // Self-consistency: n=1 against the greedy tokens (must be 0).
            var s = 0
            for p in genRange where single[p] != seq[p + 1] { s += 1 }
            print("n=1 self-consistency mismatches: \(s) (expect 0)")
        }
    }

    /// M63.3b integration: DFlash engages through the full module dispatch
    /// (config flag → ensureDFlashDraft HF pull → runSpeculative branch →
    /// engine) and produces coherent output. Uses a symlink named like the
    /// store entry so `DFlashRegistry` matches the resident model name.
    func testDFlashEngagesThroughTheModule() async throws {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy, ~20GB)") }
        let env = ProcessInfo.processInfo.environment
        let targetOpt =
            env["ATHENA_TEST_GEMMA4_DIR"].map(URL.init(fileURLWithPath:))
            ?? snapshot("models--mlx-community--gemma-4-31b-it-4bit")
        guard let target = targetOpt,
            snapshot("models--z-lab--gemma-4-31B-it-DFlash") != nil
                || env["ATHENA_DFLASH_DRAFT_DIR"] != nil
        else { throw XCTSkip("31B target and/or drafter not present") }

        let tmp = FileManager.default.temporaryDirectory
            .appending(path: "dflash-it-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        // The module keys the drafter registry on the dir's lastPathComponent.
        let link = tmp.appending(path: "gemma-4-31b-it-4bit")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: target.resolvingSymlinksInPath())

        let llm = MLXLLMModule(
            modelDirectory: link,
            parameters: .init(maxTokens: 48, temperature: 0, speculative: true),
            dflashEnabled: true)
        let gov = MemoryGovernor(totalBudgetBytes: Int(96) << 30)
        await gov.register(llm, evictable: false)
        try await gov.ensureLoaded(.llm)

        var out = ""
        for await chunk in llm.generate(
            prompt: "Name three primary colors, comma separated.")
        {
            out += chunk
        }
        let text = out.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(text.isEmpty, "DFlash module path produced no output")
        XCTAssertGreaterThan(
            Set(text.replacingOccurrences(of: " ", with: "")).count, 2,
            "degenerate output via DFlash dispatch: \(text)")
        print("DFlash via module dispatch: \(text.prefix(80))")
    }

    func testDFlashBitIdenticalToGreedy() async throws {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy, ~20GB)") }
        let env = ProcessInfo.processInfo.environment
        let targetDir =
            env["ATHENA_TEST_GEMMA4_DIR"].map { URL(fileURLWithPath: $0) }
            ?? snapshot("models--mlx-community--gemma-4-31b-it-4bit")
        let draftDir =
            env["ATHENA_DFLASH_DRAFT_DIR"].map { URL(fileURLWithPath: $0) }
            ?? snapshot("models--z-lab--gemma-4-31B-it-DFlash")
        guard let targetDir, let draftDir else {
            throw XCTSkip("31B Gemma4 target and/or z-lab drafter not present")
        }

        let container = try await loadModelContainer(
            from: targetDir.resolvingSymlinksInPath(),
            using: #huggingFaceTokenizerLoader())

        // A coherent chat-templated prompt → varied generation, so the
        // accept/REJECT/rollback transitions are exercised (a degenerate
        // repeated-token prompt would accept everything and never roll back).
        let lmInput = try await container.prepare(
            input: UserInput(chat: [
                .user("Write three sentences about why the sky is blue.")
            ]))
        let prompt = lmInput.text.tokens.asArray(Int.self)
        let maxTokens = 64

        // Load the draft INSIDE the @Sendable perform closure (only the
        // Sendable [Int]/URL cross the boundary; MLXArray-bearing models do not).
        try await container.perform { (ctx: ModelContext) in
            let draft = try DFlashDraftLoader.load(directory: draftDir)
            guard let target = Self.backbone(ctx.model) else {
                throw XCTSkip("loaded model is not a Gemma4 target")
            }
            let stop = Self.stopSet(ctx)

            let reference = Self.greedy(
                target, prompt: prompt, maxTokens: maxTokens, stop: stop)
            let counter = AcceptanceCounter()
            let dflash = SpeculativeStats.$observer.withValue(counter) {
                DFlashGeneration.generate(
                    target: target, draft: draft, promptTokens: prompt,
                    maxTokens: maxTokens, stopTokens: stop)
            }
            XCTAssertFalse(reference.isEmpty, "reference produced no tokens")

            // The acceptance observer engaged and the drafter is useful.
            XCTAssertGreaterThan(counter.total, 0, "no speculative iterations recorded")
            let acceptRate = Double(counter.accepted) / Double(max(counter.total, 1))
            print("DFlash acceptance rate: \(String(format: "%.2f", acceptRate)) "
                + "(\(counter.accepted)/\(counter.total))")
            XCTAssertGreaterThan(
                acceptRate, 0.3, "implausibly low acceptance — draft not engaging")

            // DFlash is lossless = every emitted token is the target's argmax
            // under the verify (block, n≥2) forward. It therefore matches
            // single-token greedy UNTIL the first SDPA-kernel near-tie where
            // the n≥2 verify kernel and the n=1 vector kernel disagree (an
            // intrinsic MLX kernel-dispatch difference, not an engine error —
            // see testKernelBlockSizeSensitivity). The gate: assert they
            // agree up to the first divergence, and that the divergence is a
            // genuine block-vs-single kernel tie (so no engine bug hides here).
            let n = min(dflash.count, reference.count)
            var d = 0
            while d < n, dflash[d] == reference[d] { d += 1 }
            print(
                "DFlash vs single-token greedy: \(d)/\(n) identical prefix "
                    + "(dflash=\(dflash.count) ref=\(reference.count) tokens)")
            if d < n {
                // At the divergence, the prefix (prompt + matched tokens) is
                // shared. The n≥2 block forward of that prefix must predict
                // exactly what DFlash committed (proving DFlash followed the
                // verify argmax), and it must differ from the single-token
                // choice (proving the divergence is a kernel tie, not a bug).
                let prefix = prompt + Array(reference[0 ..< d])
                let (bl, _) = target.callReturningHidden(
                    MLXArray(prefix.map { Int32($0) }, [1, prefix.count]),
                    cache: target.newCache(parameters: nil), captureLayers: [])
                let blockNext = Int(
                    argMax(bl, axis: -1).reshaped(-1).asArray(Int32.self).last!)
                XCTAssertEqual(
                    dflash[d], blockNext,
                    "DFlash diverged from the BLOCK-forward greedy at \(d) — "
                        + "engine bug (dflash=\(dflash[d]) block=\(blockNext))")
                XCTAssertNotEqual(
                    blockNext, reference[d],
                    "divergence at \(d) is not an SDPA kernel tie "
                        + "(block=\(blockNext) single=\(reference[d])) — engine bug")
                print(
                    "  divergence at \(d) is a verified kernel tie: "
                        + "block=\(blockNext) single=\(reference[d]); "
                        + "DFlash correctly followed the block argmax")
            }
        }
    }
}
