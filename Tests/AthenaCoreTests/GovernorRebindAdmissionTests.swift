import AthenaEmbedding
import AthenaLLM
import Foundation
import XCTest

@testable import AthenaCore

/// #206 — admission and accounting follow the model actually requested and
/// resident, across cold loads and warm rebinds.
final class GovernorRebindAdmissionTests: XCTestCase {

    private func llm(_ bytes: [String: Int]) -> StubLLMModule {
        StubLLMModule(
            reserveBytes: 1, modelIds: Array(bytes.keys).sorted(),
            configuredDefault: "small", modelBytes: bytes)
    }

    private func mod(_ s: GovernorSnapshot, _ id: ModuleID) -> ModuleSnapshot? {
        s.modules.first { $0.id == id }
    }

    // (a) — a cold load is admitted on the requested model's own estimate.
    func testColdLoadAdmitsOnRequestedModelEstimate() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "big": 900])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 300), evictable: true)
        try await gov.ensureLoaded(.textEmbedding)

        try await l.selectColdLoadModel("big")
        try await gov.ensureLoaded(.llm)

        let s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 900)
        let emb = mod(s, .textEmbedding)?.state
        XCTAssertTrue(
            emb == .unloading || emb == .unloaded,
            "900 + 300 > 1000: the co-tenant must be evicted, got \(String(describing: emb))")
    }

    // (a) — a footprint learned for one model never admits another.
    func testLearnedFootprintIsPerModel() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "big": 900])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 300), evictable: true)

        try await gov.ensureLoaded(.llm)  // learns small = 100
        await gov.unload(.llm)
        try await gov.ensureLoaded(.textEmbedding)
        try await l.selectColdLoadModel("big")
        try await gov.ensureLoaded(.llm)

        let s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 900)
        XCTAssertNotEqual(mod(s, .textEmbedding)?.state, .loaded)
    }

    // (b) — a rebind re-measures: residentBytes follows the new model.
    func testRebindReconcilesResidentBytes() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "big": 600])
        await gov.register(l, evictable: false)
        try await gov.ensureLoaded(.llm)
        let before = await gov.snapshot()
        XCTAssertEqual(mod(before, .llm)?.residentBytes, 100)

        try await gov.rebind(.llm, to: "big") { try await l.rebind(to: "big") }
        var s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 600)
        XCTAssertEqual(mod(s, .llm)?.measured, true)
        XCTAssertEqual(s.residentBytes, 600)

        try await gov.rebind(.llm, to: "small") { try await l.rebind(to: "small") }
        s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 100)
        XCTAssertEqual(s.residentBytes, 100)
    }

    // (b) — a rebind is admitted: a growing swap evicts co-tenants first.
    func testRebindAdmissionEvictsCoTenant() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "big": 800])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 300), evictable: true)
        try await gov.ensureLoaded(.llm)
        try await gov.ensureLoaded(.textEmbedding)

        try await gov.rebind(.llm, to: "big") { try await l.rebind(to: "big") }

        let s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 800)
        XCTAssertNotEqual(mod(s, .textEmbedding)?.state, .loaded)
        XCTAssertLessThanOrEqual(s.residentBytes, 1_000)
    }

    // (b) — a rebind to a model larger than the budget is a 400 and leaves
    // the resident model and its reservation untouched.
    func testRebindToOversizedModelIsRefusedBeforeLoad() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "huge": 2_000])
        await gov.register(l, evictable: false)
        try await gov.ensureLoaded(.llm)

        do {
            try await gov.rebind(.llm, to: "huge") { try await l.rebind(to: "huge") }
            XCTFail("expected modelExceedsBudget")
        } catch let e as AthenaError {
            XCTAssertEqual(e.httpStatus, 400)
            XCTAssertEqual(e.code, "model_exceeds_memory_budget")
        }
        let resident = await l.residentModelId()
        XCTAssertEqual(resident, "small")
        let before = await gov.snapshot()
        XCTAssertEqual(mod(before, .llm)?.residentBytes, 100)
    }

    // (c) — the LLM is admitted only with one request's KV headroom
    // (the prompt-cache cap) free on top of its weights.
    func testLLMAdmissionReservesKVHeadroom() async throws {
        let gov = MemoryGovernor(
            totalBudgetBytes: 1_000, promptCacheCapBytes: 200,
            reserveKVHeadroom: true)
        let l = llm(["small": 700, "tight": 900])
        await gov.register(l, evictable: false)

        try await l.selectColdLoadModel("tight")
        do {
            try await gov.ensureLoaded(.llm)
            XCTFail("900 weights + 200 KV headroom > 1000 must be refused")
        } catch let e as AthenaError {
            XCTAssertEqual(e.httpStatus, 400)
            XCTAssertEqual(e.code, "model_exceeds_memory_budget")
        }

        try await l.selectColdLoadModel("small")
        try await gov.ensureLoaded(.llm)
        let loaded = await gov.snapshot()
        XCTAssertEqual(mod(loaded, .llm)?.state, .loaded)
    }

    // (c) — a co-tenant cannot consume the resident LLM's KV headroom.
    func testCoTenantCannotConsumeKVHeadroom() async throws {
        let gov = MemoryGovernor(
            totalBudgetBytes: 1_000, promptCacheCapBytes: 200,
            reserveKVHeadroom: true)
        let l = llm(["small": 700])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 200), evictable: true)
        try await gov.ensureLoaded(.llm)

        do {
            try await gov.ensureLoaded(.textEmbedding)
            XCTFail("700 + 200 KV headroom + 200 > 1000 must be refused")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "memory_budget_exceeded")
        }
    }

    // (c) — without the headroom switch admission is unchanged (revert path).
    func testNoKVHeadroomWhenDisabled() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000, promptCacheCapBytes: 200)
        let l = llm(["small": 700])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 200), evictable: true)
        try await gov.ensureLoaded(.llm)
        try await gov.ensureLoaded(.textEmbedding)
        let s = await gov.snapshot()
        XCTAssertEqual(s.residentBytes, 900)
    }

    // (a) — the real LLM module estimates the requested model, not the store max.
    func testMLXLLMModuleEstimatesRequestedModel() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-206-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        for (name, bytes) in [("small-llm", 1_000), ("big-llm", 9_000)] {
            let dir = root.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            try Data(#"{"model_type":"llama"}"#.utf8).write(
                to: dir.appendingPathComponent("config.json"))
            try Data(count: bytes).write(
                to: dir.appendingPathComponent("model.safetensors"))
        }
        let m = MLXLLMModule(modelStoreRoot: root, configuredDefault: "small-llm")

        let small = try await m.admissionEstimate(forModel: nil)
        XCTAssertEqual(small.model, "small-llm")
        XCTAssertEqual(small.bytes, 1_000)
        let big = try await m.admissionEstimate(forModel: "big-llm")
        XCTAssertEqual(big.bytes, 9_000)
        let cold = await m.memoryEstimate()
        XCTAssertEqual(cold, 1_000)
        try await m.selectColdLoadModel("big-llm")
        let coldBig = await m.memoryEstimate()
        XCTAssertEqual(coldBig, 9_000)
    }
}
