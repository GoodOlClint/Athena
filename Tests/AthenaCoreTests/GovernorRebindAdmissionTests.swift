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

    // (a) — a footprint learned for one model never admits another. The
    // co-tenant is not evictable, so a wrong (small) estimate would admit and
    // load instead of refusing up front.
    func testLearnedFootprintIsPerModel() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "big": 900])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 300), evictable: false)

        try await gov.ensureLoaded(.llm)  // learns small = 100
        await gov.unload(.llm)
        try await gov.ensureLoaded(.textEmbedding)
        try await l.selectColdLoadModel("big")
        do {
            try await gov.ensureLoaded(.llm)
            XCTFail("900 + 300 > 1000 must be refused before the load")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "memory_budget_exceeded")
        }
        let resident = await l.residentModelId()
        XCTAssertNil(resident)
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

    // (b) — a growing rebind that cannot fit is refused before the swap: the
    // co-tenant is not evictable, so only admission (not a post-swap
    // reconcile) can stop it.
    func testRebindRefusedBeforeSwapWhenItCannotFit() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "big": 800])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 300), evictable: false)
        try await gov.ensureLoaded(.llm)
        try await gov.ensureLoaded(.textEmbedding)

        do {
            try await gov.rebind(.llm, to: "big") { try await l.rebind(to: "big") }
            XCTFail("100→800 with a 300 non-evictable co-tenant must be refused")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "memory_budget_exceeded")
        }
        let resident = await l.residentModelId()
        XCTAssertEqual(resident, "small")
        let s = await gov.snapshot()
        XCTAssertEqual(s.residentBytes, 400)
    }

    // (b) — the swap is admitted in place of the old reservation: 400 → 500
    // next to a 450 co-tenant needs only +100, which fits.
    func testRebindAdmitsInPlaceOfOldReservation() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 400, "big": 500])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 450), evictable: false)
        try await gov.ensureLoaded(.llm)
        try await gov.ensureLoaded(.textEmbedding)

        try await gov.rebind(.llm, to: "big") { try await l.rebind(to: "big") }
        let s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 500)
        XCTAssertEqual(s.residentBytes, 950)
    }

    // (b) — a swap that fails before it starts (e.g. the gate wait is
    // cancelled) leaves the old model resident and its reservation intact.
    func testFailedRebindRestoresOldReservation() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "big": 600])
        await gov.register(l, evictable: false)
        try await gov.ensureLoaded(.llm)

        do {
            try await gov.rebind(.llm, to: "big") { throw CancellationError() }
            XCTFail("expected the perform error")
        } catch is CancellationError {}
        var s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 100)
        XCTAssertEqual(s.residentBytes, 100)

        try await gov.rebind(.llm, to: "big") { try await l.rebind(to: "big") }
        s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 600)
    }

    // (b) — a swap that empties the slot and then fails returns its bytes.
    func testFailedRebindThatEmptiedSlotReleasesReservation() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let l = llm(["small": 100, "big": 600])
        await gov.register(l, evictable: false)
        try await gov.ensureLoaded(.llm)

        struct Boom: Error {}
        do {
            try await gov.rebind(.llm, to: "big") {
                await l.unload()
                throw Boom()
            }
            XCTFail("expected Boom")
        } catch is Boom {}
        let s = await gov.snapshot()
        XCTAssertEqual(s.residentBytes, 0)
        XCTAssertEqual(mod(s, .llm)?.state, .unloaded)
    }

    // (c) — a load measured above its estimate keeps the KV headroom:
    // reconcile evicts a co-tenant rather than eat into it.
    func testReconcileKeepsKVHeadroom() async throws {
        let probe = ProbeScript()
        let gov = MemoryGovernor(
            totalBudgetBytes: 1_000, memoryProbe: { probe.next() },
            promptCacheCapBytes: 200, reserveKVHeadroom: true)
        let l = llm(["small": 500])
        await gov.register(l, evictable: false)
        await gov.register(StubEmbeddingModule(reserveBytes: 250), evictable: true)
        try await gov.ensureLoaded(.textEmbedding)

        probe.queue([0, 0, 700])  // relief check, before, after the LLM load
        try await gov.ensureLoaded(.llm)

        let s = await gov.snapshot()
        XCTAssertEqual(mod(s, .llm)?.residentBytes, 700)
        XCTAssertNotEqual(
            mod(s, .textEmbedding)?.state, .loaded,
            "700 + 250 + 200 headroom > 1000: the co-tenant must go")
    }

    // (b) — a growing swap evicts an evictable co-tenant.
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

/// A scripted process-memory probe: returns queued values, then 0.
private final class ProbeScript: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int] = []
    func queue(_ v: [Int]) { lock.withLock { values = v } }
    func next() -> Int {
        lock.withLock { values.isEmpty ? 0 : values.removeFirst() }
    }
}
