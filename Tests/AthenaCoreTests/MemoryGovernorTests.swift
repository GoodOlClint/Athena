import AthenaEmbedding
import AthenaLLM
import AthenaTranscription
import XCTest

@testable import AthenaCore

/// Global-budget admission & eviction — the thesis subsystem.
final class MemoryGovernorTests: XCTestCase {

    func testAdmitsWithinBudget() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        await gov.register(StubLLMModule(reserveBytes: 400), evictable: false)

        try await gov.ensureLoaded(.llm)

        let s = await gov.snapshot()
        XCTAssertEqual(s.reservedBytes, 400)
        XCTAssertEqual(s.freeBytes, 600)
        XCTAssertEqual(s.modules.first { $0.id == .llm }?.state, .loaded)
    }

    func testRejectsOverBudgetUnevictable() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 50)
        await gov.register(StubLLMModule(reserveBytes: 100), evictable: false)

        do {
            try await gov.ensureLoaded(.llm)
            XCTFail("expected memoryBudgetExceeded")
        } catch let e as AthenaError {
            XCTAssertEqual(e.httpStatus, 503)
            XCTAssertEqual(e.code, "memory_budget_exceeded")
            guard case let .memoryBudgetExceeded(requested, available, module) = e
            else { return XCTFail("wrong error case: \(e)") }
            XCTAssertEqual(requested, 100)
            XCTAssertEqual(available, 50)
            XCTAssertEqual(module, .llm)
        }

        let s = await gov.snapshot()
        XCTAssertEqual(s.reservedBytes, 0)
        XCTAssertEqual(s.modules.first { $0.id == .llm }?.state, .unloaded)
    }

    func testEvictsLRUToAdmit() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 100)
        await gov.register(
            StubTranscriptionModule(reserveBytes: 60), evictable: true)
        await gov.register(
            StubEmbeddingModule(reserveBytes: 60), evictable: true)

        try await gov.ensureLoaded(.transcription)
        try await gov.ensureLoaded(.textEmbedding)  // forces eviction

        let s = await gov.snapshot()
        XCTAssertEqual(s.reservedBytes, 60)
        XCTAssertEqual(
            s.modules.first { $0.id == .textEmbedding }?.state, .loaded)
        let victim = s.modules.first { $0.id == .transcription }?.state
        XCTAssertTrue(victim == .unloading || victim == .unloaded)
    }

    func testProtectsNonEvictable() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 100)
        await gov.register(StubLLMModule(reserveBytes: 80), evictable: false)
        await gov.register(
            StubTranscriptionModule(reserveBytes: 80), evictable: true)

        try await gov.ensureLoaded(.llm)

        do {
            try await gov.ensureLoaded(.transcription)
            XCTFail("expected memoryBudgetExceeded")
        } catch is AthenaError {
            // expected
        }
        let s = await gov.snapshot()
        XCTAssertEqual(s.reservedBytes, 80)
        XCTAssertEqual(s.modules.first { $0.id == .llm }?.state, .loaded)
    }

    func testUnloadReturnsBytes() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        await gov.register(StubLLMModule(reserveBytes: 400), evictable: false)

        try await gov.ensureLoaded(.llm)
        let reserved = await gov.snapshot().reservedBytes
        XCTAssertEqual(reserved, 400)

        await gov.unload(.llm)
        let s = await gov.snapshot()
        XCTAssertEqual(s.reservedBytes, 0)
        XCTAssertEqual(s.freeBytes, 1_000)
        XCTAssertEqual(s.modules.first { $0.id == .llm }?.state, .unloaded)
    }

    func testConcurrentLoadsCoalesce() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        await gov.register(StubLLMModule(reserveBytes: 400), evictable: false)

        await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { try await gov.ensureLoaded(.llm) }
            }
        }

        let reserved = await gov.snapshot().reservedBytes
        XCTAssertEqual(reserved, 400)
    }

    func testUnknownModuleThrows() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        do {
            try await gov.ensureLoaded(.llm)
            XCTFail("expected moduleNotRegistered")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "module_not_registered")
            XCTAssertEqual(e.httpStatus, 404)
        }
    }

    // MARK: - M5.1 live footprint reconciliation

    /// Scripted process-memory probe: successive calls return the next
    /// value (last repeats). Models before/after each load sample.
    private final class FakeProbe: @unchecked Sendable {
        private var seq: [Int]
        private var i = 0
        init(_ seq: [Int]) { self.seq = seq }
        func next() -> Int {
            defer { i += 1 }
            return seq[Swift.min(i, seq.count - 1)]
        }
    }

    func testReconcilesReservationToObservedFootprint() async throws {
        let probe = FakeProbe([0, 600])  // before=0, after=600
        let gov = MemoryGovernor(
            totalBudgetBytes: 1_000,
            memoryProbe: { probe.next() })
        await gov.register(
            StubLLMModule(reserveBytes: 400), evictable: false)

        try await gov.ensureLoaded(.llm)

        let s = await gov.snapshot()
        // Estimate was 400; real footprint 600 ⇒ reservation reconciled.
        XCTAssertEqual(s.reservedBytes, 600)
        XCTAssertEqual(s.freeBytes, 400)
        XCTAssertEqual(
            s.modules.first { $0.id == .llm }?.reservedBytes, 600)
    }

    func testOverBudgetReconciliationEvictsEvictable() async throws {
        // t: before0/after60 (obs 60 == est). llm: before60/after150
        // (obs 90 vs est 30) ⇒ reserved 60+90=150 > 100 ⇒ evict t.
        let probe = FakeProbe([0, 60, 60, 150])
        let gov = MemoryGovernor(
            totalBudgetBytes: 100, memoryProbe: { probe.next() })
        await gov.register(
            StubTranscriptionModule(reserveBytes: 60), evictable: true)
        await gov.register(
            StubLLMModule(reserveBytes: 30), evictable: false)

        try await gov.ensureLoaded(.transcription)
        try await gov.ensureLoaded(.llm)

        let s = await gov.snapshot()
        XCTAssertEqual(s.reservedBytes, 90)
        XCTAssertEqual(
            s.modules.first { $0.id == .llm }?.state, .loaded)
        let v = s.modules.first { $0.id == .transcription }?.state
        XCTAssertTrue(v == .unloading || v == .unloaded)
    }
}
