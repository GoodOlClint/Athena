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
        XCTAssertEqual(s.residentBytes, 400)
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
        XCTAssertEqual(s.residentBytes, 0)
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
        XCTAssertEqual(s.residentBytes, 60)
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
        XCTAssertEqual(s.residentBytes, 80)
        XCTAssertEqual(s.modules.first { $0.id == .llm }?.state, .loaded)
    }

    func testUnloadReturnsBytes() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        await gov.register(StubLLMModule(reserveBytes: 400), evictable: false)

        try await gov.ensureLoaded(.llm)
        let reserved = await gov.snapshot().residentBytes
        XCTAssertEqual(reserved, 400)

        await gov.unload(.llm)
        let s = await gov.snapshot()
        XCTAssertEqual(s.residentBytes, 0)
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

        let reserved = await gov.snapshot().residentBytes
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

    // Note: each ensureLoaded reads the probe 3×: relievePressure,
    // then performLoad before + after.
    func testReconcilesReservationToObservedFootprint() async throws {
        // relief=0 (no relief), before=0, after=600.
        let probe = FakeProbe([0, 0, 600])
        let gov = MemoryGovernor(
            totalBudgetBytes: 1_000,
            memoryProbe: { probe.next() })
        await gov.register(
            StubLLMModule(reserveBytes: 400), evictable: false)

        try await gov.ensureLoaded(.llm)

        let s = await gov.snapshot()
        // Estimate was 400; real footprint 600 ⇒ reservation reconciled.
        XCTAssertEqual(s.residentBytes, 600)
        XCTAssertEqual(s.freeBytes, 400)
        XCTAssertEqual(
            s.modules.first { $0.id == .llm }?.residentBytes, 600)
    }

    func testOverBudgetReconciliationEvictsEvictable() async throws {
        // 3 reads/load. t: relief0, before0, after60 (obs 60 == est).
        // llm: relief0 (≤90 hi-water, no relief), before60, after150
        // (obs 90 vs est 30) ⇒ reserved 60+90=150 > 100 ⇒ evict t.
        let probe = FakeProbe([0, 0, 60, 0, 60, 150])
        let gov = MemoryGovernor(
            totalBudgetBytes: 100, memoryProbe: { probe.next() })
        await gov.register(
            StubTranscriptionModule(reserveBytes: 60), evictable: true)
        await gov.register(
            StubLLMModule(reserveBytes: 30), evictable: false)

        try await gov.ensureLoaded(.transcription)
        try await gov.ensureLoaded(.llm)

        let s = await gov.snapshot()
        XCTAssertEqual(s.residentBytes, 90)
        XCTAssertEqual(
            s.modules.first { $0.id == .llm }?.state, .loaded)
        let v = s.modules.first { $0.id == .transcription }?.state
        XCTAssertTrue(v == .unloading || v == .unloaded)
    }

    // MARK: - M5.2 unload hook (trim substrate cache)

    private final class Counter: @unchecked Sendable {
        private(set) var n = 0
        func bump() { n += 1 }
    }

    func testUnloadHookFiresOnEvictionAndExplicitUnload() async throws {
        let c = Counter()
        let gov = MemoryGovernor(
            totalBudgetBytes: 100, onUnloaded: { c.bump() })
        await gov.register(
            StubTranscriptionModule(reserveBytes: 60), evictable: true)
        await gov.register(
            StubEmbeddingModule(reserveBytes: 60), evictable: true)

        try await gov.ensureLoaded(.transcription)
        try await gov.ensureLoaded(.textEmbedding)  // evicts transcription
        // eviction unload is detached; wait for it to settle.
        for _ in 0..<50 where c.n == 0 {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(c.n, 1, "hook should fire on eviction")

        await gov.unload(.textEmbedding)
        XCTAssertEqual(c.n, 2, "hook should fire on explicit unload")
    }

    // MARK: - M5.3 proactive pressure relief (live probe)

    func testLivePressureShedsLRUEvenWhenBookkeepingHasRoom() async throws
    {
        // Constant-high probe: 950 > 90% of 1000 ⇒ relief triggers.
        // before==after ⇒ reconcile observes 0 ⇒ no-op (isolated test).
        let gov = MemoryGovernor(
            totalBudgetBytes: 1_000,
            memoryProbe: { 950 })
        await gov.register(
            StubTranscriptionModule(reserveBytes: 60), evictable: true)
        await gov.register(
            StubEmbeddingModule(reserveBytes: 60), evictable: true)

        try await gov.ensureLoaded(.transcription)
        // Bookkeeping has ample room (60+60 ≪ 1000) but live memory is
        // over high-water, so loading embedding sheds the LRU evictable.
        try await gov.ensureLoaded(.textEmbedding)
        for _ in 0..<50 {
            let st = await gov.snapshot().modules
                .first { $0.id == .transcription }?.state
            if st == .unloaded { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        let s = await gov.snapshot()
        let t = s.modules.first { $0.id == .transcription }?.state
        XCTAssertTrue(
            t == .unloading || t == .unloaded,
            "live pressure should have shed transcription")
        XCTAssertEqual(
            s.modules.first { $0.id == .textEmbedding }?.state, .loaded)
    }

    // MARK: - M5.4 learned footprints

    /// Shared "process memory" the fake modules allocate into.
    private final class MemBox: @unchecked Sendable {
        private(set) var value = 0
        func add(_ n: Int) { value += n }
    }

    /// Fake module that actually moves the probe: `load` allocates
    /// `footprint` into the shared box (deterministic — no scripted
    /// sequence). `memoryEstimate` is the deliberately-wrong static
    /// guess.
    private actor AllocatingModule: InferenceModule {
        nonisolated let id: ModuleID
        private let estimate: Int
        private let footprint: Int
        private let box: MemBox
        private var loaded = false
        init(
            id: ModuleID, estimate: Int, footprint: Int, box: MemBox
        ) {
            self.id = id
            self.estimate = estimate
            self.footprint = footprint
            self.box = box
        }
        var residentBytes: Int { loaded ? footprint : 0 }
        func memoryEstimate() -> Int { estimate }
        func load(reservation: MemoryReservation) async throws {
            box.add(footprint)
            loaded = true
        }
        func unload() async { loaded = false }
    }

    func testReadmissionUsesLearnedFootprintNotEstimate() async throws {
        let box = MemBox()
        let gov = MemoryGovernor(
            totalBudgetBytes: 1_000, memoryProbe: { box.value })
        // Static estimate 100, but loading really costs 600.
        await gov.register(
            AllocatingModule(
                id: .transcription, estimate: 100, footprint: 600,
                box: box), evictable: true)
        // Non-evictable blocker that reserves 500 (no extra alloc).
        await gov.register(
            AllocatingModule(
                id: .llm, estimate: 500, footprint: 0, box: box),
            evictable: false)

        try await gov.ensureLoaded(.transcription)  // observes 600
        let reserved1 = await gov.snapshot().residentBytes
        XCTAssertEqual(reserved1, 600)
        await gov.unload(.transcription)  // learned[.transcription]=600
        try await gov.ensureLoaded(.llm)  // reserves 500

        // Re-admit transcription. With the stale estimate (100) it
        // would fit (500+100 ≤ 1000); with the LEARNED 600 it must be
        // rejected (500+600 > 1000, blocker non-evictable).
        do {
            try await gov.ensureLoaded(.transcription)
            XCTFail("expected memoryBudgetExceeded (learned 600)")
        } catch let e as AthenaError {
            guard
                case let .memoryBudgetExceeded(requested, _, _) = e
            else { return XCTFail("wrong error: \(e)") }
            XCTAssertEqual(
                requested, 600,
                "admission must use the learned footprint, not 100")
        }
    }

    // MARK: - M62 cold-load model selection + failure surfacing

    /// A cold governor load must bind the model chosen via
    /// `selectColdLoadModel`, not the slot default — pre-M62 a request for a
    /// non-default model on a cold/just-restarted slot silently served the
    /// default (the consuming application asked for 4bit, got the 8bit default).
    func testColdLoadBindsSelectedModelNotDefault() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let stub = StubLLMModule(
            reserveBytes: 100, modelIds: ["model-A", "model-B"])
        await gov.register(stub, evictable: false)

        try await stub.selectColdLoadModel("model-B")  // non-default
        try await gov.ensureLoaded(.llm)

        let resident = await stub.residentModelId()
        XCTAssertEqual(resident, "model-B")
    }

    func testColdLoadUsesDefaultWhenSelectionCleared() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let stub = StubLLMModule(
            reserveBytes: 100, modelIds: ["model-A", "model-B"])
        await gov.register(stub, evictable: false)

        try await stub.selectColdLoadModel("model-B")
        try await stub.selectColdLoadModel(nil)  // clear ⇒ default
        try await gov.ensureLoaded(.llm)

        let resident = await stub.residentModelId()
        XCTAssertEqual(resident, "model-A")
    }

    func testSelectColdLoadModelRejectsUnknownId() async throws {
        let stub = StubLLMModule(modelIds: ["model-A"])
        do {
            try await stub.selectColdLoadModel("not-allowed")
            XCTFail("expected modelNotAvailable")
        } catch let e as AthenaError {
            guard case .modelNotAvailable = e else {
                return XCTFail("wrong error: \(e)")
            }
            XCTAssertEqual(e.httpStatus, 400)
        }
    }

    func testSelectColdLoadModelIsCaseInsensitiveCanonical() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let stub = StubLLMModule(
            reserveBytes: 100, modelIds: ["Model-B", "Model-A"])
        await gov.register(stub, evictable: false)

        try await stub.selectColdLoadModel("model-b")  // lowercased request
        try await gov.ensureLoaded(.llm)

        // The CANONICAL (stored) casing is bound, not the request's.
        let resident = await stub.residentModelId()
        XCTAssertEqual(resident, "Model-B")
    }

    /// A failed cold-load must surface the real error to the next
    /// non-blocking caller — not silently kick another doomed load and 503
    /// `module_loading` forever (a model dir missing config.json did exactly
    /// that pre-M62).
    func testColdLoadFailureSurfacedNotPerpetualLoading() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        await gov.register(FailingModule(id: .llm), evictable: false)

        // First call kicks the doomed background load → .loading.
        let first = try await gov.beginLoadIfNeeded(.llm)
        guard case .loading = first else {
            return XCTFail("first cold-load should be .loading")
        }

        // While the load is in flight it stays .loading; once it fails, the
        // next call THROWS the real error instead of another .loading.
        var surfaced: Error?
        for _ in 0..<100 {
            do {
                let s = try await gov.beginLoadIfNeeded(.llm)
                guard case .loading = s else {
                    return XCTFail("unexpected non-loading status")
                }
                try await Task.sleep(nanoseconds: 10_000_000)
            } catch {
                surfaced = error
                break
            }
        }
        let e = try XCTUnwrap(
            surfaced as? AthenaError, "the load failure should surface")
        guard case .moduleLoadFailed = e else {
            return XCTFail("expected moduleLoadFailed, got \(e)")
        }
    }

    /// Module whose `load` always throws — drives the M62 failure-surfacing
    /// path in `beginLoadIfNeeded`.
    private actor FailingModule: InferenceModule {
        nonisolated let id: ModuleID
        init(id: ModuleID) { self.id = id }
        var residentBytes: Int { 0 }
        func memoryEstimate() -> Int { 10 }
        func load(reservation: MemoryReservation) async throws {
            throw AthenaError.moduleLoadFailed(id, reason: "test boom")
        }
        func unload() async {}
    }

    // MARK: - M68.1 concurrency & lifecycle

    /// A module with a deliberately SLOW `unload()`, so the teardown window
    /// stays open while a concurrent reload runs — the NE1 race condition.
    /// `isLoaded` is the ground truth the test asserts against: if `load()`
    /// and the pending slow `unload()` race, the unload lands last and leaves
    /// `isLoaded == false` while the governor records `.loaded`.
    private actor SlowUnloadModule: InferenceModule {
        nonisolated let id: ModuleID
        private let bytes: Int
        private let unloadDelayNs: UInt64
        private(set) var isLoaded = false
        init(id: ModuleID, bytes: Int, unloadDelayNs: UInt64) {
            self.id = id
            self.bytes = bytes
            self.unloadDelayNs = unloadDelayNs
        }
        var residentBytes: Int { isLoaded ? bytes : 0 }
        func memoryEstimate() -> Int { bytes }
        func load(reservation: MemoryReservation) async throws {
            isLoaded = true
        }
        func unload() async {
            try? await Task.sleep(nanoseconds: unloadDelayNs)
            isLoaded = false
        }
        func loaded() -> Bool { isLoaded }
    }

    /// NE1 — a slot re-requested DURING its eviction teardown must wait for
    /// the pending `unload()` before re-loading, so `load()` doesn't race the
    /// still-running `unload()` on the module actor and end up torn down while
    /// the governor records it `.loaded`.
    func testReloadDuringTeardownWaitsForUnloadNE1() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 100)
        let a = SlowUnloadModule(
            id: .transcription, bytes: 60, unloadDelayNs: 80_000_000)
        let b = SlowUnloadModule(
            id: .textEmbedding, bytes: 60, unloadDelayNs: 80_000_000)
        await gov.register(a, evictable: true)
        await gov.register(b, evictable: true)

        try await gov.ensureLoaded(.transcription)  // A loaded (60)
        try await gov.ensureLoaded(.textEmbedding)  // evicts A (slow), B loaded
        // Re-request A while A's slow unload is still in flight. With the fix
        // performLoad awaits A's teardown before A.load(); without it, A.load
        // runs first and the trailing unload flips isLoaded back to false.
        try await gov.ensureLoaded(.transcription)  // evicts B, reloads A

        let aLoaded = await a.loaded()
        XCTAssertTrue(
            aLoaded,
            "A must be genuinely loaded — the teardown must have completed "
                + "before the reload")
        let s = await gov.snapshot()
        XCTAssertEqual(
            s.modules.first { $0.id == .transcription }?.state, .loaded)
        XCTAssertEqual(
            s.modules.first { $0.id == .transcription }?.residentBytes, 60)
    }

    /// NE1 (mirror) — an explicit `unload(_:)` racing a reload must not clobber
    /// the freshly-loaded slot back to `.unloaded`. The reload awaits the
    /// unload teardown, then loads; the final state is `.loaded`.
    func testExplicitUnloadThenReloadEndsLoadedNE1() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = SlowUnloadModule(
            id: .llm, bytes: 100, unloadDelayNs: 60_000_000)
        await gov.register(m, evictable: false)

        try await gov.ensureLoaded(.llm)
        // Kick the slow unload, then wait until the slot has actually entered
        // its teardown (`.unloading`) before reloading — otherwise the reload
        // could take the `.loaded` fast path and never exercise the drain.
        async let unloadDone: Void = gov.unload(.llm)
        for _ in 0..<500 {
            let st = await gov.snapshot().modules
                .first { $0.id == .llm }?.state
            if st == .unloading { break }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        // Reload while the slow `unload()` is mid-flight: the reload must drain
        // the teardown, then load — and the trailing markUnloaded must NOT
        // clobber the freshly-loaded slot back to `.unloaded`.
        try await gov.ensureLoaded(.llm)
        _ = await unloadDone

        let mLoaded = await m.loaded()
        XCTAssertTrue(mLoaded, "reload must win the final state")
        let s = await gov.snapshot()
        XCTAssertEqual(s.modules.first { $0.id == .llm }?.state, .loaded)
        XCTAssertEqual(s.residentBytes, 100)
    }

    /// NE2 — an admission (makeRoom) failure on the non-blocking path must be
    /// surfaced to the next caller as the real `memory_budget_exceeded`, not a
    /// perpetual `module_loading` 503 (pre-fix, only `module.load()` throws
    /// were recorded in `lastLoadError`; the makeRoom throw was swallowed).
    func testAdmissionFailureSurfacedNotPerpetualLoadingNE2() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 50)
        await gov.register(StubLLMModule(reserveBytes: 100), evictable: false)

        let first = try await gov.beginLoadIfNeeded(.llm)
        guard case .loading = first else {
            return XCTFail("first cold-load should be .loading")
        }
        var surfaced: Error?
        for _ in 0..<100 {
            do {
                let s = try await gov.beginLoadIfNeeded(.llm)
                guard case .loading = s else {
                    return XCTFail("unexpected non-loading status")
                }
                try await Task.sleep(nanoseconds: 5_000_000)
            } catch {
                surfaced = error
                break
            }
        }
        let e = try XCTUnwrap(
            surfaced as? AthenaError,
            "admission failure must surface, not loop on module_loading")
        guard case .memoryBudgetExceeded = e else {
            return XCTFail("expected memoryBudgetExceeded, got \(e)")
        }
    }

    /// E5 — a non-positive configured budget must clamp to the physical-memory
    /// default rather than refuse every load (a daemon that 503s everything).
    func testNonPositiveBudgetClampsToDefaultE5() async throws {
        for bad in [0, -1, Int.min] {
            let gov = MemoryGovernor(totalBudgetBytes: bad)
            let budget = await gov.snapshot().totalBudgetBytes
            XCTAssertGreaterThan(
                budget, 0, "budget \(bad) should clamp to a positive default")
            await gov.register(
                StubLLMModule(reserveBytes: 100), evictable: false)
            try await gov.ensureLoaded(.llm)
            let state = await gov.snapshot().modules
                .first { $0.id == .llm }?.state
            XCTAssertEqual(state, .loaded)
        }
    }

    /// E12 — when the process-global probe delta is deflated below the static
    /// estimate (a concurrent teardown freed bytes between before/after), the
    /// reconcile must fall back to the module's own resident self-report so
    /// `learnedFootprint` is still recorded — not skipped, leaving the slot
    /// billed at the under-counted estimate.
    func testReconcileFallsBackToSelfReportE12() async throws {
        // relief=0, before=500, after=500 ⇒ probe delta 0 (< estimate 100).
        let probe = FakeProbe([0, 500, 500])
        let gov = MemoryGovernor(
            totalBudgetBytes: 1_000, memoryProbe: { probe.next() })
        // Static estimate 100, but the module self-reports 600 resident.
        await gov.register(
            AllocatingModule(
                id: .transcription, estimate: 100, footprint: 600,
                box: MemBox()),
            evictable: true)

        try await gov.ensureLoaded(.transcription)

        let s = await gov.snapshot()
        // Without the E12 fallback, observed==0 ⇒ reconcile skipped ⇒
        // reservation stays at the 100 estimate. With it, the self-report
        // (600) drives the reconcile.
        XCTAssertEqual(
            s.modules.first { $0.id == .transcription }?.residentBytes, 600,
            "reconcile should bill the self-reported footprint")
        XCTAssertEqual(s.residentBytes, 600)
    }
}
