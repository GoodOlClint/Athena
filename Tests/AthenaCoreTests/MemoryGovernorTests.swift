import AthenaEmbedding
import AthenaLLM
import AthenaTranscription
import Foundation
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

    func testMeasuredFlagReflectsReconcile() async throws {
        // ADR 023 G3 — a load whose probe reconciles (obs 600 vs est 400) marks
        // the module `measured`; a registered-but-never-loaded module stays
        // unmeasured (its number is still the estimate / 0).
        let probe = FakeProbe([0, 0, 600])
        let gov = MemoryGovernor(
            totalBudgetBytes: 10_000, memoryProbe: { probe.next() })
        await gov.register(
            StubLLMModule(reserveBytes: 400), evictable: false)
        await gov.register(
            StubTranscriptionModule(reserveBytes: 200), evictable: true)

        try await gov.ensureLoaded(.llm)  // transcription left unloaded

        let mods = await gov.snapshot().modules
        XCTAssertEqual(
            mods.first { $0.id == .llm }?.measured, true,
            "reconcile fired ⇒ residentBytes is measured")
        XCTAssertEqual(
            mods.first { $0.id == .transcription }?.measured, false,
            "never loaded ⇒ no reconcile ⇒ unmeasured")
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

    /// NL4 — the `onUnloaded` hook fires from a DETACHED Task in
    /// `performEviction` (off-actor), while the test thread polls `n`. The
    /// previous unsynchronized `n += 1` / read was a genuine data race (TSan
    /// would flag it); guard with an NSLock like the suite's other counters.
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var _n = 0
        var n: Int {
            lock.lock()
            defer { lock.unlock() }
            return _n
        }
        func bump() {
            lock.lock()
            _n += 1
            lock.unlock()
        }
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

    // MARK: - M70.3 L9 — coalescing invokes module.load exactly once

    /// A module that counts its `load()` invocations and sleeps briefly so
    /// concurrent first-touch callers pile up on the in-flight load Task.
    private actor CountingModule: InferenceModule {
        nonisolated let id: ModuleID
        private let bytes: Int
        private(set) var loadCount = 0
        private var isLoaded = false
        init(id: ModuleID, bytes: Int) {
            self.id = id
            self.bytes = bytes
        }
        var residentBytes: Int { isLoaded ? bytes : 0 }
        func memoryEstimate() -> Int { bytes }
        func load(reservation: MemoryReservation) async throws {
            loadCount += 1
            // Widen the window so all concurrent callers coalesce on one Task.
            try? await Task.sleep(nanoseconds: 10_000_000)
            isLoaded = true
        }
        func unload() async { isLoaded = false }
        func count() -> Int { loadCount }
    }

    /// L9 — `testConcurrentLoadsCoalesce` asserts the BYTES reserve once; this
    /// pins the stronger invariant the coalescing exists for: the underlying
    /// `module.load()` runs EXACTLY ONCE even under 8 concurrent first-touch
    /// `ensureLoaded` calls (they await the same in-flight Task), so a load is
    /// never duplicated.
    func testConcurrentLoadsInvokeLoadExactlyOnceL9() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = CountingModule(id: .llm, bytes: 400)
        await gov.register(m, evictable: false)

        await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { try await gov.ensureLoaded(.llm) }
            }
        }

        let n = await m.count()
        XCTAssertEqual(
            n, 1, "concurrent first-touch loads coalesce into ONE module.load()")
        let reserved = await gov.snapshot().residentBytes
        XCTAssertEqual(reserved, 400, "reserved once, not 8×")
    }

    // MARK: - M70.3 NL1 — beginLoadIfNeeded success transition + coalescing

    /// NL1 — the only prior `beginLoadIfNeeded` CI test drove the FAILURE path
    /// (FailingModule). Pin the success path the M62 non-blocking serving
    /// entrypoint exists for: the first call returns `.loading` and starts a
    /// detached background load, a concurrent call also returns `.loading`
    /// WITHOUT a second load (in-flight coalescing), and once the background
    /// load completes a later call returns `.loaded`.
    func testBeginLoadIfNeededLoadingThenLoadedNL1() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = CountingModule(id: .llm, bytes: 400)
        await gov.register(m, evictable: false)

        let first = try await gov.beginLoadIfNeeded(.llm)
        XCTAssertEqual(first, .loading, "cold first touch starts a bg load")
        // A concurrent re-request joins the in-flight load (no 2nd load).
        let concurrent = try await gov.beginLoadIfNeeded(.llm)
        XCTAssertEqual(concurrent, .loading, "in-flight coalesces")

        // Poll until the detached load lands and the slot reports .loaded.
        var landed: MemoryGovernor.LoadStatus = .loading
        for _ in 0..<100 {
            landed = try await gov.beginLoadIfNeeded(.llm)
            if landed == .loaded { break }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(landed, .loaded, "slot becomes .loaded after the bg load")
        let n = await m.count()
        XCTAssertEqual(n, 1, "exactly one background load ran")
    }

    /// NL1/NE8 — while an operator-action pull is marked in flight,
    /// `beginLoadIfNeeded` returns `.loading` WITHOUT attempting a load (the
    /// weights aren't local yet); clearing the pull lets the load proceed.
    func testSetPullingShortCircuitsBeginLoad() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = CountingModule(id: .llm, bytes: 400)
        await gov.register(m, evictable: false)

        await gov.setPulling(.llm, true)
        let pulling = try await gov.beginLoadIfNeeded(.llm)
        XCTAssertEqual(pulling, .loading, "pull-in-flight ⇒ .loading")
        let duringPull = await m.count()
        XCTAssertEqual(duringPull, 0, "no load is attempted while pulling")

        await gov.setPulling(.llm, false)
        let after = try await gov.beginLoadIfNeeded(.llm)
        XCTAssertEqual(after, .loading, "clearing the pull starts the load")
    }

    // MARK: - ADR 015 — block-until-ready (awaitLoad) decision algebra

    /// A module whose `load()` sleeps a configurable time, so a test can make
    /// the load OUTLAST a short `awaitLoad` budget (the timeout path) or finish
    /// WITHIN a generous one (the block-until-ready path). Counts loads so the
    /// single-flight invariant is checkable.
    private actor SlowLoadModule: InferenceModule {
        nonisolated let id: ModuleID
        private let bytes: Int
        private let loadDelayNs: UInt64
        private(set) var loadCount = 0
        private var isLoaded = false
        init(id: ModuleID, bytes: Int, loadDelayNs: UInt64) {
            self.id = id
            self.bytes = bytes
            self.loadDelayNs = loadDelayNs
        }
        var residentBytes: Int { isLoaded ? bytes : 0 }
        func memoryEstimate() -> Int { bytes }
        func load(reservation: MemoryReservation) async throws {
            loadCount += 1
            try? await Task.sleep(nanoseconds: loadDelayNs)
            isLoaded = true
        }
        func unload() async { isLoaded = false }
        func count() -> Int { loadCount }
    }

    /// ADR 015 — a cold slot whose local load completes within the budget is
    /// WAITED ON and served `.loaded` in a single call (the peer-runner
    /// "block-until-ready" behavior), with exactly one underlying load.
    func testAwaitLoadBlocksUntilLoaded() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = SlowLoadModule(id: .llm, bytes: 400, loadDelayNs: 30_000_000)
        await gov.register(m, evictable: false)

        let status = try await gov.awaitLoad(.llm, within: 5.0)

        XCTAssertEqual(status, .loaded, "a within-budget load is awaited")
        let count = await m.count()
        XCTAssertEqual(count, 1, "exactly one load ran")
        let state = await gov.snapshot().modules.first { $0.id == .llm }?.state
        XCTAssertEqual(state, .loaded)
    }

    /// ADR 015 — when the load OUTLASTS the budget, `awaitLoad` returns
    /// `.loading` (caller 503s) but does NOT cancel the load: it keeps running
    /// detached so a later call finds the slot resident — and only ONE load ran.
    func testAwaitLoadTimesOutThenLandsLoaded() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = SlowLoadModule(id: .llm, bytes: 400, loadDelayNs: 200_000_000)
        await gov.register(m, evictable: false)

        // Budget far shorter than the 200ms load ⇒ timeout ⇒ .loading.
        let timed = try await gov.awaitLoad(.llm, within: 0.02)
        XCTAssertEqual(timed, .loading, "budget elapsed ⇒ .loading (503)")

        // The detached load was NOT cancelled; poll until it lands .loaded.
        var landed: MemoryGovernor.LoadStatus = .loading
        for _ in 0..<100 {
            landed = try await gov.awaitLoad(.llm, within: 0.05)
            if landed == .loaded { break }
        }
        XCTAssertEqual(landed, .loaded, "the un-cancelled load eventually lands")
        let count = await m.count()
        XCTAssertEqual(count, 1, "the timeout did not start a second load")
    }

    /// ADR 015 — `within: 0` is the revert switch: behave exactly like the
    /// legacy non-blocking gate (start the load, return `.loading` at once).
    func testAwaitLoadZeroBudgetIsImmediateLoading() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = SlowLoadModule(id: .llm, bytes: 400, loadDelayNs: 30_000_000)
        await gov.register(m, evictable: false)

        let status = try await gov.awaitLoad(.llm, within: 0)
        XCTAssertEqual(status, .loading, "zero budget ⇒ immediate .loading")
        // It still kicked the load off (legacy beginLoadIfNeeded semantics).
        var landed: MemoryGovernor.LoadStatus = .loading
        for _ in 0..<100 {
            landed = try await gov.awaitLoad(.llm, within: 0)
            if landed == .loaded { break }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(landed, .loaded)
        let count = await m.count()
        XCTAssertEqual(count, 1)
    }

    /// ADR 015 — an in-flight operator pull (download) is NOT waited on:
    /// `awaitLoad` returns `.loading` immediately and attempts no load, even
    /// with a generous budget (blocking on a multi-GB download is exactly what
    /// M43.2 forbids).
    func testAwaitLoadDoesNotWaitOnPull() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = SlowLoadModule(id: .llm, bytes: 400, loadDelayNs: 30_000_000)
        await gov.register(m, evictable: false)

        await gov.setPulling(.llm, true)
        let start = Date()
        let status = try await gov.awaitLoad(.llm, within: 10.0)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(status, .loading, "pull-in-flight ⇒ .loading")
        XCTAssertLessThan(elapsed, 1.0, "must return at once, not wait the budget")
        let count = await m.count()
        XCTAssertEqual(count, 0, "no load attempted while pulling")
    }

    /// ADR 015 — a failing local load surfaces its REAL error within the same
    /// blocking call (the poll loop observes `lastLoadError`), not a perpetual
    /// `module_loading`.
    func testAwaitLoadSurfacesLoadFailure() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        await gov.register(FailingModule(id: .llm), evictable: false)

        do {
            _ = try await gov.awaitLoad(.llm, within: 5.0)
            XCTFail("expected the load failure to surface")
        } catch let e as AthenaError {
            guard case .moduleLoadFailed = e else {
                return XCTFail("expected moduleLoadFailed, got \(e)")
            }
        }
    }

    /// ADR 015 — over-budget admission surfaces `memory_budget_exceeded` from
    /// the blocking call too (not a wait, not a perpetual loading).
    func testAwaitLoadSurfacesAdmissionFailure() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 50)
        await gov.register(StubLLMModule(reserveBytes: 100), evictable: false)

        do {
            _ = try await gov.awaitLoad(.llm, within: 5.0)
            XCTFail("expected memoryBudgetExceeded")
        } catch let e as AthenaError {
            guard case .memoryBudgetExceeded = e else {
                return XCTFail("expected memoryBudgetExceeded, got \(e)")
            }
        }
    }

    /// ADR 015 — `peekLoad` is a non-mutating disposition read: a cold slot is
    /// `.needsLoad` (and stays cold — no load kicked off), a pull is `.pulling`,
    /// a hot slot is `.loaded`.
    func testPeekLoadDispositions() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = SlowLoadModule(id: .llm, bytes: 400, loadDelayNs: 30_000_000)
        await gov.register(m, evictable: false)

        let cold = await gov.peekLoad(.llm)
        XCTAssertEqual(cold, .needsLoad, "cold slot ⇒ needsLoad")
        let kicked = await m.count()
        XCTAssertEqual(kicked, 0, "peek must not start a load")

        await gov.setPulling(.llm, true)
        let pulling = await gov.peekLoad(.llm)
        XCTAssertEqual(pulling, .pulling, "pull in flight ⇒ pulling")
        await gov.setPulling(.llm, false)

        try await gov.ensureLoaded(.llm)
        let hot = await gov.peekLoad(.llm)
        XCTAssertEqual(hot, .loaded, "resident slot ⇒ loaded")
    }

    /// ADR 015 — an already-resident slot returns `.loaded` immediately with no
    /// new load (the hot fast-path is unchanged).
    func testAwaitLoadHotSlotReturnsImmediately() async throws {
        let gov = MemoryGovernor(totalBudgetBytes: 1_000)
        let m = SlowLoadModule(id: .llm, bytes: 400, loadDelayNs: 10_000_000)
        await gov.register(m, evictable: false)

        try await gov.ensureLoaded(.llm)  // make it hot
        let status = try await gov.awaitLoad(.llm, within: 5.0)

        XCTAssertEqual(status, .loaded)
        let count = await m.count()
        XCTAssertEqual(count, 1, "hot slot triggers no second load")
    }
}
