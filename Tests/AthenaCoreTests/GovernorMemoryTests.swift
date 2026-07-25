import XCTest

@testable import AthenaCore

/// G1 (ADR 023) — the serve-path MLX cache-limit resolution is MLX-free, so it
/// runs in CI (ADR 009). Pins the default-fraction / explicit-override /
/// 0-means-unbounded rules.
final class GovernorMemoryTests: XCTestCase {
    private let budget = 96 * 1024 * 1024 * 1024  // 96 GiB

    func testDefaultIsBudgetFraction() {
        XCTAssertEqual(
            GovernorMemory.resolveCacheLimit(configured: nil, budgetBytes: budget),
            budget / 3)
    }

    func testCustomFractionDenominator() {
        XCTAssertEqual(
            GovernorMemory.resolveCacheLimit(
                configured: nil, budgetBytes: budget, fractionDenominator: 4),
            budget / 4)
    }

    func testConfiguredValueWins() {
        let lim = 8 * 1024 * 1024 * 1024
        XCTAssertEqual(
            GovernorMemory.resolveCacheLimit(
                configured: lim, budgetBytes: budget),
            lim)
    }

    func testZeroMeansUnbounded() {
        XCTAssertNil(
            GovernorMemory.resolveCacheLimit(configured: 0, budgetBytes: budget))
    }

    func testNegativeMeansUnbounded() {
        XCTAssertNil(
            GovernorMemory.resolveCacheLimit(configured: -1, budgetBytes: budget))
    }

    func testNoBudgetNoConfigLeavesDefault() {
        XCTAssertNil(
            GovernorMemory.resolveCacheLimit(configured: nil, budgetBytes: 0))
    }

    func testConfiguredWinsEvenWithZeroBudget() {
        XCTAssertEqual(
            GovernorMemory.resolveCacheLimit(configured: 1234, budgetBytes: 0),
            1234)
    }

    // MARK: - ADR 030 — default prefill prompt-token ceiling

    func testPromptCeilingScalesWithDeviceBuffer() {
        // sqrt(maxBuffer / bytesPerScoreElem). 72 GiB / 128 ⇒ ~23.7k.
        let buf = 72 * 1024 * 1024 * 1024
        let ceil = GovernorMemory.defaultPromptTokenCeiling(maxBufferBytes: buf)
        let expected = Int((Double(buf) / 128.0).squareRoot())
        XCTAssertEqual(ceil, expected)
        // The conservative default must sit safely below the observed 61k-token
        // abort on an ~80 GiB-class device.
        XCTAssertLessThan(ceil, 61_000)
    }

    func testPromptCeilingUnknownDeviceUsesSafeFloor() {
        XCTAssertEqual(
            GovernorMemory.defaultPromptTokenCeiling(maxBufferBytes: 0), 16_384)
        XCTAssertEqual(
            GovernorMemory.defaultPromptTokenCeiling(maxBufferBytes: -1), 16_384)
    }

    func testPromptCeilingNeverBelowMinForTinyDevice() {
        // A pathologically tiny buffer still permits a usable prompt.
        XCTAssertEqual(
            GovernorMemory.defaultPromptTokenCeiling(maxBufferBytes: 1024), 4_096)
    }

    func testPromptCeilingLargerDeviceAllowsLongerPrompt() {
        let small = GovernorMemory.defaultPromptTokenCeiling(
            maxBufferBytes: 16 * 1024 * 1024 * 1024)
        let large = GovernorMemory.defaultPromptTokenCeiling(
            maxBufferBytes: 128 * 1024 * 1024 * 1024)
        XCTAssertGreaterThan(large, small)
    }

    // MARK: - ADR 042 — the EFFECTIVE ceiling published on /v1/models

    /// Unset ⇒ the device-derived default, i.e. the published number is the
    /// same one the decode path enforces.
    func testEffectiveCeilingUnsetUsesDerivedDefault() {
        let buf = 48 * 1024 * 1024 * 1024
        XCTAssertEqual(
            GovernorMemory.effectivePromptTokenCeiling(
                configured: nil, maxBufferBytes: buf),
            GovernorMemory.defaultPromptTokenCeiling(maxBufferBytes: buf))
    }

    func testEffectiveCeilingConfiguredWins() {
        XCTAssertEqual(
            GovernorMemory.effectivePromptTokenCeiling(
                configured: 96_000,
                maxBufferBytes: 48 * 1024 * 1024 * 1024),
            96_000)
    }

    /// Explicit `0` (or negative) is the unbounded opt-out — nil, so the wire
    /// field is OMITTED. Absent must never be read as a zero-token ceiling.
    func testEffectiveCeilingZeroIsUnboundedNotDerived() {
        XCTAssertNil(
            GovernorMemory.effectivePromptTokenCeiling(
                configured: 0, maxBufferBytes: 48 * 1024 * 1024 * 1024))
        XCTAssertNil(
            GovernorMemory.effectivePromptTokenCeiling(
                configured: -1, maxBufferBytes: 48 * 1024 * 1024 * 1024))
    }

    /// Unknown device + unset config still publishes the safe floor, never nil
    /// (a client must not read "unbounded" off a device probe failure).
    func testEffectiveCeilingUnknownDeviceStillBounded() {
        XCTAssertEqual(
            GovernorMemory.effectivePromptTokenCeiling(
                configured: nil, maxBufferBytes: 0),
            16_384)
    }

    // MARK: - G2 — admission against the real footprint (ADR 023)

    func testAdmissionModeParseDefaultsToFootprint() {
        XCTAssertEqual(GovernorMemory.AdmissionMode.parse(nil), .footprint)
        XCTAssertEqual(GovernorMemory.AdmissionMode.parse(""), .footprint)
        XCTAssertEqual(GovernorMemory.AdmissionMode.parse("bogus"), .footprint)
        XCTAssertEqual(GovernorMemory.AdmissionMode.parse("footprint"), .footprint)
        XCTAssertEqual(GovernorMemory.AdmissionMode.parse("FOOTPRINT"), .footprint)
    }

    func testAdmissionModeParseEstimateRevertSwitch() {
        XCTAssertEqual(GovernorMemory.AdmissionMode.parse("estimate"), .estimate)
        XCTAssertEqual(GovernorMemory.AdmissionMode.parse("Estimate"), .estimate)
    }

    func testCommittedExcludesReclaimableCache() {
        // phys_footprint 96 GiB, 79 GiB of it reclaimable cache (the field
        // finding) ⇒ committed is the genuinely-pinned ~17 GiB, not 96.
        let phys = 96 * 1024 * 1024 * 1024
        let cache = 79 * 1024 * 1024 * 1024
        XCTAssertEqual(
            GovernorMemory.committedBytes(physFootprint: phys, reclaimableCache: cache),
            phys - cache)
    }

    func testCommittedClampsNonNegativeOnProbeRace() {
        // A momentary probe race (cache reported larger than footprint) must
        // not yield a negative committed.
        XCTAssertEqual(
            GovernorMemory.committedBytes(physFootprint: 10, reclaimableCache: 25),
            0)
    }

    func testFootprintDenominatorTakesLiveCeilingOverReservation() {
        // Warm steady state: committed (live, incl. ungoverned growth) exceeds
        // the reservation sum ⇒ the live ceiling wins, so admission sees the
        // real footprint the pre-G2 estimate math was blind to.
        let committed = 60 * 1024 * 1024 * 1024
        let reserved = 40 * 1024 * 1024 * 1024
        XCTAssertEqual(
            GovernorMemory.admissionDenominator(
                mode: .footprint, committed: committed, reserved: reserved),
            committed)
    }

    func testFootprintDenominatorFloorsOnReservationDuringWarmup() {
        // Just-loaded-but-cold model: its mmap'd weights haven't faulted in, so
        // committed under-reads while the reservation already booked the full
        // estimate ⇒ the floor (reserved) wins, preventing transient
        // double-admission in the warmup window.
        let committed = 18 * 1024 * 1024 * 1024
        let reserved = 40 * 1024 * 1024 * 1024
        XCTAssertEqual(
            GovernorMemory.admissionDenominator(
                mode: .footprint, committed: committed, reserved: reserved),
            reserved)
    }

    func testEstimateModeIgnoresCommitted() {
        // The revert switch / probe-nil fallback: denominator is the reservation
        // sum regardless of the live footprint — byte-identical to pre-G2.
        let committed = 90 * 1024 * 1024 * 1024
        let reserved = 40 * 1024 * 1024 * 1024
        XCTAssertEqual(
            GovernorMemory.admissionDenominator(
                mode: .estimate, committed: committed, reserved: reserved),
            reserved)
    }

    func testFreeBytesAndFits() {
        let budget = 96 * 1024 * 1024 * 1024
        let denom = 80 * 1024 * 1024 * 1024
        XCTAssertEqual(
            GovernorMemory.freeBytes(budget: budget, denominator: denom),
            16 * 1024 * 1024 * 1024)
        // Fits exactly to the budget edge.
        XCTAssertTrue(
            GovernorMemory.fits(
                request: 16 * 1024 * 1024 * 1024, denominator: denom, budget: budget))
        // One byte over the edge does not fit.
        XCTAssertFalse(
            GovernorMemory.fits(
                request: 16 * 1024 * 1024 * 1024 + 1, denominator: denom, budget: budget))
    }

    func testFreeBytesClampsNonNegativeOverBudget() {
        // An over-budget live footprint reports 0 free, never a negative.
        XCTAssertEqual(
            GovernorMemory.freeBytes(budget: 10, denominator: 25), 0)
    }

    // MARK: - ADR 039 — per-sequence KV admission

    func testSequenceKVReservationIsWorstCaseProduct() {
        XCTAssertEqual(
            GovernorMemory.sequenceKVReservation(
                maxTokens: 150, perTokenKVBytes: 256 * 1024),
            150 * 256 * 1024)
    }

    func testSequenceKVReservationClampsNegatives() {
        XCTAssertEqual(
            GovernorMemory.sequenceKVReservation(maxTokens: -5, perTokenKVBytes: 100), 0)
        XCTAssertEqual(
            GovernorMemory.sequenceKVReservation(maxTokens: 100, perTokenKVBytes: -1), 0)
    }

    func testAdmitsSequenceStacksOnDenominatorPlusActiveKV() {
        // budget 100; model denominator 40; batch already holds 30 of KV.
        // A 30-byte row fits exactly (40+30+30=100); 31 does not.
        XCTAssertTrue(
            GovernorMemory.admitsSequence(
                rowKVBytes: 30, activeSequenceKVBytes: 30, denominator: 40, budget: 100))
        XCTAssertFalse(
            GovernorMemory.admitsSequence(
                rowKVBytes: 31, activeSequenceKVBytes: 30, denominator: 40, budget: 100))
    }

    // MARK: - SequenceKVLedger (ADR 039)

    func testLedgerAdmitsUntilBudgetThenDenies() {
        // budget 100, denominator 10 → 90 of KV headroom. Rows of 40 each:
        // two admit (10+40+40=90 ≤ 100), the third is denied.
        var ledger = SequenceKVLedger()
        XCTAssertTrue(ledger.admit(uid: 1, rowKVBytes: 40, denominator: 10, budget: 100))
        XCTAssertTrue(ledger.admit(uid: 2, rowKVBytes: 40, denominator: 10, budget: 100))
        XCTAssertFalse(
            ledger.admit(uid: 3, rowKVBytes: 40, denominator: 10, budget: 100),
            "third row overcommits the budget and must be denied")
        XCTAssertEqual(ledger.activeCount, 2)
        XCTAssertEqual(ledger.totalReservedBytes, 80)
    }

    func testLedgerReleaseFreesBudgetForNextRow() {
        var ledger = SequenceKVLedger()
        XCTAssertTrue(ledger.admit(uid: 1, rowKVBytes: 40, denominator: 10, budget: 100))
        XCTAssertTrue(ledger.admit(uid: 2, rowKVBytes: 40, denominator: 10, budget: 100))
        XCTAssertFalse(ledger.admit(uid: 3, rowKVBytes: 40, denominator: 10, budget: 100))
        // Row 1 finishes → 40 bytes free → row 3 now admits.
        XCTAssertEqual(ledger.release(uid: 1), 40)
        XCTAssertTrue(ledger.admit(uid: 3, rowKVBytes: 40, denominator: 10, budget: 100))
        XCTAssertEqual(ledger.activeCount, 2)  // rows 2 and 3
    }

    func testLedgerDeniedRowReservesNothing() {
        var ledger = SequenceKVLedger()
        XCTAssertTrue(ledger.admit(uid: 1, rowKVBytes: 90, denominator: 10, budget: 100))
        XCTAssertFalse(ledger.admit(uid: 2, rowKVBytes: 1, denominator: 10, budget: 100))
        XCTAssertEqual(ledger.totalReservedBytes, 90, "a denied row must not be recorded")
        XCTAssertEqual(ledger.activeCount, 1)
    }

    func testLedgerAdmitIsIdempotentPerUID() {
        var ledger = SequenceKVLedger()
        XCTAssertTrue(ledger.admit(uid: 1, rowKVBytes: 40, denominator: 10, budget: 100))
        XCTAssertTrue(ledger.admit(uid: 1, rowKVBytes: 40, denominator: 10, budget: 100))
        XCTAssertEqual(ledger.totalReservedBytes, 40, "re-admitting a uid must not double-count")
        XCTAssertEqual(ledger.release(uid: 999), 0, "releasing an unknown uid frees nothing")
    }
}
