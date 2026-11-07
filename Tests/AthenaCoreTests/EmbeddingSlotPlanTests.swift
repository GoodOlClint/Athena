import XCTest

@testable import AthenaEmbedding

/// #236 — the embedding slot's model choice at the decision seam. Each test
/// replays the governor's `selectColdLoadModel` → `load(reservation:)` →
/// success/failure sequence without a Metal device.
final class EmbeddingSlotPlanTests: XCTestCase {
    private let dflt = "default-model"

    /// What the module does for one request on a cold slot.
    private func coldRequest(
        _ plan: inout EmbeddingSlotPlan, model: String, loads: Set<String>
    ) -> String? {
        plan.stageColdTarget(model)
        let target = plan.loadTarget(defaultModel: { dflt })
        guard loads.contains(target) else {
            plan.loadFailed()
            return nil
        }
        plan.loadSucceeded(target)
        return target
    }

    func testColdLoadBindsRequestedModelNotDefault() {
        var plan = EmbeddingSlotPlan()
        XCTAssertEqual(
            coldRequest(&plan, model: "bge", loads: ["bge", dflt]), "bge")
    }

    func testFailedModelDoesNotWedgeLaterRequests() {
        var plan = EmbeddingSlotPlan()
        let ok: Set = ["bge", "gemma2"]
        XCTAssertEqual(coldRequest(&plan, model: "bge", loads: ok), "bge")
        XCTAssertNil(coldRequest(&plan, model: "gemma300", loads: ok))
        XCTAssertEqual(coldRequest(&plan, model: "gemma2", loads: ok), "gemma2")
        XCTAssertEqual(coldRequest(&plan, model: "bge", loads: ok), "bge")
        XCTAssertEqual(coldRequest(&plan, model: "gemma2", loads: ok), "gemma2")
    }

    func testFailureOnFirstRequestDoesNotStageTheBrokenModel() {
        var plan = EmbeddingSlotPlan()
        XCTAssertNil(coldRequest(&plan, model: "broken", loads: ["bge"]))
        XCTAssertNil(plan.lastServed)
        XCTAssertEqual(
            plan.loadTarget(defaultModel: { dflt }), dflt,
            "an unrelated reload must not retry the broken model")
    }

    func testEvictReloadRestoresLastSuccessfullyServedModel() {
        var plan = EmbeddingSlotPlan()
        XCTAssertEqual(
            coldRequest(&plan, model: "bge", loads: ["bge"]), "bge")
        XCTAssertNil(coldRequest(&plan, model: "broken", loads: ["bge"]))
        XCTAssertEqual(
            plan.loadTarget(defaultModel: { dflt }), "bge",
            "NI3: a governor reload restores the last model that loaded")
    }

    /// #241 item 2 — a staged target that never loaded (no-op rebind, pruned
    /// dir) must not be bound by a later evict→reload.
    func testClearedStageIsNotBoundByLaterReload() {
        var plan = EmbeddingSlotPlan()
        XCTAssertEqual(
            coldRequest(&plan, model: "bge", loads: ["bge"]), "bge")
        plan.stageColdTarget("never-loaded")
        plan.clearColdTarget()
        XCTAssertEqual(plan.loadTarget(defaultModel: { dflt }), "bge")
    }

    /// #241 item 4 — no re-staging: the reload after a successful load must
    /// bind what loaded, and a stale stage must not survive a load success.
    func testLoadSuccessConsumesTheStage() {
        var plan = EmbeddingSlotPlan()
        plan.stageColdTarget("requested")
        plan.loadSucceeded("other")
        XCTAssertEqual(plan.loadTarget(defaultModel: { dflt }), "other")
    }

    func testLoadFailureConsumesTheStageWithoutRestaging() {
        var plan = EmbeddingSlotPlan()
        plan.stageColdTarget("broken")
        plan.loadFailed()
        XCTAssertEqual(plan.loadTarget(defaultModel: { dflt }), dflt)
    }
}
