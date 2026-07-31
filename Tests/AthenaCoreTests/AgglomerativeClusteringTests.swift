import AthenaTranscription
import Foundation
import XCTest

/// ND14 — `AgglomerativeClustering.cluster` is pure + MLX-free (UPGMA over
/// cosine distance, M25.3), so its decision branches run in CI (ADR 009).
/// Pins the auto/exact/cap/floor stop rules (D10) and the cannot-link
/// constraint propagation (ADR 018) on synthetic 2-D unit vectors, where
/// cosine distance = 1 − dot:
///   a=[1,0]·a=[1,0] ⇒ 0 (identical), a·[0,1] ⇒ 1 (orthogonal),
///   a·[-1,0] ⇒ 2 (opposite).
final class AgglomerativeClusteringTests: XCTestCase {
    private let a: [Float] = [1, 0]
    private let b: [Float] = [0, 1]  // orthogonal to a (distance 1)
    private let c: [Float] = [-1, 0]  // opposite to a (distance 2)

    /// Distinct cluster labels, regardless of which id each got.
    private func groups(_ labels: [Int]) -> Set<Int> { Set(labels) }
    private func sameCluster(_ l: [Int], _ i: Int, _ j: Int) -> Bool {
        l[i] == l[j]
    }

    func testEmptyAndSingleton() {
        XCTAssertEqual(AgglomerativeClustering.cluster([]), [])
        XCTAssertEqual(AgglomerativeClustering.cluster([a]), [0])
    }

    func testAutoMergesNearAndSplitsFar() {
        // a, a (d=0 ≤ 0.75 merge), b (d=1 > 0.75 stays) ⇒ 2 clusters.
        let l = AgglomerativeClustering.cluster([a, a, b])
        XCTAssertEqual(groups(l).count, 2)
        XCTAssertTrue(sameCluster(l, 0, 1))
        XCTAssertFalse(sameCluster(l, 0, 2))
    }

    func testAutoAllCloseCollapseToOne() {
        XCTAssertEqual(groups(AgglomerativeClustering.cluster([a, a, a])).count, 1)
    }

    func testExactCountForcesMergePastThreshold() {
        // a, b are far (d=1 > default 0.75) so auto would keep 2; force 1.
        let l = AgglomerativeClustering.cluster([a, b], numClusters: 1)
        XCTAssertEqual(groups(l).count, 1)
    }

    func testExactCountForcesSplitOfIdentical() {
        // Three identical vectors auto-collapse to 1; force exactly 2.
        let l = AgglomerativeClustering.cluster([a, a, a], numClusters: 2)
        XCTAssertEqual(groups(l).count, 2)
    }

    func testNumClustersClampedToInputCount() {
        // Asking for more clusters than points → each its own (clamped to n).
        let l = AgglomerativeClustering.cluster([a, b], numClusters: 5)
        XCTAssertEqual(groups(l).count, 2)
        // Zero/under-one clamps up to 1 cluster.
        XCTAssertEqual(groups(AgglomerativeClustering.cluster([a, a], numClusters: 0)).count, 1)
    }

    func testMaxClustersCapForcesMerge() {
        // a, b far (auto keeps 2) but a cap of 1 forces the merge past threshold.
        let l = AgglomerativeClustering.cluster([a, b], maxClusters: 1)
        XCTAssertEqual(groups(l).count, 1)
    }

    func testMinClustersFloorPreventsOverMerge() {
        // Permissive threshold (2.0) would collapse everything to 1; the
        // floor of 2 holds distinct speakers apart (D10).
        let l = AgglomerativeClustering.cluster(
            [a, a, b], threshold: 2.0, minClusters: 2)
        XCTAssertEqual(groups(l).count, 2)
    }

    func testThresholdClampedHigh() {
        // threshold 10 clamps to 2 (max cosine distance) → a,b (d=1) merge.
        let l = AgglomerativeClustering.cluster([a, b], threshold: 10)
        XCTAssertEqual(groups(l).count, 1)
    }

    func testThresholdClampedLow() {
        // Negative threshold clamps to 0 → only exact-duplicate (d=0) merges.
        XCTAssertEqual(groups(AgglomerativeClustering.cluster([a, a], threshold: -5)).count, 1)
        XCTAssertEqual(groups(AgglomerativeClustering.cluster([a, b], threshold: -5)).count, 2)
    }

    func testCannotLinkKeepsIdenticalSeparate() {
        // Identical vectors (d=0) that are cannot-linked must NOT merge (ADR 018).
        let l = AgglomerativeClustering.cluster([a, a], cannotLink: [(0, 1)])
        XCTAssertEqual(groups(l).count, 2)
    }

    func testCannotLinkPropagatesOnMerge() {
        // 0,1 identical and free to merge (no ban between them); 1 cannot-link
        // 2. The closest pair {0,1} merges first, and the merged cluster must
        // INHERIT 1's ban on 2 (propagation), so 2 stays its own cluster.
        let l = AgglomerativeClustering.cluster([a, a, a], cannotLink: [(1, 2)])
        XCTAssertTrue(sameCluster(l, 0, 1))
        XCTAssertFalse(sameCluster(l, 0, 2))
        XCTAssertEqual(groups(l).count, 2)
    }

    func testLabelsAreContiguousZeroBased() {
        let l = AgglomerativeClustering.cluster([a, b, c])
        // All far apart ⇒ 3 singletons relabelled to exactly {0,1,2}.
        XCTAssertEqual(Set(l), Set(0 ..< groups(l).count))
    }
}
