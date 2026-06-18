import Foundation
import XCTest

@testable import AthenaTranscription

/// Pure (MLX-free) decode for the pyannote PyanNet segmentation model
/// (ADR 018 / S2): powerset→multilabel, hysteresis binarization, region
/// geometry, and sliding-window center-ownership. Always runs in CI (ADR 009).
final class PyannoteSegmentationDecodeTests: XCTestCase {

    // MARK: powerset → multilabel

    func testPowersetMappingIsCanonical() {
        XCTAssertEqual(
            PyannotePowerset.classToSpeakers,
            [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]])
    }

    func testLocalSpeakerProbsSumsContainingClasses() {
        // p = [non, s0, s1, s2, s0s1, s0s2, s1s2]
        let p: [Float] = [0.1, 0.2, 0.05, 0.0, 0.3, 0.25, 0.1]
        let out = PyannotePowerset.localSpeakerProbs([p])[0]
        // s0 = p1+p4+p5 = 0.2+0.3+0.25 = 0.75
        // s1 = p2+p4+p6 = 0.05+0.3+0.1 = 0.45
        // s2 = p3+p5+p6 = 0.0+0.25+0.1 = 0.35
        XCTAssertEqual(out[0], 0.75, accuracy: 1e-5)
        XCTAssertEqual(out[1], 0.45, accuracy: 1e-5)
        XCTAssertEqual(out[2], 0.35, accuracy: 1e-5)
    }

    func testNonSpeechFrameYieldsZeroActivity() {
        let p: [Float] = [1, 0, 0, 0, 0, 0, 0]
        let out = PyannotePowerset.localSpeakerProbs([p])[0]
        XCTAssertEqual(out, [0, 0, 0])
    }

    // MARK: hysteresis binarization

    func testBinarizeOpensOnOnsetClosesOnOffset() {
        // onset 0.5, offset 0.3. Rises at idx2, dips below 0.3 at idx5.
        let s: [Float] = [0.1, 0.4, 0.6, 0.55, 0.45, 0.2, 0.1]
        let iv = PyannoteSegmentationDecode.binarizeFrames(
            s, onset: 0.5, offset: 0.3)
        XCTAssertEqual(iv.count, 1)
        XCTAssertEqual(iv[0].start, 2)
        XCTAssertEqual(iv[0].end, 5)
    }

    func testBinarizeClosesOpenIntervalAtEnd() {
        let s: [Float] = [0.1, 0.9, 0.9]
        let iv = PyannoteSegmentationDecode.binarizeFrames(
            s, onset: 0.5, offset: 0.3)
        XCTAssertEqual(iv.count, 1)
        XCTAssertEqual(iv[0].start, 1)
        XCTAssertEqual(iv[0].end, 3)
    }

    func testBinarizeHysteresisStaysOpenBetweenOnsetAndOffset() {
        // Dips to 0.4 (< onset 0.5 but >= offset 0.3) → stays open.
        let s: [Float] = [0.9, 0.4, 0.9, 0.1]
        let iv = PyannoteSegmentationDecode.binarizeFrames(
            s, onset: 0.5, offset: 0.3)
        XCTAssertEqual(iv.map { [$0.start, $0.end] }, [[0, 3]])
    }

    // MARK: region assembly + ownership clipping

    func testRegionsTaggedWithWindowAndLocalSpeaker() {
        // Two frames: frame0 speaker0 active, frame1 speaker1 active.
        let post: [[Float]] = [
            [0, 1, 0, 0, 0, 0, 0],  // s0
            [0, 0, 1, 0, 0, 0, 0],  // s1
        ]
        let regions = PyannoteSegmentationDecode.regions(
            posteriors: post, frameDuration: 1.0,
            windowStart: 10.0, ownStart: 0.0, ownEnd: 100.0,
            window: 3, params: .init(onset: 0.5, offset: 0.3, minDurationOn: 0))
        XCTAssertEqual(regions.count, 2)
        XCTAssertTrue(regions.allSatisfy { $0.window == 3 })
        XCTAssertEqual(Set(regions.map { $0.localSpeaker }), [0, 1])
        // Times are absolute (windowStart + frame*frameDuration).
        let s0 = regions.first { $0.localSpeaker == 0 }!
        XCTAssertEqual(s0.start, 10.0, accuracy: 1e-6)
        XCTAssertEqual(s0.end, 11.0, accuracy: 1e-6)
    }

    func testRegionsClippedToOwnershipZone() {
        // Speaker active across the whole window, but ownership is [12,13].
        let post = Array(
            repeating: [Float]([0, 1, 0, 0, 0, 0, 0]), count: 5)
        let regions = PyannoteSegmentationDecode.regions(
            posteriors: post, frameDuration: 1.0,
            windowStart: 10.0, ownStart: 12.0, ownEnd: 13.0,
            window: 0, params: .init(onset: 0.5, offset: 0.3, minDurationOn: 0))
        XCTAssertEqual(regions.count, 1)
        XCTAssertEqual(regions[0].start, 12.0, accuracy: 1e-6)
        XCTAssertEqual(regions[0].end, 13.0, accuracy: 1e-6)
    }

    func testRegionsDropShorterThanMinDuration() {
        let post: [[Float]] = [[0, 1, 0, 0, 0, 0, 0]]  // 1 frame * 0.1s = 0.1s
        let regions = PyannoteSegmentationDecode.regions(
            posteriors: post, frameDuration: 0.1,
            windowStart: 0, ownStart: 0, ownEnd: 100,
            window: 0, params: .init(onset: 0.5, offset: 0.3, minDurationOn: 0.3))
        XCTAssertTrue(regions.isEmpty)
    }

    // MARK: same-window cannot-link + clustering integration

    func testSameWindowCannotLinkPairsOnlyDifferentLocalSpeakers() {
        let regions = [
            SpeakerActivityRegion(start: 0, end: 1, window: 0, localSpeaker: 0),
            SpeakerActivityRegion(start: 0, end: 1, window: 0, localSpeaker: 1),
            SpeakerActivityRegion(start: 5, end: 6, window: 1, localSpeaker: 0),
        ]
        let pairs = PyannoteSegmentationDecode.sameWindowCannotLink(regions)
        // Only (0,1): same window 0, different local speakers. (0,2)/(1,2) are
        // different windows; same window+same local speaker never pairs.
        XCTAssertEqual(pairs.count, 1)
        XCTAssertEqual(pairs[0].0, 0)
        XCTAssertEqual(pairs[0].1, 1)
    }

    func testCannotLinkPreventsMergingIdenticalVectors() {
        // Two identical embeddings would normally merge at any threshold; the
        // cannot-link forces them to stay in separate clusters.
        let v: [Float] = [1, 0, 0]
        let labels = AgglomerativeClustering.cluster(
            [v, v], threshold: 1.0, cannotLink: [(0, 1)])
        XCTAssertNotEqual(labels[0], labels[1])
    }

    func testCannotLinkPropagatesThroughMerges() {
        // a≈b≈c all close; a cannot-link c. b should merge with one side, but
        // a and c must never share a cluster → at least 2 clusters remain.
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0.99, 0.01, 0]
        let c: [Float] = [0.98, 0.02, 0]
        let labels = AgglomerativeClustering.cluster(
            [a, b, c], threshold: 1.0, cannotLink: [(0, 2)])
        XCTAssertNotEqual(labels[0], labels[2])
    }

    func testMergeSameSpeakerJoinsAdjacentPreservesCrossSpeakerOverlap() {
        let turns = [
            DiarizationTurn(start: 0.0, end: 1.0, speaker: 0),
            DiarizationTurn(start: 1.1, end: 2.0, speaker: 0),  // adjacent → merge
            DiarizationTurn(start: 0.5, end: 1.5, speaker: 1),  // overlaps spk0
        ]
        let merged = PyannoteSegmentationDecode.mergeSameSpeakerTurns(
            turns, gapTolerance: 0.25)
        // spk0's two turns merge into [0,2]; spk1 stays, overlapping spk0.
        let s0 = merged.filter { $0.speaker == 0 }
        let s1 = merged.filter { $0.speaker == 1 }
        XCTAssertEqual(s0.count, 1)
        XCTAssertEqual(s0[0].start, 0.0, accuracy: 1e-6)
        XCTAssertEqual(s0[0].end, 2.0, accuracy: 1e-6)
        XCTAssertEqual(s1.count, 1)  // overlap with spk0 NOT collapsed
    }

    // MARK: min-cluster-size reassignment

    func testReassignSmallClustersCollapsesSingletonsIntoRealSpeakers() {
        // Two real speakers (A near [1,0], B near [0,1]) each with long
        // airtime, plus a noisy singleton closer to A. The singleton (short)
        // should be absorbed into A → 2 clusters.
        let embA: [Float] = [1, 0]
        let embB: [Float] = [0, 1]
        let noisy: [Float] = [0.9, 0.1]  // closer to A
        let embeddings = [embA, embA, embB, embB, noisy]
        let labels = [0, 0, 1, 1, 2]  // singleton cluster 2
        let durations = [5.0, 5.0, 5.0, 5.0, 0.5]  // cluster 2 tiny
        let out = PyannoteSegmentationDecode.reassignSmallClusters(
            embeddings: embeddings, labels: labels, durations: durations,
            minDuration: 3.0)
        XCTAssertEqual(Set(out).count, 2)
        // The noisy point joined A's cluster (same label as indices 0,1).
        XCTAssertEqual(out[4], out[0])
        XCTAssertNotEqual(out[4], out[2])
    }

    func testReassignKeepsAllWhenEveryClusterMeetsBar() {
        let embeddings = [[Float]([1, 0]), [0, 1]]
        let out = PyannoteSegmentationDecode.reassignSmallClusters(
            embeddings: embeddings, labels: [0, 1], durations: [10, 10],
            minDuration: 3.0)
        XCTAssertEqual(Set(out).count, 2)
    }

    func testReassignAnchorsLargestWhenNoneMeetBar() {
        // No cluster reaches the bar → keep the largest as the sole anchor;
        // everything collapses to 1.
        let embeddings = [[Float]([1, 0]), [0.8, 0.2], [0, 1]]
        let out = PyannoteSegmentationDecode.reassignSmallClusters(
            embeddings: embeddings, labels: [0, 1, 2], durations: [2.0, 1.0, 0.5],
            minDuration: 5.0)
        XCTAssertEqual(Set(out).count, 1)
    }

    func testCompactRenumbersFirstAppearanceOrder() {
        XCTAssertEqual(
            PyannoteSegmentationDecode.compact([5, 5, 2, 9, 2]), [0, 0, 1, 2, 1])
    }

    func testMergeSameSpeakerKeepsDistantTurnsSeparate() {
        let turns = [
            DiarizationTurn(start: 0.0, end: 1.0, speaker: 0),
            DiarizationTurn(start: 5.0, end: 6.0, speaker: 0),  // gap > tol
        ]
        let merged = PyannoteSegmentationDecode.mergeSameSpeakerTurns(
            turns, gapTolerance: 0.25)
        XCTAssertEqual(merged.count, 2)
    }

    func testOwnershipFirstMiddleLast() {
        // step 5, window 10 → halfGap 2.5. total 30.
        let first = PyannoteSegmentationDecode.ownership(
            index: 0, count: 4, step: 5, windowSeconds: 10,
            windowStart: 0, totalSeconds: 30)
        XCTAssertEqual(first.start, 0, accuracy: 1e-6)
        XCTAssertEqual(first.end, 7.5, accuracy: 1e-6)  // 0 + step + halfGap

        let mid = PyannoteSegmentationDecode.ownership(
            index: 1, count: 4, step: 5, windowSeconds: 10,
            windowStart: 5, totalSeconds: 30)
        XCTAssertEqual(mid.start, 7.5, accuracy: 1e-6)  // 5 + 2.5
        XCTAssertEqual(mid.end, 12.5, accuracy: 1e-6)  // 5 + 5 + 2.5

        let last = PyannoteSegmentationDecode.ownership(
            index: 3, count: 4, step: 5, windowSeconds: 10,
            windowStart: 15, totalSeconds: 30)
        XCTAssertEqual(last.start, 17.5, accuracy: 1e-6)
        XCTAssertEqual(last.end, 30.0, accuracy: 1e-6)  // capped to total
    }
}
