import Foundation

/// Pure, MLX-free decode for the pyannote PyanNet segmentation model (ADR 018):
/// powerset-class posteriors → per-local-speaker activity → time regions. The
/// MLX forward (`PyanNetSegmentationModel`) produces the `[frames × 7]`
/// posteriors as plain `[[Float]]`; everything from there is decision logic,
/// so it lives here and is unit-pinned under `swift test` (ADR 009) — no MLX.
///
/// The powerset head encodes up to `maxSpeakers` simultaneous *local* speakers
/// per 10 s window as 7 classes (∅, 3 singletons, 3 pairs). Local speaker ids
/// are meaningful only within one window; cross-window/global identity is
/// resolved downstream by WeSpeaker embedding + clustering (S3), never here.
public enum PyannotePowerset {
    public static let maxSpeakers = 3
    public static let numClasses = 7

    /// Class index → the set of local speakers active in that class. Matches
    /// pyannote `utils/powerset.py` (combinations of `range(3)` for k=0,1,2):
    /// 0=∅, 1={0}, 2={1}, 3={2}, 4={0,1}, 5={0,2}, 6={1,2}.
    public static let classToSpeakers: [[Int]] = [
        [], [0], [1], [2], [0, 1], [0, 2], [1, 2],
    ]

    /// Soft powerset→multilabel decode (pyannote `soft=True` path / soniqo's
    /// sum decode — numerically identical): per-speaker activity probability =
    /// Σ of the probabilities of every class containing that speaker.
    /// - Parameter posteriors: `frames × numClasses` softmax probabilities.
    /// - Returns: `frames × maxSpeakers` per-local-speaker activity in [0,1].
    public static func localSpeakerProbs(_ posteriors: [[Float]]) -> [[Float]] {
        posteriors.map { p in
            var out = [Float](repeating: 0, count: maxSpeakers)
            for (cls, speakers) in classToSpeakers.enumerated() where cls < p.count {
                for s in speakers { out[s] += p[cls] }
            }
            return out
        }
    }
}

/// Tunable thresholds for turning frame-level activity into regions. Defaults
/// track the pyannote `segmentation-3.0` pipeline (onset/offset ≈ 0.5) with a
/// small min-duration to drop single-frame flickers.
public struct PyannoteSegmentationParams: Sendable, Equatable {
    /// Enter a speaker region when activity rises to ≥ `onset`.
    public var onset: Float
    /// Leave a region when activity falls below `offset` (≤ onset gives
    /// hysteresis; equal = a plain threshold).
    public var offset: Float
    /// Drop regions shorter than this (seconds) after center-zone clipping.
    public var minDurationOn: Double

    public init(
        onset: Float = 0.5, offset: Float = 0.5, minDurationOn: Double = 0.05
    ) {
        self.onset = onset
        self.offset = offset
        self.minDurationOn = minDurationOn
    }

    public static let `default` = PyannoteSegmentationParams()
}

/// Frame-level segmentation decode for one sliding window. Kept separate from
/// the MLX model so the binarization + region geometry is CI-testable.
public enum PyannoteSegmentationDecode {

    /// Hysteresis-binarize one speaker's per-frame activity into `[start,end)`
    /// frame intervals: open at `onset`, close at `offset`.
    static func binarizeFrames(
        _ prob: [Float], onset: Float, offset: Float
    ) -> [(start: Int, end: Int)] {
        var intervals: [(Int, Int)] = []
        var active = false
        var start = 0
        for (f, v) in prob.enumerated() {
            if !active, v >= onset {
                active = true
                start = f
            } else if active, v < offset {
                active = false
                intervals.append((start, f))
            }
        }
        if active { intervals.append((start, prob.count)) }
        return intervals
    }

    /// Decode one window's posteriors into locally-tagged speaker regions in
    /// ABSOLUTE seconds, clipped to the window's center-ownership zone so
    /// overlapping windows don't double-count the same speech.
    ///
    /// - Parameters:
    ///   - posteriors: `frames × 7` softmax probabilities for this window.
    ///   - frameDuration: seconds per frame (window_seconds / frames).
    ///   - windowStart: absolute start time of the window (seconds).
    ///   - ownStart/ownEnd: absolute center-ownership bounds; regions are
    ///     clipped to `[ownStart, ownEnd]` (the midpoints to adjacent windows).
    ///   - window: window index (carried for the same-window cannot-link
    ///     constraint applied during clustering).
    ///   - params: onset/offset/min-duration.
    public static func regions(
        posteriors: [[Float]],
        frameDuration: Double,
        windowStart: Double,
        ownStart: Double,
        ownEnd: Double,
        window: Int,
        params: PyannoteSegmentationParams = .default
    ) -> [SpeakerActivityRegion] {
        guard !posteriors.isEmpty, ownEnd > ownStart else { return [] }
        let local = PyannotePowerset.localSpeakerProbs(posteriors)
        let frames = local.count
        var out: [SpeakerActivityRegion] = []
        for s in 0..<PyannotePowerset.maxSpeakers {
            let series = (0..<frames).map { local[$0][s] }
            for iv in binarizeFrames(
                series, onset: params.onset, offset: params.offset)
            {
                // [start,end) frames → absolute seconds, then clip to the
                // window's ownership zone.
                let a = max(windowStart + Double(iv.start) * frameDuration, ownStart)
                let b = min(windowStart + Double(iv.end) * frameDuration, ownEnd)
                if b - a >= params.minDurationOn {
                    out.append(
                        SpeakerActivityRegion(
                            start: a, end: b, window: window, localSpeaker: s))
                }
            }
        }
        return out
    }

    /// Merge a SINGLE speaker's overlapping/adjacent turns into contiguous
    /// turns, preserving cross-speaker overlap (turns of *different* speakers
    /// are never merged, so simultaneous speech stays as overlapping turns).
    /// `gapTolerance` (seconds) joins turns separated by a short silence.
    public static func mergeSameSpeakerTurns(
        _ turns: [DiarizationTurn], gapTolerance: Double = 0.25
    ) -> [DiarizationTurn] {
        guard !turns.isEmpty else { return [] }
        var bySpeaker: [Int: [DiarizationTurn]] = [:]
        for t in turns { bySpeaker[t.speaker, default: []].append(t) }
        var out: [DiarizationTurn] = []
        for (spk, group) in bySpeaker {
            let sorted = group.sorted { $0.start < $1.start }
            var curStart = sorted[0].start
            var curEnd = sorted[0].end
            for t in sorted.dropFirst() {
                if t.start <= curEnd + gapTolerance {
                    curEnd = max(curEnd, t.end)
                } else {
                    out.append(
                        DiarizationTurn(start: curStart, end: curEnd, speaker: spk))
                    curStart = t.start
                    curEnd = t.end
                }
            }
            out.append(
                DiarizationTurn(start: curStart, end: curEnd, speaker: spk))
        }
        return out.sorted {
            $0.start != $1.start ? $0.start < $1.start : $0.speaker < $1.speaker
        }
    }

    /// Dissolve clusters whose total region duration is below `minDuration`
    /// and reassign their members to the nearest surviving cluster by centroid
    /// cosine similarity (pyannote's `min_cluster_size`, duration-based). This
    /// stabilizes the *auto* speaker count against fragmentation from noisy /
    /// short / overlap-contaminated embeddings — the difference between ~6 and
    /// ~90 speakers on long messy audio. Skipped when an exact `num_speakers`
    /// was requested. If no cluster meets the bar, the largest is kept as the
    /// sole anchor so the result never collapses to nothing.
    ///
    /// - Parameters:
    ///   - embeddings: per-region vectors (row-aligned with `labels`).
    ///   - labels: current cluster ids (row-aligned).
    ///   - durations: per-region duration in seconds (row-aligned).
    ///   - minDuration: minimum total airtime (seconds) for a cluster to count
    ///     as a real speaker.
    /// - Returns: contiguous 0-based labels with small clusters reassigned.
    public static func reassignSmallClusters(
        embeddings: [[Float]], labels: [Int], durations: [Double],
        minDuration: Double
    ) -> [Int] {
        let n = labels.count
        guard n > 0, embeddings.count == n, durations.count == n else {
            return labels
        }
        var totalDur: [Int: Double] = [:]
        for i in 0..<n { totalDur[labels[i], default: 0] += durations[i] }
        var big = totalDur.filter { $0.value >= minDuration }.map { $0.key }
        // Never dissolve everything: anchor on the longest cluster if none
        // clear the bar.
        if big.isEmpty, let top = totalDur.max(by: { $0.value < $1.value })?.key
        {
            big = [top]
        }
        let bigSet = Set(big)
        if bigSet.count == totalDur.count { return compact(labels) }

        // L2-normalized centroid per surviving cluster.
        var centroids: [Int: [Float]] = [:]
        for label in big {
            let members = (0..<n).filter { labels[$0] == label }
            guard let dim = members.first.map({ embeddings[$0].count }) else {
                continue
            }
            var mean = [Float](repeating: 0, count: dim)
            for m in members {
                let e = embeddings[m]
                for k in 0..<min(dim, e.count) { mean[k] += e[k] }
            }
            let norm = max(sqrt(mean.reduce(0) { $0 + $1 * $1 }), 1e-9)
            centroids[label] = mean.map { $0 / norm }
        }

        var out = labels
        for i in 0..<n where !bigSet.contains(labels[i]) {
            let e = embeddings[i]
            let en = max(sqrt(e.reduce(0) { $0 + $1 * $1 }), 1e-9)
            var bestLabel = big[0]
            var bestSim = -Float.greatestFiniteMagnitude
            for (label, c) in centroids {
                var dot: Float = 0
                for k in 0..<min(c.count, e.count) { dot += c[k] * (e[k] / en) }
                if dot > bestSim {
                    bestSim = dot
                    bestLabel = label
                }
            }
            out[i] = bestLabel
        }
        return compact(out)
    }

    /// Renumber labels to contiguous 0-based ids in first-appearance order.
    static func compact(_ labels: [Int]) -> [Int] {
        var remap: [Int: Int] = [:]
        var next = 0
        return labels.map {
            if let id = remap[$0] { return id }
            remap[$0] = next
            defer { next += 1 }
            return next
        }
    }

    /// Cannot-link point-index pairs for clustering: any two regions from the
    /// SAME segmentation window with DIFFERENT local speakers are distinct
    /// people and must never merge (ADR 018).
    public static func sameWindowCannotLink(
        _ regions: [SpeakerActivityRegion]
    ) -> [(Int, Int)] {
        var pairs: [(Int, Int)] = []
        for i in 0..<regions.count {
            for j in (i + 1)..<regions.count
            where regions[i].window == regions[j].window
                && regions[i].localSpeaker != regions[j].localSpeaker
            {
                pairs.append((i, j))
            }
        }
        return pairs
    }

    /// Center-ownership bounds for window `index` given uniform `step` and
    /// `windowSeconds`, with `totalSeconds` capping the last window. Window
    /// `w` owns `[w·step + (W−step)/2, (w+1)·step + (W−step)/2]`; the first
    /// window owns from 0 and the last to `totalSeconds`.
    public static func ownership(
        index: Int, count: Int, step: Double, windowSeconds: Double,
        windowStart: Double, totalSeconds: Double
    ) -> (start: Double, end: Double) {
        let halfGap = (windowSeconds - step) / 2
        let start = index == 0 ? 0 : windowStart + halfGap
        let end =
            index == count - 1
            ? totalSeconds : windowStart + step + halfGap
        return (start, min(end, totalSeconds))
    }
}
