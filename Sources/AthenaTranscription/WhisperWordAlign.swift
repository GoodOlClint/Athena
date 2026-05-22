import Foundation
import MLX
import MLXLMCommon

/// Word-level timestamp alignment for one decoded 30 s window (M26.2).
/// Ports `openai/mlx-whisper` `find_alignment`: a single non-cached
/// decoder pass over the window's content tokens captures the cross-
/// attention scores of the checkpoint's alignment heads; DTW over those
/// (after `WordAlignMath` post-processing) maps each token to an audio
/// frame, which is grouped into words. Read-only — it never alters the
/// greedy decode, so transcripts are unchanged whether or not word
/// timestamps are requested.
public enum WhisperWordAlign {
    /// Encoder frames per second (10 ms hop × 2 conv stride ⇒ 20 ms).
    static let framesPerSecond = 1.0 / WhisperDecode.timeStep  // 50

    /// `audio` = encoder features for the window; `contentTokens` = the
    /// window's decoded text token ids (no timestamps/eot); `validFrames`
    /// = real (non-padding) encoder frames. Returns window-relative word
    /// timings (monotonically ordered) each paired with the half-open
    /// range of `contentTokens` it covers, so callers can attach words to
    /// the segment that owns those tokens.
    public static func align(
        model: WhisperModel, audio: MLXArray,
        tokenizer: any MLXLMCommon.Tokenizer,
        contentTokens: [Int], langTok: Int, validFrames: Int
    ) -> [(word: WordTiming, range: Range<Int>)] {
        let n = contentTokens.count
        guard n > 0 else { return [] }

        let prefix = [
            WhisperDecode.sot, langTok, WhisperDecode.transcribe,
            WhisperDecode.noTimestamps,
        ]
        let prefixLen = prefix.count
        let seq = prefix + contentTokens + [WhisperDecode.eot]
        let T = seq.count

        let inp = MLXArray(seq.map { Int32($0) }, [1, T])
        let crossQK = WhisperCrossQK(layers: model.config.n_text_layer)
        let logitsArr = model.logits(inp, audio: audio, crossQK: crossQK)
        logitsArr.eval()
        crossQK.evalAll()

        // Probability of each chosen content token (predicted by the
        // logits one position earlier, starting at the noTimestamps row).
        let predRows = logitsArr[0, (prefixLen - 1) ..< (prefixLen - 1 + n), 0...]
        let probs = MLX.softmax(predRows, axis: -1)
        let idx = MLXArray(contentTokens.map { Int32($0) }, [n, 1])
        let tokenProbs = MLX.takeAlong(probs, idx, axis: -1)
            .asArray(Float.self).map(Double.init)

        // Alignment-head cross-attention weights → [head][token][frame].
        let heads =
            model.alignmentHeads.isEmpty
            ? allHeads(model) : model.alignmentHeads
        let Fv = max(1, min(validFrames, audio.dim(1)))
        var weights: [[[Double]]] = []
        weights.reserveCapacity(heads.count)
        for (l, h) in heads {
            guard let qk = crossQK.perLayer[l] else { continue }
            let slice = qk[0, h, 0..., 0 ..< Fv]  // [T, Fv]
            let flat = slice.asArray(Float.self)
            var mat = [[Double]](
                repeating: [Double](repeating: 0, count: Fv), count: T)
            for r in 0 ..< T {
                let base = r * Fv
                for c in 0 ..< Fv { mat[r][c] = Double(flat[base + c]) }
            }
            weights.append(mat)
        }
        guard !weights.isEmpty else { return [] }

        let matrix = WordAlignMath.reduceWeights(weights, medfiltWidth: 7)
        guard matrix.count >= prefixLen + n else { return [] }
        let contentRows = Array(matrix[prefixLen ..< prefixLen + n])
        let cost = contentRows.map { $0.map { -$0 } }
        let (textIdx, timeIdx) = WordAlignMath.dtw(cost)
        let frames = WordAlignMath.firstFrames(
            textIdx: textIdx, timeIdx: timeIdx, n: n)
        let tokenStart = frames.map { Double($0) / framesPerSecond }
        let windowEnd = Double(Fv) / framesPerSecond

        // Group content tokens into words and time each from its tokens.
        let subwords = subwordSplit(
            tokens: contentTokens, tokenizer: tokenizer)
        let words = WordAlignMath.mergeWords(subwords)
        var out: [(word: WordTiming, range: Range<Int>)] = []
        out.reserveCapacity(words.count)
        for (wi, w) in words.enumerated() {
            let a = w.range.lowerBound
            guard a < n else { continue }
            let start = tokenStart[a]
            let end =
                wi + 1 < words.count
                ? tokenStart[min(words[wi + 1].range.lowerBound, n - 1)]
                : windowEnd
            let probSlice = tokenProbs[w.range]
            let prob =
                probSlice.isEmpty
                ? 0
                : probSlice.reduce(0, +) / Double(probSlice.count)
            let text = w.word.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            out.append(
                (WordTiming(
                    word: text, start: start, end: max(start, end),
                    probability: prob), w.range))
        }
        return out
    }

    /// Fallback when the checkpoint carried no `alignment_heads`: every
    /// (layer, head) of the decoder.
    private static func allHeads(_ model: WhisperModel) -> [(layer: Int, head: Int)] {
        var hs: [(layer: Int, head: Int)] = []
        for l in 0 ..< model.config.n_text_layer {
            for h in 0 ..< model.config.n_text_head { hs.append((l, h)) }
        }
        return hs
    }

    /// Split tokens into unicode-safe subwords with their token ranges,
    /// accumulating ids until the decode no longer ends in a replacement
    /// char (mirrors whisper `split_tokens_on_unicode`).
    static func subwordSplit(
        tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer
    ) -> [(text: String, range: Range<Int>)] {
        var result: [(text: String, range: Range<Int>)] = []
        var acc: [Int] = []
        var start = 0
        for (i, t) in tokens.enumerated() {
            acc.append(t)
            let decoded = tokenizer.decode(
                tokenIds: acc, skipSpecialTokens: false)
            if !decoded.contains("\u{FFFD}") {
                result.append((decoded, start ..< (i + 1)))
                acc = []
                start = i + 1
            }
        }
        if !acc.isEmpty {
            let decoded = tokenizer.decode(
                tokenIds: acc, skipSpecialTokens: false)
            result.append((decoded, start ..< tokens.count))
        }
        return result
    }
}
