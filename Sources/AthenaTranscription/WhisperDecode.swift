import Foundation
import MLX
import MLXLMCommon

/// Greedy KV-cached Whisper decoding with language auto-detect and
/// timestamps (M4.2c → M4.2e). Timestamps are always decoded; the
/// `.text`/json formats drop them, `verbose_json`/`srt`/`vtt` use the
/// resulting segments.
public enum WhisperDecode {
    // whisper-large-v3 special-token ids (vocab 51866, fixed).
    static let eot = 50_257  // <|endoftext|>
    static let sot = 50_258  // <|startoftranscript|>
    static let langBase = 50_259  // <|en|> ; lang i → langBase+i
    static let transcribe = 50_360
    static let noTimestamps = 50_364
    static let timestampBegin = 50_365  // <|0.00|> ; step 0.02 s
    static let timeStep = 0.02
    static let windowSeconds = 30.0

    private static let langOrder = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it",
    ]
    private static let langIndex: [String: Int] = {
        var m: [String: Int] = [:]
        for (i, c) in langOrder.enumerated() { m[c] = i }
        return m
    }()

    static func languageToken(_ code: String) -> Int {
        langBase + (langIndex[code.lowercased()] ?? 0)  // default en
    }
    static func languageCode(forToken tok: Int) -> String {
        let i = tok - langBase
        return (i >= 0 && i < langOrder.count) ? langOrder[i] : "en"
    }

    /// Whisper language auto-detection: with only `[sot]` fed, argmax
    /// the next distribution over `[langBase, transcribe)`.
    static func detectLanguageToken(
        model: WhisperModel, audio: MLXArray
    ) -> Int {
        let inp = MLXArray([Int32(sot)], [1, 1])
        let lg = model.logits(inp, audio: audio)
        let langs = lg[0..., -1, langBase ..< transcribe]
        return langBase + argMax(langs, axis: -1).item(Int.self)
    }

    /// Explicit ISO code wins; nil/"auto" ⇒ detect from `audio`.
    private static func resolveLang(
        _ language: String?, model: WhisperModel, audio: MLXArray
    ) -> Int {
        if let language, language.lowercased() != "auto" {
            return languageToken(language)
        }
        return detectLanguageToken(model: model, audio: audio)
    }

    /// Additive logit mask (size n_vocab) that −∞'s the contiguous
    /// non-timestamp special block `[eot+1, timestampBegin)` (sot / lang
    /// / transcribe / translate / notimestamps / nospeech). Leaves text
    /// + `eot` + timestamp ids selectable. Built once per decode.
    private static func specialMask(vocab: Int) -> MLXArray {
        var m = [Float](repeating: 0, count: vocab)
        for i in (eot + 1)..<timestampBegin { m[i] = -1e9 }
        return MLXArray(m)
    }

    /// A timestamp-delimited span before tokenizer decoding: its
    /// window-relative bounds, the content token ids it covers, and the
    /// mean per-token log-probability of those tokens (M26.1). Pure data
    /// so the parser is unit-testable without a model.
    struct ParsedSegment {
        let start: Double
        let end: Double
        let tokens: [Int]
        let avgLogprob: Double?
    }

    /// Split a window's greedy output into timestamp-delimited spans.
    /// `generated` is the decoded ids (text + `<|t|>` markers, no eot);
    /// `logprobs` is the parallel per-token log-probability (same count).
    /// Pure: depends only on the fixed timestamp/eot constants. M26.1.
    static func parseSegments(
        generated: [Int], logprobs: [Double]
    ) -> [ParsedSegment] {
        var segments: [ParsedSegment] = []
        var buf: [Int] = []
        var bufLogprobs: [Double] = []
        var segStart: Double?
        var lastTs: Double?
        func flush(_ end: Double) {
            guard !buf.isEmpty else { return }
            let avg =
                bufLogprobs.isEmpty
                ? nil
                : bufLogprobs.reduce(0, +) / Double(bufLogprobs.count)
            segments.append(
                ParsedSegment(
                    start: segStart ?? 0, end: end, tokens: buf,
                    avgLogprob: avg))
            buf = []
            bufLogprobs = []
        }
        for (i, t) in generated.enumerated() {
            if t >= timestampBegin {
                let ts = Double(t - timestampBegin) * timeStep
                lastTs = ts
                if segStart == nil {
                    segStart = ts
                } else {
                    flush(ts)
                    segStart = nil
                }
            } else {
                buf.append(t)
                if i < logprobs.count { bufLogprobs.append(logprobs[i]) }
            }
        }
        flush(lastTs ?? windowSeconds)
        return segments
    }

    /// Greedy-decode one embedded 30 s window → (text, window-relative
    /// segments, content token ids). KV-cached, timestamp-aware. Tracks
    /// each step's log-probability so segments carry `avgLogprob` (M26.1)
    /// and exposes the content tokens for word alignment (M26.2).
    private static func decodeWindow(
        model: WhisperModel, audio: MLXArray,
        tokenizer: any MLXLMCommon.Tokenizer,
        langTok: Int, maxTokens: Int
    ) -> (
        text: String, segments: [TranscriptionSegment],
        contentTokens: [Int]
    ) {
        let prefix = [sot, langTok, transcribe]
        var generated: [Int] = []
        var genLogprobs: [Double] = []
        let limit = min(
            model.config.n_text_ctx, prefix.count + maxTokens)
        let mask = specialMask(vocab: model.config.n_vocab)
        let cache = WhisperKVCache(layers: model.config.n_text_layer)

        func step(_ ids: [Int], _ offset: Int) -> (token: Int, logprob: Double) {
            let inp = MLXArray(ids.map { Int32($0) }, [1, ids.count])
            let lg = model.logits(
                inp, audio: audio, offset: offset, cache: cache)
            let last = lg[0..., -1, 0...] + mask
            let n = argMax(last, axis: -1).item(Int.self)
            // log-softmax of the chosen id under the decode distribution.
            let lse = logSumExp(last, axis: -1).item(Float.self)
            let logp = Double(last[0, n].item(Float.self) - lse)
            cache.evalStep()
            return (n, logp)
        }

        var produced = prefix.count
        var next = step(prefix, 0)
        while next.token != eot && produced < limit {
            generated.append(next.token)
            genLogprobs.append(next.logprob)
            produced += 1
            if produced >= limit { break }
            next = step([next.token], produced - 1)
        }

        let parsed = parseSegments(
            generated: generated, logprobs: genLogprobs)
        var segments: [TranscriptionSegment] = []
        for p in parsed {
            let t = tokenizer.decode(
                tokenIds: p.tokens, skipSpecialTokens: true
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !t.isEmpty else { continue }
            segments.append(
                TranscriptionSegment(
                    start: p.start, end: p.end, text: t,
                    avgLogprob: p.avgLogprob))
        }

        let text = segments.map(\.text).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let contentTokens = generated.filter { $0 < eot }
        return (text, segments, contentTokens)
    }

    /// Transcribe full-length PCM (mono 16 kHz) → text + globally-timed
    /// segments. >30 s is split into 30 s windows (last zero-padded);
    /// language resolved once (window 0) and reused; window `i`'s
    /// segment times are offset by `i·30 s`.
    public static func transcribeResult(
        model: WhisperModel,
        pcm: [Float],
        tokenizer: any MLXLMCommon.Tokenizer,
        language: String? = nil,
        maxTokens: Int = 224
    ) -> TranscriptionResult {
        let n = LogMel.nSamples
        let windows: [[Float]] =
            pcm.count <= n
            ? [pcm]
            : stride(from: 0, to: pcm.count, by: n).map {
                Array(pcm[$0 ..< min($0 + n, pcm.count)])
            }
        var langTok: Int?
        var allSegments: [TranscriptionSegment] = []
        var parts: [String] = []
        for (i, w) in windows.enumerated() {
            let audio = model.embedAudio(LogMel.logMel(w))
            audio.eval()
            let lt =
                langTok
                ?? resolveLang(language, model: model, audio: audio)
            langTok = lt
            let (text, segs, _) = decodeWindow(
                model: model, audio: audio, tokenizer: tokenizer,
                langTok: lt, maxTokens: maxTokens)
            let off = Double(i) * windowSeconds
            allSegments.append(
                contentsOf: segs.map {
                    TranscriptionSegment(
                        start: $0.start + off, end: $0.end + off,
                        text: $0.text, avgLogprob: $0.avgLogprob)
                })
            if !text.isEmpty { parts.append(text) }
        }
        return TranscriptionResult(
            text: parts.joined(separator: " "),
            language: languageCode(forToken: langTok ?? langBase),
            duration: Double(pcm.count) / Double(LogMel.sampleRate),
            segments: allSegments)
    }

    /// Text-only convenience (PCM). Back-compat for callers/tests.
    public static func transcribe(
        model: WhisperModel, pcm: [Float],
        tokenizer: any MLXLMCommon.Tokenizer,
        language: String? = nil, maxTokens: Int = 224
    ) -> String {
        transcribeResult(
            model: model, pcm: pcm, tokenizer: tokenizer,
            language: language, maxTokens: maxTokens
        ).text
    }

    /// Text-only convenience (single ≤30 s log-mel window).
    public static func transcribe(
        model: WhisperModel, mel: MLXArray,
        tokenizer: any MLXLMCommon.Tokenizer,
        language: String? = nil, maxTokens: Int = 224
    ) -> String {
        let audio = model.embedAudio(mel)
        audio.eval()
        let langTok = resolveLang(language, model: model, audio: audio)
        return decodeWindow(
            model: model, audio: audio, tokenizer: tokenizer,
            langTok: langTok, maxTokens: maxTokens
        ).text
    }
}
