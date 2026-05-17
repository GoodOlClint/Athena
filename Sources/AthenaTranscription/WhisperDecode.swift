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

    /// Greedy-decode one embedded 30 s window → (text, window-relative
    /// segments). KV-cached, timestamp-aware.
    private static func decodeWindow(
        model: WhisperModel, audio: MLXArray,
        tokenizer: any MLXLMCommon.Tokenizer,
        langTok: Int, maxTokens: Int
    ) -> (text: String, segments: [TranscriptionSegment]) {
        let prefix = [sot, langTok, transcribe]
        var generated: [Int] = []
        let limit = min(
            model.config.n_text_ctx, prefix.count + maxTokens)
        let mask = specialMask(vocab: model.config.n_vocab)
        let cache = WhisperKVCache(layers: model.config.n_text_layer)

        func step(_ ids: [Int], _ offset: Int) -> Int {
            let inp = MLXArray(ids.map { Int32($0) }, [1, ids.count])
            let lg = model.logits(
                inp, audio: audio, offset: offset, cache: cache)
            let last = lg[0..., -1, 0...] + mask
            let n = argMax(last, axis: -1).item(Int.self)
            cache.evalStep()
            return n
        }

        var produced = prefix.count
        var next = step(prefix, 0)
        while next != eot && produced < limit {
            generated.append(next)
            produced += 1
            if produced >= limit { break }
            next = step([next], produced - 1)
        }

        // Parse timestamp-delimited segments.
        var segments: [TranscriptionSegment] = []
        var buf: [Int] = []
        var segStart: Double?
        var lastTs: Double?
        func flush(_ end: Double) {
            guard !buf.isEmpty else { return }
            let t = tokenizer.decode(
                tokenIds: buf, skipSpecialTokens: true
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty {
                segments.append(
                    TranscriptionSegment(
                        start: segStart ?? 0, end: end, text: t))
            }
            buf = []
        }
        for t in generated {
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
            }
        }
        flush(lastTs ?? windowSeconds)

        let text = segments.map(\.text).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (text, segments)
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
            let (text, segs) = decodeWindow(
                model: model, audio: audio, tokenizer: tokenizer,
                langTok: lt, maxTokens: maxTokens)
            let off = Double(i) * windowSeconds
            allSegments.append(
                contentsOf: segs.map {
                    TranscriptionSegment(
                        start: $0.start + off, end: $0.end + off,
                        text: $0.text)
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
