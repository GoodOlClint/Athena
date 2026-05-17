import Foundation
import MLX
import MLXLMCommon

/// Greedy, single-30 s-window, no-timestamps transcription (M4.2c) with
/// language auto-detect (M4.2e-1). KV-cache, >30 s chunking, timestamps
/// remain M4.2e.
public enum WhisperDecode {
    // whisper-large-v3 special-token ids (vocab 51866, fixed).
    static let eot = 50_257  // <|endoftext|>
    static let sot = 50_258  // <|startoftranscript|>
    static let langBase = 50_259  // <|en|> ; lang i → langBase+i
    static let transcribe = 50_360
    static let noTimestamps = 50_364

    /// Whisper language order (subset). Index → `langBase + index`.
    private static let langIndex: [String: Int] = [
        "en": 0, "zh": 1, "de": 2, "es": 3, "ru": 4, "ko": 5, "fr": 6,
        "ja": 7, "pt": 8, "tr": 9, "pl": 10, "ca": 11, "nl": 12,
        "ar": 13, "sv": 14, "it": 15,
    ]

    static func languageToken(_ code: String) -> Int {
        langBase + (langIndex[code.lowercased()] ?? 0)  // default en
    }

    /// Whisper language auto-detection: with only `[sot]` fed, the next
    /// distribution is over the language tokens — argmax restricted to
    /// `[langBase, transcribe)` (the 100 `<|lang|>` ids) → the detected
    /// language token id.
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

    /// Greedy decode of one already-embedded 30 s window. argmax is
    /// restricted to `[0, eot]` so special/lang/timestamp ids are
    /// structurally excluded.
    private static func decodeWindow(
        model: WhisperModel, audio: MLXArray,
        tokenizer: any MLXLMCommon.Tokenizer,
        langTok: Int, maxTokens: Int
    ) -> String {
        let prefix = [sot, langTok, transcribe, noTimestamps]
        var tokens = prefix
        let limit = min(
            model.config.n_text_ctx, prefix.count + maxTokens)
        // KV-cached incremental decode (M4.2e-3): prime the cache with
        // the whole prefix, then feed one token at a time. Restricting
        // argmax to [0, eot] and the cached-attention math being
        // identical to a full re-forward keeps this bit-identical to
        // the uncached greedy — only faster.
        let cache = WhisperKVCache(layers: model.config.n_text_layer)
        func step(_ ids: [Int], _ offset: Int) -> Int {
            let inp = MLXArray(ids.map { Int32($0) }, [1, ids.count])
            let lg = model.logits(
                inp, audio: audio, offset: offset, cache: cache)
            let last = lg[0..., -1, 0 ..< (eot + 1)]
            let n = argMax(last, axis: -1).item(Int.self)
            cache.evalStep()
            return n
        }
        var next = step(prefix, 0)
        while next != eot && tokens.count < limit {
            tokens.append(next)
            if tokens.count >= limit { break }
            next = step([next], tokens.count - 1)
        }
        let generated = Array(tokens[prefix.count...])
        return tokenizer.decode(
            tokenIds: generated, skipSpecialTokens: true
        ).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Transcribe a single ≤30 s log-mel (`[n_mels, 3000]`) → text.
    public static func transcribe(
        model: WhisperModel,
        mel: MLXArray,
        tokenizer: any MLXLMCommon.Tokenizer,
        language: String? = nil,
        maxTokens: Int = 224
    ) -> String {
        let audio = model.embedAudio(mel)
        audio.eval()
        let langTok = resolveLang(language, model: model, audio: audio)
        return decodeWindow(
            model: model, audio: audio, tokenizer: tokenizer,
            langTok: langTok, maxTokens: maxTokens)
    }

    /// Transcribe full-length PCM (mono 16 kHz). Audio longer than 30 s
    /// is split into consecutive 30 s windows (the last is zero-padded
    /// by `LogMel`), each decoded independently and the texts joined.
    /// Language is resolved once (detected on window 0 if not given)
    /// and reused. Cross-window prompt conditioning is a future
    /// refinement (M4.2e). M4.2e-2.
    public static func transcribe(
        model: WhisperModel,
        pcm: [Float],
        tokenizer: any MLXLMCommon.Tokenizer,
        language: String? = nil,
        maxTokens: Int = 224
    ) -> String {
        let n = LogMel.nSamples
        let windows: [[Float]]
        if pcm.count <= n {
            windows = [pcm]
        } else {
            windows = stride(from: 0, to: pcm.count, by: n).map {
                Array(pcm[$0 ..< min($0 + n, pcm.count)])
            }
        }
        var langTok: Int?
        var parts: [String] = []
        for w in windows {
            let audio = model.embedAudio(LogMel.logMel(w))
            audio.eval()
            let lt =
                langTok
                ?? resolveLang(language, model: model, audio: audio)
            langTok = lt  // resolve once, reuse across windows
            parts.append(
                decodeWindow(
                    model: model, audio: audio, tokenizer: tokenizer,
                    langTok: lt, maxTokens: maxTokens))
        }
        return parts.filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
