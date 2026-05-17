import Foundation
import MLX
import MLXLMCommon

/// Greedy, single-30 s-window, no-timestamps transcription (M4.2c).
/// KV-cache, >30 s chunking, language auto-detect, timestamps and
/// suppression nuance are deferred to M4.2e.
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

    /// Transcribe a single ≤30 s log-mel (`[n_mels, 3000]`) → text.
    /// Greedy argmax restricted to text ids + `eot` (special/lang/
    /// timestamp ids are structurally excluded by the argmax range).
    public static func transcribe(
        model: WhisperModel,
        mel: MLXArray,
        tokenizer: any MLXLMCommon.Tokenizer,
        language: String = "en",
        maxTokens: Int = 224
    ) -> String {
        let audio = model.embedAudio(mel)
        audio.eval()

        let prefix = [
            sot, languageToken(language), transcribe, noTimestamps,
        ]
        var tokens = prefix
        let limit = min(
            model.config.n_text_ctx, prefix.count + maxTokens)

        while tokens.count < limit {
            let inp = MLXArray(
                tokens.map { Int32($0) }, [1, tokens.count])
            let lg = model.logits(inp, audio: audio)
            // Restrict to [0, eot] → text tokens + <|endoftext|>;
            // everything ≥ sot (specials/langs/timestamps) excluded.
            let last = lg[0..., -1, 0 ..< (eot + 1)]
            let next = argMax(last, axis: -1).item(Int.self)
            if next == eot { break }
            tokens.append(next)
        }

        let generated = Array(tokens[prefix.count...])
        return tokenizer.decode(
            tokenIds: generated, skipSpecialTokens: true
        ).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
