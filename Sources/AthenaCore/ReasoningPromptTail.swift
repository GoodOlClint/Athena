import Foundation

/// #198 (ADR 035 amendment) — pure, MLX-free check of whether a rendered
/// chat-template prompt ends inside an open reasoning block: the last
/// occurrence of the open marker has no matching close marker after it.
/// Qwen3.5's own template does exactly this (`<think>\n` with no `</think>`)
/// whenever thinking is active, so the model's completion starts already
/// "inside" the block with no start delimiter of its own to key on.
///
/// Lives in `AthenaCore` (not `AthenaServerKit`, where the markers/filter
/// live, and not `AthenaLLM`) because it is the one module both depend on:
/// `AthenaLLM` calls this on the decoded prompt tail (it has the tokenizer),
/// `AthenaServerKit`'s filter consumes the resulting `Bool` the server
/// threads through — neither of those two targets depends on the other.
public enum ReasoningPromptTail {
    /// How many of the prompt's trailing tokens to decode and scan. Small:
    /// the markers this checks for (`<think>\n`, or the disabled form
    /// `<think>\n\n</think>\n\n`) are a handful of tokens at most, and this
    /// runs once per generation, not per token.
    public static let tailTokenCount = 32

    /// True iff `tail` (the decoded text of the prompt's last
    /// `tailTokenCount` tokens) contains `openMarker` with no `closeMarker`
    /// anywhere after that last occurrence.
    public static func startsInOpenBlock(
        _ tail: String, openMarker: String = "<think>",
        closeMarker: String = "</think>"
    ) -> Bool {
        guard let lastOpen = tail.range(of: openMarker, options: .backwards)
        else { return false }
        return tail.range(
            of: closeMarker,
            range: lastOpen.upperBound ..< tail.endIndex) == nil
    }
}
