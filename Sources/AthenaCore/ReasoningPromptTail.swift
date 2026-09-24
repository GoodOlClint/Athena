import Foundation

/// #198 (ADR 035 amendment) — pure, MLX-free check of whether a rendered
/// chat-template prompt ends inside an open reasoning block: the prompt's
/// own text ends with the open marker (nothing but whitespace after it).
/// Qwen3.5's own template does exactly this (`<think>\n`, then nothing —
/// generation starts right there) whenever thinking is active, so the
/// model's completion starts already "inside" the block with no start
/// delimiter of its own to key on.
///
/// Requiring nothing-but-whitespace AFTER the marker (not just "contains the
/// marker with no close after it" — Codex adversarial review, PR #213 round
/// 3) is deliberate: a prompt whose LAST USER MESSAGE happens to end with the
/// literal text `<think>` is followed by the template's own generation-
/// prompt boilerplate (e.g. `<|im_start|>assistant\n`) before generation
/// starts — real, non-whitespace text after the marker — so it is correctly
/// rejected here, rather than misclassifying an ordinary user prompt as an
/// opened reasoning block and losing the entire response into
/// `reasoning_content`.
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
    /// `tailTokenCount` tokens — i.e. the prompt's own true suffix, since
    /// nothing follows it) ends with `openMarker` followed by nothing but
    /// whitespace, with no `closeMarker` in between.
    public static func startsInOpenBlock(
        _ tail: String, openMarker: String = "<think>",
        closeMarker: String = "</think>"
    ) -> Bool {
        guard let lastOpen = tail.range(of: openMarker, options: .backwards)
        else { return false }
        let afterOpen = tail[lastOpen.upperBound...]
        guard afterOpen.range(of: closeMarker) == nil else { return false }
        return afterOpen.trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
    }
}
