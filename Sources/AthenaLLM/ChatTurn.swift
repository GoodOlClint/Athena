import Foundation

/// One chat turn carried into the LLM module — role + content, decoupled
/// from any HTTP DTO or substrate type. The serve path builds these from
/// the request's full message list so the model sees the WHOLE
/// conversation (system, user, assistant, tool), not a user-only join.
public struct ChatTurn: Sendable, Equatable {
    /// "system" | "user" | "assistant" | "tool" (anything else ⇒ user).
    public let role: String
    public let content: String
    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

extension Array where Element == ChatTurn {
    /// Fallback flattening for conformers without a native chat path
    /// (the stub): newline-join every turn's content in order. Role-aware
    /// conformers (the MLX module) ignore this and map to the substrate
    /// chat template instead.
    public func flattenedPrompt() -> String {
        map(\.content).joined(separator: "\n")
    }
}
