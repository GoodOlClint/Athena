import CoreImage
import Foundation

/// One chat turn carried into the LLM module — role + content (+ any image
/// inputs), decoupled from any HTTP DTO or substrate type. The serve path
/// builds these from the request's full message list so the model sees the
/// WHOLE conversation (system, user, assistant, tool), not a user-only join.
public struct ChatTurn: Sendable, Equatable {
    /// "system" | "user" | "assistant" | "tool" (anything else ⇒ user).
    public let role: String
    public let content: String
    /// Decoded image inputs for this turn (M71.1, vision input). Empty for a
    /// text-only turn — every existing text path treats `ChatTurn` exactly as
    /// before. Populated only from OpenAI `image_url` content-parts on the chat
    /// path; consumed by the VLM generate path (M71.2).
    public let images: [ChatImage]
    /// ADR 034 — tool calls on an ASSISTANT turn (the model's prior request),
    /// carried so the chat template renders a coherent call→result history.
    /// Empty for non-tool turns.
    public let toolCalls: [ChatToolCall]
    /// ADR 034 — the assistant tool-call id a TOOL-result turn answers
    /// (OpenAI `tool_call_id`). nil for non-tool turns.
    public let toolCallID: String?
    public init(
        role: String, content: String, images: [ChatImage] = [],
        toolCalls: [ChatToolCall] = [], toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.images = images
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }
}

/// One tool call on an assistant turn (ADR 034). `argumentsJSON` is the OpenAI
/// stringified-args form, parsed back to an object when handed to the substrate
/// chat template.
public struct ChatToolCall: Sendable, Equatable {
    public let id: String?
    public let name: String
    public let argumentsJSON: String
    public init(id: String?, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

/// A decoded inline image carried into the LLM module (M71.1). Holds the raw
/// bytes + the declared media type; the VLM processor turns these into pixel
/// tensors (M71.2). Decoding lives here (not in the HTTP layer) so it is pure,
/// MLX-free, and unit-testable under `swift test`.
public struct ChatImage: Sendable, Equatable {
    public let data: Data
    /// e.g. "image/png", "image/jpeg" — the media type from the data: URL.
    public let mediaType: String
    public init(data: Data, mediaType: String) {
        self.data = data
        self.mediaType = mediaType
    }
}

/// Why an image content-part could not be accepted. Every case maps to a 400
/// at the HTTP boundary — never a silent drop.
public enum ChatImageError: Error, Equatable, Sendable {
    /// An `http(s)://` image URL. Rejected by the passive-oracle rule: the
    /// daemon performs NO outbound image fetch. The client must inline the
    /// image as a base64 `data:` URL.
    case remoteURLUnsupported
    /// A URL that is neither `data:` nor `http(s):` (unknown scheme).
    case unsupportedScheme
    /// A `data:` URL that is malformed, empty, or not an `image/*` payload.
    case malformedDataURL
}

extension ChatImage {
    /// Decode an OpenAI `image_url.url` string into a `ChatImage`.
    ///
    /// Passive-oracle (ADR 010): ONLY inline `data:` URLs are accepted. An
    /// `http(s)://` URL throws `remoteURLUnsupported` (the daemon never fetches
    /// images outbound); any other scheme throws `unsupportedScheme`.
    ///
    /// Accepts `data:[<media-type>][;base64],<payload>`. The media type must be
    /// `image/*`. Base64 payloads are decoded; a non-base64 (percent-encoded)
    /// payload is supported for completeness.
    public static func fromImageURL(_ url: String) throws -> ChatImage {
        let trimmed = url.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        if lower.hasPrefix("http://") || lower.hasPrefix("https://") {
            throw ChatImageError.remoteURLUnsupported
        }
        guard lower.hasPrefix("data:") else {
            throw ChatImageError.unsupportedScheme
        }
        let afterScheme = trimmed.dropFirst("data:".count)
        guard let comma = afterScheme.firstIndex(of: ",") else {
            throw ChatImageError.malformedDataURL
        }
        let meta = afterScheme[..<comma]  // e.g. "image/png;base64"
        let payload = String(afterScheme[afterScheme.index(after: comma)...])
        let isBase64 = meta.lowercased().contains(";base64")
        let mediaType =
            meta.split(separator: ";").first.map(String.init)?
            .trimmingCharacters(in: .whitespaces) ?? ""
        guard mediaType.lowercased().hasPrefix("image/") else {
            throw ChatImageError.malformedDataURL
        }
        let bytes: Data
        if isBase64 {
            guard let d = Data(base64Encoded: payload) else {
                throw ChatImageError.malformedDataURL
            }
            bytes = d
        } else {
            guard let decoded = payload.removingPercentEncoding,
                let d = decoded.data(using: .utf8)
            else {
                throw ChatImageError.malformedDataURL
            }
            bytes = d
        }
        guard !bytes.isEmpty else { throw ChatImageError.malformedDataURL }
        // Validate the bytes are a decodable image at the HTTP boundary so a
        // corrupt-but-base64 payload is a clean 400 here, not a 500 deep in
        // the VLM prepare path (M71.2). CIImage is lazy — this parses the
        // header, it does not render pixels.
        guard CIImage(data: bytes) != nil else {
            throw ChatImageError.malformedDataURL
        }
        return ChatImage(data: bytes, mediaType: mediaType)
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
