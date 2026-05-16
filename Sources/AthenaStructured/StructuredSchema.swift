import Foundation

/// Type-erased JSON value — captures an arbitrary user-supplied schema
/// object so it can be round-tripped and re-serialized to the string the
/// outlines-core shim expects.
public indirect enum JSONValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let b = try? c.decode(Bool.self) {
            self = .bool(b)
        } else if let d = try? c.decode(Double.self) {
            self = .number(d)
        } else if let s = try? c.decode(String.self) {
            self = .string(s)
        } else if let a = try? c.decode([JSONValue].self) {
            self = .array(a)
        } else {
            self = .object(try c.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let n): try c.encode(n)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    /// Compact JSON string (sorted keys ⇒ deterministic).
    public func jsonString() -> String? {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? enc.encode(self) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

/// Maps an OpenAI `response_format` into the schema string the
/// structured-output Guide compiles, or nil for unconstrained output.
public enum StructuredSchema {
    /// - `json_schema` → the supplied schema object, serialized.
    /// - `json_object` → a permissive object schema.
    /// - anything else (incl. "text"/nil) → nil (no constraint).
    public static func schemaJSON(
        responseFormatType type: String?, jsonSchema: JSONValue?
    ) -> String? {
        switch type {
        case "json_schema":
            return jsonSchema?.jsonString()
        case "json_object":
            return #"{"type":"object"}"#
        default:
            return nil
        }
    }
}
