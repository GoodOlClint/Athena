import CoreFoundation
import Foundation

/// Float-faithful JSON emission for `config.json` rewrites.
///
/// `athena convert` rewrites a model's `config.json` (to stamp the
/// `quantization` block). The obvious round-trip — `JSONSerialization.jsonObject`
/// then `JSONSerialization.data` (or `JSONEncoder`) — is **lossy on whole-valued
/// floats**: every Foundation JSON serializer emits a `Double` like `30.0` as the
/// integer `30`. That breaks the upstream MLX/transformers ecosystem, whose
/// strict-config validation rejects an int where a `float` field is declared
/// (e.g. gemma4 `final_logit_softcapping: 30.0`). An Athena convert must load
/// anywhere a standard `mlx-community` convert does.
///
/// The *parse* keeps the distinction: `JSONSerialization.jsonObject` stores a
/// JSON float in an `NSNumber` for which `CFNumberIsFloatType` is true, even when
/// the value is whole. This emitter reads that flag and prints whole floats with
/// a trailing `.0`, so a field that was a float in the source stays a float —
/// regardless of architecture (gemma4, qwen3_5, …). Fix at the single emission
/// point so every arch benefits.
///
/// MLX-free (ADR 008/009): pure Foundation, unit-pinned without a model load.
public enum ConfigJSONEmit {
    /// Serialize a JSON object parsed by `JSONSerialization.jsonObject`,
    /// preserving the int/float distinction the parse captured. Keys are sorted
    /// (stable diffs); indentation is 2-space. Accepts both `NSNumber` (from the
    /// parse) and bridged Swift scalars (values the caller injected), so the
    /// `quantization` block built from Swift `Int`/`String` emits correctly too.
    public static func data(from object: [String: Any]) throws -> Data {
        var out = ""
        try emit(object, indent: 0, into: &out)
        out += "\n"
        guard let d = out.data(using: .utf8) else {
            throw EmitError.encoding
        }
        return d
    }

    public enum EmitError: Error { case encoding, unsupportedValue }

    private static func emit(
        _ value: Any, indent: Int, into out: inout String
    ) throws {
        let pad = String(repeating: " ", count: indent * 2)
        let inner = String(repeating: " ", count: (indent + 1) * 2)

        switch value {
        case let dict as [String: Any]:
            if dict.isEmpty { out += "{}"; return }
            out += "{\n"
            let keys = dict.keys.sorted()
            for (i, k) in keys.enumerated() {
                out += inner + encodeString(k) + ": "
                try emit(dict[k]!, indent: indent + 1, into: &out)
                out += i == keys.count - 1 ? "\n" : ",\n"
            }
            out += pad + "}"

        case let arr as [Any]:
            if arr.isEmpty { out += "[]"; return }
            out += "[\n"
            for (i, el) in arr.enumerated() {
                out += inner
                try emit(el, indent: indent + 1, into: &out)
                out += i == arr.count - 1 ? "\n" : ",\n"
            }
            out += pad + "]"

        case is NSNull:
            out += "null"

        case let s as String:
            out += encodeString(s)

        case let n as NSNumber:
            out += encodeNumber(n)

        default:
            throw EmitError.unsupportedValue
        }
    }

    private static func encodeNumber(_ n: NSNumber) -> String {
        // JSON `true`/`false` parse to `__NSCFBoolean` (an NSNumber); emit as
        // booleans, never 1/0.
        if CFGetTypeID(n) == CFBooleanGetTypeID() {
            return n.boolValue ? "true" : "false"
        }
        // A float-typed number — keep it a float. Swift's `Double` description
        // already renders whole values with `.0` (`30.0`, `1000000.0`) and
        // fractional/exponent values round-trippably (`0.5`, `1e-06`).
        if CFNumberIsFloatType(n) {
            return String(n.doubleValue)
        }
        return n.stringValue  // integer
    }

    private static func encodeString(_ s: String) -> String {
        var r = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": r += "\\\""
            case "\\": r += "\\\\"
            case "\n": r += "\\n"
            case "\r": r += "\\r"
            case "\t": r += "\\t"
            case "\u{08}": r += "\\b"
            case "\u{0C}": r += "\\f"
            case let c where c.value < 0x20:
                r += String(format: "\\u%04x", c.value)
            default:
                r.unicodeScalars.append(scalar)
            }
        }
        r += "\""
        return r
    }
}
