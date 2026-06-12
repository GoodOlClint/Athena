import Foundation

/// Minimal `multipart/form-data` reader — just enough for OpenAI
/// `/v1/audio/transcriptions` (one binary `file` part + a few short
/// text fields). Not a general RFC 7578 implementation: no nested
/// multipart, no transfer-encoding. Hand-rolled to avoid a dependency.
struct MultipartForm {
    struct Part {
        let name: String
        let filename: String?
        let data: Data
    }
    let parts: [Part]

    /// Cap on the number of inter-boundary segments parsed from one body.
    /// A real transcription form is one `file` part plus a handful of
    /// short text fields; this bounds a body packed with boundaries from
    /// spawning unbounded segment/part work (A9).
    static let maxParts = 256

    /// `boundary` is the value from the `Content-Type` header
    /// (`multipart/form-data; boundary=...`).
    init?(body: Data, boundary: String) {
        let dashBoundary = Data("--\(boundary)".utf8)
        let crlf = Data("\r\n".utf8)
        let headerSep = Data("\r\n\r\n".utf8)

        // Split on the boundary delimiter using the stdlib substring
        // search (`firstRange`) instead of an O(n·m) per-byte compare on
        // the request thread, and cap the segment count (A9).
        var segments: [Data] = []
        var searchStart = body.startIndex
        while segments.count < Self.maxParts,
            let r = body.firstRange(
                of: dashBoundary, in: searchStart ..< body.endIndex)
        {
            if r.lowerBound > searchStart {
                segments.append(body[searchStart ..< r.lowerBound])
            }
            searchStart = r.upperBound
        }

        var out: [Part] = []
        for seg in segments {
            // A real part begins with CRLF then headers; the closing
            // delimiter segment starts with "--". Skip non-parts.
            guard seg.count > crlf.count else { continue }
            let afterPrefix = seg[(seg.startIndex + crlf.count)...]
            guard
                let hsRange = afterPrefix.firstRange(of: headerSep)
            else { continue }
            let headerData = afterPrefix[afterPrefix.startIndex ..< hsRange.lowerBound]
            var body = afterPrefix[hsRange.upperBound...]
            // Trim the trailing CRLF that precedes the next boundary.
            if body.count >= crlf.count,
                body[(body.endIndex - crlf.count)...] == crlf
            {
                body = body[..<(body.endIndex - crlf.count)]
            }
            guard
                let headers = String(data: headerData, encoding: .utf8)
            else { continue }

            var name: String?
            var filename: String?
            for line in headers.split(separator: "\r\n") {
                let l = String(line)
                guard
                    l.lowercased().hasPrefix("content-disposition:")
                else { continue }
                for token in l.split(separator: ";") {
                    let t = token.trimmingCharacters(in: .whitespaces)
                    if t.hasPrefix("name=") {
                        name = Self.unquote(String(t.dropFirst(5)))
                    } else if t.hasPrefix("filename=") {
                        filename = Self.unquote(String(t.dropFirst(9)))
                    }
                }
            }
            guard let name else { continue }
            out.append(
                Part(
                    name: name, filename: filename,
                    data: Data(body)))
        }
        if out.isEmpty { return nil }
        self.parts = out
    }

    private static func unquote(_ s: String) -> String {
        var v = s
        if v.hasPrefix("\"") && v.hasSuffix("\"") && v.count >= 2 {
            v = String(v.dropFirst().dropLast())
        }
        return v
    }

    func first(_ name: String) -> Part? {
        parts.first { $0.name == name }
    }

    func text(_ name: String) -> String? {
        first(name).flatMap { String(data: $0.data, encoding: .utf8) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    /// All values supplied for a (possibly repeated) field, e.g.
    /// `timestamp_granularities[]`.
    func texts(_ name: String) -> [String] {
        parts.filter { $0.name == name }.compactMap {
            String(data: $0.data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    /// Parse the boundary token out of a Content-Type header value.
    static func boundary(fromContentType ct: String) -> String? {
        for token in ct.split(separator: ";") {
            let t = token.trimmingCharacters(in: .whitespaces)
            if t.lowercased().hasPrefix("boundary=") {
                return unquote(String(t.dropFirst(9)))
            }
        }
        return nil
    }
}
