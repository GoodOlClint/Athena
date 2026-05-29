import CAthenaStructured
import Foundation

/// Thin Swift surface over the Rust `llguidance` structured-output shim
/// (M53 — replaced the `outlines-core` engine; same C ABI). Exposes the
/// engine version + last-error; the safe `StructuredGuide`
/// (vocab/index/guide lifecycle, allowed-token mask, advance) wraps the
/// rest.
public enum StructuredShim {
    /// The pinned structured-output engine version the shim is built
    /// against (e.g. `llguidance-1.7.5`).
    public static var version: String {
        guard let c = oc_version() else { return "" }
        return String(cString: c)
    }

    /// The last error recorded by a failed shim call (empty if none).
    public static func lastError() -> String {
        let needed = oc_last_error(nil, 0)
        guard needed > 1 else { return "" }
        var buf = [CChar](repeating: 0, count: needed)
        _ = oc_last_error(&buf, needed)
        return String(cString: buf)
    }
}
