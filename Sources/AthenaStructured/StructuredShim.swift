import CAthenaStructured
import Foundation

/// Thin Swift surface over the Rust `outlines-core` structured-output
/// shim. M3.1 is FFI bring-up only (version + last-error); the safe
/// `StructuredGuide` (vocab/index/guide lifecycle, allowed-token mask,
/// advance/rollback) lands in M3.2.
public enum StructuredShim {
    /// The pinned outlines-core version the shim is built against.
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
