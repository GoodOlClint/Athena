import Foundation

/// Errors surfaced by the governor and module lifecycle. Each carries an HTTP
/// classification so the serve path never turns a memory-budget event into an
/// unhandled Metal abort — it becomes a classified 503 instead.
public enum AthenaError: Error, Sendable, Equatable {
    /// Admission was refused: the request would exceed the global budget and
    /// nothing evictable could be freed. Governed backpressure → 503.
    case memoryBudgetExceeded(requested: Int, available: Int, module: ModuleID)
    /// A module load failed in the substrate.
    case moduleLoadFailed(ModuleID, reason: String)
    /// No module is registered under this id.
    case moduleNotRegistered(ModuleID)
    /// An MLX/Metal allocation failed (genuine device OOM, distinct
    /// from governed admission). Classified to 503 so the client sees
    /// retryable backpressure, never a bare 500 / process abort.
    case metalOutOfMemory(module: ModuleID?, detail: String)
    /// The request's prompt would need more KV/prompt-cache bytes than
    /// the governor-owned global cap allows. Governed 503 (refuse big
    /// contexts before they OOM the box), not a bare failure.
    case promptCacheCapExceeded(requestedBytes: Int, capBytes: Int)
    /// The audio exceeds a model's single-pass capacity (e.g. the
    /// offline diarizer's learned positional table). A client error
    /// (400) — split the audio into shorter segments — NOT a silent
    /// empty result.
    case audioTooLong(module: ModuleID, seconds: Double, maxSeconds: Double)
    /// A requested audio segment is empty/out-of-range or too short to
    /// yield any feature frame. A client error (400) — never a silent
    /// zero embedding.
    case audioSegmentInvalid(module: ModuleID, detail: String)
    /// Generation outlived the per-request inference deadline
    /// (`request_timeout_secs`) and was cancelled. A gateway timeout
    /// (504) so a runaway decode bounds the caller's wait (and frees the
    /// worker) instead of being capped only by `max_tokens`.
    case requestTimedOut(seconds: Int)
    /// The request named a model that is not in the configured/selectable
    /// set for its module (e.g. per-request embedding selection). A client
    /// error (400) — NEVER a silent fallback to a different model, which
    /// for embeddings would return wrong-dimension vectors, and never an
    /// arbitrary on-request download.
    case modelNotAvailable(requested: String, available: [String])

    /// HTTP status the serve path should return for this error.
    public var httpStatus: Int {
        switch self {
        case .memoryBudgetExceeded: return 503
        case .moduleLoadFailed: return 500
        case .moduleNotRegistered: return 404
        case .metalOutOfMemory: return 503
        case .promptCacheCapExceeded: return 503
        case .audioTooLong: return 400
        case .audioSegmentInvalid: return 400
        case .requestTimedOut: return 504
        case .modelNotAvailable: return 400
        }
    }

    /// Stable machine-readable code (OpenAI-style error `code`).
    public var code: String {
        switch self {
        case .memoryBudgetExceeded: return "memory_budget_exceeded"
        case .moduleLoadFailed: return "module_load_failed"
        case .moduleNotRegistered: return "module_not_registered"
        case .metalOutOfMemory: return "metal_oom"
        case .promptCacheCapExceeded: return "prompt_cache_cap_exceeded"
        case .audioTooLong: return "audio_too_long"
        case .audioSegmentInvalid: return "audio_segment_invalid"
        case .requestTimedOut: return "inference_timeout"
        case .modelNotAvailable: return "model_not_available"
        }
    }

    public var message: String {
        switch self {
        case let .memoryBudgetExceeded(requested, available, module):
            return "Insufficient governed memory for \(module.rawValue): "
                + "requested \(requested) B, \(available) B available after eviction."
        case let .moduleLoadFailed(module, reason):
            return "Module \(module.rawValue) failed to load: \(reason)"
        case let .moduleNotRegistered(module):
            return "No module registered for \(module.rawValue)."
        case let .metalOutOfMemory(module, detail):
            let who = module.map { " for \($0.rawValue)" } ?? ""
            return "Metal/MLX out of memory\(who): \(detail)"
        case let .promptCacheCapExceeded(requested, cap):
            return "Prompt too large for the governed prompt-cache "
                + "cap: needs ~\(requested) B, cap \(cap) B."
        case let .audioTooLong(module, seconds, maxSeconds):
            return String(
                format:
                    "Audio (%.0fs) exceeds the %@ model's single-pass "
                    + "limit of ~%.0fs. Split it into shorter segments "
                    + "and submit them separately.",
                seconds, module.rawValue, maxSeconds)
        case let .audioSegmentInvalid(module, detail):
            return "Invalid audio segment for \(module.rawValue): \(detail)"
        case let .requestTimedOut(seconds):
            return "Inference exceeded the \(seconds)s request timeout "
                + "and was cancelled."
        case let .modelNotAvailable(requested, available):
            return "Model '\(requested)' is not available. Configured "
                + "models: \(available.joined(separator: ", "))."
        }
    }

    /// Does `error` look like a genuine MLX/Metal allocation failure
    /// (vs. governed admission, which is `memoryBudgetExceeded`)?
    /// Substring match on the substrate/Metal failure vocabulary —
    /// focused to avoid matching incidental "metal" text.
    public static func isMetalOOM(_ error: any Error) -> Bool {
        if error is AthenaError { return false }
        let s = String(describing: error).lowercased()
        let needles = [
            "out of memory", "insufficient memory",
            "failed to allocate", "cannot allocate",
            "metal allocation", "mtlbuffer", "newbufferwithlength",
            "[metal] out", "vm_allocate",
        ]
        return needles.contains { s.contains($0) }
    }

    /// Map an arbitrary thrown error to a classified `AthenaError`:
    /// pass through existing ones, route Metal OOM to the 503 case,
    /// else a 500 `moduleLoadFailed` (the generic substrate failure).
    public static func classify(
        _ error: any Error, module: ModuleID?
    ) -> AthenaError {
        if let a = error as? AthenaError { return a }
        if isMetalOOM(error) {
            return .metalOutOfMemory(
                module: module, detail: String(describing: error))
        }
        return .moduleLoadFailed(
            module ?? .llm, reason: String(describing: error))
    }
}
