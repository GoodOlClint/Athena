import Foundation
import Hummingbird

/// In-process request metrics for the monitoring UI (M11.1). Pure
/// Swift, zero deps: counters + a bounded latency reservoir for
/// percentiles. Health/UI/metrics paths are excluded so the numbers
/// reflect real inference/admin work, not dashboard polling.
actor AthenaMetrics {
    struct Snapshot: Codable, Sendable {
        let totalRequests: Int
        let totalErrors: Int
        let byKind: [String: Int]
        let avgMs: Double
        let p50Ms: Double
        let p95Ms: Double
        let llmTokens: Int
        let sinceEpoch: Double
        /// Observations in the latency reservoir the quantiles/avg are
        /// computed over (bounded; M37 — backs the summary _count/_sum).
        let latencyWindow: Int
    }

    private var total = 0
    private var errors = 0
    private var byKind: [String: Int] = [:]
    private var tokens = 0
    private let started = Date().timeIntervalSince1970
    /// Last N request durations (ms); bounded so memory is flat.
    private var lat: [Double] = []
    private let cap = 1024

    func record(kind: String, ms: Double, isError: Bool) {
        total += 1
        byKind[kind, default: 0] += 1
        if isError { errors += 1 }
        lat.append(ms)
        if lat.count > cap { lat.removeFirst(lat.count - cap) }
    }

    func addTokens(_ n: Int) { tokens += n }

    func snapshot() -> Snapshot {
        let sorted = lat.sorted()
        func pct(_ p: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            let i = min(
                sorted.count - 1,
                Int((Double(sorted.count) * p).rounded(.down)))
            return sorted[i]
        }
        let avg =
            sorted.isEmpty
            ? 0 : sorted.reduce(0, +) / Double(sorted.count)
        return Snapshot(
            totalRequests: total, totalErrors: errors,
            byKind: byKind, avgMs: avg, p50Ms: pct(0.5),
            p95Ms: pct(0.95), llmTokens: tokens,
            sinceEpoch: started, latencyWindow: sorted.count)
    }

    /// Render a snapshot as Prometheus text-exposition format 0.0.4
    /// (M37). Counters end in `_total`; latency is a summary over the
    /// recent bounded window; gauges report start/uptime. Metric names
    /// are namespaced `athena_` (the project/goddess name). Pure, so it
    /// is testable without the actor.
    static func prometheus(_ s: Snapshot, now: Double) -> String {
        func esc(_ v: String) -> String {
            v.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n")
        }
        var out = ""
        out += "# HELP athena_requests_total Inference/admin requests "
        out += "counted (excludes health/UI/metrics).\n"
        out += "# TYPE athena_requests_total counter\n"
        out += "athena_requests_total \(s.totalRequests)\n"
        out += "# HELP athena_request_errors_total Requests that "
        out += "returned >= 400 or threw.\n"
        out += "# TYPE athena_request_errors_total counter\n"
        out += "athena_request_errors_total \(s.totalErrors)\n"
        out += "# HELP athena_requests_by_kind_total Requests by "
        out += "coarse kind.\n"
        out += "# TYPE athena_requests_by_kind_total counter\n"
        for k in s.byKind.keys.sorted() {
            out +=
                "athena_requests_by_kind_total{kind=\"\(esc(k))\"} "
                + "\(s.byKind[k] ?? 0)\n"
        }
        out += "# HELP athena_llm_tokens_total Cumulative LLM tokens "
        out += "(prompt+completion) metered.\n"
        out += "# TYPE athena_llm_tokens_total counter\n"
        out += "athena_llm_tokens_total \(s.llmTokens)\n"
        out += "# HELP athena_request_latency_ms Request latency (ms) "
        out += "over a recent bounded window.\n"
        out += "# TYPE athena_request_latency_ms summary\n"
        out += "athena_request_latency_ms{quantile=\"0.5\"} \(s.p50Ms)\n"
        out += "athena_request_latency_ms{quantile=\"0.95\"} \(s.p95Ms)\n"
        out +=
            "athena_request_latency_ms_sum "
            + "\(s.avgMs * Double(s.latencyWindow))\n"
        out += "athena_request_latency_ms_count \(s.latencyWindow)\n"
        out += "# HELP athena_start_time_seconds Daemon start time "
        out += "(unix epoch seconds).\n"
        out += "# TYPE athena_start_time_seconds gauge\n"
        out += "athena_start_time_seconds \(s.sinceEpoch)\n"
        out += "# HELP athena_uptime_seconds Seconds since the daemon "
        out += "started.\n"
        out += "# TYPE athena_uptime_seconds gauge\n"
        out += "athena_uptime_seconds \(now - s.sinceEpoch)\n"
        return out
    }
}

/// Times every routed request and records it by a coarse `kind`
/// derived from the path. Dashboard/health/metrics traffic is not
/// counted (returns nil ⇒ pass through untimed).
struct MetricsMiddleware<Context: RequestContext>: RouterMiddleware {
    let metrics: AthenaMetrics

    static func kind(_ path: String) -> String? {
        if path == "/v1/chat/completions" { return "chat" }
        if path == "/v1/embeddings" { return "embeddings" }
        if path == "/v1/audio/transcriptions" { return "transcription" }
        if path == "/v1/audio/diarizations" { return "diarization" }
        if path == "/v1/audio/embeddings" { return "speaker_embeddings" }
        if path.hasPrefix("/v1/vectors") { return "vectors" }
        if path.hasPrefix("/v1/queue") { return "queue" }
        if path.hasPrefix("/v1/store") { return "store" }
        if path.hasPrefix("/api/") { return "ollama" }
        return nil  // /healthz, /ui*, /metrics, etc. — not work
    }

    func handle(
        _ request: Request, context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard let kind = Self.kind(request.uri.path) else {
            return try await next(request, context)
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        func ms() -> Double {
            Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        }
        do {
            let resp = try await next(request, context)
            await metrics.record(
                kind: kind, ms: ms(),
                isError: resp.status.code >= 400)
            return resp
        } catch {
            await metrics.record(kind: kind, ms: ms(), isError: true)
            throw error
        }
    }
}
