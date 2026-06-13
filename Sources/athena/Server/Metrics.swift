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
        /// H14 (M66.1) — cumulative audit-log write failures. The M30 audit
        /// trail is a security record; a failed append is non-fatal to the
        /// mutation that triggered it but MUST be visible, so it surfaces
        /// here (scrapeable via /metrics) as well as in the error log.
        /// Non-zero ⇒ the trail has gaps; alert on any increase.
        let auditWriteFailures: Int
    }

    private var total = 0
    private var errors = 0
    private var byKind: [String: Int] = [:]
    private var tokens = 0
    private let started = Date().timeIntervalSince1970
    /// Last N request durations (ms); bounded so memory is flat.
    private var lat: [Double] = []
    private let cap = 1024
    /// M43.1 — live request count (incremented on handler entry,
    /// decremented on exit/throw via `MetricsMiddleware`). Surfaces in
    /// `/healthz` so a hung daemon is legible without scraping
    /// `/metrics`.
    private var inflight = 0
    /// Unix-epoch seconds the most recent request entered the
    /// metered surface; 0 ⇒ none since boot.
    private var lastRequestAt: Double = 0
    /// M60.1 — most recent decode throughput (tok/s), recorded by the
    /// `collectMetered` heartbeat each interval it observes the decode
    /// phase. Surfaced on `/healthz` so a client can read the live rate
    /// (alongside `thermalState`) and back off BEFORE submitting a call
    /// that would cross its deadline, instead of eating a 540 s cancel.
    /// Holds the last observed value while idle; 0 ⇒ none since boot.
    private var lastDecodeTokensPerSec: Double = 0
    /// H14 (M66.1) — count of audit-log writes that failed (the store
    /// `addAudit` threw). Surfaced on /metrics so a gap in the security
    /// trail is observable, not silently swallowed.
    private var auditWriteFailures = 0

    func record(kind: String, ms: Double, isError: Bool) {
        total += 1
        byKind[kind, default: 0] += 1
        if isError { errors += 1 }
        lat.append(ms)
        if lat.count > cap { lat.removeFirst(lat.count - cap) }
    }

    func addTokens(_ n: Int) { tokens += n }

    /// H14 (M66.1) — record one failed audit-log write.
    func recordAuditWriteFailure() { auditWriteFailures += 1 }

    /// M43.1 — call on handler entry. Increments the live counter and
    /// stamps `lastRequestAt`.
    func enter() {
        inflight += 1
        lastRequestAt = Date().timeIntervalSince1970
    }

    /// M43.1 — call on handler exit (or throw). Never goes negative.
    func leave() {
        if inflight > 0 { inflight -= 1 }
    }

    /// M60.1 — record the live decode rate observed by the heartbeat.
    /// Called only while in the decode phase, so the value reflects real
    /// generation throughput (not a setup/prefill zero).
    func recordDecodeRate(_ tps: Double) { lastDecodeTokensPerSec = tps }

    /// M43.1 — snapshot of the live signals the /healthz response
    /// surfaces. `lastRequestAt == 0` ⇒ never-served sentinel.
    /// M60.1 adds the most recent decode tok/s (0 ⇒ none since boot).
    func healthFields()
        -> (inflight: Int, lastRequestAt: Double, decodeTokensPerSec: Double)
    {
        (inflight, lastRequestAt, lastDecodeTokensPerSec)
    }

    func snapshot() -> Snapshot {
        let sorted = lat.sorted()
        func pct(_ p: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            // A25 — nearest-rank: rank = ceil(p·n), index = rank-1, clamped to
            // [0, n-1]. The prior `floor(p·n)` under-counted the rank (e.g.
            // p95 over 20 samples picked index 19 correctly but p50 over 10
            // picked the 6th, not the 5th, value).
            let rank = Int((Double(sorted.count) * p).rounded(.up))
            let i = min(sorted.count - 1, max(0, rank - 1))
            return sorted[i]
        }
        let avg =
            sorted.isEmpty
            ? 0 : sorted.reduce(0, +) / Double(sorted.count)
        return Snapshot(
            totalRequests: total, totalErrors: errors,
            byKind: byKind, avgMs: avg, p50Ms: pct(0.5),
            p95Ms: pct(0.95), llmTokens: tokens,
            sinceEpoch: started, latencyWindow: sorted.count,
            auditWriteFailures: auditWriteFailures)
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
        out += "# HELP athena_audit_write_failures_total Audit-log writes "
        out += "that failed (security trail gaps).\n"
        out += "# TYPE athena_audit_write_failures_total counter\n"
        out += "athena_audit_write_failures_total \(s.auditWriteFailures)\n"
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
        // `@Sendable` so it can be called from the body-completion closure
        // (NA3) as well as the synchronous error path; it captures only the
        // Sendable `t0`.
        @Sendable func ms() -> Double {
            Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        }
        await metrics.enter()
        do {
            var resp = try await next(request, context)
            let isError = resp.status.code >= 400
            // NA3 — record latency + drop the inflight gauge at body
            // COMPLETION (which, for a streamed response, is after the GPU
            // decode that fills it), not when the handler returns its lazy
            // body. Otherwise /healthz `inflight` and p50/p95 read ~0 while
            // the daemon is actively streaming — defeating the "is it hung or
            // working" legibility goal — and the latency window misses the
            // decode entirely.
            let metrics = self.metrics
            resp.body = resp.body.onBodyComplete {
                await metrics.record(kind: kind, ms: ms(), isError: isError)
                await metrics.leave()
            }
            return resp
        } catch {
            await metrics.record(kind: kind, ms: ms(), isError: true)
            await metrics.leave()
            throw error
        }
    }
}
