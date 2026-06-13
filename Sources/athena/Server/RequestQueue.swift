import AthenaServerKit
import AthenaStore
import Foundation
import Logging

/// Async request queue (M8.1). Jobs persist in the shared
/// `AthenaStore`; a single serial worker drains them through the same
/// governed module paths as the sync endpoints — the queue is for
/// non-blocking *submission*, not a separate priority lane (both
/// queued and sync work hit the one governor; natural FIFO there).
/// Survives restarts (queued/running jobs are re-drained).
actor RequestQueue {
    /// (kind, request bytes, owner) → (result bytes, error message).
    /// Set by `AthenaServer` to dispatch to chat/embeddings/etc.
    /// `owner` is the submitting principal (M12.6) so the executor can
    /// meter usage per principal (M27.2); nil when auth is disabled.
    typealias Executor =
        @Sendable (_ kind: String, _ request: Data, _ owner: String?)
            async -> (result: Data?, error: String?)

    /// Kinds a client may submit via the public `/v1/queue/:kind`
    /// route. Model-store ops are deliberately NOT here — they are
    /// enqueued only by the perm-gated `/api/models/*` handlers
    /// (M16.3), so a `queue.submit`-only caller cannot bypass the
    /// `model.write` gate via the generic queue endpoint.
    static let publicKinds: Set<String> = [
        "conversation", "embeddings",
    ]
    /// Every kind the executor can run (public + the internally-
    /// dispatched long-running model ops).
    static let kinds: Set<String> = publicKinds.union([
        "model_pull", "model_convert", "model_prune",
    ])

    private let store: AthenaStore
    private var executor: Executor?
    private let wake: AsyncStream<Void>
    private let signal: AsyncStream<Void>.Continuation
    private let log = Logger(label: AthenaLog.daemonLabel)
    /// Set by `stop()` on graceful shutdown (M33.2): the worker finishes
    /// the in-flight job, then exits instead of starting the next one.
    private var stopping = false
    /// Retention bounds for terminal (done/error/canceled) job results
    /// (M34.1). `ttlSecs` > 0 ⇒ prune results older than that; `maxRows`
    /// > 0 ⇒ cap total job rows (oldest terminal first). Both 0 ⇒ keep
    /// forever (opt-in, off by default). Swept on the worker idle path.
    private var ttlSecs = 0
    private var maxRows = 0
    /// Content opt-out (M34.2): when true, a job's `request` (prompt)
    /// blob is cleared once it reaches a terminal state, so inference
    /// inputs don't persist past completion.
    private var dropRequestContent = false

    init(store: AthenaStore) {
        self.store = store
        (wake, signal) = AsyncStream.makeStream()
    }

    func setExecutor(_ e: @escaping Executor) { executor = e }

    /// Configure result retention (M34.1) + content opt-out (M34.2).
    /// Called once at startup. 0 for either bound disables that
    /// dimension; `dropRequestContent=false` keeps prompts (default).
    func setRetention(
        ttlSecs: Int, maxRows: Int, dropRequestContent: Bool
    ) {
        self.ttlSecs = ttlSecs
        self.maxRows = maxRows
        self.dropRequestContent = dropRequestContent
    }

    /// Enqueue; returns the job id. `owner` = the submitting
    /// principal (M12.6); nil when auth is disabled.
    func submit(
        kind: String, request: Data, owner: String?
    ) async throws -> String {
        let id = UUID().uuidString
        try await store.insertJob(
            id: id, kind: kind, request: request, owner: owner)
        // M45.2: promoted from .info → .notice so queue submit
        // events survive `log show` without --info. Queue state
        // transitions are exactly the events an operator wants
        // visible at post-incident triage (per-job count is low
        // enough that persisted volume isn't an issue).
        log.notice("queue submit kind=\(kind) id=\(id)")
        signal.yield(())
        return id
    }

    /// `owner` (M65.6 / audit H6) is the optional defense-in-depth scope
    /// passed straight to the store: nil for admin / worker callers (see
    /// every job), the caller's principal for a scoped tenant.
    func status(id: String, owner: String? = nil) async -> JobRow? {
        await store.getJob(id: id, owner: owner)
    }

    func list(status: String?, owner: String? = nil) async -> [JobRow] {
        await store.listJobs(status: status, owner: owner)
    }

    /// A15 (M69.1) — queued depth via a `COUNT(*)` query (no blob
    /// materialization), not `listJobs(status:).count`.
    func depth() async -> Int {
        await store.queuedJobCount()
    }

    /// NA5 (M69.1) — the most recent `limit` job summaries (blob-free) for
    /// the /ui dashboard, newest first.
    func recentSummaries(limit: Int) async -> [JobSummary] {
        await store.recentJobSummaries(limit: limit)
    }

    /// Remove a job row entirely (cancel-if-queued is implicit — a
    /// deleted row is simply never picked up). Returns whether it
    /// existed.
    func remove(id: String) async -> Bool {
        await store.deleteJob(id: id)
    }

    /// Cancel a not-yet-running job. Running jobs can't be interrupted
    /// (the governed work is atomic); returns false then.
    func cancel(id: String) async -> Bool {
        // A23 — one atomic conditional UPDATE (cancel iff still queued) so the
        // worker can't pick the job up between a separate check and write and
        // have a running/done job clobbered back to `canceled`.
        (try? await store.cancelQueuedJob(id: id)) ?? false
    }

    /// Serial worker: drain on startup (re-pick interrupted jobs),
    /// then on each submit signal. One job at a time. Returns when
    /// `stop()` finishes the wake stream (graceful shutdown, M33.2).
    func runWorker() async {
        await drain()
        for await _ in wake { await drain() }
    }

    /// Graceful-shutdown signal (M33.2): stop after the in-flight job and
    /// end `runWorker`'s wake loop so the managed Service exits. The queue
    /// is durable — any job left queued/running re-drains on next start.
    func stop() {
        stopping = true
        signal.finish()
    }

    private func drain() async {
        guard let executor else { return }
        while true {
            // M33.2: on graceful shutdown, stop before picking up new
            // work (a job already running below still completes).
            if stopping { return }
            // A4/A24 (M69.1) — the oldest job still needing work (a prior
            // run may have left one in `running`), via a single indexed
            // LIMIT-1 query instead of loading every row's BLOBs to pick the
            // head.
            guard let job = await store.nextPendingJob() else {
                // Queue idle: bound retained results before sleeping
                // (M34.1). Runs on startup and each time the worker
                // drains empty after a wake.
                await sweep()
                return
            }
            if job.status != "running" {
                try? await store.updateJob(
                    id: job.id, status: "running", result: nil,
                    error: nil)
            }
            log.notice("queue job running kind=\(job.kind) id=\(job.id)")
            let (result, error) = await executor(
                job.kind, job.request, job.owner)
            try? await store.updateJob(
                id: job.id,
                status: error == nil ? "done" : "error",
                result: result, error: error)
            // M34.2: drop the prompt from disk once the job is finished
            // (the result the client polls for is retained separately).
            if dropRequestContent {
                try? await store.clearJobRequest(id: job.id)
            }
            if let error {
                log.warning(
                    "queue job error id=\(job.id) detail=\(error)")
            } else {
                log.notice("queue job done id=\(job.id)")
            }
        }
    }

    /// Bound retained terminal results (M34.1). Age-based TTL first, then
    /// the row cap. Non-fatal — a sweep hiccup must never sink the worker.
    /// 0 for a bound skips it.
    private func sweep() async {
        if ttlSecs > 0 {
            let cutoff = Date().timeIntervalSince1970 - Double(ttlSecs)
            let removed =
                (try? await store.pruneJobs(olderThan: cutoff)) ?? 0
            if removed > 0 {
                log.notice(
                    "queue retention: pruned \(removed) result(s) older than \(ttlSecs)s")
            }
        }
        if maxRows > 0 {
            let removed =
                (try? await store.trimJobs(maxRows: maxRows)) ?? 0
            if removed > 0 {
                log.notice(
                    "queue retention: trimmed \(removed) result(s) over cap \(maxRows)")
            }
        }
    }
}
