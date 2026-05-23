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

    init(store: AthenaStore) {
        self.store = store
        (wake, signal) = AsyncStream.makeStream()
    }

    func setExecutor(_ e: @escaping Executor) { executor = e }

    /// Enqueue; returns the job id. `owner` = the submitting
    /// principal (M12.6); nil when auth is disabled.
    func submit(
        kind: String, request: Data, owner: String?
    ) async throws -> String {
        let id = UUID().uuidString
        try await store.insertJob(
            id: id, kind: kind, request: request, owner: owner)
        log.info("queue submit kind=\(kind) id=\(id)")
        signal.yield(())
        return id
    }

    func status(id: String) async -> JobRow? {
        await store.getJob(id: id)
    }

    func list(status: String?) async -> [JobRow] {
        await store.listJobs(status: status)
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
        guard let j = await store.getJob(id: id) else { return false }
        guard j.status == "queued" else { return false }
        try? await store.updateJob(
            id: id, status: "canceled", result: nil, error: nil)
        return true
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
            // Oldest job still needing work (a prior run may have left
            // one in `running`).
            let pending = await store.listJobs().first {
                $0.status == "queued" || $0.status == "running"
            }
            guard let job = pending else { return }
            if job.status != "running" {
                try? await store.updateJob(
                    id: job.id, status: "running", result: nil,
                    error: nil)
            }
            log.info("queue job running kind=\(job.kind) id=\(job.id)")
            let (result, error) = await executor(
                job.kind, job.request, job.owner)
            try? await store.updateJob(
                id: job.id,
                status: error == nil ? "done" : "error",
                result: result, error: error)
            if let error {
                log.warning(
                    "queue job error id=\(job.id) detail=\(error)")
            } else {
                log.info("queue job done id=\(job.id)")
            }
        }
    }
}
