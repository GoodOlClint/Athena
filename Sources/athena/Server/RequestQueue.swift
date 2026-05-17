import AthenaStore
import Foundation

/// Async request queue (M8.1). Jobs persist in the shared
/// `AthenaStore`; a single serial worker drains them through the same
/// governed module paths as the sync endpoints — the queue is for
/// non-blocking *submission*, not a separate priority lane (both
/// queued and sync work hit the one governor; natural FIFO there).
/// Survives restarts (queued/running jobs are re-drained).
actor RequestQueue {
    /// (kind, request bytes) → (result bytes, error message). Set by
    /// `AthenaServer` to dispatch to chat/embeddings/etc.
    typealias Executor =
        @Sendable (_ kind: String, _ request: Data) async -> (
            result: Data?, error: String?
        )

    static let kinds: Set<String> = ["conversation", "embeddings"]

    private let store: AthenaStore
    private var executor: Executor?
    private let wake: AsyncStream<Void>
    private let signal: AsyncStream<Void>.Continuation

    init(store: AthenaStore) {
        self.store = store
        (wake, signal) = AsyncStream.makeStream()
    }

    func setExecutor(_ e: @escaping Executor) { executor = e }

    /// Enqueue; returns the job id.
    func submit(kind: String, request: Data) async throws -> String {
        let id = UUID().uuidString
        try await store.insertJob(
            id: id, kind: kind, request: request)
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
    /// then on each submit signal. One job at a time.
    func runWorker() async {
        await drain()
        for await _ in wake { await drain() }
    }

    private func drain() async {
        guard let executor else { return }
        while true {
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
            let (result, error) = await executor(
                job.kind, job.request)
            try? await store.updateJob(
                id: job.id,
                status: error == nil ? "done" : "error",
                result: result, error: error)
        }
    }
}
