import ServiceLifecycle

/// Wraps the serial queue worker as a ServiceLifecycle `Service` (M33.2)
/// so it joins the application's `ServiceGroup` instead of running as a
/// detached `Task`. On graceful shutdown (SIGTERM) the group signals the
/// worker, which finishes its in-flight job and exits within the stop
/// window — alongside the HTTP server draining its in-flight requests.
/// Anything still queued/running is durable and re-drains on next start.
struct QueueWorkerService: Service {
    let queue: RequestQueue

    func run() async throws {
        await withGracefulShutdownHandler {
            await queue.runWorker()
        } onGracefulShutdown: {
            // Synchronous hook → hop onto the actor to flip the stop flag
            // and end the wake loop; `runWorker` then returns.
            Task { await queue.stop() }
        }
    }
}
