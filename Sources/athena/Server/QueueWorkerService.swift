import AthenaCore
import Logging
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
            // M46.1 operator legibility: log the start of graceful
            // shutdown. ServiceLifecycle 2.x triggers this hook when it
            // catches SIGTERM/SIGINT — the same moment that previously
            // surfaced only as macOS's `(CoreAnalytics) Entering exit
            // handler` system line, with nothing on Athena's side to say
            // "we got a signal." Now an operator reading `log show
            // --predicate 'subsystem == "athena"'` sees a daemon-warning
            // line at the cutoff timestamp; correlating with
            // launchd / system logs gives the sender. The Logger is
            // local to keep the closure @Sendable without capturing
            // anything from QueueWorkerService.
            Logger(label: AthenaLogLabel.daemon).warning(
                """
                daemon received graceful-shutdown signal; draining \
                in-flight requests and queue worker
                """)
            // Synchronous hook → hop onto the actor to flip the stop flag
            // and end the wake loop; `runWorker` then returns.
            Task { await queue.stop() }
        }
    }
}
