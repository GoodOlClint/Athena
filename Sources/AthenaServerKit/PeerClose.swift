import NIOCore

/// #209 — cancel a request's work when its client goes away.
///
/// Hummingbird runs a responder inline on the connection task and reads
/// nothing from the socket until it returns, so a client that disconnects
/// mid-request never cancels a handler that is awaiting a result (a
/// non-streaming decode drain). `PeerClose.cancelling` watches the channel for
/// the close itself and cancels the handler's work, which reaches the decode
/// loops through their existing task-cancellation handlers (ADR 029).
public enum PeerClose {
    /// Run `operation`; if `channel`'s peer closes first, cancel it and return
    /// whatever it returns once cancelled. `nil` channel ⇒ just run it.
    ///
    /// The server enables `allowRemoteHalfClosure`, so a client FIN arrives as
    /// `ChannelEvent.inputClosed`, not a channel close. It is treated as
    /// abandonment: a client that half-closes its write side after sending the
    /// request and still waits for the response gets its request cancelled.
    public static func cancelling<T: Sendable>(
        _ channel: (any Channel)?,
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        guard let channel else { return try await operation() }
        let (closed, signal) = AsyncStream<Void>.makeStream()
        let watcher = Watcher(signal)
        do {
            try await channel.pipeline.addHandler(watcher, position: .first)
                .get()
        } catch {
            // Only fails when the channel is already closed: the peer is gone.
            signal.yield()
        }
        defer {
            signal.finish()
            channel.pipeline.removeHandler(watcher, promise: nil)
        }
        return try await cancelling(on: closed, operation)
    }

    /// The MLX-free core: run `operation` until it finishes, cancelling it as
    /// soon as `closed` yields. Returns (or rethrows) the operation's own
    /// outcome either way, so a handler can still build its error response.
    public static func cancelling<T: Sendable>(
        on closed: AsyncStream<Void>,
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: Outcome<T>.self) { group in
            group.addTask { .value(try await operation()) }
            group.addTask {
                for await _ in closed { return .peerClosed }
                return .peerClosed
            }
            var result: T?
            while let next = try await group.next() {
                group.cancelAll()
                if case .value(let v) = next { result = v }
            }
            guard let result else { throw CancellationError() }
            return result
        }
    }

    private enum Outcome<T: Sendable>: Sendable {
        case value(T)
        case peerClosed
    }

    /// Sits first in the pipeline so it sees the close before any codec.
    final class Watcher: ChannelInboundHandler, RemovableChannelHandler,
        Sendable
    {
        typealias InboundIn = NIOAny
        private let signal: AsyncStream<Void>.Continuation

        init(_ signal: AsyncStream<Void>.Continuation) { self.signal = signal }

        func userInboundEventTriggered(
            context: ChannelHandlerContext, event: Any
        ) {
            if let e = event as? ChannelEvent, e == .inputClosed {
                signal.yield()
            }
            context.fireUserInboundEventTriggered(event)
        }

        func channelInactive(context: ChannelHandlerContext) {
            signal.yield()
            context.fireChannelInactive()
        }
    }
}
