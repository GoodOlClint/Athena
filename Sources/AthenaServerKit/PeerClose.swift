import Foundation
import NIOCore

/// #209 — cancel a request's work when its client goes away.
///
/// Hummingbird runs a responder inline on the connection task and reads
/// nothing from the socket until it returns, so a client that disconnects
/// mid-request never cancels a handler that is awaiting a result (a
/// non-streaming decode drain). `PeerClose.cancelling` watches the connection
/// for the close itself and cancels the handler's work, which reaches the
/// decode loops through their existing task-cancellation handlers (ADR 029).
public enum PeerClose {
    /// Run `operation`; if `channel`'s peer has closed or closes before it
    /// finishes, cancel it and return whatever it returns once cancelled.
    /// A channel without a `PeerCloseLatch` (or `nil`) just runs it.
    ///
    /// The server enables `allowRemoteHalfClosure`, so a client FIN arrives as
    /// `ChannelEvent.inputClosed`, not a channel close. It is treated as
    /// abandonment: a client that half-closes its write side after sending the
    /// request and still waits for the response gets its request cancelled.
    public static func cancelling<T: Sendable>(
        _ channel: (any Channel)?,
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        guard
            let channel,
            let latch = try? await channel.pipeline.handler(
                type: PeerCloseLatch.self
            ).get()
        else { return try await operation() }
        let closed = latch.subscribe()
        defer { latch.unsubscribe() }
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
}

/// Per-connection record of whether the client has gone away, installed with
/// the connection (`additionalChannelHandlers`) so a close that lands before a
/// request's handler runs (a client that sends and closes in one burst) is
/// still seen. On `handlerAdded` it puts a forwarder first in the pipeline:
/// the HTTP pipelining handler can hold back `inputClosed` while a request is
/// in flight, so the latch must see the event before any codec does. It also
/// records the events itself, a backstop should the forwarder be missing.
/// `@unchecked Sendable`: every mutable field is read and written under `lock`.
public final class PeerCloseLatch: ChannelInboundHandler, RemovableChannelHandler,
    @unchecked Sendable
{
    public typealias InboundIn = NIOAny
    private let lock = NSLock()
    private var closed = false
    private var waiter: AsyncStream<Void>.Continuation?

    public init() {}

    public func handlerAdded(context: ChannelHandlerContext) {
        try? context.pipeline.syncOperations.addHandler(
            Front(self), position: .first)
    }

    public func userInboundEventTriggered(
        context: ChannelHandlerContext, event: Any
    ) {
        if let e = event as? ChannelEvent, e == .inputClosed { markClosed() }
        context.fireUserInboundEventTriggered(event)
    }

    public func channelInactive(context: ChannelHandlerContext) {
        markClosed()
        context.fireChannelInactive()
    }

    /// A stream that yields once the peer has closed — at once if it already
    /// has. One subscriber at a time (HTTP/1 requests on a connection are
    /// sequential); a new subscription replaces the previous one.
    func subscribe() -> AsyncStream<Void> {
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        let already = lock.withLock {
            waiter?.finish()
            waiter = continuation
            return closed
        }
        if already { continuation.yield() }
        return stream
    }

    func unsubscribe() {
        lock.withLock {
            waiter?.finish()
            waiter = nil
        }
    }

    fileprivate func markClosed() {
        let w = lock.withLock {
            closed = true
            return waiter
        }
        w?.yield()
    }

    private final class Front: ChannelInboundHandler, RemovableChannelHandler,
        Sendable
    {
        typealias InboundIn = NIOAny
        private let latch: PeerCloseLatch

        init(_ latch: PeerCloseLatch) { self.latch = latch }

        func userInboundEventTriggered(
            context: ChannelHandlerContext, event: Any
        ) {
            if let e = event as? ChannelEvent, e == .inputClosed {
                latch.markClosed()
            }
            context.fireUserInboundEventTriggered(event)
        }

        func channelInactive(context: ChannelHandlerContext) {
            latch.markClosed()
            context.fireChannelInactive()
        }
    }
}
