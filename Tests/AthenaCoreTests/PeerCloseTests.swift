import Foundation
import NIOCore
import NIOEmbedded
import XCTest

@testable import AthenaServerKit

/// #209 — a client disconnect cancels the in-flight handler work.
final class PeerCloseTests: XCTestCase {

    /// Stands in for a decode drain: runs until cancelled (bounded, so a
    /// regression fails instead of hanging the suite).
    private static func untilCancelled() async -> String {
        let deadline = Date().addingTimeInterval(3)
        while !Task.isCancelled {
            if Date() > deadline { return "timed-out" }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return "cancelled"
    }

    private func connectedChannel() async throws -> NIOAsyncTestingChannel {
        let channel = NIOAsyncTestingChannel()
        try await channel.pipeline.addHandler(PeerCloseLatch()).get()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 1))
        return channel
    }

    func testCloseCancelsOperation() async throws {
        let (closed, signal) = AsyncStream<Void>.makeStream()
        Task {
            try? await Task.sleep(nanoseconds: 50_000_000)
            signal.yield()
        }
        let r = try await PeerClose.cancelling(on: closed) {
            await Self.untilCancelled()
        }
        XCTAssertEqual(r, "cancelled")
    }

    func testOperationFinishingFirstIsNotCancelled() async throws {
        let (closed, _) = AsyncStream<Void>.makeStream()
        let r = try await PeerClose.cancelling(on: closed) {
            Task.isCancelled ? "cancelled" : "done"
        }
        XCTAssertEqual(r, "done")
    }

    func testOperationErrorPropagates() async {
        struct Boom: Error {}
        let (closed, _) = AsyncStream<Void>.makeStream()
        do {
            _ = try await PeerClose.cancelling(on: closed) { () -> Int in
                throw Boom()
            }
            XCTFail("expected Boom")
        } catch {
            XCTAssertTrue(error is Boom)
        }
    }

    /// The server allows remote half-closure, so a client FIN is an
    /// `inputClosed` event, not a channel close.
    func testInputClosedDuringRequestCancels() async throws {
        let channel = try await connectedChannel()
        let pipeline = channel.pipeline
        let fire = Task {
            try? await Task.sleep(nanoseconds: 50_000_000)
            pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        }
        let r = try await PeerClose.cancelling(channel) {
            await Self.untilCancelled()
        }
        _ = await fire.result
        XCTAssertEqual(r, "cancelled")
    }

    /// A client that sends its request and closes in one burst: the close
    /// lands before the handler runs, and the latch still reports it.
    func testInputClosedBeforeRequestCancels() async throws {
        let channel = try await connectedChannel()
        channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        let active = channel.isActive
        XCTAssertTrue(active, "half-closure leaves the channel active")
        let r = try await PeerClose.cancelling(channel) {
            await Self.untilCancelled()
        }
        XCTAssertEqual(r, "cancelled")
    }

    func testChannelInactiveCancels() async throws {
        let channel = try await connectedChannel()
        let pipeline = channel.pipeline
        let fire = Task {
            try? await Task.sleep(nanoseconds: 50_000_000)
            pipeline.close(promise: nil)
        }
        let r = try await PeerClose.cancelling(channel) {
            await Self.untilCancelled()
        }
        _ = await fire.result
        XCTAssertEqual(r, "cancelled")
    }

    /// The close is seen ahead of any codec: a handler that swallows
    /// `inputClosed` (as the HTTP pipelining handler can while a request is
    /// in flight) must not hide it from the latch.
    func testLatchSeesCloseAheadOfCodec() async throws {
        final class Swallow: ChannelInboundHandler, Sendable {
            typealias InboundIn = NIOAny
            func userInboundEventTriggered(
                context: ChannelHandlerContext, event: Any
            ) {}
        }
        let channel = NIOAsyncTestingChannel()
        try await channel.pipeline.addHandler(Swallow()).get()
        try await channel.pipeline.addHandler(PeerCloseLatch()).get()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 1))
        channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        let r = try await PeerClose.cancelling(channel) {
            await Self.untilCancelled()
        }
        XCTAssertEqual(r, "cancelled")
    }

    /// Keep-alive: completed requests on one live connection are not
    /// cancelled by the ones before them.
    func testSequentialRequestsOnLiveConnectionComplete() async throws {
        let channel = try await connectedChannel()
        for _ in 0 ..< 3 {
            let r = try await PeerClose.cancelling(channel) {
                Task.isCancelled ? "cancelled" : "done"
            }
            XCTAssertEqual(r, "done")
        }
    }

    func testChannelWithoutLatchJustRuns() async throws {
        let channel = NIOAsyncTestingChannel()
        let r = try await PeerClose.cancelling(channel) { "done" }
        XCTAssertEqual(r, "done")
    }
}
