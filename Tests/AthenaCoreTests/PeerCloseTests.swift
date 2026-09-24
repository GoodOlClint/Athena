import Foundation
import NIOCore
import NIOEmbedded
import XCTest

@testable import AthenaServerKit

/// #209 — a client disconnect cancels the in-flight handler work.
final class PeerCloseTests: XCTestCase {

    /// Stands in for a decode drain: runs until cancelled, then reports it.
    private static func untilCancelled() async -> String {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return "cancelled"
    }

    func testCloseCancelsOperation() async throws {
        let (closed, signal) = AsyncStream<Void>.makeStream()
        Task {
            try? await Task.sleep(nanoseconds: 50_000_000)
            signal.yield()
        }
        let started = Date()
        let r = try await PeerClose.cancelling(on: closed) {
            await Self.untilCancelled()
        }
        XCTAssertEqual(r, "cancelled")
        XCTAssertLessThan(Date().timeIntervalSince(started), 5)
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
    func testInputClosedOnChannelCancels() async throws {
        let channel = NIOAsyncTestingChannel()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 1))
        let work = Task {
            try await PeerClose.cancelling(channel) {
                await Self.untilCancelled()
            }
        }
        try await Task.sleep(nanoseconds: 50_000_000)
        channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        let r = try await work.value
        XCTAssertEqual(r, "cancelled")
    }

    func testChannelInactiveCancels() async throws {
        let channel = NIOAsyncTestingChannel()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 1))
        let work = Task {
            try await PeerClose.cancelling(channel) {
                await Self.untilCancelled()
            }
        }
        try await Task.sleep(nanoseconds: 50_000_000)
        try await channel.close()
        let r = try await work.value
        XCTAssertEqual(r, "cancelled")
    }

    /// The watcher leaves the pipeline once the request is done, so a
    /// keep-alive connection does not accumulate one per request.
    func testWatcherRemovedAfterCompletion() async throws {
        let channel = NIOAsyncTestingChannel()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 1))
        for _ in 0 ..< 3 {
            _ = try await PeerClose.cancelling(channel) { "done" }
        }
        try await Task.sleep(nanoseconds: 20_000_000)
        let names = try await channel.pipeline.handler(
            type: PeerClose.Watcher.self
        ).map { _ in true }.recover { _ in false }.get()
        XCTAssertFalse(names)
    }
}
