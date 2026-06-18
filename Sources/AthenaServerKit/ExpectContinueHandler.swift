import HTTPTypes
import NIOCore
import NIOHTTPTypes

/// ADR 017 — answer `Expect: 100-continue` so a client that waits for it
/// never hangs.
///
/// Swift HTTP clients (URLSession, AsyncHTTPClient) add `Expect:
/// 100-continue` **for large request bodies** and then wait for the server
/// to send `100 Continue` before streaming the body. Hummingbird sends one
/// nowhere, so such a client deadlocks until its own timeout (the
/// "oversized audio upload hangs" report). This handler fixes it for every
/// route.
///
/// It is deliberately **stateless**: on a request head carrying `Expect:
/// 100-continue` it writes an interim `100 Continue` and forwards the head
/// unchanged; every other part is passthrough. No cap awareness, no body
/// draining, no request short-circuit — size enforcement stays app-side
/// (`UploadLimit` + the handlers' `collect(upTo:)`), so a worst-case
/// problem here cannot reintroduce the oversized hang.
///
/// Why the interim write is safe in Hummingbird's pipeline: it is added via
/// `HTTP1Channel.Configuration.additionalChannelHandlers`, sitting above the
/// `HTTP1ToHTTPServerCodec` (which converts and passes any response head —
/// including an informational `100` — through unconditionally) and below
/// `HTTPConnectionStateHandler` in the outbound direction (so the interim
/// never reaches the connection-state machine). Hummingbird also disables
/// `withPipeliningAssistance` and `withErrorHandling`, so the NIO state
/// machines that historically gated interim 1xx responses are absent.
public final class ExpectContinueHandler: ChannelDuplexHandler,
    RemovableChannelHandler
{
    public typealias InboundIn = HTTPRequestPart
    public typealias InboundOut = HTTPRequestPart
    public typealias OutboundIn = HTTPResponsePart
    public typealias OutboundOut = HTTPResponsePart

    public init() {}

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        if case .head(let head) = unwrapInboundIn(data),
            Self.expectsContinue(head)
        {
            context.writeAndFlush(
                wrapOutboundOut(.head(HTTPResponse(status: .continue))),
                promise: nil)
        }
        context.fireChannelRead(data)
    }

    /// True iff the request head carries `Expect: 100-continue`
    /// (case-insensitive value, per RFC 9110 §10.1.1).
    public static func expectsContinue(_ head: HTTPRequest) -> Bool {
        guard let expect = head.headerFields[.expect] else { return false }
        return expect.caseInsensitiveCompare("100-continue") == .orderedSame
    }
}
