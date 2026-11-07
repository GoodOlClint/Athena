import AthenaCore
import Foundation
import MLXLMCommon
import XCTest

@testable import AthenaLLM

final class RejectedToolCallTests: XCTestCase {
    private let secret = "SECRET-ARG-VALUE"

    private func truncated() -> RejectedToolCall {
        RejectedToolCall(
            reason: .incompleteOutput, format: .json,
            rawText: String(repeating: secret, count: 50),
            previewByteLimit: 64)
    }

    func testRejectionClassifiesWithoutPreviewText() {
        let r = truncated()
        XCTAssertTrue(r.isPreviewTruncated)
        let err = MLXLLMModule.athenaError(for: r)
        XCTAssertEqual(err.httpStatus, 502)
        XCTAssertEqual(err.code, "rejected_tool_call")
        XCTAssertEqual(err.type, "server_error")
        XCTAssertNotEqual(err.code, "module_load_failed")
        XCTAssertFalse(err.message.contains(secret))
        XCTAssertFalse((err.serverDetail ?? "").contains(secret))
        XCTAssertTrue(err.serverDetail?.contains("incomplete_output") == true)
        XCTAssertTrue(err.serverDetail?.contains("\(r.rawTextByteCount)") == true)
    }

    func testTruncatedRejectionEmitsRealUsageBeforeFailing() {
        let usage = TokenUsage(promptTokens: 120, completionTokens: 512)
        var emitted: [GenChunk] = []
        let err = MLXLLMModule.failTruncatedRejection(
            truncated(), usage: usage
        ) { emitted.append($0) }
        XCTAssertEqual(err.code, "rejected_tool_call")
        guard emitted.count == 1, case .usage(let u) = emitted[0] else {
            return XCTFail("expected exactly one .usage chunk, got \(emitted)")
        }
        XCTAssertEqual(u, usage)
        XCTAssertGreaterThan(u.totalTokens, 0)
    }
}
