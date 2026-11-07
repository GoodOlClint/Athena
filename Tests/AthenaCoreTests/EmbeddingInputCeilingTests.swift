import AthenaCore
import XCTest

@testable import AthenaEmbedding

/// #235 — the per-input ceiling is min(32_768, the model's position table).
final class EmbeddingInputCeilingTests: XCTestCase {
    func testInputPastPositionTableIs400NamingTheCeiling() throws {
        XCTAssertThrowsError(
            try MLXEmbeddingModule.checkInputLengths(
                [10, 513], maxPositionEmbeddings: 512)
        ) { error in
            guard let e = error as? AthenaError,
                case let .inputTooLong(_, tokens, maxTokens) = e
            else { return XCTFail("expected inputTooLong, got \(error)") }
            XCTAssertEqual(tokens, 513)
            XCTAssertEqual(maxTokens, 512)
            XCTAssertEqual(e.httpStatus, 400)
            XCTAssertTrue(e.message.contains("512"))
        }
    }

    func testInputAtThePositionTableIsAccepted() throws {
        try MLXEmbeddingModule.checkInputLengths(
            [512], maxPositionEmbeddings: 512)
    }

    func testCeilingNeverExceedsTheFlatCap() {
        XCTAssertEqual(
            MLXEmbeddingModule.inputCeiling(maxPositionEmbeddings: 131_072),
            32_768)
        XCTAssertEqual(
            MLXEmbeddingModule.inputCeiling(maxPositionEmbeddings: nil), 32_768)
    }

    /// RoBERTa numbers positions from pad_token_id + 1: 514 entries, pad 1.
    func testPaddingAwarePositionTableReservesItsOffset() throws {
        let reserved = MLXEmbeddingModule.reservedPositions(
            modelType: "roberta", padTokenId: 1)
        XCTAssertEqual(reserved, 2)
        try MLXEmbeddingModule.checkInputLengths(
            [512], maxPositionEmbeddings: 514, reservedPositions: reserved)
        XCTAssertThrowsError(
            try MLXEmbeddingModule.checkInputLengths(
                [513], maxPositionEmbeddings: 514, reservedPositions: reserved))
        XCTAssertEqual(
            MLXEmbeddingModule.reservedPositions(
                modelType: "bert", padTokenId: 0), 0)
    }
}
