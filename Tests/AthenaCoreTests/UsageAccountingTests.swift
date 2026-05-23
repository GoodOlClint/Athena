import AthenaCore
import Foundation
import XCTest

@testable import AthenaEmbedding
@testable import AthenaLLM

/// M27.1 — true per-request token accounting threaded out of the
/// generate/embed paths. These exercise the model-free stub conformers
/// (CI-safe); the real MLX counts are validated against a live daemon.
final class UsageAccountingTests: XCTestCase {

    func testTokenUsageMath() {
        let u = TokenUsage(promptTokens: 12, completionTokens: 5)
        XCTAssertEqual(u.totalTokens, 17)
        XCTAssertEqual(TokenUsage.zero.totalTokens, 0)
    }

    /// The stub's metered stream must yield text chunks AND exactly one
    /// terminal usage with non-zero counts, so `usage`/metrics are live
    /// end-to-end under `--engine stub`.
    func testStubGenerateMeteredYieldsTerminalUsage() async {
        let llm = StubLLMModule(reserveBytes: 1)
        var text = ""
        var usages: [TokenUsage] = []
        for await event in llm.generateMetered(
            messages: [ChatTurn(role: "user", content: "hello there")],
            schemaJSON: nil, tools: nil, maxTokens: nil, temperature: nil)
        {
            switch event {
            case .text(let t): text += t
            case .usage(let u): usages.append(u)
            case .finish: break
            }
        }
        XCTAssertFalse(text.isEmpty, "stub still streams text")
        XCTAssertEqual(usages.count, 1, "exactly one terminal usage")
        XCTAssertEqual(usages[0].promptTokens, 2, "‘hello there’ = 2")
        XCTAssertGreaterThan(usages[0].completionTokens, 0)
        XCTAssertEqual(
            usages[0].totalTokens,
            usages[0].promptTokens + usages[0].completionTokens)
    }

    /// M31.2: a positive `max_tokens` truncates the stub stream and the
    /// terminal `.finish` reports `.length`; an absent cap reports
    /// `.stop`. The server maps these to the OpenAI `finish_reason`.
    func testStubGenerateMeteredReportsFinishReason() async {
        let llm = StubLLMModule(reserveBytes: 1)
        let turns = [ChatTurn(role: "user", content: "hello")]

        func run(maxTokens: Int?) async -> (
            text: String, completion: Int, finish: FinishReason?
        ) {
            var text = ""
            var completion = 0
            var finish: FinishReason?
            for await event in llm.generateMetered(
                messages: turns, schemaJSON: nil, tools: nil,
                maxTokens: maxTokens, temperature: nil)
            {
                switch event {
                case .text(let t): text += t
                case .usage(let u): completion = u.completionTokens
                case .finish(let r): finish = r
                }
            }
            return (text, completion, finish)
        }

        let capped = await run(maxTokens: 2)
        XCTAssertEqual(capped.finish, .length, "hit max_tokens ⇒ length")
        XCTAssertEqual(capped.completion, 2, "truncated at the cap")

        let full = await run(maxTokens: nil)
        XCTAssertEqual(full.finish, .stop, "natural end ⇒ stop")
        XCTAssertGreaterThan(
            full.completion, capped.completion,
            "uncapped emits more than the truncated run")
    }

    /// The String `generate` filter drops the usage event but preserves
    /// the same text as the metered stream (single source of truth).
    func testStringGenerateMatchesMeteredText() async {
        let llm = StubLLMModule(reserveBytes: 1)
        let turns = [ChatTurn(role: "user", content: "hi")]
        var metered = ""
        for await e in llm.generateMetered(
            messages: turns, schemaJSON: nil, tools: nil,
            maxTokens: nil, temperature: nil)
        {
            if case .text(let t) = e { metered += t }
        }
        var plain = ""
        for await t in llm.generate(
            messages: turns, schemaJSON: nil, tools: nil,
            maxTokens: nil, temperature: nil)
        {
            plain += t
        }
        XCTAssertEqual(metered, plain)
    }

    /// Embeddings report a non-zero input token count (== total tokens;
    /// no completion side).
    func testStubEmbedReportsPromptTokens() async throws {
        let batch = try await StubEmbeddingModule().embed([
            "alpha beta", "gamma",
        ])
        XCTAssertEqual(batch.vectors.count, 2)
        XCTAssertEqual(batch.promptTokens, 3, "2 + 1 whitespace tokens")
    }

    func testStubEmbedEmptyHasZeroTokens() async throws {
        let batch = try await StubEmbeddingModule().embed([])
        XCTAssertTrue(batch.vectors.isEmpty)
        XCTAssertEqual(batch.promptTokens, 0)
    }
}
