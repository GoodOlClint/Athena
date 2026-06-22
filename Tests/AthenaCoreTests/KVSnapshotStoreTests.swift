import Crypto
import Foundation
import XCTest

@testable import AthenaCore

/// ADR 027 S2 — the disk store is MLX-free `Data` + Foundation, so its
/// round-trip, skip-on-skew, ciphertext-at-rest, and retention behavior are
/// unit-pinned here (ADR 008/009). The end-to-end bit-identical-across-restart
/// guarantee is the S3 model-host gate.
final class KVSnapshotStoreTests: XCTestCase {

    private var dir: URL!
    private var store: KVSnapshotStore!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("kvsnap-test-\(UUID().uuidString)")
        store = KVSnapshotStore(directory: dir)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func kek(_ seed: UInt8 = 7) throws -> KeyfileKEK {
        try KeyfileKEK(keyfile: Data((0 ..< 32).map { UInt8($0) &+ seed }))
    }

    private func save(
        _ store: KVSnapshotStore, prefixHash: Data, body: Data,
        model: String = "Qwen/X", quant: String = "4bit",
        lastUsed: UInt64 = 1_000, kek: KEKProvider
    ) throws {
        try store.save(
            prefixHash: prefixHash, modelID: model, quantTag: quant,
            scopeKey: "p=alice", tokenCount: 100, contextSize: 4096,
            saveReason: .cold, createdUnix: 900, lastUsedUnix: lastUsed,
            body: body, kek: kek)
    }

    func testSaveLoadRoundTrip() throws {
        let k = try kek()
        let body = Data((0 ..< 2048).map { UInt8($0 & 0xff) })
        try save(store, prefixHash: Data([1, 2, 3]), body: body, kek: k)
        let loaded = store.load(
            prefixHash: Data([1, 2, 3]), requireModel: "Qwen/X", requireQuant: "4bit", kek: k)
        XCTAssertEqual(loaded, body)
    }

    func testCiphertextAtRest() throws {
        let k = try kek()
        let secret = Data("SECRET-KV-PLAINTEXT-PHI".utf8)
        let body = secret + Data(repeating: 0, count: 2000) + secret
        try save(store, prefixHash: Data([9]), body: body, kek: k)
        let raw = try Data(contentsOf: store.fileURL(forPrefixHash: Data([9])))
        XCTAssertNil(raw.range(of: secret), "plaintext must never appear on disk")
    }

    func testModelQuantMismatchSkips() throws {
        let k = try kek()
        try save(store, prefixHash: Data([5]), body: Data("kv".utf8), model: "M", quant: "4bit", kek: k)
        XCTAssertNil(
            store.load(prefixHash: Data([5]), requireModel: "OTHER", requireQuant: "4bit", kek: k),
            "model mismatch ⇒ cold")
        XCTAssertNil(
            store.load(prefixHash: Data([5]), requireModel: "M", requireQuant: "8bit", kek: k),
            "quant mismatch ⇒ cold")
    }

    func testWrongKEKFailsLoad() throws {
        try save(store, prefixHash: Data([5]), body: Data("kv".utf8), kek: try kek(1))
        XCTAssertNil(
            store.load(prefixHash: Data([5]), requireModel: "Qwen/X", requireQuant: "4bit", kek: try kek(2)),
            "wrong keyfile ⇒ cold")
    }

    func testTamperedFileFailsLoad() throws {
        let k = try kek()
        try save(store, prefixHash: Data([5]), body: Data((0 ..< 512).map { UInt8($0 & 0xff) }), kek: k)
        let url = store.fileURL(forPrefixHash: Data([5]))
        var raw = try Data(contentsOf: url)
        raw[raw.index(raw.endIndex, offsetBy: -1)] ^= 0xff  // flip a body byte
        try raw.write(to: url)
        XCTAssertNil(
            store.load(prefixHash: Data([5]), requireModel: "Qwen/X", requireQuant: "4bit", kek: k),
            "tampered body ⇒ cold")
    }

    func testMissingFileLoadsNil() throws {
        XCTAssertNil(
            store.load(prefixHash: Data([0xde]), requireModel: "M", requireQuant: "q", kek: try kek()))
    }

    func testEmptyPrefixHashRejected() throws {
        XCTAssertThrowsError(try save(store, prefixHash: Data(), body: Data("x".utf8), kek: try kek())) {
            XCTAssertEqual($0 as? KVSnapshotStore.Failure, .prefixHashEmpty)
        }
    }

    func testFile0600Permissions() throws {
        try save(store, prefixHash: Data([7]), body: Data("x".utf8), kek: try kek())
        let attrs = try FileManager.default.attributesOfItem(
            atPath: store.fileURL(forPrefixHash: Data([7])).path)
        XCTAssertEqual((attrs[.posixPermissions] as? NSNumber)?.intValue, 0o600)
    }

    func testIndexAndDelete() throws {
        let k = try kek()
        try save(store, prefixHash: Data([1]), body: Data("a".utf8), kek: k)
        try save(store, prefixHash: Data([2]), body: Data("bb".utf8), kek: k)
        XCTAssertEqual(Set(store.index().map(\.prefixHash)), [Data([1]), Data([2])])
        store.delete(prefixHash: Data([1]))
        XCTAssertEqual(store.index().map(\.prefixHash), [Data([2])])
    }

    func testEnforceRetentionByCount() throws {
        let k = try kek()
        try save(store, prefixHash: Data([1]), body: Data("a".utf8), lastUsed: 100, kek: k)
        try save(store, prefixHash: Data([2]), body: Data("b".utf8), lastUsed: 200, kek: k)
        try save(store, prefixHash: Data([3]), body: Data("c".utf8), lastUsed: 300, kek: k)
        let evicted = store.enforceRetention(maxEntries: 2, maxBytes: nil, maxAgeSecs: nil, now: 400)
        XCTAssertEqual(evicted, [Data([1])], "oldest-used evicted first")
        XCTAssertEqual(Set(store.index().map(\.prefixHash)), [Data([2]), Data([3])])
    }

    // MARK: - Pure retention algebra

    private func item(_ id: UInt8, _ bytes: Int, _ lastUsed: UInt64) -> KVSnapshotRetention.Item {
        .init(id: Data([id]), bytes: bytes, lastUsedUnix: lastUsed)
    }

    func testRetentionAgeFirst() {
        // item 1 idle 900s (> 300 cap ⇒ evict); item 2 idle 200s (≤ cap ⇒ keep).
        let items = [item(1, 10, 100), item(2, 10, 800)]
        let evicted = KVSnapshotRetention.toEvict(
            items, maxEntries: nil, maxBytes: nil, maxAgeSecs: 300, now: 1000)
        XCTAssertEqual(evicted, [Data([1])], "only the age-expired entry is evicted")
    }

    func testRetentionByBytesLRU() {
        let items = [item(1, 100, 10), item(2, 100, 20), item(3, 100, 30)]
        let evicted = KVSnapshotRetention.toEvict(
            items, maxEntries: nil, maxBytes: 250, maxAgeSecs: nil, now: 100)
        XCTAssertEqual(evicted, [Data([1])], "evict oldest until total ≤ 250")
    }

    func testRetentionUnboundedKeepsAll() {
        let items = [item(1, 100, 10), item(2, 100, 20)]
        XCTAssertEqual(
            KVSnapshotRetention.toEvict(items, maxEntries: 0, maxBytes: 0, maxAgeSecs: 0, now: 100),
            [], "zero/absent caps ⇒ unbounded")
    }

    func testRetentionCombinedAgeAndCount() {
        let items = [item(1, 10, 100), item(2, 10, 200), item(3, 10, 5_000)]
        // now=6000, age cap 1000 expires 1 & 2; count cap 1 is already satisfied by survivor 3.
        let evicted = KVSnapshotRetention.toEvict(
            items, maxEntries: 1, maxBytes: nil, maxAgeSecs: 1_000, now: 6_000)
        XCTAssertEqual(Set(evicted), [Data([1]), Data([2])])
    }
}
