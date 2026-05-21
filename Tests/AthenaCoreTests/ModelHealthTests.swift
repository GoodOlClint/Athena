import Foundation
import XCTest

@testable import AthenaLLM

/// Regression coverage for `ModelHealth` on the HF-cache layout that
/// `athena pull` produces: snapshot shards are SYMLINKS to ../../blobs/<sha>.
/// The size check must follow the link (real blob size), not lstat it
/// (the link path is tens of bytes) — otherwise every pulled model is
/// spuriously flagged "corrupt header" and `prune` removes it.
final class ModelHealthTests: XCTestCase {

    /// Minimal valid .safetensors: 8-byte LE header length + JSON header
    /// (one tensor + __metadata__) + a small payload.
    private func writeSafetensors(to url: URL) throws {
        let header =
            #"{"__metadata__":{"format":"mlx"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#
        var data = Data()
        var len = UInt64(header.utf8.count).littleEndian
        withUnsafeBytes(of: &len) { data.append(contentsOf: $0) }
        data.append(contentsOf: Array(header.utf8))
        data.append(contentsOf: [0, 0, 0, 0])
        try data.write(to: url)
    }

    private func tmpDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("mh-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: d, withIntermediateDirectories: true)
        return d
    }

    /// HF-cache layout: shard is a relative symlink to a blob. Healthy.
    func testBlobSymlinkShardIsHealthy() throws {
        let root = try tmpDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let blobs = root.appendingPathComponent("blobs", isDirectory: true)
        let snap = root.appendingPathComponent(
            "snapshots/rev", isDirectory: true)
        try FileManager.default.createDirectory(
            at: blobs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: snap, withIntermediateDirectories: true)

        try writeSafetensors(to: blobs.appendingPathComponent("deadbeef"))
        try Data(#"{"model_type":"x"}"#.utf8).write(
            to: snap.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(
            to: snap.appendingPathComponent("tokenizer.json"))
        try FileManager.default.createSymbolicLink(
            atPath: snap.appendingPathComponent("model.safetensors").path,
            withDestinationPath: "../../blobs/deadbeef")

        XCTAssertEqual(
            ModelHealth.check(snap), [],
            "blob-symlinked shard must read as healthy")
    }

    /// A genuinely bogus header (claims a huge length in a tiny file) is
    /// still caught — the fix must not blanket-pass everything.
    func testTruncatedHeaderStillDetected() throws {
        let dir = try tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_type":"x"}"#.utf8).write(
            to: dir.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(
            to: dir.appendingPathComponent("tokenizer.json"))
        var len = UInt64(1 << 30).littleEndian  // 1 GiB header claim
        var d = Data()
        withUnsafeBytes(of: &len) { d.append(contentsOf: $0) }
        try d.write(to: dir.appendingPathComponent("model.safetensors"))

        let problems = ModelHealth.check(dir)
        XCTAssertTrue(
            problems.contains {
                $0.contains("corrupt header") || $0.contains("truncated")
            }, "expected a header problem, got \(problems)")
    }
}
