import AthenaCore
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon

/// Fetch an HF repo snapshot and link it into the model store as
/// `storeRoot/<name>`, so `serve --model <name>` and `athena list`
/// pick it up without a multi-GB copy. M6-cli-3.
public enum ModelPull {
    /// `progress` (0…1 download fraction) is optional — default nil
    /// keeps the daemon/queue callers unchanged; the CLI passes a
    /// renderer.
    ///
    /// A dropped/stalled connection on a multi-GB fetch is the common
    /// failure, so transient network errors are retried with backoff
    /// rather than aborting the whole pull. The Hub cache skips files
    /// already fully downloaded, so each retry resumes at file
    /// granularity — completed shards are never re-fetched. (An
    /// interrupted file still restarts; true byte-level resume needs the
    /// substrate to persist the partial.) `onRetry` lets the caller
    /// surface the wait; the daemon/queue path passes nil.
    public static func pull(
        id: String, revision: String? = nil, into storeRoot: URL,
        progress: (@Sendable (ModelOpProgress) -> Void)? = nil,
        maxAttempts: Int = 5,
        onRetry: (
            @Sendable (
                _ attempt: Int, _ maxAttempts: Int,
                _ error: any Error
            ) -> Void
        )? = nil
    ) async throws -> URL {
        var attempt = 1
        let snapshot: URL
        // Bypass the `#hubDownloader` macro (its MLXLMCommon.Downloader protocol
        // pins the single-`Progress` shape) and call HubClient.downloadSnapshot
        // directly so the fork's optional per-file handler surfaces one row per
        // shard (audit §2). No substrate edit — the client is ours to construct.
        let client = HuggingFace.HubClient(
            session: AthenaProxy.proxiedURLSession())
        guard let repo = HuggingFace.Repo.ID(rawValue: id) else {
            throw ModelStoreOps.OpError.invalidName(id)
        }
        while true {
            do {
                snapshot = try await client.downloadSnapshot(
                    of: repo, kind: .model,
                    revision: revision ?? "main",
                    matching: [
                        "*.json", "*.safetensors", "*.txt", "*.jinja",
                        "tokenizer*", "*.model",
                    ],
                    progressHandler: { p in
                        progress?(
                            .download(
                                fraction: p.fractionCompleted,
                                bytes: p.completedUnitCount,
                                total: p.totalUnitCount))
                    },
                    perFileHandler: { files in
                        let n = files.count
                        for (i, f) in files.enumerated() {
                            progress?(
                                .file(
                                    name: (f.path as NSString).lastPathComponent,
                                    index: i + 1, count: n,
                                    bytes: f.completedUnitCount,
                                    total: f.totalUnitCount,
                                    done: f.totalUnitCount > 0
                                        && f.completedUnitCount >= f.totalUnitCount))
                        }
                    })
                break
            } catch {
                guard attempt < maxAttempts,
                    isTransientNetworkError(error)
                else { throw error }
                onRetry?(attempt, maxAttempts, error)
                // Exponential backoff capped at 30s: 2, 4, 8, 16, 30…
                let secs = min(30.0, pow(2.0, Double(attempt)))
                try? await Task.sleep(for: .seconds(secs))
                attempt += 1
            }
        }

        let name = id.split(separator: "/").last.map(String.init) ?? id
        // C5: an id whose last path component is `..` (e.g. `org/..`)
        // would make `dest` the store root's PARENT and the `removeItem`
        // below would wipe it. Reject anything that isn't a bare child name.
        guard ModelStoreOps.isValidName(name) else {
            throw ModelStoreOps.OpError.invalidName(name)
        }
        try FileManager.default.createDirectory(
            at: storeRoot, withIntermediateDirectories: true)
        let dest = storeRoot.appendingPathComponent(
            name, isDirectory: true)
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.createSymbolicLink(
            at: dest, withDestinationURL: snapshot)
        return dest
    }

    /// A retryable network blip (vs. a terminal error like 401/404 or a
    /// full disk). Matches the `NSURLErrorDomain` codes a flaky link
    /// produces — connection lost, timeout, no route, DNS, TLS reset —
    /// so a momentary drop self-heals instead of aborting a long fetch.
    public static func isTransientNetworkError(_ error: any Error) -> Bool {
        let ns = error as NSError
        guard ns.domain == NSURLErrorDomain else { return false }
        switch ns.code {
        case NSURLErrorTimedOut,
            NSURLErrorNetworkConnectionLost,
            NSURLErrorNotConnectedToInternet,
            NSURLErrorCannotConnectToHost,
            NSURLErrorCannotFindHost,
            NSURLErrorDNSLookupFailed,
            NSURLErrorResourceUnavailable,
            NSURLErrorSecureConnectionFailed,
            NSURLErrorHTTPTooManyRedirects:
            return true
        default:
            return false
        }
    }

    /// Concise, human-readable rendering of a pull failure for the CLI.
    /// `NSError`'s full `description` dumps the signed CDN URL + resume
    /// blob (hundreds of unreadable chars); the localized description is
    /// the one sentence a human needs ("The network connection was
    /// lost.").
    public static func friendlyError(_ error: any Error) -> String {
        (error as NSError).localizedDescription
    }
}
