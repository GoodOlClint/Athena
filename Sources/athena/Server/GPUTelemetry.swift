import Foundation
import SoCMetrics

/// M60.3 — sudoless GPU clock + die-temperature telemetry for `/healthz`, via
/// the `swift-soc-metrics` package's `SoCMetrics` module (IOReport for clock/
/// residency, SMC for temperature; no root, no subprocess).
///
/// A single `SoCSampler` holds one IOReport subscription for the daemon's
/// lifetime. A detached background thread refreshes a cached reading ~1 Hz so
/// the `/healthz` handler reads it **non-blocking** (a direct `sample()` would
/// block ~0.1 s on two IOReport snapshots — unacceptable on a frequently
/// scraped health endpoint, and it would stall a cooperative-pool thread).
///
/// Everything is best-effort: if IOReport is unavailable (non–Apple-Silicon,
/// or the private framework drifts), `SoCSampler()` throws, the probe holds
/// `nil`, no thread starts, and `current` reports `(nil, nil)` so `/healthz`
/// simply omits the GPU fields. Lock-guarded `@unchecked Sendable`, mirroring
/// the codebase's `PowerAssertion` / rate-limit idiom.
final class GPUTelemetryProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var cachedMHz: Double?
    private var cachedActive: Double?
    private var cachedTempC: Double?
    private let sampler: SoCSampler?

    /// Refresh cadence and per-sample window. 0.1 s of sampling once per
    /// ~1.1 s is negligible load and keeps the cache fresh enough for a health
    /// endpoint.
    private static let refreshInterval: TimeInterval = 1.0
    private static let sampleWindow: TimeInterval = 0.1

    init() {
        self.sampler = try? SoCSampler()
        if sampler != nil { startBackgroundSampling() }
    }

    /// Most recent cached reading. `(nil, nil)` until the first sample lands or
    /// when GPU telemetry is unavailable.
    var current: (mhz: Double?, activeResidency: Double?, temperatureC: Double?) {
        lock.lock(); defer { lock.unlock() }
        return (cachedMHz, cachedActive, cachedTempC)
    }

    /// Whether GPU telemetry is available on this host (the sampler opened).
    var isAvailable: Bool { sampler != nil }

    private func startBackgroundSampling() {
        Thread.detachNewThread { [weak self] in
            while let self, let sampler = self.sampler {
                let s = sampler.sample(interval: Self.sampleWindow)
                self.lock.lock()
                self.cachedMHz = s.gpuFrequencyMHz
                self.cachedActive = s.gpuActiveResidency
                self.cachedTempC = s.gpuTemperatureC
                self.lock.unlock()
                Thread.sleep(forTimeInterval: Self.refreshInterval)
            }
        }
    }
}
