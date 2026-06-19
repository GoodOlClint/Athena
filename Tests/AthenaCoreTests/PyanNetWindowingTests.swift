import Foundation
import XCTest

@testable import AthenaTranscription

/// Regression for the daemon-crash on a too-short audio file (EXC_BREAKPOINT in
/// `mlx_conv1d` under `PyanSincNet`). A tiny/empty/corrupt clip — e.g. an
/// accidental record+delete that decodes to a handful of samples — used to be
/// fed to the SincNet as a window shorter than the first Conv1d's 251-sample
/// kernel, making the convolution degenerate; MLX's default error handler then
/// ABORTED the whole process. The window math is now MLX-free and unit-pinned:
/// every window handed to the network is exactly `windowSamples`, zero-padded,
/// so the conv can never go degenerate.
final class PyanNetWindowingTests: XCTestCase {
    private let W = PyanNetSegmentationModel.windowSamples  // 160_000
    /// The SincNet's first Conv1d kernel — the hard floor a window must clear.
    private let sincKernel = 251

    // MARK: paddedWindow — always exactly W samples

    func testTinyClipPadsToFullWindow() {
        // 50 samples — far below the 251 conv kernel that crashed the daemon.
        let win = PyanNetSegmentationModel.paddedWindow(
            [Float](repeating: 0.5, count: 50), start: 0, windowSamples: W)
        XCTAssertEqual(win.count, W)
        XCTAssertGreaterThanOrEqual(win.count, sincKernel)
        XCTAssertEqual(Array(win.prefix(50)), [Float](repeating: 0.5, count: 50))
        XCTAssertEqual(win[50], 0, "tail must be zero-padded")
        XCTAssertEqual(win.last, 0)
    }

    func testStartBeyondEndIsAllZeroFullWindow() {
        let win = PyanNetSegmentationModel.paddedWindow(
            [Float](repeating: 1, count: 10), start: 1000, windowSamples: W)
        XCTAssertEqual(win.count, W)
        XCTAssertEqual(win.max(), 0)
    }

    func testFullWindowSliceIsUnchanged() {
        let samples = (0..<(W + 5)).map { Float($0) }
        let win = PyanNetSegmentationModel.paddedWindow(
            samples, start: 0, windowSamples: W)
        XCTAssertEqual(win.count, W)
        XCTAssertEqual(win.first, 0)
        XCTAssertEqual(win.last, Float(W - 1))  // not padded
    }

    // MARK: windowStarts

    func testEmptyClipHasNoWindows() {
        XCTAssertEqual(
            PyanNetSegmentationModel.windowStarts(
                total: 0, windowSamples: W, stepSamples: W / 2),
            [])
    }

    func testShortClipIsOneWindow() {
        XCTAssertEqual(
            PyanNetSegmentationModel.windowStarts(
                total: 50, windowSamples: W, stepSamples: W / 2),
            [0])
    }

    func testLongClipTilesAndRightAlignsTail() {
        // 25 s at 16 kHz, 50% hop → windows at 0, 5s, 10s, +right-aligned tail.
        let total = 400_000
        let starts = PyanNetSegmentationModel.windowStarts(
            total: total, windowSamples: W, stepSamples: W / 2)
        XCTAssertEqual(starts.first, 0)
        XCTAssertEqual(starts.last! + W, total, "tail right-aligned to the end")
        for i in 1..<starts.count {
            XCTAssertGreaterThan(starts[i], starts[i - 1])
        }
    }

    // MARK: the crash-safety invariant

    /// For ANY clip length ≥ 1, every window the segmenter would feed the conv
    /// is exactly W samples — never below the 251-sample receptive field. This
    /// is the property whose absence aborted the daemon.
    func testEveryWindowClearsTheConvReceptiveField() {
        for total in [1, 50, sincKernel - 1, sincKernel, 999, W - 1, W, W + 1, 400_000]
        {
            let samples = [Float](repeating: 0.1, count: total)
            let starts = PyanNetSegmentationModel.windowStarts(
                total: total, windowSamples: W, stepSamples: W / 2)
            XCTAssertFalse(starts.isEmpty, "total=\(total) produced no windows")
            for s in starts {
                let win = PyanNetSegmentationModel.paddedWindow(
                    samples, start: s, windowSamples: W)
                XCTAssertEqual(
                    win.count, W,
                    "total=\(total) start=\(s) window not full-width")
                XCTAssertGreaterThanOrEqual(win.count, sincKernel)
            }
        }
    }
}
