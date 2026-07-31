# Compiler-warning audit — 2026-07-31

Athena's own targets built clean except for one deprecation class, which this audit fixed at 15 of 18 sites and justified at the remaining 3. This document is the standing record of *why* the survivors are acceptable, so a future reader does not re-litigate them.

## Scope

First-party code only: `Sources/`, `Tests/`, and the `clients/` package. Warnings from `.build/checkouts/` (mlx-swift, mlx-swift-lm, hummingbird, swift-nio, swift-log, …) are out of scope — we do not control that source.

## Method

Incremental builds hide warnings for files that did not change, so every first-party source was touched before each capture.

```sh
find Sources Tests -name '*.swift' -print0 | xargs -0 touch
swift build --build-tests 2>&1 | tee /tmp/athena-warnings.log
(cd clients && find Sources -name '*.swift' -print0 | xargs -0 touch && swift build)
./deploy/build.sh Release          # the xcodebuild path — the only one that links, so the only one that shows `ld` warnings
```

The `swift build` path and the `xcodebuild` path do not produce the same set: linker warnings appear only under `deploy/build.sh`. Both were captured.

Baseline: **18 unique first-party compiler warnings** (`swift build`), **9 linker warnings** (`deploy/build.sh Release`), **1 SwiftPM manifest warning** (dependency resolution). The `clients/` package was already clean.

Known blind spot: a compiler only warns about code it compiles. The non-Darwin arms of the `#if canImport(Darwin)` / `#if canImport(FoundationNetworking)` blocks in `AthenaCore`, `clients/`, and the test target are compiled on no machine today — CI lints on Linux but builds and tests only on macOS. Whatever warnings live there are unmeasured, not absent.

## Fixed (15)

| Site | Warning | Fix |
|---|---|---|
| `Sources/AthenaLLM/MLXLLMModule.swift` ×3 | `samplingTemp` / `samplingTopP` / `samplingSeed` never used | Deleted, with the `requestTopP` / `requestSeed` parameters they fed. Dead since publication S0 removed the vendored Qwen3.5 decode fork — the in-closure M40.2/M40.3 sampling branch went with it, and an unstructured sampling request now resolves the same knobs onto `GenerateParameters` in `beginGeneration`. |
| `Sources/AthenaLLM/MLXLLMModule.swift:483` | redundant `try` | `container.perform` is `rethrows` and the closure does not throw — dropped `try`. |
| `Sources/AthenaLLM/MLXLLMModule.swift:883` | redundant `await` | `effectiveMaxTokens` is a plain static function — dropped `await`. |
| `Sources/AthenaLLM/MLXLLMModule.swift:1537` | redundant `try` | Same `rethrows` case as above. The outer `try` on `InferenceGate.withExclusiveExecution` is load-bearing and stays — that one is `throws`, not `rethrows`. |
| `Sources/AthenaServerKit/AthenaLogging.swift` ×2 | deprecated default `log(event:)` bridge | `TerminalLogHandler` and `OSUnifiedLogHandler` now implement `log(event:)` directly (swift-log 1.12 renamed the flat-parameter method). Field-for-field identical for what the old signature carried: `event.level` / `event.message` / `event.metadata` / `event.function` (`source`, `file` and `line` were declared-but-unused in both handlers). `event.error` is new surface the old bridge dropped on the floor; since the handlers now own the emit path, both fold it into the metadata as `error=` rather than let it vanish. No call site passes `error:` today, so live output is unchanged. |
| `Sources/AthenaStructured/StructuredShim.swift:23` | `String(cString: [CChar])` deprecated | Truncate at the shim's NUL, then `String(decoding:as: UTF8.self)` — exactly what the deprecation message prescribes. |
| `Sources/AthenaTranscription/Parakeet/ParakeetModel.swift:482` | `var ifgo` never mutated | `let`. |
| `Sources/athena/Commands/AuthCmd.swift` ×2 | redundant `try` | `parseTTLSeconds` does not throw; the failure path is `FailableExit.die` (`Never`). |
| `Tests/AthenaCoreTests/DecodeProgressTests.swift` ×2 | redundant `await` | `TaskLocal.withValue` resolves to the synchronous overload; the two tests no longer need to be `async`. |
| `Tests/AthenaCoreTests/GuidedDecoderTests.swift:133` | duplicate `Equatable` conformance | Deleted the retroactive extension. `CommitResult` is a payload-free enum, so `AthenaLLM` already synthesizes `Equatable` — the extension restated a conformance that exists in the type's own module. |

Verification: `./deploy/test.sh` → 789 tests, 0 failures, 35 skipped (the heavy `ATHENA_RUN_MODEL_TESTS` tier). `./deploy/build.sh Release` → BUILD SUCCEEDED, hardening gate OK. Logging was additionally smoke-tested against a live daemon because it has no unit coverage and the change rewrites the emit path: both sinks (stderr and the macOS unified log) produce byte-identical line format, including the `function=` field and the request-scoped `req=` / `principal=` metadata.

## Accepted, with justification

### 1. `MLX.setErrorHandler` is deprecated — 3 sites

`Sources/athena/Commands/Load.swift:543`, `Tests/AthenaCoreTests/MetalFaultDegradeE2ETests.swift:34,42`.

The suggested replacements are not substitutes. In `mlx-swift`'s `ErrorHandler.swift`, the handler stack `withErrorHandler` / `withError` push onto is `@TaskLocal`, and `dispatch` consults the task-local stack first, falling back to a process-global handler only when that stack is empty. MLX raises `async_eval` faults on its own worker thread (`default-qos.cooperative`), which has no view of the raising task's task-locals — so a scoped handler provably cannot see the fault class ADR 030 Part 2 exists to catch. The deprecated global setter is the only lever that reaches it.

Alternatives considered and rejected:

- Calling `mlx_set_error_handler` from `Cmlx` directly would overwrite `mlx-swift`'s own trampoline and break `withError` / `withErrorHandler` for the substrate as well as for us. Strictly worse.
- Wrapping the call in a locally-deprecated shim to silence the diagnostic is suppression wearing a fix's clothes. Not done.

The real fix is upstream: `mlx-swift` needs a non-deprecated way to install a process-global handler, or must keep this one. Until then the call stays, and the `ponytail:` comment at the call site carries the same reasoning. The two test sites install the identical handler on purpose — the test's whole point is to pin that the *production* mechanism survives a genuine `[metal::malloc]` device-cap fault.

### 2. `ld: object file was built for newer 'macOS' version (26.0) than being linked (14.0)` — 9 warnings

All nine name objects from the rustup prebuilt sysroot: `std`, `core`, `alloc`, `panic_unwind`, `object`, `addr2line`, `gimli`, `hashbrown`, `rustc_demangle`. Zero name `athena_structured_shim` — our own crate's objects link clean at the package's macOS 14 floor.

That distinction is the whole justification. `MACOSX_DEPLOYMENT_TARGET=14.0` only affects crates cargo compiles locally; the sysroot rlibs ship prebuilt against the host SDK and are not rebuilt without `-Z build-std` (nightly-only). The two ways to zero these out are to require a nightly toolchain for a release build, or to raise Athena's deployment target to macOS 26 — both cost more than the warnings do. `LC_BUILD_VERSION` mismatch is a min-version note, not a link error.

`rust-shim/build.sh` already documents this; this audit re-derived it from the warning text rather than taking the comment's word for it.

### 3. SwiftPM: conflicting identity for `swift-huggingface` — 1 warning

`swift-transformers` depends on `huggingface/swift-huggingface`; Athena depends on `GoodOlClint/swift-huggingface`. Same package identity, different URL, so SwiftPM warns — and says it "will be escalated to an error in future versions."

This is a compiler-adjacent build warning about a deliberate, already-tracked state: the fork carries the unmerged upstream PR #50 (per-byte download progress and resume), and `Package.swift` documents the revert-to-upstream condition. It resolves itself the day #50 merges.

Two ways to close it sooner, neither belonging in a warnings-cleanup change: commit a `.swiftpm/configuration/mirrors.json` mapping the upstream URL onto the fork, or vendor the patch. Both alter dependency resolution and want their own change with a resolution diff and a fresh `Package.resolved`.

## Follow-up (found here, not fixed here)

Deleting the dead sampling locals exposed an adjacent inefficiency in `runSpeculative`, out of scope for a warnings change because fixing it alters control flow.

For an unstructured request with `speculative: true` and no logprobs, `greedyEligible` / `samplingEligible` keep the request off the early-return at line 979 — it then runs `container.prepare` (template render plus tokenization), reaches the `guide == nil && logprobSink == nil` check, and returns nil anyway, so `beginGeneration` prepares the same prompt a second time. Two consequences: a redundant prepare per speculative unstructured request, and a `dispatch path=speculative-greedy` / `speculative-sampling` debug line naming a path that no longer exists in that closure. The generated tokens are correct either way — `beginGeneration` still drives the MTP drafter overload when one is resident.

## Re-checking

```sh
find Sources Tests -name '*.swift' -print0 | xargs -0 touch
swift build --build-tests 2>&1 | grep -E '^/.*/(Sources|Tests)/.*warning:' | sort -u
```

Expected: the three `setErrorHandler` sites and nothing else.
