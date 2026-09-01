# 008 — Testable server seam: extract `AthenaServerKit`, don't `@testable import` the executable

**Status:** Accepted — Implemented (M70.1, v0.10.144)
**Date:** 2026-06-13
**Milestone:** M70.1 (audit-remediation; resolves audit NA2/NB4 — "CI blindness")

## Context

The single test target (`AthenaCoreTests`) depends on the library targets but **not** on the `athena` executable target, and nothing does `@testable import athena`. So the security boundary of the whole daemon — `Auth.constantTimeEqual`, bearer `resolve()`/expiry math, `AuthPolicy.required` (the route→permission map), the `RateLimiter`/`ConcurrencyLimiter` token buckets, `Session.validate`/CSRF HMAC, the `MultipartForm` reader, and `AthenaMetrics.prometheus`/percentile math — had **zero** automated coverage. These were reachable only via the host-bound `e2e-rbac.sh` scripts, which the project notes mark explicitly **not** a CI unit tier (audit NA2). The entire `Commands/` layer is likewise unreachable by the test suite (audit NB4).

Two structural options were on the table (the NA2 fix sketch lists both):

- **(a)** Extract the pure logic into a new library target both the executable and a test target depend on.
- **(b)** Add the `athena` executable as a test-target dependency and `@testable import` it (SE-0294 permits testing an `@main` executable that uses `@main struct`, which this one does).

## Decision

**Option (a).** Create a new MLX-free library target **`AthenaServerKit`** that holds the daemon's pure HTTP-server primitives; the executable and `AthenaCoreTests` both depend on it. Moved in: `Auth.swift`, `RateLimit.swift`, `Metrics.swift`, `Session.swift`, `MultipartForm.swift`, `Passwords.swift`, `AppRequestContext.swift`, and `AthenaLogging.swift` (which `Auth`/`AppRequestContext` depend on via `LogScope`). Its dependencies are `AthenaCore`, `AthenaStore`, `Hummingbird`, `Crypto`, `Logging` — **no MLX**, so its tests run under `swift test` (`./deploy/test.sh`).

Option (b) was rejected: `@testable import athena` co-links the **entire executable graph** — Hummingbird + HummingbirdTLS (swift-nio-ssl) + MLX/Metal + AppleSiliconMetrics
+ the `AthenaServer` god-object — into the test bundle. Even though pure-logic tests wouldn't *execute* Metal, that bundle is slow to link, fragile (any Metal-touching static-init breaks CI), and improves no architecture. It is also the only path here that would force the test bundle to link an `@main` module, while `xcodebuild test` is already broken in this environment (so `swift test` would be the sole runner).

This is a **pure refactor**: the moved types are unchanged except for `public` access control and one behavior-preserving extraction — `AthenaMetrics`'s former local `pct` became a testable `static func percentile(_:_:)` with byte-identical nearest-rank math. No route changes (the e2e spec↔routes drift-guard is untouched). The `AthenaServer` god-object and every MLX-linked handler stay in the executable; `AthenaServerKit` is their testable substrate. The split also takes the first concrete step toward the audit's A6 ("server god-object") decoupling.

## Consequences

- `AthenaServerKit` is unit-testable under `./deploy/test.sh` with no model weights and no metallib; M70.2/.3 build their stub-tier tests on top of it.
- The moved server primitives are now `public` *within the package*. This is an application package, not a published SDK, so the wider surface is acceptable; it is the cost of making the security boundary testable.
- `Commands/` (audit NB4 — `ConfigEditor`/`parseTTLSeconds`/`isValidLabel`) was **not** addressed in this slice: `ConfigEditor` coupled to `Engine` (then in `Commands/Load.swift`) and `KVCompression` (in the MLX-linked `AthenaLLM`), which had to be relocated to an MLX-free target first. **DONE (M70.1b):** `Engine`/`KVCompression` now live in `AthenaCore`, and a new MLX-free `AthenaDeploy` target holds `ConfigEditor.swift`/`CLIParse.swift`/ `LaunchdPlist.swift`, unit-pinned by `ConfigEditorTests`/`AthenaDeployTests`/`CLIParseTests`.
- `Package.swift` gains one target; `deploy/build.sh` (xcodebuild) and `deploy/test.sh` (swift test) both still work; the `clients/` package and the rust-shim linkage are unaffected (the moved code reaches neither).
