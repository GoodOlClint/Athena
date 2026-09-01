# 003 — rust-shim FFI panic strategy: per-entry `catch_unwind`

**Status:** Accepted — Implemented (M65.1 shipped v0.10.117)
**Date:** 2026-06-12
**Milestone:** M65 (audit-remediation; resolves standing DECISION #4; audit G1)

## Context

`rust-shim/src/lib.rs` is a C-ABI `staticlib` over `llguidance`. The schema string it compiles crosses the FFI from a **remote** caller (`/v1` `response_format.json_schema`). A panic inside llguidance (or our own code) would unwind across the `extern "C"` boundary — undefined behaviour that, in practice, aborts the whole daemon: a remote-triggerable DoS (audit **G1**, the program's only Critical).

Two ways to make a panic non-UB at a C boundary:

1. **`panic = "abort"`** in the crate profile — a panic immediately aborts. Simple, but it converts every panic (including ones we could otherwise turn into a clean error return) into a process kill, and `catch_unwind` becomes a no-op, so we lose the ability to degrade gracefully.
2. **Per-entry `catch_unwind`** — each `extern "C"` entry runs its body inside `std::panic::catch_unwind`; a caught panic becomes the function's normal failure sentinel (NULL / -1 / false) plus a stashed error message. Requires the crate to stay `panic = "unwind"`.

## Decision

**Per-entry `catch_unwind` (option 2). The crate stays `panic = "unwind"`; we do NOT set `panic = "abort"`.**

A hostile or buggy schema must degrade to an ordinary error the caller sees as a failed structured-output request (which the server maps to the standard `{"error":{…}}` envelope), never a daemon abort. A single `ffi_guard(label, default, …)` helper wraps every entry; on a caught panic it records `"<label>: panic caught at FFI boundary: <payload>"` for `oc_last_error` and returns `default`. The helper asserts unwind-safety at the boundary (`AssertUnwindSafe`) because most handles carry an `llguidance::TokenParser`; this is sound because a caught panic is reported as failure and the handle is discarded on that path — post-panic state is never observed.

## Consequences

- A panicking schema can no longer take down the daemon; it returns a typed error.
- The crate must remain `panic = "unwind"` — a future `panic = "abort"` would silently neuter every guard. A compile-time note in `Cargo.toml`/`lib.rs` documents this.
- Caught panics still print the default panic hook to stderr (captured by the unified log), so the cause remains diagnosable.
- Shipped alongside the other M65.1 input caps (G2/G3/G6/G10). Rust suite: 9/9.
