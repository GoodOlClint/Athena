//! Athena structured-output shim — M3 skeleton.
//!
//! Intentionally empty in M0. M3 fills this with the `extern "C"` surface
//! described in Cargo.toml. Kept in the tree so the milestone boundary and
//! the build-vs-adopt decision (do not reimplement regex->DFA; depend on
//! `outlines-core`) are visible from day one.
//!
//! Named risk to carry into M3: Swift's `LogitProcessor` has no rejection
//! callback, so under MTP/speculative rollback the processor must diff the
//! committed token array itself and roll the Guide back.
