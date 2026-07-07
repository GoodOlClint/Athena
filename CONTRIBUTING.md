# Contributing to Athena

Thanks for your interest. Athena is a single native macOS/MLX daemon; this guide covers what you need to build it, how it's tested, and how changes land.

## How this project is developed

Athena is built primarily by autonomous AI agents (Anthropic's Claude) under human direction — see the "How Athena is built" section of the [README](README.md). External contributions are welcome and are reviewed by the human operator the same way agent-authored changes are: against the tests and the architectural-decision record. Expect review to focus on whether a change fits the existing design, not just whether it works.

## Build prerequisites

- **macOS on Apple Silicon.** The daemon is Apple/MLX-only. (The `clients/` package is cross-platform, but the daemon is not.)
- **Full Xcode** — not just the Command-Line Tools. MLX's Metal shaders cannot be compiled by CLT alone.
- **The Metal Toolchain component**, downloaded once:
  ```sh
  xcodebuild -downloadComponent MetalToolchain
  ```
- **Rust** (for the structured-output shim in `rust-shim/`).

Build the shim, then the daemon:
```sh
rust-shim/build.sh
./deploy/build.sh Release    # xcodebuild → .build/xcode/.../Release/athena
```

## Tests

- **Unit tier:** `./deploy/test.sh`. Pure decision-logic tests (governor, auth/RBAC, server primitives, decode-path algebra). Per ADRs 008/009, MLX *numerics* are not evaluated under test — the pure logic is extracted into MLX-free seams so it's testable, while `MLXArray` math stays in the inference targets.
- **RBAC end-to-end:** `./deploy/e2e-rbac.sh`.
- **Heavy model gates:** several `deploy/e2e-*.sh` scripts exercise real models end-to-end (transcription, tool-calling, batching, KV snapshots). These need a model in the store and real hardware; they are run manually, not in CI.

A change to non-trivial logic should come with the smallest test that fails if the logic breaks.

## Architectural decisions

Athena records significant design choices as ADRs under [`docs/decisions/`](docs/decisions/). **Read the relevant ADRs before proposing a change that touches the passive-oracle rule, the OpenAPI spec, the Metal memory governor, or the API surface** — the design space may already have been explored. If your change makes a new architectural decision, record it as an ADR in the same change.

Two binding rules worth knowing up front:
- **All HTTP routes live in `Sources/athena/Server/OpenAPISpec.swift`** — update the spec in the same change as any route change.
- **All errors use the envelope `{"error":{"message","type","code"}}`.** No ad-hoc error shapes.

## Pull requests

- Keep changes small and focused; one concern per PR.
- Make sure `./deploy/test.sh` passes.
- Fill out the PR template (test evidence + ADR-discipline checkboxes).
- Don't introduce outbound network calls — the passive-oracle rule is binding (model-weight fetches and the opt-in remote-syslog sink are the only exceptions).
