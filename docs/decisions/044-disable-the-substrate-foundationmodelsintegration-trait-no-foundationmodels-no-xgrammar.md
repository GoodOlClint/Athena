# ADR 044 — Disable the substrate FoundationModelsIntegration trait (no FoundationModels, no xgrammar)

- **Status:** Accepted (operator directed the #91 bump; this records the trait decision #91's acceptance criteria required)
- **Date:** 2026-08-01
- **Deciders:** operator + agent
- **Context source:** issue #91 (substrate bump to `integration-2026-07-31`), blast-radius inventory from the #89 review

## Context

The mlx-swift-lm substrate at `integration-2026-07-31` declares a SwiftPM trait, `FoundationModelsIntegration`, **default-on**. With the trait enabled, every consumer compiles and links:

- `MLXFoundationModels` — an adapter bridging MLX inference to Apple's FoundationModels framework (macOS 27+ surface), pulled in conditionally by `MLXHuggingFace`, which Athena links.
- `MLXGuidedGeneration` + `MLXCXGrammar` — a grammar-constrained-decoding engine over **~30k lines of vendored xgrammar C++17**, linked only under the trait.

Athena uses neither. No daemon surface touches FoundationModels, and no ADR covers it appearing in the binary. Structured output is the rust-shim llguidance engine (M53, ADR list "M53 structured-engine swap") — a second, unused grammar engine in the same binary is exactly the "parallel implementation" class the canonical-pipelines rule treats as a defect, and unvetted build/link surface is a real cost for a passive-oracle daemon whose hardened-runtime posture (ADR 024 T1) is part of the product.

Athena's `Package.swift` is `swift-tools-version: 6.1`, so the `traits:` parameter (SE-0450) is available on every `.package(...)` form.

## Decision

**Pass `traits: []` on the mlx-swift-lm dependency — both the SCM declaration and the `ATHENA_LOCAL_DEV=1` path declaration** — so the default trait is disabled and neither FoundationModels adapter code nor the vendored xgrammar C++ enters Athena's build or link graph. The two declarations share one `substrateTraits` constant so they cannot drift.

The substrate is designed for this: disabling the trait compiles `MLXFoundationModels` to an empty library and drops the `MLXGuidedGeneration`/`MLXCXGrammar` targets from the graph; `MLXLLM`/`MLXVLM`/`MLXLMCommon`/`MLXEmbedders`/`MLXHuggingFace` are unaffected.

## Rejected alternatives

- **Accept the default trait** (do nothing): compiles ~30k lines of C++ we never call into a security-hardened daemon, adds a second grammar engine beside llguidance with no ADR covering it, and grows cold-build time. Rejected — surface must be deliberate, not inherited.
- **Adopt xgrammar and retire the rust-shim** (llguidance → MLXGuidedGeneration): a real option someday — it would delete the rust-shim FFI boundary (ADR 003) — but it is a structured-output *engine swap* (the M53 class of change) with its own correctness gates, not a substrate-bump rider. Rejected here; re-open as its own ADR if the substrate's engine matures into a reason.
- **Fork the substrate manifest to remove the trait**: pointless — the trait mechanism exists precisely so consumers opt out without forking.

## Consequences

- **Honesty boundary (measured 2026-08-01, Xcode 26.5):** the trait disable is honored fully by **open-source SwiftPM** (`swift build`/`swift test`: the athena debug binary and the test bundle carry **zero** xgrammar symbols; `MLXFoundationModels` compiles to empty `#if`-gated stubs) but **NOT by xcodebuild**, which honors `traits: []` at *resolution* (it errors against a manifest that declares no traits — that is how a stale `SourcePackages` checkout of the previous pin breaks the build; clear it) yet ignores `.when(traits:)` conditions at *build planning*, so the Release daemon still links MLXCXGrammar (~509 xgrammar symbols) and the FoundationModels adapter as dead code. Since MLX's Metal shaders force the Release binary through xcodebuild (no OSS-SPM path exists for it), the shipped binary keeps that dead weight until Xcode's package support catches up. **Tripwire: re-run the symbol check (`nm -gU athena | grep -c xgrammar`) on each Xcode major; when it hits 0 the disable is fully effective and this paragraph should be updated.**
- The manifest states the intended surface either way: unit-tier builds prove the daemon compiles without the trait's code, so nothing in Athena can silently grow a dependency on it.
- A future substrate feature gated on the trait will not light up in Athena until this ADR is superseded — that is the point (deliberate adoption).
- If Athena ever wants the substrate's guided-generation engine, that lands via a new ADR superseding the M53 llguidance choice, not by flipping the trait in passing.
- `ATHENA_LOCAL_DEV=1` builds behave identically to release builds with respect to the trait (both declarations share `substrateTraits`).
