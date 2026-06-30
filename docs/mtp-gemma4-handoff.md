# Handoff: wire up Gemma 4 MTP speculative decoding (testable now)

**From:** mlx-swift-lm owning agent · **Date:** 2026-06-30
**Status of the substrate:** ready. The Gemma 4 MTP path (including E-series) is
complete in `../mlx/mlx-swift-lm` on the **`integration`** branch.

## TL;DR

Gemma 4 ships tiny "assistant" **drafter** checkpoints (`gemma4_assistant`) that
do multi-token prediction for speculative decoding — 1.6×–3× decode speedup.
mlx-swift-lm now supports them end-to-end. Athena needs to: load a Gemma 4
target + its matching drafter, and call the `mtpDrafter:` overload of
`generate(...)`. **No `Package.swift` change** — your path dep already sees it.

## Build wiring (nothing to change, just verify)

`Athena/Package.swift` already has `.package(path: "../mlx/mlx-swift-lm")`, which
resolves to `~/Source/mlx/mlx-swift-lm` and builds against its **working tree**.
That clone is currently on `integration`, which contains the MTP work
(commit `0bec134`, "Implement Gemma 4 E-series MTP centroid embedder", on top of
upstream `e145aca` "Add Gemma 4 MTP speculative decoding (#308)").

- Confirm before building: `git -C ~/Source/mlx/mlx-swift-lm rev-parse --abbrev-ref HEAD` → `integration`.
- Do a clean build so MLXVLM recompiles (the drafter lives in the `MLXVLM` product, which you already depend on).
- `import MLXVLM` somewhere in the linked target so the `gemma4_assistant`
  drafter type self-registers (the `registerModelType("gemma4_assistant")` call
  lives in `MLXVLM/Gemma4AssistantRegistration.swift`).

## The API to call

Public, in `MLXLMCommon/Evaluate.swift`:

```swift
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,           // the Gemma 4 TARGET (verifier)
    mtpDrafter: any MTPDrafterModel, // the gemma4_assistant DRAFTER
    blockSize: Int = 4,              // tokens/round; 4 matches mlx-vlm default
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation>
```

There's also a `generateTokens(... mtpDrafter: ...) -> AsyncStream<TokenGeneration>`
raw-token variant. The drafter holds **no** target-derived state (the target is
threaded in per round via `draftBlock(target:...)`), so a loaded drafter is safe
to reuse/share across generations.

## Loading the drafter

Two paths — pick one.

**A. Factory (preferred for Athena's registry style).**
`MTPDrafterModelFactory.shared.load(...)` (inherited `GenericModelFactory.load`)
returns an `MTPDrafterContainer`; `container.context.model` is the
`any MTPDrafterModel` you pass as `mtpDrafter:`. The 26B/31B drafter
`ModelConfiguration`s are pre-registered in `MTPDrafterRegistry`
(`gemma4_26B_assistant_bf16`, `gemma4_31B_assistant_bf16`); for **E2B/E4B**
construct `ModelConfiguration(id: "mlx-community/gemma-4-E4B-it-assistant-bf16")`.

**B. Manual (what the substrate's integration test does — simplest to read).**
```swift
let cfg = try JSONDecoder().decode(
    Gemma4AssistantConfiguration.self,
    from: Data(contentsOf: drafterDir.appendingPathComponent("config.json")))
let drafter = Gemma4AssistantDraftModel(cfg)
try loadWeights(modelDirectory: drafterDir, model: drafter)
// drafter is `any MTPDrafterModel`
```
The target loads exactly as you load any Gemma 4 model today (it IS your normal
`ModelContext`). No special target setup — the target's emit hooks are already
in the merged `Gemma4.swift`.

## Model pairs (target ↔ drafter, all `mlx-community`)

| Target | Drafter | Notes |
|---|---|---|
| `gemma-4-31b-it-8bit` (or `-bf16`) | `gemma-4-31B-it-assistant-bf16` | bit-exact verified upstream; **start here** |
| `gemma-4-26B-A4B-it-bf16` | `gemma-4-26B-A4B-it-assistant-bf16` | MoE target |
| `gemma-4-e4b-it-4bit` | `gemma-4-E4B-it-assistant-bf16` (~78 MB) | E-series; uses the new centroid head |
| `gemma-4-e2b-it-4bit` | `gemma-4-E2B-it-assistant-bf16` (~78 MB) | E-series |

Drafters are **bf16 only** for now. Pair each target with its **matching-size**
drafter — they share K/V geometry; mismatched pairs won't work.

## Suggested test (smallest thing that proves it)

1. Load `gemma-4-E4B-it-assistant-bf16` drafter + the `e4b-it-4bit` target.
2. Run the `mtpDrafter:` overload at `temperature: 0`, `maxTokens: 32`,
   `blockSize: 4` on a simple prompt; drain the stream.
3. Read the `.info` `GenerateCompletionInfo`: expect `proposedDraftTokens > 0`,
   `acceptedDraftTokens > 0`, `passthroughReason == nil`, coherent text.
4. Compare tok/s vs. a plain non-speculative `generate(...)` on the same target
   for the speedup.

The substrate already has this exact shape as a gated integration test
(`MTPIteratorEndToEndDiagnosticTests.testMTPE4BPairProducesAcceptedDrafts`,
enabled by `TEST_E4B_PAIR`) — mirror its body.

## Known limits (don't file these as Athena bugs)

- bf16 drafter weights only (no quantized drafter yet).
- Single-stream only (no batched B>1 MTP).
- If the target's KV cache quantizes **mid-stream**, the iterator transparently
  falls back to single-token generation (`passthroughReason` set) — correctness
  preserved, speedup dropped. So don't combine MTP with mid-stream KV-quant
  onset if you want the speedup.
- temp=0 byte-identity to non-speculative is bounded (~64 tokens) by an MLX
  fused-SDPA numerical quirk; the speculation-correctness guarantee still holds.

## Open upstream item (FYI, not blocking)

`gemma4_assistant` loads via `MTPDrafterModelFactory`, **not** the standard
`LLMModelFactory`/`draftModel:` path (issue ml-explore/mlx-swift-lm#279). Use the
MTP factory/overload above — that's the supported entry point.
