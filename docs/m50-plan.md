# M50 — MLX clearCache audit for non-LLM modules

Planning doc captured 2026-05-28 after the M48-M49 program closed.
Implements [A-1 from the deferred-items review](
../README.md): hunt the M46.6 leak pattern across the modules the
audit list flagged but never inspected.

## Status

**Shipped (M50.1–.5, v0.10.80).** `clearCache()` added across the flagged non-LLM
modules (`WhisperDecode`, `WeSpeakerModel`, `Sortformer`, `TriAttentionScorer`,
`VectorStore`). Audit list lives in
[project_mlx-clearcache-per-call](
../../.internal/projects/-Users-goodolclint-Source-Athena/memory/project_mlx-clearcache-per-call.md)
and was captured immediately after M46.6 found the 37.9 GiB embedder
leak. Verified 2026-05-28 against current source — every audit target
still has explicit `.eval()` calls AND no `MLX.Memory.clearCache()`,
so the leak risk is real and unhunted.

## Trigger

M46.6 (v0.10.58, M46 program) chased a the consuming application E2E failure
mode in which the 4 B embedder's `residentBytes` drifted from
~157 MiB initial to ~37.9 GiB after a handful of mixed-length batch
calls (~4-5× the weight footprint). Root cause: `result.eval()`
forces lazy-graph materialization but does NOT release the
allocator pool — MLX-swift keeps freed allocations in a cache for
reuse, and that cache grows unbounded if nothing flushes it. The
LLM decode loops had figured this out and call `clearCache()` every
256 tokens; the embedder did not. Adding a per-bucket `clearCache()`
after `bucketVectors` materialization fixed it.

The same shape is structurally present in every other MLX module
Athena ships. The audit list captured the suspects; nothing has
chased them.

## Verified audit targets (2026-05-28)

Confirmed against current source — each has `eval()` calls AND
zero `clearCache()`:

| File | Eval site(s) | Shape | Risk |
|---|---|---|---|
| `Sources/AthenaTranscription/WhisperDecode.swift` | `audio.eval()` at L218, L304 | Token-by-token Whisper decode loop | HIGH — same shape as LLM decode; long transcriptions × many chunks |
| `Sources/AthenaTranscription/Whisper.swift` | encoder + KV at L51, L52, L67 (`evalAll()`) | Per-layer KV materialization | MED — one-shot per audio frame |
| `Sources/AthenaTranscription/WhisperWordAlign.swift` | DTW alignment path | Per-segment scoring | LOW-MED |
| `Sources/AthenaTranscription/LogMel.swift` | `mel.eval()` at L122 | One-shot mel-spectrogram per audio chunk | LOW |
| `Sources/AthenaModels/TriAttention/TriAttentionScorer.swift` | `reduced.eval()` at L32 | Per-layer eviction scoring | MED — fires every `divideLength` decode tokens at long context |
| `Sources/AthenaTranscription/Sortformer/*` | (vendored — no top-level eval visible at module grep depth) | M4.3 diarization | MED — vendored Python port; eval likely inside the vendored code |
| `Sources/AthenaTranscription/WeSpeaker/*` | (vendored) | M25 speaker embedding | MED — same |
| `Sources/AthenaStore/VectorStore.swift` | implicit via `.item()` on cosine top-k score | Per-query vector search | LOW — single matmul + reduce |

## Validation procedure (from the memory)

For each module:

1. Load just that module via
   `athena load --<class>-model <id> ...` (or the module-specific
   path).
2. Hit `/healthz` (or `athena status`), record per-module
   `residentBytes` baseline (should be near the module's weight
   bytes).
3. Drive 5-10 realistic workload calls — mixed-length audio for
   transcription, varied speaker count for diarization, mixed
   prompts for TriAttention, varied vector dims for VectorStore.
4. Re-read `/healthz` `residentBytes` for that module.
5. If it grew **>1.5× the weight footprint**, you've found a leak.
6. Add a `clearCache()` at the appropriate loop boundary, mirror
   the LLM-decode pattern, re-validate.

## Sequencing

Four slices, ordered by impact × risk × validation difficulty.

### M50.1 — Whisper / transcription (HIGH priority)

Audit + fix as a single ship covering the four transcription-class
files:

- `WhisperDecode.swift` (token-by-token decode — likely needs
  every-N-token periodic clear, exactly mirroring LLM decode)
- `Whisper.swift` (encoder per-frame eval — one-shot end-of-frame
  clear)
- `WhisperWordAlign.swift` (DTW — per-window clear if leak)
- `LogMel.swift` (one-shot per audio chunk — end-of-call clear)

These four cover the entire transcription pipeline; the leak risk
is highest because long-audio transcriptions iterate many chunks ×
many tokens. Validation workload: 5 audio files of varying length
(short / medium / hour-long), check `residentBytes` between calls.

### M50.2 — Speaker embedding (M25 / WeSpeaker)

`MLXSpeakerEmbeddingModule.embed()` — single-module audit. Workload:
varied-length audio segments × varied speaker counts. The vendored
WeSpeaker code path is the likely leak source; the wrapper module is
silent on clearCache. End-of-`embed()` clear is the analog if leak
present.

### M50.3 — Diarization (M4.3 / Sortformer)

`MLXDiarizationModule.diarize()` — single-module audit. Same pattern
as M50.2. Workload: varied audio lengths × varied speaker counts.

### M50.4 — TriAttention scorer

`TriAttentionScorer.swift` fires per-attention-layer during eviction
runs. Audit needs a long-context decode (>= the configured
`kvBudget`) to engage the scorer at all. Workload: drive a
TriAttention-configured LLM on a long-prompt decode, watch
`residentBytes` over the decode iterations. Fix at the per-scorer-
call boundary if leak present.

### M50.5 — VectorStore

Lower-priority audit because cosine top-k is a single matmul +
reduce per query, not a loop. But the M5.5 RSS probe (M42) made the
governor see VectorStore's working set, and any drift here under
sustained vector-search load would compound with embedding pressure.
Workload: 100+ top-k queries against a populated store; check
`residentBytes`. End-of-query clear if leak present.

## Implementation pattern

The fix shape is identical for every module — mirror M46.6:

```swift
// Token-by-token / per-iteration loop:
if iteration % N == 0 { MLX.Memory.clearCache() }

// Per-batch / per-call boundary:
let result = computeResult()  // ← .eval() materializes
let extracted = result.asArray(Float.self)
MLX.Memory.clearCache()  // ← AFTER conversion out of MLXArray
return extracted
```

The clear must happen **AFTER** the result is converted out of an
`MLXArray` into a Swift-side type (`[Float]`, `Data`, etc.) —
clearing while a downstream reader still holds an `MLXArray`
reclaims something the caller needs. The clear is essentially free
when the cache is empty; over-eager clearing is benign.

## Tests

The clearCache call has no observable behavior change beyond memory
posture — output bits are identical with and without the clear. So
tests for M50 are observational, not unit:

- Each slice's e2e test (existing test infra) must still pass.
- New gated `ATHENA_RUN_MODEL_TESTS=1` regression: drive 20+ calls
  on the audited module, assert final `residentBytes` ≤ 1.5×
  baseline. One test per audited module class — parallels the
  validation procedure.

CI-safe pure unit tests don't add value here — the leak is
allocator-level behavior that can't be asserted without a real MLX
runtime.

## Risks

| Risk | Mitigation |
|---|---|
| Over-eager `clearCache` while a downstream reader holds an `MLXArray` | The fix pattern explicitly clears AFTER conversion to Swift-side types. Verified by tests that drive the module end-to-end and check output validity. |
| Vendored module internals (Sortformer, WeSpeaker) hide the real eval point | Inspect the vendored code's `.eval()` / lazy-materialization touchpoints. If the leak is inside the vendored layer, the clear goes in our wrapper module at the call boundary, after the vendored call returns. |
| False positives — `residentBytes` may drift for reasons other than allocator pool growth (e.g. KV cache growth in a model with long context) | Validation workload picks call sequences that should hold context length steady; any drift past 1.5× weights with steady-state context is allocator pool, not KV. |

## Definition of done for the M50 program

- [ ] All 5 slices shipped with their respective module's
      `residentBytes` ≤ 1.5× weights after a sustained workload
      (gated e2e regression).
- [ ] `project_mlx-clearcache-per-call.md` memory updated to move
      each audited module from "audit targets" to "known instances
      (correctly handled)".
- [ ] `MEMORY.md` index entry for M50.
- [ ] No regression in the LLM decode path (the existing
      every-256-token clear continues to work — M50 should only
      ADD clears in the audited paths).

## Out of scope (deferred follow-ups)

- A daemon-wide watchdog that monitors `residentBytes` drift and
  alerts when any module crosses a threshold. Could be M51-class
  observability work after the audits land.
- Auto-tuning the clear cadence (every-N) based on workload —
  current LLM N=256 was chosen by inspection; M50 modules adopt
  similar boundaries empirically.
- Vendored-module upstream fixes (if leaks are found inside
  Sortformer / WeSpeaker, we fix at the wrapper level for M50;
  upstream PRs are a separate workstream).
