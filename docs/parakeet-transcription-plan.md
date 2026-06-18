# Multi-backend transcription (Parakeet) — change plan (M76)

**Status:** Shipped M76 (v0.10.171–175). All slices S1–S5 landed; pairs with
ADR 020 (Accepted). Usage: `docs/transcription.md`. The ADR-019 spike
(`Sources/AthenaTranscription/Parakeet/*`) was the validated foundation this
hardened.

## Goal

Add **Parakeet-TDT-0.6B-v3** as a second transcription engine behind the
existing `/v1/audio/transcriptions`, selected by model class — higher
multilingual ASR quality, ~63× real-time (ADR 019). Whisper stays the default;
existing callers are byte-unchanged.

## Spike → production gap (what the milestone adds over ADR 019's spike)

| Area | Spike (have) | Production (add) |
|---|---|---|
| Forward | faithful, ~63× RT | mel-exactness (reference int16-view magnitude); numeric parity pin |
| Routing | standalone model | `TranscriptionArch` positive router (whisper/parakeet/unsupported); load dispatch |
| Module | throwaway loader | governed `MLXTranscriptionModule` second backend; estimate + reconcile; evictable; cold-load |
| Tokenizer | inline `joint.vocabulary` | SentencePiece detokenizer + special-token handling |
| Timestamps | none | word/segment from TDT durations (0.08 s/frame) for verbose_json/SRT/VTT |
| Audio | one clip | long-audio chunking + overlap stitch; language handling (25-lang v3) |
| Tests | one heavy bench | MLX-free unit pins + gated parity/throughput + e2e |

## Architecture (mirrors ADR 018 diarization)

```
MLXTranscriptionModule (actor, ModuleID.transcription)
 ├─ allowlist spans: [whisper ids…, parakeet id]
 ├─ TranscriptionArch router (MLX-free): config.json → .whisper | .parakeet | .unsupported
 ├─ load → WhisperModel (existing)  OR  ParakeetModel (hardened spike)
 └─ transcribe(audio, opts) → TranscriptionResult   (engine-dispatched)
```

`/v1/audio/transcriptions` route unchanged; `model=` selects weights, the
resident model's class selects the engine. `.unsupported` → the existing
`unsupported_transcription_arch` 4xx.

## Slices (each: annotated tag direct-to-main, `appVersion` bump IN the slice
commit, ≥1 regression test; pre-commit pipeline Tests → Security → Quality →
Refactor)

**S1 — `TranscriptionArch` router + backend-dispatched load (MLX-free seam).**
Turn the denylist into `.whisper`/`.parakeet`/`.unsupported` (config.json
`model_type`/`architectures`); `MLXTranscriptionModule.loadModel` dispatches to
the Whisper or Parakeet engine; Parakeet path returns `notImplemented` until S2.
Unit-pin the router table. *Test:* router classification (whisper, parakeet,
bert→unsupported); load dispatch.

**S2 — Harden the Parakeet engine.** Mel-exactness (match the reference
int16-view magnitude trick; pin a numeric-parity test on a fixture), governed
`ParakeetModel` (memory estimate + reconcile, evictable, cold-load), proper
SentencePiece detokenizer (`tokenizer.model`) + special-token stripping. Wire
into the module's `transcribe`. *Test:* gated heavy — transcript parity + ~RT
throughput on real audio (vs the spike baseline).

**S3 — Timestamps + formats.** Word/segment timestamps from TDT durations
(0.08 s/frame) → `verbose_json` segments/words, SRT, VTT. Language field for the
25-lang model. *Test:* timestamps monotonic + within clip bounds (gated heavy);
format serialization (unit).

**S4 — Long-audio + robustness.** Chunking with overlap + stitch for clips
beyond a single encoder pass; decode anti-stall (`max_symbols`); empty/edge
audio. *Test:* long-clip e2e; anti-stall unit.

**S5 — Plumbing: routing, allowlist, OpenAPI, docs, A/B.** Allowlist seed +
`athena init` aux-pull for the Parakeet id; `/v1/models` + `athena ps` show it;
OpenAPI note (same edit as any route touch; drift-guard green); docs. **WER +
throughput A/B** Parakeet vs `whisper-large-v3-turbo` on a labeled set →
**informs the open "make Parakeet default?" decision** (ADR 020 point 2).
*Test:* OpenAPI drift-guard; e2e-rbac unaffected.

## Test bar

- MLX-free logic (router, timestamp math, detokenizer, chunk stitch) → pure unit
  tests under `./deploy/test.sh` (ADR 008/009).
- MLX numerics → gated heavy (`ATHENA_RUN_MODEL_TESTS=1`, run via `xcrun
  xctest`) on real audio; pin mel parity + transcript coherence + throughput.
- Route → e2e; `./deploy/e2e-rbac.sh` stays green.

## Risks

- **R1 mel parity.** The spike's `|X|` vs the reference int16-view trick. S2
  matches it exactly + pins a numeric-parity test (mitigated; coherent already).
- **R2 tokenizer fidelity.** Inline vocab worked for the spike; production needs
  the SentencePiece detokenizer + special-token handling for all 25 languages.
- **R3 long-audio.** Encoder is O(T²) in attention; very long clips need
  chunking (S4). The 100 MiB cap bounds input; chunking bounds memory.
- **R4 governor estimate.** ~2.4 GB weights mmap'd + activations; conservative
  estimate, M5 reconciles.
- **R5 default-engine regression risk.** Keep Whisper default until the S5 A/B
  proves parity; making Parakeet default is a separate, gated decision.

## Out of scope

- Beam search; streaming/online decode (greedy TDT only).
- Replacing Whisper (stays default + fallback).
- CoreML/ANE (thesis-breaking, ADR 020 rejected).

## Open decision for review

**Should Parakeet become the default transcription engine** once S5's A/B proves
parity? Recommendation: ship additive (Whisper default) first; decide default on
the data. Flagged here so the reviewer rules on it before S5.
