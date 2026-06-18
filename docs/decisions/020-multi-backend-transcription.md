# ADR 020 — multi-backend transcription (Whisper + Parakeet)

**Status:** Proposed (M76) — awaiting operator approval. No production code yet
(the ADR-019 spike foundation exists but is not wired in). Pairs with
`docs/parakeet-transcription-plan.md`.

## Context

The transcription module is **Whisper-only**: `MLXTranscriptionModule` loads
exclusively via `WhisperLoader` and rejects non-Whisper vocab. A consumer wants
**Parakeet-TDT-0.6B-v3** for higher multilingual ASR quality. ADR 019's spike
proved an MLX-Swift Parakeet port is **faithful and ~63× real-time** on Apple
Silicon — a clear GO — so the only question left is the production architecture.

This is the same shape as diarization's ADR 018: one modality, a default engine,
and an additive higher-capability backend selected per request — solved there by
**model-class routing** through a single governed slot. Transcription should
follow that proven pattern rather than invent a new one.

## Decision

1. **Transcription becomes multi-backend, selected by model class (ADR-016/018
   pattern).** The `transcription` allowlist spans Whisper and Parakeet
   checkpoints. A `TranscriptionArch` detector — extended from today's denylist
   (ADR-019/v0.10.170) into a **positive router** (`whisper` / `parakeet` /
   `unsupported`, MLX-free, from `config.json`) — routes `loadModel` to the
   Whisper engine or the Parakeet engine. Single governed `transcription` slot
   (ADR 011); the resident model's class decides the engine. No CoreML (breaches
   the governor thesis / WhisperKit precedent).

2. **Whisper stays the default; Parakeet is additive.** `whisper-large-v3-turbo`
   remains `transcription` default; Parakeet is an allowlist entry the operator
   pulls (`athena init` aux-pull + seed), selected via `model=`. **Open decision
   for review:** whether Parakeet should *become* the default once parity is
   proven (higher multilingual quality + ~63× RT argue for it; Whisper is the
   established, feature-complete path). Recommendation: ship additive first,
   revisit default after a real WER/throughput A/B (see plan).

3. **The `/v1/audio/transcriptions` surface is unchanged.** Same multipart route,
   `response_format` (`json`/`text`/`verbose_json`/`srt`/`vtt`), `model=`
   selection, 100 MiB cap, cold-load behavior. Parakeet word/segment timestamps
   come from the **TDT durations** (`time_ratio` = 0.08 s/encoder-frame) for
   `verbose_json`/SRT/VTT. `diarize=true` still uses the diarization slot
   (orthogonal; ADR 018/the v0.10.170 409 stands).

4. **The spike foundation is hardened, not re-written.** `Parakeet/*` from ADR
   019 is the base; the milestone adds mel-exactness (match the reference
   int16-view magnitude), governor integration, the proper SentencePiece
   detokenizer, long-audio chunking, language handling, and tests. MLX-free
   decode/detector logic is unit-pinned (ADR 008/009); MLX numerics validated on
   real audio via the gated heavy test.

### Rejected / deferred

- **CoreML/ANE Parakeet** (FluidAudio) — fast but not Metal-governed; breaches
  ADR 011. Rejected (same as WhisperKit).
- **Replace Whisper outright** — no; keep it as default + fallback until a parity
  A/B justifies otherwise. Whisper's maturity and the conservative default
  matter.
- **Beam search / streaming decode** — deferred within the milestone (greedy TDT
  is what the spike validated and what production ships first).

## Consequences

- Passive-oracle preserved (ungated HF weight fetch only). OpenAPI updated in the
  same edit as any route change (canonical-pipeline rule). Error envelope
  unchanged; `unsupported_transcription_arch` 4xx becomes the *router's*
  "neither Whisper nor Parakeet" path instead of a flat denylist.
- New vendored subtree hardened under `Sources/AthenaTranscription/Parakeet/`
  (Apache-2.0/MIT-attributed reference lineage), matching the Sortformer/pyannote
  vendor pattern; substrate pristine.
- Governor: Parakeet ~0.6B (~2.4 GB fp; weights mmap'd) gets a real estimate +
  M5 reconciliation, evictable, single-tenant with Whisper.
- A consumer can pick higher-quality multilingual ASR by `model=`; Whisper
  callers are byte-unchanged.

Plan + slices: `docs/parakeet-transcription-plan.md`.
