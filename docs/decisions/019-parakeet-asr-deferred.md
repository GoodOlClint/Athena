# ADR 019 — Parakeet ASR: deferred (keep Whisper)

**Status:** Deferred (M74) — operator decision. Researched, not built. Whisper
remains the sole transcription engine. No code beyond the interim
`unsupported_transcription_arch` 4xx (v0.10.170).

## Context

A consumer asked for Parakeet (`parakeet-tdt-0.6b-v3`) transcription — higher
multilingual ASR quality than `whisper-large-v3-turbo` (Parakeet-TDT-0.6B is
SOTA on several multilingual benchmarks, lower WER, faster on GPU). The id had
been allowlisted ahead of any port, so `/v1/audio/transcriptions` returned a
bare 500 (the Whisper-only engine can't load a Parakeet checkpoint). That 500
is now a cause-naming 400 (`unsupported_transcription_arch`, v0.10.170).

**Feasibility — yes, it is portable:**
- Weights exist MLX-native: `mlx-community/parakeet-tdt-0.6b-v3` (~2 GB, 25 lang).
- Live reference: `senstella/parakeet-mlx` (Python MLX — TDT/RNN-T/CTC + the
  FastConformer encoder). MIT Swift starting point:
  `FluidInference/swift-parakeet-mlx`.
- Athena already vendors a **FastConformer encoder** (in Sortformer) — the same
  NeMo family Parakeet's encoder uses, so part of the port is in-house.

**The catch (decisive):**
- `FluidInference/swift-parakeet-mlx` is **archived** — its authors abandoned
  MLX-Swift for CoreML because the **performance was not good enough**. TDT /
  RNN-T decoding is an inherently sequential loop, and MLX-Swift's per-op
  overhead hurts there. A pure-MLX Parakeet in Athena may not be fast enough to
  justify it over Whisper.
- **Thesis tension (ADR 011):** Athena exists for the unified Metal memory
  governor, which requires MLX (one allocator). CoreML runs on the ANE / a
  separate path — **not Metal-governed** — the same reason WhisperKit was
  rejected. So the fast option (CoreML) breaks the thesis, and the
  thesis-aligned option (MLX) carries the archived-for-perf risk.

This is a milestone-scale port (FastConformer encoder + TDT decoder + BPE
tokenizer + mel features + decode loop) gated on an unresolved
performance/governance fork.

## Decision

**Defer.** Keep Whisper (`whisper-large-v3-turbo`) as the sole transcription
engine. Do not port Parakeet now, and do not adopt CoreML (it would breach the
ADR-011 governor thesis / the WhisperKit-rejection precedent without an explicit
exception). The interim behavior is the honest 4xx: a Parakeet/non-Whisper
transcription id is refused with `unsupported_transcription_arch` pointing at a
Whisper model.

Rejected-for-now alternatives: (a) full MLX port — real milestone, perf-risk
unquantified; (b) CoreML sidecar — fast but ungoverned, thesis-breaking. Both
remain reopenable.

## Consequences

- No transcription-engine change; the multi-backend transcription substrate is
  **not** built (unlike diarization's ADR 018, which had a clear win).
- The `TranscriptionArch` denylist + `unsupported_transcription_arch` 4xx stand
  as the durable "not supported yet" surface; when/if Parakeet lands, route its
  arch to a Parakeet backend instead of denylisting it.
- Consumers needing higher multilingual ASR run Whisper for now.

## Tripwire — reopen when any of:

- A **fast** MLX-Swift Parakeet/TDT decode appears (someone solves the
  sequential-loop overhead the archived port hit), removing the perf risk.
- Multilingual-ASR quality demand justifies a milestone **spike-first**: a
  minimal MLX encoder + TDT-greedy decode on one clip, benchmarked for tok/s +
  WER on-device, before committing to the full port. (This was the recommended
  path; the operator chose to backlog instead.)
- The governor thesis is revisited such that a non-Metal ASR tenant (CoreML on
  ANE) becomes acceptable as an explicit, scoped exception.
