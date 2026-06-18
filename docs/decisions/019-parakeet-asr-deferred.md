# ADR 019 — Parakeet ASR: MLX feasibility spike (was: deferred)

**Status:** Reopened — **MLX spike in progress (M75)**. The original deferral
(M74) rested on "the Swift-MLX reference was archived for performance." That
premise was **corrected by the operator**: the reference team archived for an
**iOS / ANE power-efficiency strategy**, which has no bearing on a plugged-in
macOS GPU daemon — so MLX-Swift Parakeet performance on a Mac is *unquantified,
not known-bad*. Since CoreML/ANE is off the table for Athena (governor thesis),
MLX is the only path, and a **spike** is the way to quantify it before any
milestone. Whisper remains the sole *shipped* transcription engine until the
spike reports. Interim error surface (`unsupported_transcription_arch` 4xx,
v0.10.170) stands. Spike scope at the end of this ADR.

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

**Why the Swift-MLX reference was archived — corrected:**
`FluidInference/swift-parakeet-mlx` is archived, but the team's stated reason is
**strategic and platform-specific, not a perf verdict**: *"Our team's goal is to
run more workloads on the ANE, so MLX doesn't really align with that right now"*
— and they were **specifically targeting iOS**, where they wanted the **ANE for
its power efficiency** (a mobile battery-budget concern). They pivoted to CoreML
to hit the Neural Engine on phones, not because MLX-Swift Parakeet was proven
too slow.

**This has essentially zero bearing on Athena.** Athena is a plugged-in macOS
daemon on Apple Silicon where the **GPU/MLX is the correct target** and mobile
power efficiency is a non-constraint. The iOS/ANE-efficiency motivation that
drove the archival simply does not apply. So the **performance of an MLX-Swift
TDT decode on a Mac is UNQUANTIFIED** — the reference archival gives no signal
about it either way. (TDT/RNN-T decoding *is* a sequential loop where MLX-Swift
per-op overhead *could* hurt, but that's a hypothesis to measure, not a result.)

**This actually clarifies Athena's path** rather than blocking it: the ANE-vs-GPU
tradeoff that drove the reference team **does not apply to Athena**. CoreML/ANE
is off the table here regardless — it runs outside the unified Metal memory
governor (ADR 011), the same reason WhisperKit was rejected — so **MLX is
Athena's only viable path** to Parakeet. The reference team's archival tells us
nothing about whether that path is fast enough.

So the only real unknown is **MLX-Swift TDT decode performance on-device**, and
the only way to retire it is a **spike** (encoder + TDT-greedy on one clip,
measure tok/s + WER). This remains a milestone-scale port (FastConformer encoder
+ TDT decoder + BPE tokenizer + mel features + decode loop), but it is **not**
blocked by a known performance wall.

## Decision

**Run a time-boxed MLX feasibility spike before any milestone commitment.** The
only open question is *does MLX-Swift Parakeet (FastConformer + TDT greedy) run
fast enough on a Mac to beat Whisper on quality-per-second?* Build the minimum
needed to measure that, throwaway-grade, off to the side of the daemon. Do **not**
adopt CoreML (breaches the ADR-011 governor thesis / WhisperKit precedent). Keep
Whisper as the shipped engine until the spike reports; the
`unsupported_transcription_arch` 4xx stays the interim surface.

## Spike scope (M75)

Minimum to produce a real number; **not** production:

- Load `mlx-community/parakeet-tdt-0.6b-v3` (config + safetensors) — confirm
  tensor keys / arch up front (the pyannote-style de-risk).
- Port the **FastConformer encoder** to MLX-Swift — reuse what's possible from
  the vendored Sortformer encoder (`ConvSubsampling` + conformer layers; same
  NeMo family) rather than porting from scratch.
- Port the **TDT decoder**: prediction network (LSTM) + joint network + the
  **greedy TDT decode loop** (token + duration prediction). Greedy only — no
  beam, no streaming, no chunking.
- Mel features + the model's BPE/SentencePiece tokenizer (multilingual v3).
- Harness: a **gated heavy test** (or throwaway target) that transcribes one
  real clip and prints **transcript + decode tok/s + wall-clock**; A/B the WER
  and speed against `whisper-large-v3-turbo` on the same clip.

**Exit criteria → informs the full-port decision:**
- *Go* (plan the full multi-backend transcription port, model-class routed like
  ADR 018): MLX Parakeet decodes at usable speed (target: ≥ Whisper-large-turbo
  throughput, ideally faster) AND lower/comparable WER.
- *No-go* (return to Deferred, this ADR records why): decode is meaningfully
  slower than Whisper with no clear optimization path.

**Explicitly out of scope for the spike:** governor integration, allowlist/API
wiring, multi-backend transcription routing, robustness, broad tests, beam
search, streaming. Those belong to the full port if the spike says *go*.

## Consequences

- No shipped transcription-engine change yet; the multi-backend transcription
  substrate is built only if the spike says *go*.
- The `TranscriptionArch` denylist + `unsupported_transcription_arch` 4xx stand
  as the "not supported yet" surface; when/if Parakeet lands, route its arch to
  the Parakeet backend instead of denylisting it.
- Consumers needing higher multilingual ASR run Whisper until then.
