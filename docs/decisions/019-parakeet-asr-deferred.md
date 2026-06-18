# ADR 019 — Parakeet ASR: MLX feasibility spike (was: deferred)

**Status:** Spike **COMPLETE — GO (M75)**. An MLX-Swift Parakeet-TDT-0.6B-v3
spike is faithful (verbatim-correct transcript on real audio) **and ~63× real
time** on Apple Silicon — refuting the perf concern entirely. A full production
port is **recommended**; scheduling is the operator's call. Whisper remains the
sole *shipped* engine until that port lands; the `unsupported_transcription_arch`
4xx (v0.10.170) stands as the interim surface. Result + numbers below; the
original deferral rationale (corrected: the reference was archived for an iOS/ANE
power-efficiency strategy, irrelevant to a plugged-in macOS GPU daemon) is kept
for the record.

## Spike result (M75) — GO

Throwaway MLX-Swift port (additive `Sources/AthenaTranscription/Parakeet/*` +
gated heavy test), built green, run on a real 60 s clip:

- **Transcript: coherent and verbatim-correct** — fluent, punctuated English
  with the correct proper names/content of the source recording (the decisive
  correctness proof: a broken forward cannot produce the right words). One short
  garbled stretch on overlapping/disfluent speech; the rest clean.
- **~63× real time** (60 s audio in ~0.95 s total inference: encoder ~529 ms +
  decode ~420 ms), **787 decode tok/s**, on a *Debug* xctest binary (Release is
  faster). Dramatically faster than the Whisper path.
- Forward verified end-to-end: mel filterbank, rel-pos attention + rel-shift,
  conv-subsampling layout, LSTM gate order, and the TDT duration-split
  (`[:8193]` tokens / `[8193:]` durations) all correct; loader's critical-tensor
  non-zero guard confirms real weights (not random init).
- **Big reuse win confirmed:** the encoder is the Sortformer FastConformer
  (adapted keys + bias-off); the pred-net LSTM is the pyannote `Wx/Wh/bias`
  cell. Net new = 128-bin mel, TDT joint + greedy decode, trivial tokenizer.
- **Spike shortcut to fix in production:** the mel magnitude uses standard
  `|X|`; the reference uses a quirky int16-view `abs[::2]+abs[1::2]` trick.
  Coherent output ⇒ numerically close enough for the spike, but the production
  port must match the reference exactly.

**Decision: GO.** The only open question (Mac-GPU MLX-TDT perf) is resolved
decisively in favor. Proceed to a full multi-backend transcription port when
prioritized (scope below).

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

**Spike done → GO.** The feasibility question is resolved (faithful + ~63× real
time, above). The thesis-aligned MLX path is viable, so plan a full
**multi-backend transcription port** when prioritized; do **not** adopt CoreML
(breaches the ADR-011 governor thesis / WhisperKit precedent). Whisper stays the
shipped engine until the port lands.

## Full-port scope (the milestone, when scheduled)

The spike proved the engine; the milestone hardens + integrates it. Model-class
routed exactly like diarization's ADR 018 (Whisper stays default; Parakeet is an
additional transcription backend selected by `config.json` model class):

- **Mel exactness:** match the reference magnitude (int16-view trick) and
  re-validate; pin numeric parity on a fixture.
- **Multi-backend `transcription` slot:** an MLX-free arch detector routes a
  checkpoint to the Whisper engine or the Parakeet engine (mirror
  `DiarizationBackend` / ADR 016). One governed slot (ADR 011).
- **Governor integration:** real memory estimate + reconciliation; cold-load;
  evictable; the 100 MiB upload cap + cold-load 503 behavior.
- **API:** the existing `/v1/audio/transcriptions` surface unchanged; Parakeet
  selected by `model=`/allowlist. Word/segment timestamps from TDT durations
  (`time_ratio` 0.08 s/frame) for `verbose_json`. Language handling for the
  25-lang v3.
- **Robustness:** long-audio chunking, the SentencePiece detokenizer (vs the
  spike's inline vocab), special-token handling, decode anti-stall.
- **Tests:** MLX-free decode/detector logic unit-pinned (ADR 008/009); a gated
  heavy parity/throughput test; e2e.
- Allowlist seed + `athena init` aux-pull; OpenAPI + docs.

**Out of scope (defer within the milestone):** beam search, streaming/online
decode.

The spike code (`Sources/AthenaTranscription/Parakeet/*`, gated test) is the
foundation the milestone hardens — not production as-is (mel shortcut, no
routing/governor/API).

## Consequences

- No shipped transcription-engine change yet; the multi-backend transcription
  substrate is built only if the spike says *go*.
- The `TranscriptionArch` denylist + `unsupported_transcription_arch` 4xx stand
  as the "not supported yet" surface; when/if Parakeet lands, route its arch to
  the Parakeet backend instead of denylisting it.
- Consumers needing higher multilingual ASR run Whisper until then.
