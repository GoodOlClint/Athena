# ADR 018 — multi-backend diarization (pyannote pipeline for >4 / overlapping speakers)

**Status:** Accepted — shipped v0.10.168 (M74). The `pyannote` method is live
behind `POST /v1/audio/diarizations`; PyanNet segmentation ported to MLX,
forward validated on real audio (overlap detected, no 4-cap). Auto speaker
counting on long/messy audio is approximate — pass `num_speakers`/`max_speakers`
for an exact count (see plan + `docs/diarization.md`). Implementation:
`docs/multi-backend-diarization-plan.md`.

## Context

Athena's default diarizer is NVIDIA Sortformer
(`mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16`), wrapped end-to-end
by `MLXDiarizationModule`. Sortformer is **architecturally fixed at 4 speakers**
— this is a design constraint of the entire Sortformer family
(`diar_sortformer_4spk-v1`, `diar_streaming_sortformer_4spk-v2/v2.1`), confirmed
unchanged as of Feb 2026. There is no 8- or N-speaker Sortformer checkpoint to
re-point the module at, so "select a different Sortformer model" cannot solve
recordings with more than 4 speakers.

Athena already ships a second, cap-free path: `POST /v1/audio/diarizations` with
form field `method=cluster` (M25.3) runs WeSpeaker speaker-embeddings over a
naive 1.5 s/0.75 s sliding window (relative-RMS silence gate) → UPGMA
agglomerative clustering (`AgglomerativeClustering.swift`). On real ICSI meeting
audio it recovered 5 and 8 speakers where Sortformer reported 4. Its weakness is
the **front-end**: fixed windows with a crude energy gate give no learned speech
boundaries and **no overlap attribution** — a window straddling a speaker change
or holding two simultaneous voices is forced to a single label.

The operator's recordings have >4 speakers **with significant overlapping
speech that must be attributed**. That is precisely the gap the naive windowing
cannot close and a learned segmentation front-end can.

The standard arbitrary-speaker, overlap-aware approach on Apple Silicon is the
pyannote pipeline: a learned **segmentation** model emits per-window
speaker-activity (including overlap), each locally-active region is embedded,
and embeddings are clustered across windows into global speakers. Athena already
owns two of the three stages in MLX (WeSpeaker embeddings + agglomerative
clustering). The missing stage is the segmentation model, which exists as
MLX-native, ungated, **safetensors** weights:

- [`aufklarer/Pyannote-Segmentation-MLX`](https://huggingface.co/aufklarer/Pyannote-Segmentation-MLX)
  — PyanNet (SincNet 3×80 learnable bandpass + 4-layer BiLSTM 128/dir + linear),
  `model.safetensors` (5.96 MB) + `config.json`, **MIT**. Output `[batch,
  frames, 7]`: a 7-class **powerset** over ≤3 simultaneous local speakers per
  10 s window (∅, 3 singletons, 3 pairs). Same publisher as Athena's existing
  WeSpeaker mirror (`aufklarer/WeSpeaker-ResNet34-LM-MLX`).

This sidesteps the durable M25 blocker — the canonical
`mlx-community/pyannote-segmentation-3.0-mlx` ships only `weights.npz`, and MLX's
`loadArrays(url:)` reads safetensors/gguf, **not npz**. The aufklarer mirror is
safetensors + config, so no npz extractor is needed.

A clean MLX-Swift reference for the PyanNet forward + powerset decoding exists in
[soniqo/speech-swift](https://github.com/soniqo/speech-swift) (its WeSpeaker
piece is CoreML — irrelevant here, Athena's WeSpeaker is already MLX).

## Decision

1. **Diarization becomes multi-backend, selected by model class (ADR-016
   pattern).** `MLXDiarizationModule`'s allowlist spans both Sortformer ids and
   the pyannote-segmentation id. A new MLX-free `DiarizationBackend` detector
   (`config.json` `model_type` / `pyannet` markers; generative vs sortformer vs
   pyannoteSegmentation) routes `loadModel` to the right engine. The
   `diarization` slot stays a **single governed tenant** (ADR 011); selecting a
   pyannote model evicts Sortformer and vice versa — both models are tiny
   (Sortformer fp16 / PyanNet 6 MB) so rebind is cheap.

2. **The pyannote method is orchestrated at the route, reusing the M25.3
   stages.** A new `method=pyannote` on `POST /v1/audio/diarizations`:
   segment (new PyanNet engine) → embed each locally-active region via the
   existing `speakerEmbedding` module → `AgglomerativeClustering` with a
   **same-window cannot-link** constraint (two local speakers in one window are
   never merged) → **overlap-aware** turn assembly (a frame may belong to two
   global speakers). Clustering is **global over the whole file**, so emitted
   speaker ids are stable end-to-end (the same person = the same integer from
   0:00 to EOF); `SpeakerActivityRegion.localSpeaker` is an internal per-window
   id only. This mirrors exactly how `method=cluster` already composes
   `speakerEmbedding` + clustering at the route layer — the module owns only the
   model-specific segmentation step (`segment(audio:) -> [SpeakerActivityRegion]`).
   Whole-file is processed in one call (pyannote is internally windowed — no
   Parakeet-style attention-window limit); client-side chunking is rejected
   because it would break global ids. The full consumer contract (response
   shape, hints, overlap, embeddings stability, ASR pairing, shared cap) is
   recorded in the plan's "Consumer contract (confirmed M74)".

3. **Selection is explicit (no magic).** `method` picks the algorithm family —
   `sortformer` (default / absent, ≤4, fast, end-to-end), `cluster` (existing
   naive-window embedding cluster, no segmentation model), `pyannote` (new,
   overlap-aware). `model` selects the weights within the family. A
   method/model mismatch (e.g. `method=pyannote` with a Sortformer model id, or
   `method=sortformer` with the segmentation id) returns a cause-naming **4xx**
   in the standard error envelope, never a 500. Defaults are unchanged
   (Sortformer remains the default model); the segmentation model is an
   **additive** allowlist entry pulled by the operator (`athena init` aux-pull /
   allowlist-add), so no behavior changes for existing callers.

4. **Overlap is representable with no DTO break.** `DiarizationTurn`
   (start/end/speaker) and `DiarizationResponse` already permit overlapping
   turns (two turns may cover the same time span with different speakers).
   Sortformer and `cluster` continue to emit non-overlapping turns; only the
   pyannote path produces overlaps. No schema change to the response shape.

5. **MLX-free decision logic is unit-pinned (ADR 008/009).** Backend detection,
   powerset decoding tables, same-window cannot-link clustering wiring, and
   overlap turn-assembly live in pure Swift in `AthenaServerKit` /
   `AthenaCore` / a pure helper, pinned by `./deploy/test.sh`. MLX numerics
   (PyanNet forward) stay in `AthenaTranscription` and are validated on real
   audio via the Release binary + curl (the M25/M26 xcodebuild-sandboxes-env
   gotcha).

### Rejected / deferred alternatives

- **Add a higher-speaker Sortformer.** None exists; the cap is architectural.
- **Separate `diarizationSegmentation` ModuleID (both backends resident at
  once).** Cleaner concurrency for alternating requests, but adds a 6th module
  across ~8 enum switch sites (Load / OpenAPISpec enums / allowlist / lifecycle
  API) and a parallel allowlist surface. Rejected because both models are tiny
  (single-slot rebind is cheap) and the single-tenant-per-class model (ADR 011)
  is simpler. Re-open if alternating Sortformer/pyannote rebind latency proves
  to matter in practice (tripwire).
- **Full pyannote port including its own VAD/embedding.** Unnecessary — Athena's
  MLX WeSpeaker + clustering already cover stages 2–3; only segmentation is new.
- **CoreML segmentation (soniqo as a dep).** CoreML is not Metal-governed —
  same thesis tension that rejected WhisperKit. Use as a code reference only.

## Consequences

- **Passive-oracle preserved.** Both models are ungated HF fetches; no new
  outbound. The OpenAPI spec is updated in the same edit as the route
  (canonical-pipeline rule).
- New vendored subtree `Sources/AthenaTranscription/Pyannote/` (MIT-attributed),
  matching the Sortformer/WeSpeaker vendor pattern; substrate stays pristine.
- `cluster` is retained (works with no segmentation model; useful when the
  operator hasn't pulled PyanNet). `pyannote` supersedes it for quality.
- Governor: PyanNet adds a small (~6 MB weights + short activations) reservation;
  estimate conservatively, M5 reconciles post-load.
- Quality of the pyannote path is bounded by clustering threshold tuning and
  PyanNet port fidelity (see plan risks R2/R3); validated A/B vs `cluster` and
  Sortformer on real ICSI audio before the slice ships.

Plan: `docs/multi-backend-diarization-plan.md`.
