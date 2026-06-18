# Multi-backend diarization — change plan (M74)

**Status:** Accepted — operator-approved. Pairs with ADR 018. Implemented as
stacked slices S1–S4 below.

Confirmed during de-risk: the existing `cluster` path is *tunable* on messy
6-speaker audio but the right `threshold` is strongly audio-dependent (0.75→65,
0.85→12, 0.90→5 speakers on one real file) — non-transferable across recordings.
That sensitivity is exactly what the learned segmentation front-end removes;
hence the build proceeds rather than shipping threshold tuning alone.

## Goal

Support diarizing recordings with **>4 speakers and overlapping speech**, which
the default Sortformer backend cannot (hard 4-speaker cap, no higher-speaker
checkpoint exists). Deliver a learned, overlap-aware **pyannote** backend
alongside Sortformer and the existing `cluster` path, all behind
`POST /v1/audio/diarizations`, selected explicitly per request.

## Research summary (what to build, what to reuse)

| Stage | Status in Athena | Source |
|---|---|---|
| Segmentation (learned, overlap) | **MISSING — build this** | `aufklarer/Pyannote-Segmentation-MLX` (safetensors+config, MIT, 6 MB) |
| Speaker embedding | **Have it** (MLX WeSpeaker, `speakerEmbedding` module) | M25.1 |
| Clustering | **Have it** (`AgglomerativeClustering.swift`) | M25.3 |
| Overlap-aware turn assembly | **MISSING — build this** | new pure helper |
| End-to-end ≤4 (fast) | **Have it** (Sortformer, default) | M4.3 |
| Naive-window cluster (>4, no overlap) | **Have it** (`method=cluster`) | M25.3 |

PyanNet = SincNet (3 conv, 80 learnable bandpass) → 4-layer BiLSTM (128/dir) →
linear → `[batch, frames, 7]` powerset over ≤3 simultaneous local speakers per
10 s window (∅ · 3 singletons · 3 pairs). Process 10 s windows, 50% overlap,
~23× real-time on M2 Max. Swift/MLX forward + powerset decode reference:
soniqo/speech-swift.

## API (after the change)

`POST /v1/audio/diarizations` (multipart) — unchanged route, response shape
unchanged.

- `method` (new, optional): `sortformer` (default / absent) | `cluster` |
  `pyannote`.
- `model` (existing): selects weights within the chosen method's family from the
  `diarization` allowlist. Mismatch → 4xx `model_not_available` /
  `invalid_method_model` (cause-naming, standard envelope).
- `num_speakers` (exact) / `min_speakers` (floor) / `max_speakers` (cap) /
  `threshold` (default 0.75) — clustering hints for `pyannote`. `min_speakers`
  maps to `AgglomerativeClustering.minClusters`; `max_speakers` to
  `maxClusters`; `num_speakers` to the exact count. Optional `min_duration`,
  `onset`/`offset` segmentation thresholds if tuning needs them. (Also accepted
  on `method=cluster` for parity.)
- Response: `{num_speakers, segments:[{start,end,speaker}]}`. The `pyannote`
  path may emit **overlapping** segments (same span, different speaker); no
  schema change.

Default behavior for every existing caller is byte-unchanged.

## Consumer contract (confirmed M74)

Locked with the audio-consumer agent that is decoupling ASR (Parakeet) from
diarization. All confirmed against the ADR-018 design:

1. **Response shape unchanged** — `{num_speakers, segments:[{start,end,speaker}]}`,
   OpenAI multipart, Bearer auth. No change.
2. **Globally-stable speaker ids.** Speaker integers are **global across the
   whole file**, not per-window. `SpeakerActivityRegion.localSpeaker` is an
   internal per-window track id; the route's cross-window agglomerative
   clustering maps every region to a **global** label before emitting turns, so
   the same person carries the same integer from 0:00 to end-of-file. The
   same-window cannot-link constraint guarantees two simultaneous local tracks
   never collapse to one global id.
3. **No speaker cap.** The pyannote/cluster path has no architectural ceiling
   (the 4-cap is Sortformer-only). `num_speakers` / `min_speakers` /
   `max_speakers` hints accepted (all optional; absent ⇒ auto via `threshold`).
4. **Whole-file, single call.** pyannote is internally windowed (10 s / 50%
   overlap) and stream-friendly, so there is **no Parakeet-style 24-min
   attention window** — a 77-min file diarizes in one call. Binding limits:
   (a) the **100 MiB upload cap** shared with all `/v1/audio/*` — note raw
   16 kHz mono WAV of 77 min is ~148 MB and will 413; use compressed
   (m4a/opus); (b) clustering cost grows with the number of speech regions
   (O(n²) memory, O(n³) time in the current UPGMA). **Commitment: keep
   whole-file server-side — client-side chunking is explicitly rejected because
   it would break global ids.** S4 benchmarks a real ~77-min file and reports a
   concrete region-count/latency number; if clustering dominates, mitigate with
   coarser hop / region cap / faster linkage — never by chunking.
5. **Overlap = multiple segments.** Overlapping speech is emitted as multiple
   segments with intersecting time spans and **different** speaker ids; overlaps
   are **not** collapsed. (Sortformer and `cluster` keep emitting
   non-overlapping turns.)
6. **Embeddings untouched.** `POST /v1/audio/embeddings` keeps its
   `segments`-JSON contract, **256-d WeSpeaker**, stable model id — this work
   does not change the embedding model or dimension (would invalidate enrolled
   voiceprints). The pyannote pipeline reuses that same WeSpeaker module
   internally.
7. **ASR+diar pairing = option (a), confirmed.** `/v1/audio/diarizations` is the
   single authoritative diarization source; the client calls `/transcriptions`
   (text) and `/diarizations` separately and merges by timestamp.
   `/transcriptions?diarize=true` stays the **legacy Sortformer convenience**
   (≤4, max-overlap tagging) and its ids are **not** reconciled with a
   standalone pyannote `/diarizations` call — do not rely on it for >4 or for
   id-consistency.
8. **Shared cap + cold-load.** `/v1/audio/diarizations` shares the 100 MiB audio
   cap and the same `governor.awaitLoad` cold-load 503/keep-alive behavior as
   the other audio endpoints, for every method.

## Architecture

Single `diarization` governed slot (ADR 011), backend chosen by model class
(ADR-016 pattern):

```
MLXDiarizationModule (actor, ModuleID.diarization)
 ├─ allowlist spans: [sortformer ids…, pyannote-segmentation id]
 ├─ DiarizationBackend detector (MLX-free): config.json model_type
 │     → .sortformer | .pyannoteSegmentation
 ├─ load → SortformerModel  (existing)  OR  PyanNetSegmentationModel (new)
 ├─ diarize(audio:)  → end-to-end turns        [sortformer-class resident]
 └─ segment(audio:)  → [SpeakerActivityRegion] [pyannote-class resident]
```

`method=pyannote` route orchestration (mirrors `method=cluster`):

```
1. ensure diarization resident is pyannote-class (auditedRebind if model= given;
   4xx if the requested model is a Sortformer id)
2. regions = diarization.segment(audio)          // powerset-decoded, overlap, stitched
3. embs    = speakerEmbedding.embed(regions)      // existing WeSpeaker
4. labels  = AgglomerativeClustering.cluster(embs, cannotLink: sameWindowPairs,
                                             numClusters/threshold/maxClusters)
5. turns   = overlapAwareTurns(regions, labels)   // new pure helper; allows overlap
```

`cannotLink` (same-window cannot-link) is a small additive parameter to
`AgglomerativeClustering` — two local speakers detected in one window are
distinct people and must never merge.

## Slices (each: annotated tag direct-to-main, `Athena.appVersion` bumped IN the
slice commit, ≥1 regression test pinned; pre-commit pipeline Tests → Security →
Quality → Refactor)

**S1 — Backend-class routing seam (MLX-free, no numerics).**
`DiarizationBackend` enum + detector from `config.json`; `MLXDiarizationModule`
routes `loadModel` by class; pyannote engine is a stub returning `notImplemented`
so the architecture + selection land first. Add `segment(...)` to the
`DiarizationModule` protocol (Sortformer impl throws `unsupportedForBackend`).
Unit-pinned: detection table, method/model-mismatch → correct 4xx. Wire
`method=pyannote` route returning a clean 501/4xx until S2.
*Test:* detector classification cases; route returns cause-naming error pre-port.

**S2 — Vendor PyanNet segmentation (MLX numerics).**
`Sources/AthenaTranscription/Pyannote/` — SincNet + BiLSTM + powerset head ported
from soniqo reference (MIT-attributed); load `aufklarer/Pyannote-Segmentation-MLX`
(safetensors) via `#hubDownloader`. `segment(audio:) -> [SpeakerActivityRegion]`:
10 s windows / 50% overlap → powerset decode → stitch overlapping windows into
local speaker regions. Governor reservation (~conservative; M5 reconciles).
*Test:* component-correlation / output sanity on real ICSI audio via Release
binary + curl (NOT `say` — TTS collapses to 1 speaker, M4.3/M25 lesson; heavy
tests gated `ATHENA_RUN_MODEL_TESTS=1`, run via `xcrun xctest`).

**S3 — `method=pyannote` end-to-end orchestration.**
Route composes segment → embed → cluster (same-window cannot-link) →
overlap-aware turn assembly. Add `cannotLink` to `AgglomerativeClustering`
(pure, unit-pinned). New pure `overlapAwareTurns` helper (unit-pinned).
*Test:* clustering with cannot-link constraint; overlap turn assembly; route e2e
recovering >4 speakers with an overlapping span on real audio.

**S4 — Selection plumbing, OpenAPI, CLI, docs, A/B.**
OpenAPISpec: add `method` enum + segmentation model to allowlist note (same edit
as route, drift-guard passes). `athena init` aux-pull + allowlist seed for the
segmentation model. `athena` CLI surfacing of `--method`. `docs/` how-to.
A/B `pyannote` vs `cluster` vs Sortformer on ICSI clip150/clip600; record DER /
speaker-count accuracy in the slice notes.
*Test:* OpenAPI drift-guard; e2e-rbac unaffected; CLI smoke.

## Test bar

- MLX-free decision logic (backend detection, powerset table, cannot-link,
  overlap assembly, method/model validation) → pure unit tests under
  `./deploy/test.sh` (ADR 008/009).
- Model numerics → Release binary + curl on **real** audio (`/tmp/audio` ICSI
  Bdb001); heavy tests gated on `ATHENA_RUN_MODEL_TESTS=1` and run via
  `xcrun xctest` so env is inherited (xcodebuild test sandboxes it).
- Route behavior → e2e; `./deploy/e2e-rbac.sh` must stay green.

## Risks

- **R1 npz vs safetensors — RESOLVED.** aufklarer mirror ships
  `model.safetensors` + `config.json`; no npz extractor needed (the M25 blocker
  does not recur).
- **R2 PyanNet port fidelity.** Powerset decoding + precomputed SincNet filters
  must match the reference. Mitigate: component-correlation check vs published
  numbers (mlx-community cites >99.99% component / 88.6% output corr vs PyTorch)
  + ICSI validation. soniqo provides a working forward to diff against.
- **R3 clustering quality on overlap.** Same-window cannot-link + threshold
  tuning; reuse M25's empirically-chosen 0.75 as the start, sweep in S4.
- **R4 single-slot rebind latency** when alternating Sortformer/pyannote.
  Both models tiny → cheap. Tripwire (ADR 018): if it bites, split out a
  `diarizationSegmentation` ModuleID so both stay resident.
- **R5 governor accounting** for a tiny model + transient activations on long
  audio — conservative estimate, M5 reconciles to real footprint.

## Out of scope

- Cross-recording speaker *identity* (ADR 014 — client-side; unchanged).
- Streaming/online pyannote (offline windowed pass only this milestone).
- Removing or changing the Sortformer default or the `cluster` path.
