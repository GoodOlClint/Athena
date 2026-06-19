# Video support — workload plan (M78)

**Status:** Proposed, awaiting operator review (change gate). Pairs with ADR 022.
No production code until approved. Operator framing: **transcription first**, a
**dedicated `/v1/video/*` surface**, **keyframe-level** description is fine, and
there is a **concrete consumer** (plan to-build-next with an integration
contract).

## Goal

Accept video locally and (1) **transcribe** it via the existing
Whisper/Parakeet tenant and (2) **describe** it via the existing Gemma4 VLM
tenant — orchestration over modalities Athena already serves, under the one
Metal governor (ADR 011), with no substrate change.

## Existing state (reuse, don't reinvent)

| Piece | Status | Role for video |
|---|---|---|
| `AudioDecode` (AVFoundation) | shipped | Pattern for the demux; **opens audio only** (`AVAudioFile`) — video needs an `AVAssetReader` track extractor. |
| Transcription tenant (Whisper/Parakeet, ADR 020) | shipped | Consumes 16 kHz mono PCM; engine routed by `ModelSupport` (ADR 021). Reused **unchanged**. |
| Vision tenant (Gemma4 VLM, ADR 010/012) | shipped | Describes a single image; reused per sampled frame. |
| LLM tenant | shipped | Optional roll-up summary of frame captions. |
| Decode floor + max-samples (v0.10.182 / D4) | shipped | The audio-track PCM funnels through these → degenerate video = 4xx, never a crash. |
| Modality upload caps (ADR 017) | shipped | Pattern for `max_video_upload_bytes`. |
| `/openapi.json` drift-guard, RBAC route→permission map | shipped | New routes/permissions plug in here. |
| AVFoundation | linked | `AVAssetReader` (audio demux), `AVAssetImageGenerator` (frames). No new dependency. |

## Architecture

```
POST /v1/video/transcriptions [native]      POST /v1/video/descriptions [native]
        │                                            │
   AVAssetReader: extract audio track          AVAssetImageGenerator: sample frames
        │  → 16 kHz mono PCM                         │  (interval; max-frame cap)
   (shared decode floor + max-samples)               │
        │                                       per-frame → VLM caption (vision tenant)
   transcription tenant (Whisper/Parakeet)            │
        │                                       optional → LLM summary
   OpenAI-shaped transcript                     {frames:[{start,caption}], summary?}
   (json/text/srt/vtt/verbose_json, diarize)
```

One upload → existing governed tenant(s). Orchestration only; never a second
allocator on the Metal pool (ADR 011).

## Milestone M78.1 — video transcription

**S1 — audio-track extractor.** New `VideoAudioTrack.extractPCM(from:)`
(AthenaTranscription): `AVAsset` → first audio track → `AVAssetReader` /
`AVAssetReaderTrackOutput` → 16 kHz mono Float32 PCM, reusing the **same decode
floor + max-samples ceiling** as `AudioDecode`. No audio track ⇒ a cause-naming
400 (`video_no_audio_track`). *Test:* MLX-free decision logic (track-present /
absent, floor/ceiling reuse) unit-pinned; AVFoundation extraction validated on a
**real short video fixture**, gated (`ATHENA_RUN_MODEL_TESTS=1`). A degenerate
video (no audio / sub-0.1 s / bomb) → the matching 4xx, never an abort.

**S2 — transcription PCM seam.** Add a PCM-input entry to the transcription
tenant so both the audio route (post-`AudioDecode`) and the video route
(post-`VideoAudioTrack`) feed the engine without re-encoding. Engine selection,
chunking, timestamps, `diarize=true` all reused. *Test:* the seam dispatches
Whisper/Parakeet identically to the audio path (gated, real fixture).

**S3 — route + DTO + cap + RBAC + OpenAPI.** `POST /v1/video/transcriptions`:
multipart upload, new `max_video_upload_bytes` cap (D2), new `video.transcribe`
RBAC permission, response **identical to `/v1/audio/transcriptions`**.
`OpenAPISpec.swift` updated **in the same edit** (drift-guard) and the operation
`description` **labels the route Athena-native** per the CLAUDE.md `/v1`
compatibility rule. Passive-oracle: multipart only, 400 on any URL field.
*Test:* drift-guard green; RBAC e2e for the new permission; 413 over cap;
413/400 envelopes.

**S4 — long video + docs + e2e.** Validate the existing chunking on a long
extracted track; bound peak memory (full-PCM extraction vs. the max-samples
ceiling). `docs/video.md` (what it does, the native-surface note, the honesty
boundary). `deploy/e2e-rbac.sh` stays green. *Test:* a multi-minute real video →
coherent transcript + SRT.

## Milestone M78.2 — keyframe description

**S5 — frame sampler.** `VideoFrameSampler` via `AVAssetImageGenerator`: fixed
interval (default D4) with a **max-frame cap** (governor bound). MLX-free
schedule logic unit-pinned; extraction gated. *Test:* schedule (interval, cap,
short-clip) pinned; real-video frame extraction gated.

**S6 — VLM captioning orchestration.** Each sampled frame → the vision tenant →
caption, governed (admission per VLM call; not double-counted against the audio
slot). The frame-count cap bounds total Metal work per request. *Test:* gated —
a real video → per-frame captions; cap enforced.

**S7 — optional summary.** Roll up captions via the LLM when requested. *Test:*
gated — summary present iff requested.

**S8 — route + DTO + OpenAPI + RBAC + docs.** `POST /v1/video/descriptions`:
params (frame interval, max frames, summarize, prompt), response
`{frames:[{start, caption}], summary?}`, `video.describe` permission, **[native]
in OpenAPI** + drift-guard. `docs/video.md` updated with the keyframe honesty
boundary. *Test:* drift-guard + RBAC e2e.

## Test bar

- MLX-free decision logic (track selection, frame schedule, caps, ModelSupport
  routing) → unit tests under `./deploy/test.sh` (ADR 008/009).
- AVFoundation extraction + tenant forwards → gated (`ATHENA_RUN_MODEL_TESTS=1`)
  on **real video fixtures** (never synthetic — the audio "real fixtures only"
  lesson applies; a real short clip with a known audio track + visible content).
- **Crash-hardening regression:** a corpus of degenerate videos (no audio track,
  sub-floor audio, zero-frame, corrupt container) must yield clean 4xx on every
  video route and **abort no pipeline** — the video analogue of the M77.x
  `QuarantineAudioSweepTests`.
- Drift-guard green for both new routes; RBAC e2e for both new permissions.

## Integration contract (concrete consumer)

To confirm with the consumer **before S3** (drives D1/D3):

- **Transcription I/O:** multipart `file=` (video) → the OpenAI
  `/v1/audio/transcriptions` response shape (so an existing transcription
  consumer reuses its parser). Confirm needed `response_format`(s) and whether
  `diarize=true` is required.
- **Sync vs async:** is synchronous acceptable for the consumer's typical video
  length, or is an async `/v1/queue` job (submit → poll) required for large
  files? (D1.)
- **Description shape & granularity:** `{frames:[{start, caption}], summary?}` —
  confirm frame interval expectations and whether the summary is required (D3/D4).
- **Sizes:** typical/max video size → sets `max_video_upload_bytes` (D2).

(Contract kept consumer-agnostic in code/docs; Athena stays a passive oracle.)

## Risks

- **R1 hostile/corrupt media.** AVFoundation parses untrusted containers. Bound
  by `max_video_upload_bytes`, the max-samples ceiling, and the max-frame cap;
  note the AVFoundation attack surface in `docs/video.md`.
- **R2 long-video memory.** Full-PCM extraction of a multi-hour track is large
  but bounded by the max-samples ceiling; streaming extraction is a follow-up.
- **R3 description latency/cost.** N frames = N VLM forwards; the frame cap
  bounds it, but large videos may want the async-queue wrap (D1).
- **R4 keyframe ≠ temporal.** Honesty boundary, documented in the response and
  `docs/video.md`; no motion/event reasoning (ADR 022 decision 3).
- **R5 governor accounting for a composite request.** One request spans the
  audio slot then the vision slot (sequentially); ensure each is admitted as its
  own tenant, not double-counted — the thesis-relevant invariant (ADR 011).
- **R6 surface-compatibility drift.** Every new `/v1` route must be tagged
  `[native]`/`[oai]` (CLAUDE.md rule); a follow-up could add a drift-guard check
  that every `/v1` operation carries an explicit compatibility marker.

## Open decisions for review

- **D1 — sync vs async transcription.** *Rec:* synchronous for M78.1 (reuses the
  existing chunking); async `/v1/queue` wrap as a follow-up **iff** the consumer
  needs it. Confirm against the consumer's video length.
- **D2 — `max_video_upload_bytes` default.** *Rec:* **1 GiB** (video dwarfs
  audio's 100 MiB), modality-scoped per ADR 017; tune to the consumer's sizes.
- **D3 — description response shape.** *Rec:* `{frames:[{start, caption}],
  summary?}`, native JSON; mark `[native]`.
- **D4 — frame-sampling default.** *Rec:* fixed interval (start at **1 frame /
  2 s**) + a max-frame cap (e.g. 64) per request; scene-detection a follow-up.
- **D5 — `diarize=true` on `/v1/video/transcriptions`.** *Rec:* yes — the
  extracted audio track feeds the diarization slot exactly like audio. Confirm.
- **D6 — native-marker enforcement.** *Rec:* CLAUDE.md prose rule now (done);
  a mechanical drift-guard (every `/v1` op must declare `[native]`/`[oai]`) as a
  small standalone follow-up, not blocking M78.

## Out of scope

- Native temporal video models (video-LLM) — substrate-gated (ADR 022).
- Server-side scene/shot segmentation — follow-up.
- Streaming extraction / live video — follow-up.
