# ADR 022 — video support: transcription + keyframe description

**Status:** Proposed (M78) — awaiting operator review. No production code yet.
Pairs with `docs/video-support-plan.md`. Operator-set framing (interview):
**transcription first**, a **dedicated `/v1/video/*` surface**, **keyframe-level**
description is acceptable. Driver clarified after the interview: **video exists
and needs ingesting, but the downstream consumer's requirements are not yet
known.** This sharpens the phasing rather than changing it — M78.1
(transcription) is **requirement-independent** (every consumer wants the words
out; the output is the established OpenAI transcript shape), so it is the
committed first slice; M78.2 (description) is **requirement-gated** — its shape
(frame rate, summary, response contract) depends on needs that don't exist yet,
so it stays designed-but-deferred until a consumer defines them.

## Context

The operator wants Athena to accept video and (eventually) both **transcribe**
it and **describe** it — locally, under the one Metal governor. Today neither
exists: `AudioDecode` opens files via `AVAudioFile(forReading:)`, which cannot
read a video container, so an `.mp4`/`.mov` upload 400s; and the vision tenant
(Gemma4 VLM, ADR 010/012) describes single **images**, not video.

The key realization that shapes everything: **video is not a new model — it is
orchestration over the modalities Athena already serves.** A video file is a
container carrying an audio track and a frame sequence:

- **Transcription** = demux the audio track → the existing Whisper/Parakeet
  transcription tenant (ADR 020), unchanged. This is an audio-extraction
  problem, not a video problem.
- **Description** = sample frames → the existing Gemma4 VLM vision tenant
  (ADR 010/012) → per-frame captions, optionally summarized by the LLM tenant.

This is the unified-governor thesis (ADR 011) demonstrated across modalities: a
single upload fans out into **two existing tenants** multiplexed on one Metal
budget. Critically it composes at the **orchestration** layer (extract → call a
governed tenant), never at the inference layer — the line ADR 011 draws.

## Decision

1. **A dedicated `/v1/video/*` surface, not an overload of `/v1/audio/*` or a
   chat content-part.** Two routes, landing in two milestones:
   - **`POST /v1/video/transcriptions`** (M78.1) — multipart video upload →
     audio-track demux → the transcription tenant → the **same response shapes
     as `/v1/audio/transcriptions`** (`json`/`text`/`srt`/`vtt`/`verbose_json`,
     word timestamps, `diarize=true`). Engine chosen by the resident
     transcription model (Whisper/Parakeet), exactly as the audio route.
   - **`POST /v1/video/descriptions`** (M78.2) — multipart video upload → frame
     sampler → per-frame VLM captions (+ optional LLM summary). Keyframe-level.

   A dedicated surface (vs. accepting video on the audio route) is chosen so the
   contract is self-describing in `/openapi.json`, carries its own RBAC
   permissions (`video.transcribe`, `video.describe`) and its own upload cap,
   and has room for video-specific parameters (frame interval, max frames)
   without polluting the audio DTOs. The audio routes keep rejecting video, so
   the modality boundary stays crisp.

   **Both `/v1/video/*` routes are Athena-native, NOT OpenAI-compatible**
   (OpenAI has no video API) — marked per the CLAUDE.md `/v1` compatibility
   rule: tagged **[native]** in the endpoint list, noted here, and labelled
   Athena-native in each `OpenAPISpec.swift` operation `description`. Note that
   `/v1/video/transcriptions` reusing the OpenAI **response shape** of
   `/v1/audio/transcriptions` (verbose_json/SRT/VTT) is consumer convenience
   only — the route itself is a native extension, so it is `[native]`.

2. **Video is orchestration over existing tenants — no new inference model for
   M78.** Transcription reuses the governed `transcription` slot byte-for-byte;
   description reuses the governed VLM (`servesVision`) and, for the summary, the
   LLM. The only net-new code is **front-ends** (AVFoundation demux + frame
   sampling) and **routing/plumbing** (routes, DTOs, caps, RBAC, OpenAPI). No
   substrate change.

3. **Description is keyframe-level, and we say so.** M78.2 samples frames on a
   schedule (fixed interval default; scene-detection a follow-up), captions each
   through the VLM, and optionally rolls them up via the LLM. The honest
   boundary, stated in the docs and the response: this reports **what is on
   screen at sampled moments** — not motion, not events over time. True
   audio-visual temporal reasoning needs a **video-native model the substrate
   does not expose** and is **out of scope / substrate-gated** (the same posture
   as audio-in-chat, ADR/issue #5). Reject `model_type`s we can't serve via the
   ModelSupport predicate (ADR 021), don't over-promise.

4. **Bounded, passive, governed.** A new modality-scoped cap
   `max_video_upload_bytes` (ADR 017 pattern; video is large — default proposed
   in the plan) bounds the upload; a **max-frame cap** bounds the VLM work a
   single description request can schedule (governor protection). Passive-oracle
   preserved: video is multipart/base64 only — **no URL fetch** (mirrors the
   image rule, ADR 010: 400 on `http(s)`), no outbound. The audio-track PCM
   funnels through the **same decode floor + max-samples ceiling** as audio
   (v0.10.182 / D4), so a degenerate video (no audio track, sub-0.1 s audio,
   decompression bomb) becomes a **cause-naming 4xx**, never a daemon abort —
   carrying forward the M77.x audio crash-hardening lesson.

5. **Honesty + reuse over novelty.** Transcription output is byte-identical to
   the audio route's (a consumer that already parses `/v1/audio/transcriptions`
   gets the same shape). Description is a new, explicitly frame-based contract.
   Both are governed tenants under the one budget — never a second uncoordinated
   allocator (ADR 011).

### Rejected / deferred

- **Accept video on `/v1/audio/transcriptions`** (overload) — rejected; the
  operator chose a dedicated surface (self-describing, separately capped,
  separately permissioned).
- **Single `/v1/video/analyze` returning transcript + descriptions together** —
  deferred; couples two tenants into one response contract and one latency
  envelope. Two focused routes compose better; a consumer that wants both calls
  both.
- **Native temporal video model** (video-LLM) — out of scope; substrate-gated
  research, no governed path today. Revisit when the substrate exposes one.
- **Server-side scene-detection / shot segmentation** for frame sampling —
  deferred to a follow-up; M78.2 ships fixed-interval sampling first.
- **Streaming/async transcription of multi-hour video** — M78.1 is synchronous
  with the existing chunking; an async-queue wrap (reusing `/v1/queue`) is a
  follow-up if the consumer needs it (plan open decision D1).

## Consequences

- No new external dependency: AVFoundation (`AVAsset`/`AVAssetReader`/
  `AVAssetImageGenerator`) is already linked (used by `AudioDecode`). macOS-only,
  consistent with the daemon.
- The thesis gains its clearest cross-modal demonstrator: one request, two
  governed tenants, one Metal budget — orchestrated, never composed at inference.
- All new decision logic (track selection, frame schedule, caps, ModelSupport
  routing) stays MLX-free and unit-pinned (ADR 008/009); AVFoundation extraction
  and the VLM/transcription forwards are validated by gated real-fixture tests.
- The crash-hardening invariant extends to video for free: the audio half goes
  through the shared decode floor; the description half is bounded by the
  max-frame cap.

Plan, slices, test bar, integration contract, and open decisions:
`docs/video-support-plan.md`.
