# M72 — Audio input (audio-in-chat) — change plan

**Status:** DEFERRED (2026-06-17) — tracked as GitHub issue #5, gated on upstream
`ml-explore/mlx-swift-lm#207` (substrate audio tower). Not scheduled; no current consumer
for audio *reasoning*. This doc is the design-on-file for when it's revisited.
**Companion ADR:** a companion ADR (**to be written**, next free number) — mirrors ADR 010 for the audio modality; also refines ADR 010's deferral rationale).
**Milestone number is proposed** — operator picks the actual number and tags (versioning is operator-driven). Numbered M72 here only because vision was M71.

## ⚠️ Magnitude — read first

Unlike vision (M71), **this is a substrate port, not a wiring job.** Verified state:

- `~/Source/mlx/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift:1750-1753` — `sanitize()`
  **strips** `audio_tower` and `embed_audio` from the weights at load time. Comment:
  *"This port currently supports text + vision only."*
- `Gemma4.swift:393` — config carries `audioTokenId`, and `getInputEmbeddings`
  (~1683-1689) detects an `audioMask`, **but there is nothing to fill those positions** —
  no `audioTower(...)` call, no audio encoder, no `embed_audio` projection. Vision has a
  full stack (`Gemma4VisionModel`/`Gemma4MultimodalEmbedder`, ~1184-1640); audio has none.

So M72's dominant cost is **porting the Gemma4 audio encoder into the MLX-swift fork and
proving numerical parity against the Python `mlx-vlm` reference** — research-grade, on the
order of the MTP/TurboQuant ports, not the ~2-slice reuse M71 was. The daemon wiring (the
content-parts plumbing) is the *easy, known* part — it mirrors M71's vision wiring almost
line for line.

**Priority conflict (must be resolved before slice 2):** ADR 011 names *governor
accounting truthfulness* as the next milestone, above feature work; and there is no current
consumer for audio **reasoning** (transcription/diarization are served better by the
dedicated `/v1/audio/*` endpoints). This plan therefore stops after a cheap **slice-1
parity spike** and re-gates before the full port.

## Why (one line)

Audio-in-chat lets the chat model *reason over* an audio clip ("summarize this voice memo",
"what's the tone?") — a distinct job from the `/v1/audio/*` *analysis* endpoints
(transcribe / diarize / speaker-embed). It is the symmetric twin of M71 vision.

## Division of labor (NOT a conflict — settled in the API review)

| "Send audio to Athena" intent | Surface | Models |
|---|---|---|
| "Give me the words / who spoke when / a voice vector" (**analysis**) | `/v1/audio/transcriptions`, `/diarizations`, `/embeddings` | Whisper / Sortformer / WeSpeaker (dedicated, best-in-class — stay canonical forever) |
| "Reason about this audio in conversation" (**reasoning**) | `/v1/chat/completions` `input_audio` content-part | Gemma4 audio tower (this milestone) |

These are complementary, exactly as OpenAI ships both Whisper *and* `gpt-4o-audio`. The
"overlap with `/v1/audio/*`" worry is a category confusion: dedicated endpoints own
analysis; chat owns reasoning. **You would never route transcription through the omni
tower** — dedicated Whisper beats it. ADR 010's "overlaps `/v1/audio/*`" deferral clause is
inaccurate and ADR 012 should restate the real reason (tower stripped = infeasible until
ported; dedicated models win analysis = the endpoints stay canonical).

## Standard shape (nothing to invent)

OpenAI's audio-in-chat = `input_audio` content parts, the twin of vision's `image_url`:

```json
"content": [{ "type": "input_audio",
              "input_audio": { "data": "<base64>", "format": "wav" } }]
```

base64/`data:` only; **400 on `http(s)`** (passive-oracle, no outbound fetch) — identical
posture to M71 images.

## Scope

**In:** audio input on `/v1/chat/completions` via OpenAI `input_audio` content-parts;
base64/`data:` decode + codec validation; the substrate audio-tower port for the
`gemma4_audio` family; governor accounting of the audio tower; e2e + OpenAPI drift-guard.

**Out (this milestone):** routing transcription/diarization through chat (dedicated
endpoints stay canonical); audio **output**/TTS (`modalities:["audio"]` — separate, no
substrate support, no TTS model); outbound `http(s)` audio fetch (400); non-Gemma audio
arches; native `/api/chat` (stays text-only, like vision — audio goes through `/v1`).

## Slices (each: xcodebuild Release → e2e phase → annotated tag direct-to-main; appVersion bump IN the slice commit)

- **M72.1 — substrate audio-tower PARITY SPIKE (de-risk; STOP-and-review after).** No
  daemon code. In the `~/Source/mlx/mlx-swift-lm` fork: (a) identify the Python `mlx-vlm`
  Gemma audio-encoder reference and map its architecture (Conformer/USM blocks, subsampling,
  `embed_audio` projection); (b) confirm a real `gemma4_audio` checkpoint actually ships
  audio-tower weights (the daemon's checkpoint, `gemma-4-e2b-it-4bit`, validated for vision
  in M71.2 — verify its audio weights exist and their key layout); (c) confirm whether
  mlx-swift `UserInput`/the `Gemma4Processor` can even *carry* audio (likely **not** — the
  processor and `UserInput.Image` are vision-shaped; an audio carrier is part of the port);
  (d) prototype the encoder forward in Swift and **numerically parity-check** one short clip
  against Python (cosine/maxabs on the audio embeddings, in-distribution clip).
  **Deliverable:** a go/no-go parity number + a written port shape (modules, weight-key map,
  `UserInput` audio-carrier design, `getInputEmbeddings` splice point, `sanitize()` change)
  + a revised effort estimate. **This is the only slice authorized by this plan; M72.2+
  require re-ratification against the ADR-011 priority.**

- **M72.2 — substrate audio-tower port (gated on M72.1 go).** Port the encoder +
  `embed_audio` into `Gemma4.swift`; stop stripping `audio_tower`/`embed_audio` in
  `sanitize()`; fill `audioMask` positions in `getInputEmbeddings`; extend `UserInput`/
  processor to carry audio. Tracked substrate delta (path-dep fork; record the commit).
  Bit-/parity-gate vs Python reference. No daemon surface yet.

- **M72.3 — wire protocol (mirrors M71.1).** `ContentPart` gains `type:"input_audio"`;
  `ChatMessage` gains `audioURLs`; new `ChatAudio` (decode `data:`, 400 `http(s)`, codec
  validate) + `ChatAudioError`; `ChatTurn` gains `audios: [ChatAudio]`; `chatTurns(from:)`
  decodes them; audio with no audio-capable model → 400 `audio_not_supported`. `OpenAPISpec`
  `ChatContentPart` gains the `input_audio` variant + drift-guard stays green.

- **M72.4 — load/generate path (mirrors M71.2).** `MLXLLMModule` gains `residentIsAudio`/
  `servesAudio`; the VLM container detects an audio tower; `ChatTurn.audios` →
  `UserInput` audio → processor. Serve path (sync + queued) allows `input_audio` only when
  `servesAudio`. Governor counts the one resident copy (`estimateBytes` already sums all
  safetensors — verify it picks up the now-unstripped audio tower).

- **M72.5 — capability surface + e2e.** `athena show` reports an audio capability; real-model
  RUNBOOK scenario (curl an audio clip → coherent answer) + a stub-tier e2e-rbac.sh phase;
  OpenAPI example; quickstart "audio input" section.

## Test bar

- **Per-slice CI gate:** `./deploy/build.sh Release` (xcodebuild — MLX needs full Xcode) +
  a new `deploy/e2e-rbac.sh` phase (stub engine, loopback, curl) from M72.3 on.
- **Substrate parity gate (M72.1/.2):** Swift audio embeddings parity-checked against the
  Python `mlx-vlm` reference on an in-distribution clip (the M64.2 "use an in-distribution
  prompt" lesson applies to audio too).
- **Drift-guard:** spec↔routes test green across the `ChatContentPart` schema change.
- **Text/vision regression:** text-only and image-only chat must be byte-/behaviour-unchanged
  (assert the audio link didn't perturb the existing MLXLLM/MLXVLM paths).
- **Real-model validation:** audio-in → answer on a `gemma4_audio` checkpoint via the Release
  binary + curl, recorded in `deploy/integration/RUNBOOK.md` (manual pre-release tier — the
  stub can't exercise the audio tower, same as vision scenario J).
- **Cross-platform client:** no audio/MLXVLM code enters `clients/`; re-prove the standalone
  client builds on Linux if any `AthenaClient` shape changes.

## Rollout / risk

- **Passive-oracle:** preserved with zero carve-out — audio rides inbound in the request
  body, like images and like the transcription endpoints; no outbound fetch.
- **Backward-compat:** text-only and image chat are wire-unchanged; the audio path engages
  only when `input_audio` parts are present against an audio-capable model.
- **Substrate:** a **tracked delta** to the `~/Source/mlx/mlx-swift-lm` fork (precedent:
  TurboQuant/MTP ports). Record the commit; reproducibility pins to it.
- **Biggest risk:** the audio-encoder port is the unknown. M72.1 exists to convert that
  unknown into a number before any further commitment.

## Approval gate

This plan + (forthcoming) ADR 012 are the approval artifacts. **Only M72.1 (the parity
spike) is authorized by ratifying this plan** — it is deliberately cheap and ends in a
go/no-go. M72.2 onward require re-ratification *against ADR 011's stated next-milestone
priority (governor accounting)*: decide explicitly whether the audio port jumps the queue,
or waits. No implementation code past the spike until then.
