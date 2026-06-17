# 010 — Vision input on chat: wire the substrate VLM path, base64-only images, defer audio-in-chat

**Status:** Accepted — implementing. M71.1 (wire protocol) shipped v0.10.159; M71.2 (VLM load/generate) shipped v0.10.160; M71.3 (capability surface in `athena show` + real-model e2e) pending. Load-path fork resolved: **single VLM container** for vision checkpoints (text + image both via `MLXVLM.Gemma4`), DFlash-on-gemma4 deferred (re-wire onto the VLM's inner text backbone is a follow-on).
**Date:** 2026-06-17
**Milestone:** M71 — resolves issue #3 (vision/image input)

## Context

Issue #3 asks for multimodal **image input** on `/v1/chat/completions` (and native
`/api/chat`) so Athena can serve vision-language models (Gemma 4 unified, etc.) for
image understanding — closing the one capability gap that currently forces LM Studio to
run alongside Athena on the box. Athena already covers the other two legs of the
operator's text+audio+vision "evidence vault" use case (transcription / diarization /
speaker embeddings); vision is the missing leg.

Current state (verified against live code 2026-06-17, not the 5-day-old memory):

- **Wire protocol is text-only.** `AthenaLLM/ChatTurn.swift` is `role: String` +
  `content: String`. `Server/OpenAIDTO.swift:19` `ChatMessage.content` is a plain
  `String?`. `Server/OpenAPISpec.swift` `ChatCompletionRequest` carries no image parts.
  Only the `/v1/audio/*` endpoints accept media, via `multipart/form-data`.
- **The Gemma-4 backbone already ships.** `gemma4_unified` text inference shipped
  v0.10.107 (ADR 002 + the M64 MoE program, v0.10.113–116). Today the text load path
  (`MLXLLMModule` → `LLMModelFactory.shared`, the substrate's `MLXLLM/Gemma4.swift`)
  loads the unified checkpoint and its `sanitize()` **strips the `vision_embedder`
  tower** — i.e. the image-encoder weights are in the files we already download, and we
  discard them.
- **The Swift vision tower already exists in the substrate.** The path-dep fork
  (`~/Source/mlx-swift-lm`) ships a complete `MLXVLM` library with `Models/Gemma4.swift`:
  `Gemma4VisionModel`, patch embedder, vision attention, `Gemma4MultimodalEmbedder` (the
  `vision_embedder` projection), the `prepare()` image-embedding splice, and
  `Gemma4Processor` (`UserInputProcessor`, takes `[CIImage]`). `VLMModelFactory` exists
  alongside `LLMModelFactory`. So the heavy ML port is **already done in Swift** — it is
  simply not linked into Athena (`Package.swift` pulls only `MLXLLM`).
- **No audio-in-chat tower exists.** The same `MLXVLM/Gemma4.swift` `sanitize()` (≈ line
  1752) **strips `audio_tower` and `embed_audio`**; only audio-token *positions*
  (`audioMask`) are handled, with nothing to fill them. Audio-in-chat would be a net-new
  Swift tower port (research-grade, not the wiring vision was). It does **not** overlap the
  dedicated `/v1/audio/*` endpoints: those serve audio *analysis* (transcribe/diarize/
  speaker-embed via Whisper/Sortformer/WeSpeaker), whereas audio-in-chat would serve audio
  *reasoning* (the model reasons over a clip) — complementary jobs, exactly as OpenAI ships
  both Whisper and `gpt-4o-audio`. The dedicated endpoints stay canonical for analysis
  regardless (dedicated models beat an omni tower there).
- **Governor accounting already counts the tower.** `MLXLLMModule.estimateBytes` sums
  every `*.safetensors` in the model dir, so the on-disk admission estimate already
  includes the vision-tower weights even though the text path discards them at load —
  no governor under-count is introduced by actually using them.

## Decision

Add image input to the chat path by **wiring Athena to the substrate's existing
`MLXVLM` vision path**, not by porting a new encoder. Four sub-decisions:

1. **Reuse the substrate VLM path.** Link the `MLXVLM` product; route a chat request
   that carries image parts through `VLMModelFactory` + `Gemma4Processor` +
   `MLXVLM.Gemma4`, rather than the text-only `MLXLLM.Gemma4`. Text-only requests stay
   on the existing `MLXLLM` path **byte-unchanged** (the same discipline as the M64
   dense/MoE gate). This keeps the spec-decode / structured-output / KV-codec investment
   on the text path untouched.

2. **OpenAI-compatible content parts.** Extend `ChatMessage.content` to decode **either**
   a `String` (unchanged) **or** an array of parts
   (`[{type:"text",text} | {type:"image_url",image_url:{url}}]`) via a custom `Codable`,
   so existing OpenAI SDKs work unchanged and text-only callers are unaffected.
   `ChatTurn` grows an optional image payload; the substrate chat-template /
   message-generator path consumes it. The `/api/*` native dialect mirrors the minimal
   shape in `NativeAPIDTO.swift`.

3. **base64 / `data:` images only — passive-oracle preserved with zero carve-out.**
   Accept inline base64 and `data:` URLs; **reject `http(s)://` image URLs with a clear
   400.** The daemon performs **no** outbound fetch for images. This keeps the
   passive-oracle rule (CLAUDE.md: "Outbound network is forbidden except model-weight
   fetches from Hugging Face") intact without amendment — the client fetches remote
   images itself. (A future outbound-image carve-out, if a consumer ever needs it, would
   require its own ADR amending the passive-oracle rule; it is explicitly out of scope.)

4. **Defer audio-in-chat.** Audio-in-chat is **not** in this milestone — deferred on
   *feasibility*, not overlap: it needs a net-new Swift audio tower (the substrate strips
   it; research-grade port). It is **complementary to**, not a duplicate of, the dedicated
   `/v1/audio/*` endpoints — those own audio *analysis* (transcribe/diarize/speaker-embed),
   audio-in-chat would own audio *reasoning*. Captured as a separate future milestone with
   its own gate (plan: `docs/audio-input-chat-plan.md`; tracked as GitHub issue #5, gated on
   upstream `mlx-swift-lm#207`). Audio analysis remains available via the dedicated
   transcription / diarization endpoints in the meantime.

## Consequences

- The LM-Studio-forcing gap (image description) closes; Athena becomes the single
  passive oracle for text + audio + vision on the box.
- `Package.swift` gains the `MLXVLM` product dependency on the daemon target only — it
  must **not** leak into the cross-platform `clients/` package (`MLXVLM` is
  Apple/Metal-bound, like `MLXLLM`).
- A vision request loads a *different module instance* than the text path for the same
  checkpoint (VLM vs LLM factory). The governor must not double-count, and model-swap
  discipline must treat "gemma4 text" and "gemma4 vision" as the same resident weights
  where possible — a load-path design point for slice M-vis.2.
- `ChatTurn` is no longer a pure `role`+`content` string pair; conformers without a
  vision path (the stub, non-VLM arches) must degrade gracefully (ignore / 400 on image
  parts for a non-vision model).
- Structured output, spec-decode (MTP/DFlash), and the KV codecs are **text-path only**
  and stay byte-unchanged; whether constrained decoding composes with the VLM generate
  path is an open question deferred with the audio leg (not required by issue #3's
  image-description use case).
- The substrate VLM delta is consumed via the existing path-dep (no pin); reproducibility
  pins to the substrate commit recorded in the slice that links `MLXVLM`.

## Alternatives rejected

- **Port a vision encoder into `MLXLLM.Gemma4` from scratch** — rejected: the substrate
  already has a complete, parity-tracked `MLXVLM.Gemma4` vision tower. Re-porting is a
  parallel implementation of an existing pipeline (a defect, not a feature).
- **Allow the daemon to fetch `http(s)` image URLs** — rejected for this milestone:
  breaks the passive-oracle rule and needs an ADR amendment; the use case is satisfied by
  inline base64.
- **Full omni (image + audio-in-chat) in one milestone** — rejected: the audio leg is
  research-grade (no substrate tower) and overlaps existing endpoints; bundling it would
  sink the milestone. Operator confirmed deferral 2026-06-17.
