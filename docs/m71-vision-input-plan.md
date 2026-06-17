# M71 — Vision input (VLM chat) — change plan

**Status:** Proposed change plan — approval gate (do not implement until ratified).
**Resolves:** issue #3 (vision/image input). **ADR:** [`docs/decisions/010-vision-input-vlm-chat.md`](decisions/010-vision-input-vlm-chat.md).
**Milestone number is proposed** — the operator picks the actual number and tags (versioning is operator-driven).

## Why (one line)

Vision is the only functional capability that forces LM Studio to stay on the box;
closing it makes Athena the single passive oracle for text + audio + vision.

## LM-Studio gap analysis (the roadmap framing)

| Capability | Athena | LM Studio | Verdict |
|---|---|---|---|
| **Vision / image input** | ❌ (this milestone) | ✅ | **the gap — close it** |
| GUI / one-click model UX | control-console `/ui` | ✅ | not Athena's thesis — skip |
| GGUF / llama.cpp breadth | ❌ (MLX, curated arches) | ✅ | not Athena's thesis — skip |
| Audio transcription / diarization / speaker-embed | ✅ | ❌ | Athena-only |
| Built-in vector DB / async queue | ✅ | ❌ | Athena-only |
| One Metal governor across all modalities | ✅ | ❌ | Athena-only |
| RBAC / audit / rate-limit / metering / TLS / at-rest | ✅ | ❌ | Athena-only |
| Self-describing OpenAPI, dual `/v1`+`/api` | ✅ | ❌ | Athena-only |

Everything in the "Athena-only" rows is why Athena exists and LM Studio can't replace it.
The only two LM-Studio advantages (GUI polish, GGUF breadth) are explicitly out of thesis.

## Scope

**In:** image input (single + multi-image) on `/v1/chat/completions` and native
`/api/chat`, OpenAI `image_url` content-parts, base64/`data:` decode, the `gemma4_unified`
vision target via the substrate's `MLXVLM` path, governor accounting, e2e + OpenAPI
drift-guard.

**Out (this milestone):** audio-in-chat (deferred — separate gate; needs a net-new Swift
audio tower, overlaps `/v1/audio/*`); outbound `http(s)` image fetch (passive-oracle —
400 instead); video input; non-Gemma VLM arches (Qwen-VL etc. — generalize later);
constrained/structured decoding over the VLM path (open question, not needed by issue #3).

## Verified current-state seams (the map)

- `Sources/AthenaLLM/ChatTurn.swift` — `role`+`content: String`, text-only.
- `Sources/athena/Server/OpenAIDTO.swift:19` — `ChatMessage.content: String?`.
- `Sources/athena/Server/AthenaServer.swift:886, 2890, 3164` — the three `ChatTurn(...)`
  construction sites the new content-parts path must flow through.
- `Sources/athena/Server/OpenAPISpec.swift` — `ChatCompletionRequest` schema (+ the
  spec↔routes drift-guard that any schema change must satisfy).
- `Package.swift:131,150` — daemon targets pull `MLXLLM` only; add `MLXVLM` here, **not**
  to `clients/`.
- Substrate `~/Source/mlx-swift-lm/Libraries/MLXVLM/Models/Gemma4.swift` —
  `Gemma4VisionModel`, `Gemma4MultimodalEmbedder`, `Gemma4Processor`, `VLMModelFactory`
  (already implemented; just unlinked). `sanitize()` strips `audio_tower`/`embed_audio`.
- `Sources/AthenaLLM/MLXLLMModule.swift:234,309,312,1102` — governor byte estimate
  (`estimateBytes` sums `*.safetensors`, already counts the vision tower).

## Slices (each: xcodebuild Release → e2e phase → annotated tag direct-to-main → graphify update)

- **M71.1 — wire protocol. ✅ SHIPPED v0.10.159.** `ChatMessage.content` decodes
  `String | [part]` (custom `Codable`, text-only callers unchanged); `ChatTurn` gains an
  optional `images: [ChatImage]` payload (defaults empty); `ChatImage.fromImageURL`
  decodes base64/`data:` → bytes and **400s `http(s)` image URLs** (passive-oracle, pure +
  unit-tested in `ChatImageTests`). Sync `/v1/chat/completions` + the queued conversation
  path both decode images via `chatTurns(from:)`; an image carried with no VLM path yet →
  400 `vision_not_supported` (M71.2 flips this per-model). `OpenAPISpec` gains
  `ChatContentPart` + `content` oneOf. e2e phase 2.3 asserts the two 400s + the unchanged
  200 text path. (Native `/api/chat` minimal dialect left text-only — vision goes through
  `/v1`; revisit if a native consumer needs it.)
- **M71.2 — VLM load/generate path. ✅ SHIPPED v0.10.160** (substrate path-dep @ `7eb154c`).
  Linked `MLXVLM` (daemon graph only). A checkpoint with a top-level `vision_config`
  (`ModelConfigInfo.hasVisionConfig`) loads via `VLMModelFactory.shared.loadContainer`
  instead of the text factory — **single VLM container** serves text + image
  (`MLXLLMModule.residentIsVision`/`servesVision`). Images flow `ChatTurn.images` →
  `Chat.Message(images:)` → `UserInput` → `Gemma4Processor` (CIImage). The serve path
  (sync + queued) allows image parts only when `servesVision`, else 400
  `vision_not_supported`. DFlash is skipped for VLM loads (seam bound to the text model;
  re-wire deferred). Governor counts one resident copy (`estimateBytes` already sums all
  safetensors incl. the tower). Non-vision text models are byte-unchanged. Unit tests:
  `ModelConfigInfoTests` vision cases + `ChatImage` decodable-image validation. Real
  image→text validation = RUNBOOK scenario **J** (stub can't exercise the VLM path).
  **VALIDATED 2026-06-17** on `mlx-community/gemma-4-e2b-it-4bit` (`gemma4_audio`, full
  `vision_tower`): accurate image description + all J negatives pass. The substrate VLM
  port serves the gemma-4 **VLM** family (`gemma4`/`gemma4_audio`); the **omni** family
  (`gemma4_unified_audio`, e.g. `gemma-4-12B-it`) has a different minimal vision arch
  (`vision_embedder`/`embed_vision`) the substrate does NOT implement → **omni vision is a
  separate future substrate-port milestone**, tracked outside M71.
- **M71.3 — capability surface + e2e.** `athena show` reports a vision capability for
  `gemma4_unified`; real-model e2e (curl an image → coherent description) in
  `deploy/integration/RUNBOOK.md` (real-model tier) + a stub-tier e2e-rbac.sh phase;
  OpenAPI example. Docs: quickstart "image input" section.

## Test bar

- **Per-slice CI gate:** `./deploy/build.sh Release` (xcodebuild — MLX needs full Xcode,
  not `swift build`) + a new `deploy/e2e-rbac.sh` phase (stub engine, loopback, curl).
- **Drift-guard:** the spec↔routes test must stay green across the `ChatCompletionRequest`
  schema change.
- **Text-path regression:** a text-only chat request must produce byte-identical output to
  pre-M71 on the `MLXLLM` path (assert the VLM link didn't perturb it).
- **Real-model validation:** image-in → description on `gemma4-*-it` via the Release binary
  + curl, recorded in `deploy/integration/RUNBOOK.md` (manual pre-release tier, not CI).
- **Cross-platform client:** `MLXVLM` must not enter `clients/`; re-prove the standalone
  client package builds on Linux if any `AthenaClient` shape changes.

## Rollout / risk

- **Passive-oracle:** preserved with zero carve-out (no outbound image fetch). Image rides
  inbound in the request body, like audio on the transcription endpoints.
- **Backward-compat:** text-only chat is wire-unchanged (string `content` still decodes);
  the VLM path engages only when image parts are present.
- **Substrate:** consumed via the existing path-dep (no pin bump mechanism); reproducibility
  pins to the substrate commit recorded in M71.2.
- **Open questions deferred:** audio-in-chat (own milestone); structured output over VLM;
  multi-arch VLM (Qwen-VL) generalization of the content-parts plumbing.

## Approval gate

This plan + ADR 010 are the approval artifacts. On ratification: confirm the milestone
number + first tag, then implement M71.1 as the first stacked slice. No implementation
code until then.
