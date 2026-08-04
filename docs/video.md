# Video transcription

`POST /v1/video/transcriptions` turns an uploaded **video** into text by
demuxing its audio track and transcribing it with the same engine as
`/v1/audio/transcriptions`. ADR 022 (M78.1).

> **Athena-native, NOT OpenAI-compatible.** OpenAI has no video API. The route
> reuses the OpenAI transcript *response shape* for consumer convenience, but it
> is an Athena extension under the `/v1` namespace (marked `[native]` in the
> AGENTS.md endpoint list and in `/openapi.json`).

## Request

Multipart form, Bearer auth (inference-tier, like `/v1/audio/*`). Shared upload
cap `max_video_upload_bytes` (default **1 GiB**); over it ⇒ `413
payload_too_large`.

| Field | Meaning |
|---|---|
| `file` | The video (any container AVFoundation reads — mp4/mov/m4v/…). Required. |
| `model` | Selects among the transcription allowlist — a Whisper or Parakeet-TDT id (ADR 020). Omit ⇒ default (Whisper). |
| `language` | ISO code; omit ⇒ auto-detect (Whisper). |
| `response_format` | `json` (default), `text`, `srt`, `vtt`, `verbose_json`. |
| `timestamp_granularities[]` | `word` (verbose_json only) adds word timings. |

```console
$ curl -s -H "Authorization: Bearer $TOKEN" \
    -F file=@clip.mp4 -F response_format=verbose_json \
    http://127.0.0.1:7447/v1/video/transcriptions
{"task":"transcribe","language":"en","duration":3.9,"text":"…","segments":[…]}
```

The response is byte-identical to `/v1/audio/transcriptions`, so a consumer that
already parses audio transcripts reuses its parser unchanged.

## How it works

Video transcription is an **audio-extraction problem, not a video problem**: the
file is just a container with an audio track.

```
upload → AVAssetReader: demux the first audio track → 16 kHz mono PCM
       → (shared decode floor + ceiling) → Whisper/Parakeet tenant → transcript
```

The extracted PCM funnels through the **same** floor/ceiling as audio uploads
(`AudioDecode.sampleBoundError`) and the **same** governed transcription slot
(ADR 011) — video is orchestration over an existing tenant, not a new model.

## Error cases (all 4xx, never a crash)

| Condition | Response |
|---|---|
| No audio track in the video | `400 video_no_audio_track` |
| Audio shorter than 0.1 s | `400 audio_too_short` |
| Unreadable / truncated / corrupt container | `400 invalid_audio` |
| Over the upload cap | `413 payload_too_large` |
| `diarize=true` (not yet wired) | `501 not_implemented` |

A degenerate video can never abort the daemon — the audio track goes through the
same crash-hardened decode path as audio (validated by `deploy/e2e-video.sh`,
which sweeps no-audio / tiny / corrupt videos and asserts the daemon survives).

## Not (yet) supported

- **`diarize=true` on video** — returns `501`. Workaround: transcribe, then POST
  the audio to `/v1/audio/diarizations`. A fast-follow will wire it directly.
- **Visual description** (what's *on screen*) — a separate, deferred milestone
  (ADR 022 M78.2, `POST /v1/video/descriptions`), gated on a consumer defining
  its needs. Keyframe-level only when it lands; no temporal/event reasoning
  (that needs a video-native model the substrate doesn't expose).

## Validating locally

`./deploy/e2e-video.sh` synthesizes real fixtures with `ffmpeg`, runs a
dev-mode daemon against the model store, and checks the full matrix above. The
audio-track extractor also has a gated unit hook: point `ATHENA_VIDEO_FIXTURE`
at a real video and run with `ATHENA_RUN_MODEL_TESTS=1`.
