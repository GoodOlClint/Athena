# Audio transcription

`POST /v1/audio/transcriptions` turns an uploaded audio file into text.
Multipart form, Bearer auth, shared 100 MiB upload cap (`max_audio_upload_bytes`).
OpenAI-compatible: `response_format` of `json` (default), `text`, `srt`, `vtt`,
or `verbose_json`; `timestamp_granularities[]=word` adds word timings;
`diarize=true` tags each segment with a speaker (uses the diarization slot).

## Backends (ADR 020)

Transcription is **multi-backend**, selected by the resident model's class —
the same shape as diarization (ADR 018). The route is unchanged; the `model`
form field picks the weights, and the model's `config.json` picks the engine:

| Engine | Example model | Notes |
|---|---|---|
| **Whisper** *(default)* | `mlx-community/whisper-large-v3-turbo` | The established, feature-complete path. Detects language. |
| **Parakeet-TDT** | `mlx-community/parakeet-tdt-0.6b-v3` | NVIDIA Parakeet-TDT, MLX port. Multilingual (25 langs), ~0.6 B params, very fast. Greedy TDT decode; word/segment timestamps from the TDT durations. |

The single governed `transcription` slot holds one model at a time (ADR 011).
Selecting a different `model` rebinds the slot (unload + load) under the same
governor reservation. A checkpoint that is neither Whisper nor Parakeet ⇒
`400 unsupported_transcription_arch`.

### Selecting Parakeet

Both models are seeded into the allowlist and pulled by `athena init`. Pick
Parakeet per request:

```sh
curl -s http://127.0.0.1:7447/v1/audio/transcriptions \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@clip.wav \
  -F model=mlx-community/parakeet-tdt-0.6b-v3 \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word'
```

Whisper stays the default when `model` is omitted.

## Timestamps

Both engines populate `verbose_json` `segments` (with `start`/`end`/`text`/
`avg_logprob`) and, when `timestamp_granularities[]=word` is set, top-level and
per-segment `words`. SRT/VTT render the same segments. For Parakeet the timing
comes from the **TDT durations** (`time_ratio` = 0.08 s per encoder frame):
each emitted token carries a start + duration, grouped into words (at the `▁`
SentencePiece boundary) and sentences (at terminal punctuation). `avg_logprob`
is the mean per-token log-probability over the segment.

## Long audio

Clips longer than 120 s are split into overlapping 120 s windows (15 s overlap),
decoded independently, and stitched back into one ordered timeline — the
Parakeet encoder's attention is O(T²), so a single pass over a very long clip
would exceed the Metal budget. Shorter clips run a single pass. Whisper uses its
own 30 s windowing. The 100 MiB upload cap bounds total input.

## Language

- **Whisper** auto-detects the language (or honours the `language` form field)
  and returns it.
- **Parakeet** greedy decode does not surface a detected language (there is no
  forced language prompt); the response echoes the requested `language`, or
  `auto`. The model is multilingual regardless — it transcribes in the spoken
  language.

## Whisper vs Parakeet — choosing

Whisper is the conservative default: mature, language-detecting, feature-complete.
Parakeet is additive — pick it for multilingual material or when you want the
higher-throughput path.

**Default-engine decision (open — ADR 020).** Whether Parakeet should *become*
the default is gated on a WER + throughput A/B against `whisper-large-v3-turbo`
on a labelled set:

- **Throughput** (measured this milestone, Parakeet-TDT-0.6B-v3, Apple Silicon,
  60 s English clip): ~75–135× real-time (encoder ~85–125 ms + greedy TDT decode
  ~360–680 ms; ~500–900 decode tok/s). Comfortably ≫ real-time.
- **WER**: run both engines over a labelled multilingual corpus and compare word
  error rate per language. This needs ground-truth transcripts (operator-side
  data) — run it on your own set before flipping the default. Procedure: POST
  each clip to both `model=` values, score the returned `text` against the
  reference with a standard WER tool, aggregate per language.

Until that A/B lands, **Whisper remains the default** and Parakeet is opt-in via
`model=`.
