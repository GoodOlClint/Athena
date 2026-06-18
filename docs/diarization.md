# Speaker diarization

`POST /v1/audio/diarizations` answers "who spoke when" over an uploaded audio
file. Multipart form, Bearer auth, shared 100 MiB upload cap. Response:

```json
{"num_speakers": 5, "segments": [{"start": 1.2, "end": 4.8, "speaker": 0}, …]}
```

## Methods

Pick the engine with the `method` form field (ADR 018):

| `method` | Engine | Speakers | Overlap | Notes |
|---|---|---|---|---|
| `sortformer` *(default)* | NVIDIA Sortformer, end-to-end | **≤ 4** (hard cap) | yes | Fast. Architecturally capped at 4. |
| `cluster` | WeSpeaker embed + agglomerative cluster over a naive sliding window | unbounded | no | `threshold` is audio-dependent; tune per file. |
| `pyannote` | Learned PyanNet segmentation → WeSpeaker embed → **global** cluster | unbounded | **yes** | Overlap-aware, file-stable ids. Recommended for >4 / overlapping speech. |

Speaker integers are **global across the whole file** for every method — the
same person keeps the same id from `0:00` to end. The `pyannote` path emits
**overlapping segments** (same time span, different `speaker`) when people talk
over each other; `sortformer` and `cluster` emit non-overlapping turns.

### Speaker-count hints (cluster / pyannote)

All optional; omit to auto-detect.

- `num_speakers` — exact count.
- `min_speakers` — floor.
- `max_speakers` — cap.
- `threshold` — cosine-distance merge bound (default `0.75`).
- `min_cluster_seconds` — *pyannote auto mode only*: minimum total airtime for a
  cluster to count as a speaker; smaller clusters are folded into the nearest
  speaker (default `6`). Ignored when `num_speakers` is set.

**Auto-counting on messy/long audio is approximate.** Unsupervised speaker
counting is hard: on long recordings with crosstalk, music, or background
voices, auto mode returns a sane ballpark but can over-count (e.g. ~14 distinct
sources where you expect 6). When you know roughly how many speakers there are,
**pass `num_speakers` (exact) or `max_speakers` (cap)** for a correct count —
the clustering then targets that number directly. Raising `min_cluster_seconds`
also trims borderline speakers in auto mode.

## Examples

```sh
# pyannote — recommended for meetings / >4 speakers / crosstalk
curl -F "file=@meeting.m4a" -F method=pyannote \
     -H "Authorization: Bearer $ATHENA_KEY" \
     http://127.0.0.1:7447/v1/audio/diarizations

# pyannote with a known speaker count
curl -F "file=@meeting.m4a" -F method=pyannote -F num_speakers=6 ...

# default Sortformer (≤4, fastest)
curl -F "file=@call.wav" -H "Authorization: Bearer $ATHENA_KEY" \
     http://127.0.0.1:7447/v1/audio/diarizations
```

Note the `@` — `-F "file=@path"` uploads the file's bytes; without it curl
sends the path string and you get `400 invalid_audio`.

## Whole-file processing

`pyannote` is internally windowed (10 s / 50% overlap), so there is **no
attention-window duration limit** — a full-length recording diarizes in one
call, bounded only by the 100 MiB upload cap. Use compressed audio for long
files: raw 16 kHz mono WAV is ~1.9 MB/min, so ~52 min fills the cap, whereas
m4a/opus fit hours. Do **not** chunk diarization client-side — it would break
global speaker ids.

## Enabling the pyannote backend (operator)

The `pyannote` method needs a pyannote-segmentation model in the `diarization`
allowlist (the Sortformer default stays the default). It ships in the seed and
in `athena init`, so a fresh appliance has it. On an existing install:

```sh
athena allowlist add diarization aufklarer/Pyannote-Segmentation-MLX
athena pull aufklarer/Pyannote-Segmentation-MLX     # ~6 MB
```

The model is selected per request by backend class — `method=pyannote` routes to
the resident segmentation model automatically (or pass `model=` explicitly). A
method/model mismatch returns `400 invalid_method`.

## Pairing with transcription

`/v1/audio/diarizations` is the authoritative diarization source. To attach
speaker labels to a transcript, call `/v1/audio/transcriptions` (text) and
`/v1/audio/diarizations` separately and merge by timestamp. The
`/transcriptions?diarize=true` flag is a legacy Sortformer convenience (≤4, not
reconciled with a standalone pyannote call) — don't rely on it for >4 or for
id-consistency.
