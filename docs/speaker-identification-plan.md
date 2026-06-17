# Cross-file speaker identification — design & change plan

**Status:** Design — awaiting operator review (do not implement yet)
**Date:** 2026-06-17
**Decision of record:** ADR 014 — identity stays client-side; the daemon is not extended.
**Scope:** a client-side tool over today's `/v1/audio/*` + (not) the vector DB.

This document is the approval gate. It describes the durable client workflow the operator
chose at the design interview (client-side / ≤10 speakers / hybrid enrollment / conservative
unknowns). Nothing here touches the Athena daemon.

---

## 1. Goal

Given a folder of `.m4a` recordings sharing a small recurring cast, produce **stable speaker
names** across all files — e.g. every recording's local `speaker0/1/…` mapped to `Alice`,
`Bob`, or `unknown-N`. Deliverables per file: a speaker→name map, and optionally a
name-relabelled transcript.

## 2. The primitives (verified against the code)

| Call | In | Out | Notes |
|---|---|---|---|
| `POST /v1/audio/diarizations` | `file` (+ `num_speakers?`) | `{num_speakers, segments:[{start,end,speaker:int}]}` | **Only** `file`+`num_speakers` are exposed (verified `OpenAPISpec.swift:198`). Sortformer caps at 4 speakers; no clustering knob on this endpoint — a known limitation, not client-overridable |
| `POST /v1/audio/embeddings` | `file`, `segments`=`[{start,end},…]` (JSON) | `{data:[{index,segment,embedding:[256 float],duration_seconds}], dimension:256}` | **one 256-d L2-normalized vector per segment**; arbitrary time-ranges accepted |
| `POST /v1/audio/transcriptions` | `file`, `response_format=verbose_json`, `diarize=true` | `segments:[{…,speaker:int}]` | only used if a named transcript is wanted |

Auth: loopback dev mode needs none; otherwise `Authorization: Bearer <token>`. The vector
DB is **not** used (ADR 014: 256-vs-2560 dimension collision + owner scoping; local JSON
instead).

## 3. Core algorithm

### 3.1 Per-file → per-speaker voiceprint

```
for each file:
  turns = POST /v1/audio/diarizations(file)               # [{start,end,speaker}]
  turns = [t for t in turns if t.end - t.start >= MIN_TURN_SEC]   # drop noisy shorts
  embs  = POST /v1/audio/embeddings(file, segments=[{start,end} for t in turns])  # one call
  for each local speaker label s in turns:
    vecs = embeddings of s's turns
    centroid[s] = normalize( sum(duration_w * v for v in vecs) )  # duration-weighted mean
```

One diarization call + one embeddings call per file. Embeddings are already L2-normalized;
the duration-weighted mean is re-normalized so cosine stays well-defined.

### 3.2 Matching (conservative)

```
for each local speaker s in a file:
  best_name, best_sim = argmax over identities of cosine(centroid[s], identity.centroid)
  label[s] = best_name        if best_sim >= THRESHOLD
           = "unknown-{k}"     otherwise   # k stable within this file
```

`cosine` is a plain dot product (both sides unit-norm). `THRESHOLD` defaults conservatively
and is calibratable (§4). No second-best tie-breaking beyond argmax; below threshold is
always `unknown` — never a guessed name (operator's choice).

### 3.3 Voiceprint store (`voiceprints.json`)

```json
{
  "model": "WeSpeaker-ResNet34-LM",
  "dim": 256,
  "threshold": 0.50,
  "identities": [
    { "name": "Alice",
      "centroid": [/* 256 floats, unit-norm */],
      "exemplars": 14,
      "sources": ["2026-05-01.m4a#spk0", "2026-05-08.m4a#spk1"] }
  ]
}
```

Enrolling more samples for a name folds them into the running centroid (and bumps
`exemplars`/`sources`). Plain file, version-control-friendly, diffable.

## 4. Threshold calibration

WeSpeaker cosine has **no universal same-speaker cutoff** — it depends on recording
conditions and turn length. The tool ships a default (≈0.45–0.55, to be pinned during the
real-folder validation below) and a `calibrate` mode:

- Embed all per-file-speaker centroids in the folder, compute the pairwise cosine
  distribution, and report the bimodal gap (same-speaker vs different-speaker mass) so the
  operator can pick a threshold sitting in the valley. The chosen value is written into
  `voiceprints.json` so runs are reproducible.

## 5. Enrollment UX (hybrid)

- **Bootstrap (unsupervised):** `cluster ./recordings/` runs §3.1 across the whole folder,
  then agglomerative-clusters all per-file-speaker centroids by cosine distance
  (`1 - sim`, average linkage, cut at `1 - THRESHOLD`). Prints each cluster with its member
  count and total speech seconds and a representative `file#spkN`. Operator then
  `name <cluster-id> Alice`.
- **Supervised (pin/correct):** `enroll --name Alice clip.m4a [--speaker N]` embeds a clean
  sample and adds/updates Alice's centroid. Used to seed before the first run, or to fix a
  cluster the bootstrap split or merged.
- **Recurring unknowns** surface in the cluster view (a large cluster matching no identity)
  as enrollment candidates.

## 6. Commands (proposed CLI shape)

```
speakerid enroll   --name <NAME> <clip.m4a> [--speaker N]      # supervised add
speakerid cluster  <folder/>                                   # unsupervised bootstrap
speakerid name     <cluster-id> <NAME>                         # name a bootstrap cluster
speakerid identify <file.m4a | folder/>  [--transcript]        # the main verb → name map (+ named transcript)
speakerid calibrate <folder/>                                  # threshold guidance
speakerid list                                                 # show enrolled identities
```

`--base-url` (default `http://127.0.0.1:7447`), `--token`, `--store voiceprints.json`,
`--threshold`, `--min-turn-sec` as global flags.

## 7. Edge cases & how they're handled

- **>4 speakers in a file:** Sortformer caps at 4 and the daemon exposes **no** clustering
  override on `/v1/audio/diarizations` (only `file`+`num_speakers`). This is a documented
  limitation, not client-overridable; the tool forwards `--num-speakers` when known to help
  the model within its limit. (If a future daemon adds clustering params, they slot in behind
  the same config keys without changing callers.)
- **Short/overlapping turns:** turns under `MIN_TURN_SEC` (default ~1.0s) are dropped before
  embedding; overlap regions yield mixed-voice embeddings → naturally low cosine → pushed to
  `unknown`, never to a confident wrong name.
- **A file-speaker matching two identities closely:** argmax wins; if the margin to
  second-best is tiny the tool can warn (optional `--warn-margin`), but still never invents
  a name below threshold.
- **Empty/failed diarization:** surfaced as a per-file error, folder run continues.

## 8. What this explicitly does NOT do

- No daemon change, no enrollment table, no new HTTP route (ADR 014).
- No use of the built-in vector DB (dimension collision; ADR 014).
- No automatic enrollment of unknowns (operator chose conservative `unknown-N`).
- No cross-file linking of unknowns except via the explicit `cluster` view.

## 9. Delivery: a handoff prompt, not in-repo code (resolved)

The tool is **not built in this repo**. It is delivered as a self-contained brief —
[`docs/speaker-identification-agent-prompt.md`](speaker-identification-agent-prompt.md) —
that the operator hands to a client-side coding agent, which implements it in whatever
language/repo it likes (Python+numpy is the obvious fast path, but the prompt is
language-agnostic). Athena's tree carries only this design + the prompt. This is the most
literal reading of ADR 014's "the daemon is not extended": the spec lives here, the
implementation lives client-side.

The prompt is the contract. It is self-contained (inline wire shapes + algorithm + JSON
schema + CLI + edge cases + step plan) so the implementing agent needs no access to this
repo, and it points at `GET /openapi.json` for the agent to re-verify shapes against the
running daemon.

## 10. Step plan (carried inside the handoff prompt)

The same small, end-to-end-validated steps live in the prompt as the implementing agent's
build order:

1. **`identify` (single file)** — diarize→embed→centroid→match→print map against a manually
   seeded `voiceprints.json`. Validates the wire shapes end-to-end.
2. **`enroll` (supervised)** — build/update an identity from a clip; pin the recurring pair.
3. **`identify` (folder) + `--transcript`** — batch the folder; optional named transcript via
   `verbose_json`+`diarize`. **Immediate practical need met here.**
4. **`cluster` + `name` (bootstrap) + `calibrate`** — unsupervised path + threshold guidance;
   pin `THRESHOLD`/`MIN_TURN_SEC` from the observed distribution.
5. **README** — usage + calibration procedure.
```
