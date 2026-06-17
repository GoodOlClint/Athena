# Handoff prompt — build a cross-file speaker-identification tool over Athena

> Copy everything below the line into a fresh coding-agent session. It is self-contained:
> it does not assume access to the Athena source repo. The agent talks to a running Athena
> daemon over HTTP only.

---

You are building a small **client-side** command-line tool that gives **stable speaker names
across many audio files**. The user has a folder of `.m4a` recordings sharing a small,
mostly-fixed cast of recurring people (≤10 distinct voices). Audio diarization labels
speakers per-file and arbitrarily (`speaker0/1/2…`), so `speaker2` in one file is unrelated
to `speaker2` in another. Your tool maps those per-file labels to **persistent names**
(e.g. `Alice`, `Bob`) or flags them `unknown-N`.

## Hard constraints

- **You only consume HTTP endpoints of a local "Athena" daemon** (default
  `http://127.0.0.1:7447`). Do not modify the daemon. Do not call any cloud/online service —
  everything is local.
- **Pick any language** (Python + `numpy` + `requests` is the obvious fast path).
- **Conservative naming:** never emit a wrong name to avoid a missing one. Below the
  similarity threshold → `unknown-N`, full stop. No "best guess."
- **Voiceprints live in a local JSON file** you manage (do NOT use the daemon's vector DB —
  it is single-dimension and owner-scoped, wrong tool for this).
- Verify the wire shapes below against the live spec at `GET /openapi.json` before coding; if
  anything differs, trust the running daemon and adapt.

## The three endpoints you use

All audio endpoints are `multipart/form-data`. If the daemon has auth enabled, send
`Authorization: Bearer <token>`; on a loopback dev daemon no auth is needed.

**1. Diarize — `POST /v1/audio/diarizations`**
- form fields: `file` (required, the audio bytes); optional `num_speakers` (int hint).
  **That is the entire request surface** — there is no `method`/`max_speakers`/`threshold`
  field; do not send one (the daemon will reject unknown fields or ignore them). Verify
  against `GET /openapi.json` if in doubt.
- response: `{"num_speakers": int, "segments": [{"start": sec, "end": sec, "speaker": int}]}`
  — `speaker` is an integer label **valid only within this file**.
- **Known limitation:** the underlying model (Sortformer) tops out at **4 speakers**. A file
  with >4 distinct voices cannot be split correctly, and there is **no client-side override**
  — the daemon does not expose a clustering mode on this endpoint. Document this for the user;
  do not invent fields to work around it.

**2. Speaker-embed — `POST /v1/audio/embeddings`**
- form fields: `file` (required); `segments` = a JSON string `[{"start":sec,"end":sec},…]`
  (time-ranges within the clip; omit → whole clip as one segment); optional `model`.
- response:
  ```json
  { "object":"list",
    "data":[ {"object":"speaker_embedding","index":i,
              "segment":{"start":sec,"end":sec},
              "embedding":[/* 256 floats, L2-normalized */],
              "duration_seconds":sec } ],
    "model":"…","dimension":256 }
  ```
- **Key:** one **256-d, already L2-normalized** WeSpeaker vector per requested segment. You
  may pass all of a file's diarization turns as `segments` in **one call** and get one
  embedding per turn back in order.

**3. (Optional) Named transcript — `POST /v1/audio/transcriptions`**
- form fields: `file`, `response_format=verbose_json`, `diarize=true` (string).
- response `segments[]` include a per-segment integer `speaker`. Use your speaker→name map
  to relabel these into a human transcript when the user passes `--transcript`.

## Algorithm

**Per file → per-speaker voiceprint:**
1. Diarize the file → `turns`.
2. Drop turns shorter than `MIN_TURN_SEC` (default 1.0s) — short/overlap turns give noisy,
   mixed-voice embeddings.
3. One `embeddings` call passing the surviving turns as `segments` → one 256-d vector each.
4. For each local `speaker` label, compute the **duration-weighted mean** of its turn vectors
   and **L2-normalize** the result → that file-speaker's centroid (unit vector).

**Matching (conservative):** for each file-speaker centroid, `cosine = dot product` (both
unit-norm) against every enrolled identity centroid. `argmax`; if best ≥ `THRESHOLD` assign
that name, else `unknown-{k}` (k stable within the file). Below threshold is ALWAYS unknown.

**Voiceprint store** (`voiceprints.json`):
```json
{ "model":"WeSpeaker-ResNet34-LM", "dim":256, "threshold":0.50,
  "identities":[
    {"name":"Alice","centroid":[/*256 unit-norm floats*/],"exemplars":14,
     "sources":["2026-05-01.m4a#spk0"]} ] }
```
Enrolling more samples for a name folds them into the running centroid (re-normalize) and
bumps `exemplars`/`sources`.

## Threshold calibration

WeSpeaker cosine has **no universal same-speaker cutoff**. Ship a default of ~0.50 but
provide a `calibrate` mode: embed every per-file-speaker centroid in the folder, compute the
pairwise-cosine distribution, and report the valley between the same-speaker and
different-speaker mass so the user can pick a threshold there. Write the chosen value into
`voiceprints.json` for reproducible runs.

## CLI shape (suggested)

```
speakerid enroll   --name <NAME> <clip.m4a> [--speaker N]   # supervised add/update
speakerid cluster  <folder/>                                # unsupervised bootstrap: group all file-speakers
speakerid name     <cluster-id> <NAME>                      # name a bootstrap cluster
speakerid identify <file|folder> [--transcript]             # MAIN VERB → speaker→name map (+ named transcript)
speakerid calibrate <folder/>                               # threshold guidance
speakerid list                                              # show enrolled identities
```
Global flags: `--base-url` (default `http://127.0.0.1:7447`), `--token`,
`--store voiceprints.json`, `--threshold`, `--min-turn-sec`, `--num-speakers N` (forwarded to
diarization as the `num_speakers` hint when the operator knows the count).

**Enrollment is hybrid:** `cluster` bootstraps a never-labelled folder by agglomerative
clustering of all file-speaker centroids on cosine distance (`1 - sim`, average linkage, cut
at `1 - THRESHOLD`); print each cluster's member count, total speech seconds, and a
representative `file#spkN`, then let the user `name` it. `enroll` pins/corrects a specific
voice. A large cluster matching no identity is a recurring-unknown enrollment candidate.

## Edge cases

- **>4 speakers in a file** → not solvable client-side (Sortformer 4-speaker cap, no
  clustering knob on the endpoint). Document the cap; pass `--num-speakers` when known to help
  the model within its limit.
- **Short / overlapping turns** → dropped by `MIN_TURN_SEC`; overlap → low cosine → unknown.
- **Empty/failed diarization on a file** → report a per-file error, continue the folder run.
- **Close call between two identities** → argmax wins; optionally warn on a small margin, but
  never invent a name below threshold.

## Build order (each step runs end-to-end against the real folder)

1. `identify` on a **single file** against a hand-seeded `voiceprints.json` — proves the
   diarize→embed→centroid→match→map chain and the wire shapes.
2. `enroll` (supervised) — build/update an identity from a clip; pin the recurring pair.
3. `identify` on a **folder** + `--transcript` — **this meets the user's immediate need:
   consistent names across the `.m4a` folder.**
4. `cluster` + `name` + `calibrate` — the unsupervised bootstrap and threshold tuning; pin
   `THRESHOLD`/`MIN_TURN_SEC` from the observed distribution.
5. A short README: usage + the calibration procedure.

Deliver a working tool plus the `voiceprints.json` it produces. Keep it small and readable.
