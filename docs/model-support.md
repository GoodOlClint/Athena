# Model support & the pull preflight

*"Will this model work before I pull it?"* Athena answers that for **every
modality it serves** — chat (LLM/vision), embeddings, transcription,
diarization, and speaker-embedding — from a **config-only** check, before any
multi-GB download. This is ADR 021 (M77).

## The one predicate

A single MLX-free classifier, `ModelSupport`, is the source of truth for *"what
is this checkpoint, and can Athena load it?"* It returns:

```
(modality, loadability)
  modality    ∈ llm | vision | embedding
              | transcription(whisper|parakeet)
              | diarization(sortformer|pyannote)
              | speakerEmbedding | unsupported
  loadability ∈ loadable | unknown | unsupported(reason, guidance)
```

It **composes** the focused detectors (`ModelClass`, `TranscriptionArch`,
`DiarizationBackend`) into one modality router and adds a **loadability** layer:
a packaging check, distinct from family, that asks whether the fields the
*loader* actually requires are present.

| Modality | Loadable when… |
|---|---|
| transcription / whisper | `model_type: whisper` **and** `n_vocab == 51866` (the decoder is pinned to the large-v3 family) |
| transcription / parakeet | the NeMo `joint.vocabulary` array is present (a transformers-format Parakeet without it is **not** loadable) |
| diarization | a Sortformer or pyannote-segmentation `model_type` |
| embedding | sentence-transformers markers or an embedder-only `model_type` |
| speaker-embedding | a WeSpeaker `model_type` |
| generative / vision | **best-effort** — classified, but architecture coverage is inherited from the substrate, so loadability is `unknown` (the loader is the authority) |

Three consumers share this one predicate, so a verdict can never drift from
what actually happens:

- **Module loaders** refuse an unsupported packaging with a cause-naming **4xx**
  (e.g. a transformers-format Parakeet → `400 unsupported_transcription_arch`
  naming the missing `joint.vocabulary`), instead of an opaque 500.
- **`athena convert`** redirects the source-precision modalities (embedding,
  transcription, diarization, speaker-embedding) to `pull` — `convert` only
  quantizes generative and vision models.
- **`athena pull`** runs a config-only preflight (below).

## `athena pull --check`

A **dry run**: fetch only the model's config, classify it, print the verdict,
download nothing. Exits non-zero when the packaging is unsupported, so scripts
can gate on it.

```console
$ athena pull mlx-community/whisper-large-v3-turbo --check
mlx-community/whisper-large-v3-turbo
  modality:    transcription (whisper)
  loadability: loadable
verdict: Athena can load this model (routing + required fields confirmed from
config; this is not a correctness check).

$ athena pull nvidia/parakeet-tdt-0.6b-v3 --check
nvidia/parakeet-tdt-0.6b-v3
  modality:    transcription (parakeet)
  loadability: unsupported
  reason:      the checkpoint's config.json has no joint.vocabulary array, which
               the Parakeet loader reads to build the token table
  fix:         use a NeMo-format Parakeet-TDT export — an RNN-T or TDT checkpoint
               whose config carries the joint vocabulary; a transformers-format
               Parakeet checkpoint is not loadable
verdict: Athena cannot load this model as packaged.
```

`--check` is host-independent (it classifies from metadata, never contacts the
daemon), so it behaves identically whether or not you target a remote daemon.

## The in-`pull` gate

A plain `athena pull` runs the same check first and:

- **refuses early** on a known-unsupported packaging — naming the structural
  requirement, downloading nothing;
- **warns and proceeds** on `unknown` (a best-effort generative/vision arch the
  substrate may still load — it's the authority);
- **proceeds silently** on `loadable`.

The gate is best-effort: if the preflight fetch itself fails (offline, a 404, a
gated repo), the normal pull runs anyway and surfaces the real error.

## What it does and does **not** prove

The preflight is a **routing + packaging** check: it confirms Athena will route
the checkpoint and that the loader's required fields are present. It says
**"Athena can load this,"** never **"this model is correct."** Numerical
correctness — that the forward pass produces right answers — is the job of the
gated heavy tests (`ATHENA_RUN_MODEL_TESTS=1`), which load and run real weights.

The error/guidance strings name the **structural requirement** a checkpoint
fails (e.g. a missing `joint.vocabulary`), never a specific model id or HF repo:
repos move, get gated, or vanish, but the requirement is stable.

## `verify` vs. preflight

Two different questions — they don't overlap:

| | Question | When |
|---|---|---|
| `athena verify [NAME]` | Is an **already-stored** model **intact**? (config + safetensors headers + tokenizer) | after a pull, against local files |
| `athena pull --check ID` | Will **this id** load at all? (modality + packaging) | before a pull, against config metadata |
