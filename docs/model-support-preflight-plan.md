# Unified model-support classification + pre-pull preflight — change plan (M77)

**Status:** Proposed, awaiting operator approval (brownfield change gate). Pairs
with ADR 021. No production code until approved.

## Goal

One MLX-free `ModelSupport` predicate that answers, from **config-only**
metadata, *"what modality is this checkpoint, and can Athena load it?"* — used by
the module loaders (cause-naming 4xx, not opaque 500), by `convert` (redirect
non-convertible modalities to `pull`), and by a new `pull` preflight (`--check`
dry run + early refuse) so an operator can **confirm a model will work before
pulling it**, for **every modality** Athena serves.

## Existing state (map first — do not fork)

| Detector | Covers | Gap this plan closes |
|---|---|---|
| `ModelClass` (ADR 016) | generative / vision / embedding / unknown | no ASR / diarization / speaker class → Parakeet mis-files as generative |
| `TranscriptionArch` (ADR 020) | whisper / parakeet / unsupported | routes by family, no loadability (nvidia vs mlx-community Parakeet both `.parakeet`) |
| `DiarizationBackend` (ADR 018) | sortformer / pyannote / unknown | separate, not composed |
| `athena verify` | integrity of **stored** model | can't answer pre-pull |
| `convert` config-only pre-fetch (ADR 016) | embedder redirect | the **mechanism** to reuse for the pull preflight |

`ModelSupport` **composes** these (keeps each focused + unit-pinned) and adds the
modality router + loadability signals.

## Architecture

```
ModelSupport (MLX-free)
 ├─ modality(config) → llm | vision | embedding
 │                     | transcription(whisper|parakeet)
 │                     | diarization(sortformer|pyannote)
 │                     | speakerEmbedding | unsupported
 │     (delegates to ModelClass / TranscriptionArch / DiarizationBackend)
 └─ loadability(config, modality) → loadable | unsupported(reason, guidance) | unknown
        whisper:   model_type==whisper && n_vocab==51866
        parakeet:  joint.vocabulary present (NeMo)  — else "use mlx-community NeMo build"
        embedding: ST markers / embedder-only type
        generative/vision: best-effort (.unknown for named-but-unverified; substrate is ground truth)

consumers (one predicate, no drift):
  loaders   → unsupported ⇒ cause-naming 4xx (subsumes M76 Parakeet 500→400)
  convert   → transcription/diarization/speakerEmbedding ⇒ redirect to `pull` (like embedders)
  pull      → config-only pre-fetch: --check dry-run; refuse-early on unsupported; warn on unknown
```

## Slices (each: annotated tag direct-to-main, `appVersion` bump IN the slice
commit, ≥1 regression test; pre-commit pipeline Tests → Security → Quality →
Refactor)

**S1 — `ModelSupport` predicate (MLX-free seam).** New `ModelSupport` composing
the three detectors + the loadability signals. No behavior change at call sites
yet. *Test:* classification + loadability matrix — whisper (good vocab / wrong
vocab), parakeet NeMo vs transformers (the incident case), embedder, sortformer,
pyannote, speaker-embedding, generative, vision, unknown.

**S2 — Loader consumers.** Each module's `loadModel` consults `ModelSupport` and
refuses `.unsupported` packaging with a cause-naming 4xx (generalizes the M76
Parakeet 500→400; folds in the whisper-vocab guard). *Test:* a `parakeet_tdt`
transformers-format dir → `400 unsupported_transcription_arch` naming the
mlx-community build, **not** a 500.

**S3 — `convert` consumer.** `convert` uses `ModelSupport`: transcription /
diarization / speaker-embedding modalities are redirected to `pull` with
guidance (the path embedders already take), instead of mis-routing to the
generative factory. *Test:* `convert nvidia/parakeet-tdt-0.6b-v3` → clean
`unsupported_convert_class`-style redirect ("pull it; it serves via the
transcription path"), not the misleading "bump the substrate" message.

**S4 — `pull` preflight.** Config-only pre-fetch gate in `pull`: refuse early on
`.unsupported`, warn-and-proceed on `.unknown`, proceed on `.loadable`. Add
`athena pull --check <id>` — a dry run that fetches only the config, prints the
verdict, and downloads nothing. *Test:* unit on the gate decision (refuse / warn
/ proceed); gated/e2e config-only fetch + verdict print.

**S5 — Surface + docs.** `--check` in the CLI help + OpenAPI note if any control
route is touched (drift-guard green); `docs/model-support.md` (what preflight
confirms vs the gated-test correctness boundary); cross-link `verify` (integrity)
vs preflight (support). ADR 021 → Accepted; CLAUDE.md index updated.

## Test bar

- MLX-free logic (modality router, loadability signals, pull-gate decision) →
  pure unit tests under `./deploy/test.sh` (ADR 008/009).
- Pre-fetch / preflight over a real id → gated (`ATHENA_RUN_MODEL_TESTS=1`) +
  e2e; `./deploy/e2e-rbac.sh` stays green.
- Regression pins for both incident cases (Parakeet nvidia load 500→400; convert
  ASR misroute → redirect).

## Risks

- **R1 false confidence.** Preflight confirms config/packaging, not numeric
  correctness. Mitigate: messaging says "Athena can load this"; gated tests
  remain the correctness gate (ADR 021 honesty boundary).
- **R2 generative arch coverage is inherited.** We can't enumerate every
  `model_type`; `.unknown` ⇒ warn-and-proceed, substrate raises the precise
  error. Don't over-promise.
- **R3 detector drift.** Compose, don't fork — one predicate feeds all three
  consumers so a verdict can't disagree with an actual load.
- **R4 signal location.** Some signals live outside `config.json` (ST markers,
  `tokenizer.model`); the pre-fetch must grab the same file set `convert` does.

## Out of scope

- Numeric/forward validation (stays in the gated heavy tests).
- Auto-substituting a working alternative id.
- Converting/quantizing ASR/diarization/speaker models (not convert targets).

## Open decisions for review

- **D1 — compose vs replace** the three detectors. *Rec: compose* (keep focused
  detectors, layer `ModelSupport` on top).
- **D2 — pull gate strictness.** Refuse-early on known-unsupported,
  warn-and-proceed on unknown, proceed on loadable. *Rec: as stated* (mirrors
  convert's early-refuse for the known-bad case without blocking best-effort
  generative loads).
- **D3 — surface.** `athena pull --check` (dry run) + the same gate inside
  `pull`. *Rec: as stated*; not a separate `preflight` verb, not folded into
  `verify` (different question: support vs integrity).
