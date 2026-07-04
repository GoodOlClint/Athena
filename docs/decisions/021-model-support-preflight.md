# ADR 021 — unified model-support classification + pre-pull preflight

**Status:** **Accepted — shipped M77** (v0.10.176–179). Pairs with
`docs/model-support-preflight-plan.md`; usage in `docs/model-support.md`.
Motivated by an M76 field incident (a `nvidia/parakeet-tdt-0.6b-v3` checkpoint
produced an opaque `module_load_failed` 500, and `athena convert` on the same id
failed with a misleading "bump the substrate" message).

Shipped as four stacked slices: S1 the MLX-free `ModelSupport` predicate
(detectors relocated to `AthenaCore` so one predicate can compose them; v0.10.176),
S2 the transcription loader gate (Parakeet 500→400; v0.10.177), S3 the `convert`
redirect for the source-precision modalities (v0.10.178), S4 the `pull --check`
dry run + in-pull gate (v0.10.179). E2E-validated on the real incident pair
(nvidia transformers Parakeet → unsupported/exit 1 naming the missing
`joint.vocabulary`; mlx-community NeMo Parakeet + whisper-large-v3-turbo →
loadable).

## Context

Athena's "can this checkpoint be used, and how?" logic is **fragmented across
several detectors that don't agree and don't cover every modality**, and there
is **no way to find out before a multi-GB `athena pull`**:

- `ModelClass` (ADR 016, used by `convert`) knows `generative` / `vision` /
  `embedding` / `unknown` — but has **no notion of ASR (transcription),
  diarization, or speaker-embedding** models. A Parakeet checkpoint
  (`model_type: parakeet_tdt`) therefore falls through to `.generative`, convert
  routes it to the LLM factory, and the operator gets
  `unsupported_convert_class: parakeet_tdt (generative) … bump the substrate
  pin` — **wrong**: Parakeet is supported, just not via convert.
- `TranscriptionArch` (ADR 020) routes Parakeet by *family* but does **not check
  loadability**: both `nvidia/parakeet-tdt-0.6b-v3` (transformers packaging, no
  `joint.vocabulary`) and `mlx-community/parakeet-tdt-0.6b-v3` (NeMo packaging,
  has `joint.vocabulary`) classify as `.parakeet`, yet only the NeMo build
  loads. The router gives a **false green**, the loader then fails deep → opaque
  **500** instead of a cause-naming 4xx.
- `DiarizationBackend` (ADR 018) is a third, separate detector.
- `athena verify` only checks **integrity of an already-stored** model — it
  cannot answer "will this *new* id work?" before pulling.

The common thread: classification is per-feature and incomplete, loadability is
conflated with family, and the verdict is only discovered *after* download +
load. The operator wants to **confirm a model will work before pulling it**, and
wants that to hold for **every modality Athena serves**, not just transcription.

## Decision

1. **One MLX-free `ModelSupport` classifier — the single source of truth for
   "what is this checkpoint, and can Athena load it?"** From config-only
   metadata (`config.json` + sentence-transformers markers + a few
   packaging-signal files), it returns a verdict:

   ```
   (modality, loadability)
     modality ∈ { llm, vision, embedding,
                  transcription(whisper|parakeet),
                  diarization(sortformer|pyannote),
                  speakerEmbedding, unsupported }
     loadability ∈ { loadable, unsupported(reason, guidance), unknown }
   ```

   It **composes the existing focused detectors** (`ModelClass`,
   `TranscriptionArch`, `DiarizationBackend`) rather than forking a fourth
   parallel implementation (that would be a defect per the canonical-pipeline
   rule). The detectors stay; `ModelSupport` adds the **modality router** that
   covers all modalities and the **loadability signals** they were missing.

2. **Loadability is a packaging check, distinct from family.** Per modality,
   `ModelSupport` checks the specific signals the loader actually requires:

   - **transcription / whisper** — `model_type: whisper` **and** `n_vocab ==
     51866` (Athena's decoder is pinned to the large-v3 family; a v1/v2/medium
     mis-decodes — already enforced at load).
   - **transcription / parakeet** — the **NeMo** signal `joint.vocabulary`
     present. A `parakeet_tdt` transformers checkpoint without it →
     `unsupported(reason: "config has no joint.vocabulary — the Parakeet loader
     needs a NeMo-format export")`. The message names the **structural
     requirement**, never a repo id (see the guidance rule below).
   - **diarization** — `sortformer` / `pyannote-segmentation` per ADR 018.
   - **embedding** — sentence-transformers markers or an embedder-only type
     (ADR 016).
   - **generative / vision** — **best-effort**: classifies the class, but arch
     coverage is *inherited from the substrate* (we do not enumerate every
     `model_type`; ADR 016's stance stands). Loadability is `.unknown` for a
     named-but-unverified generative type — the substrate raises the precise
     error if its factory lacks that arch. `ModelSupport` does **not**
     over-promise correctness it can't prove from config alone.

3. **Three consumers, one predicate — so the verdict can never drift from
   reality:**

   - **Module loaders** (transcription/diarization/embedding/LLM): refuse a
     `.unsupported` packaging with a **cause-naming 4xx** (`code:
     unsupported_transcription_arch`, etc.) instead of an opaque 500. *(This
     subsumes the M76 Parakeet 500→400 fix.)*
   - **`convert`**: recognize the non-convertible modalities (transcription,
     diarization, speaker-embedding) and **redirect to `pull`** with clear
     guidance — exactly as it already redirects embedders (ADR 016) — instead of
     mis-routing ASR to the generative factory and emitting the misleading
     "bump the substrate" error.
   - **`pull` preflight**: a **config-only pre-fetch** (same sanctioned HF
     metadata egress `convert` already uses) classifies before any weight
     download. `athena pull --check <id>` is a **dry run** that prints the
     verdict and downloads nothing; a plain `pull` runs the same gate and
     **refuses early on a known-unsupported** packaging (no multi-GB waste),
     **warns-and-proceeds on `.unknown`** (lets the substrate try a
     best-effort generative load), and proceeds silently on `.loadable`.

4. **Honesty boundary (binding).** A config-only preflight proves *"Athena will
   route this and the loader's required fields are present"* — **not** that the
   forward pass is numerically correct. Full correctness remains the job of the
   gated heavy tests (`ATHENA_RUN_MODEL_TESTS=1`). Messaging must say "Athena can
   load this," never "this model is correct."

5. **Guidance names the requirement, never a repo (binding).** A
   `reason`/`guidance` string in code describes the **structural requirement**
   the checkpoint fails — what the packaging lacks or what Athena needs (e.g.
   "config has no `joint.vocabulary`"; "decoder requires the large-v3 vocab
   51866, this has N") — and **must not hard-code a specific model id or HF
   repo**. Repos are renamed, gated, or deleted; a message pointing at one rots,
   and baking a vendor's id into error paths couples Athena to that vendor. The
   requirement is stable; the repo that satisfies it is the operator's choice.
   (Historical incident examples in *docs* may name repos for context; **code**
   must not.)

### Rejected / deferred

- **Replace the three detectors with one monolith** — rejected; keep the
  focused, unit-pinned detectors and layer `ModelSupport` over them.
- **Numeric/forward validation in preflight** — out of scope; that needs a real
  load + run (gated tests), not a config fetch.
- **Auto-pulling a working alternative** (e.g. silently swap nvidia→mlx-community
  Parakeet) — rejected; surface the correct id, let the operator pull it.
- **Extending `convert` to quantize ASR** — rejected; ASR/diarization/speaker
  models serve in source precision via their module paths, like embedders.

## Consequences

- Passive-oracle preserved: the preflight reuses the **config-only** HF fetch
  (`matching: ["config.json", …]`) that `convert` already performs — no new
  outbound surface.
- All detection logic stays **MLX-free and unit-pinned** (ADR 008/009); the
  cross-modality classification matrix is pure-Swift tests.
- The operator gains a real answer to "will this work before I pull it?" across
  **every** modality, and the three failure modes from the M76 incident
  (convert mis-route, loader 500, no preflight) collapse to one shared predicate.
- `athena verify` (integrity of stored models) and `ModelSupport` (support of
  any id, pre- or post-pull) are complementary, not overlapping.

Plan + slices: `docs/model-support-preflight-plan.md`.

## Amendment (2026-07-03, usability audit §4) — fused-MTP weight-index probe

The typed model listing (`athena ls` TYPE column) needs to distinguish a Qwen3.5 `-mtp` build (fused Multi-Token-Prediction head) from a stock same-family checkpoint. This is an *attribute of a servable LLM*, not a modality, and it is the one signal `ModelSupport` cannot derive from config: real `-mtp` converts drop `mtp_num_hidden_layers` from config.json entirely, so **the safetensors weight index is the only reliable detector** (the `mtp.` substring matches the real `language_model.mtp.*` prefix). So `MTPCheckpoint.hasFusedMTP` (relocated to AthenaCore from the load-time detector) reads the weight index — **one deliberate step past the "config-only" honesty boundary of decision 4** — restricted by its caller to the `.llm` modality so no non-generative architecture is ever probed. It stays pure filesystem I/O (no MLX, no model load), remains unit-pinned (both `-mtp` and stock fixtures), and does not change any load/convert/preflight verdict — it only annotates a listing. Separately, drafter detection (`isMTPDrafterType`) is widened from an exact-match list to the `<family>_assistant` suffix convention, so the real `gemma4_unified_assistant` converts classify as `draft` (never `llm`). `fused_mtp` is surfaced additively on `ModelEntry`; the OpenAI `/v1/models` surface is byte-unchanged.
