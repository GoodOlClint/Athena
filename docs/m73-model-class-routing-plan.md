# M73 — model-class-aware convert + cause-naming load errors

## Problem

`athena convert` fails on embedding models with raw substrate errors:

```
athena convert google/embeddinggemma-300m
  → keyNotFound(path: ["model","norm","weight"], modules: ["Gemma3TextModel", …])
athena convert mixedbread-ai/mxbai-embed-large-v1
  → unsupportedModelType("bert")
```

Neither model is actually unsupported. The substrate Athena rides has **three**
model-class factories, each with its own `model_type` registry:

| Factory | Class | Registry |
|---|---|---|
| `LLMModelFactory` | generative / causal LM (~50 archs) | `LLMTypeRegistry` |
| `VLMModelFactory` | vision-language | `VLMTypeRegistry` |
| `EmbedderModelFactory` | **embedding** | `EmbedderTypeRegistry` |

`EmbedderTypeRegistry` already handles `bert`, `roberta`, `xlm-roberta`,
`distilbert`, `nomic_bert`, `qwen3`, and `gemma3`/`gemma3_text`/`gemma3n`
(→ `EmbeddingGemma`). It even ships `mixedbread-ai/mxbai-embed-large-v1` as a
named default. Athena's **serve** path already loads through it
([`MLXEmbeddingModule.swift:183`](../Sources/AthenaEmbedding/MLXEmbeddingModule.swift#L183)).

`athena convert` is the only path that is class-blind. It hand-routes just two
of the three factories — [`ModelConvert.swift:92-98`](../Sources/AthenaLLM/ModelConvert.swift#L92-L98)
picks `isVision ? VLMModelFactory : loadModelContainer` (the LLM factory) — so:

- **bert** → LLM factory has no `bert` → `unsupportedModelType("bert")`.
- **embeddinggemma** (`gemma3_text`) → LLM factory *does* have `gemma3_text`,
  so it builds the **causal-LM** Gemma3 arch, then fails on a missing
  `model.norm.weight` an embedding checkpoint doesn't carry. This is the worse
  failure: a silent wrong-arch pick, surfaced as an opaque `keyNotFound`.

## Decision (operator-confirmed)

**Redirect, don't convert.** Embedding models do not need `convert` — the serve
path loads them in source precision directly from the store. `convert` is a
*generative*-model quantization pipeline. So:

1. `convert` **detects** an embedding model and refuses it with a cause-naming
   4xx that points at the right verb (`athena pull` + set-as-embedding-model),
   instead of mis-routing to the LLM factory and dumping a substrate trace.
2. The general load/convert error path **names the class** it detected and what
   to do, never a raw `unsupportedModelType` / `keyNotFound`.

Quantizing embedders through a third convert route is **out of scope** (the
deferred alternative in ADR 016).

## Scope

### 1. Shared model-class detector (`AthenaLLM`, MLX-free)

New value type — call it `ModelClass` — derived purely from on-disk metadata,
no MLX / Metal dependency so it is unit-testable under `swift test` (ADR 009):

```
enum ModelClass { case generative, vision, embedding, unknown }
```

Detection inputs (extend `ModelConfigInfo`, which already reads `config.json`):

- **vision** — top-level `vision_config` present (existing `hasVisionConfig`).
- **embedding** — sentence-transformers artifacts in the snapshot
  (`modules.json`, `config_sentence_transformers.json`, or a `1_Pooling/`
  directory) **or** a `model_type` claimed *only* by `EmbedderTypeRegistry`
  (`bert`/`roberta`/`xlm-roberta`/`distilbert`/`nomic_bert`).
- **generative** — otherwise (any `model_type` the LLM factory claims).
- **unknown** — no factory claims it.

The sentence-transformers artifact probe is what disambiguates the overlap:
`gemma3_text` and `qwen3` live in *both* the LLM and embedder registries; the
pooling artifacts say "this checkpoint is meant to be an embedder."

### 2. `convert` refuses embedders early + fast

In `ModelConvert.convert`, after metadata is available, branch on `ModelClass`:

- `.embedding` → throw a cause-naming error (e.g. `AthenaError.moduleLoadFailed`
  or a dedicated case) whose message is actionable:
  > `google/embeddinggemma-300m` is an embedding model — `convert` only handles
  > generative/vision models. Run `athena pull google/embeddinggemma-300m`,
  > then set it as the embedding model (`--embedding-model` / config).
- `.vision` → existing VLM route (unchanged).
- `.generative` → existing LLM route (unchanged).
- `.unknown` → cause-naming error naming the unrecognized `model_type`.

**Fail-fast:** today `convert` downloads the whole repo *before* the load
fails. Pre-fetch `config.json` (+ the sentence-transformers metadata files)
and run the class check **before** pulling weights, so an embedding model is
rejected in seconds, not after a multi-GB download. (The detector only needs
those small files.)

### 3. Cause-naming instead of raw substrate errors

`Convert.run()` currently prints `error: convert failed — \(error)`
([`Convert.swift:90`](../Sources/athena/Commands/Convert.swift#L90)) — a raw
dump. Surface the detector's friendly message. If a generative/vision load
still throws `unsupportedModelType`/`keyNotFound` after class routing, wrap it
with the detected class and a one-line hint rather than the substrate trace.

### 4. Tests (ADR 009 pure-Swift tier)

Pin the detector decision table under `swift test`:

| Fixture (config.json + artifacts) | Expected |
|---|---|
| `model_type: bert` | `.embedding` |
| `gemma3_text` + `1_Pooling/` + `modules.json` | `.embedding` |
| `gemma3_text`, no ST artifacts | `.generative` |
| top-level `vision_config` | `.vision` |
| `qwen3_5` | `.generative` |
| unknown `model_type`, no artifacts | `.unknown` |

The detector is MLX-free, so this is a real numeric-free decision test, not a
stub-tier seam.

## Out of scope / deferred

- **Quantizing embedders via a third convert route** — ADR 016 records this as
  the rejected alternative; revisit only if a consumer needs quantized
  embedding checkpoints in the store.
- **Auto-pull on convert-redirect** — convert stays operator-action; it tells
  the user to pull, it doesn't pull for them.

## Validation (end-to-end)

1. `athena convert mixedbread-ai/mxbai-embed-large-v1` → fast cause-naming
   redirect, **no** multi-GB download, exit non-zero.
2. `athena convert google/embeddinggemma-300m` → same.
3. `athena pull mixedbread-ai/mxbai-embed-large-v1`, set as embedding model,
   `POST /v1/embeddings` → 200 with the expected dimension (confirms the
   substrate embedder route Athena already ships actually serves it).
4. A generative convert (e.g. an `-it` LLM) still works byte-for-byte (no
   regression to the LLM/VLM routes).

## Answer to "will we always chase models?"

Two axes, different answers:

- **Class routing (LLM/VLM/Embedder)** — finite and ours. This plan closes the
  gap; it does not grow with the model count.
- **Architecture (`model_type` → forward pass)** — inherently Swift code, but
  it lives **upstream** in the substrate. Athena inherits its ~50 LLM archs +
  the embedder registry for free as the pin is bumped. Athena's own
  `SupportedModels` set is a *validated label*, not a gate — best-effort loads
  regardless. A genuinely novel architecture the substrate lacks still needs
  code; that is the signed-arch-plugin future workstream, not this milestone.
