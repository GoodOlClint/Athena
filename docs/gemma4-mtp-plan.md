# Gemma 4 MTP speculative decoding — change plan (M83)

**Status:** APPROVED — **IMPLEMENTED + E2E VERIFIED**. **S1 v0.10.227**, **S2–S5
v0.10.228**. Full logic suite green (761/0). Pairs with **ADR 032**.

### End-to-end DoD — all 3 Gemma 4 families tested (2026-06-30, Release binary)

| Family | Target ↔ drafter | MTP result |
|---|---|---|
| **Dense** | `gemma-4-31b-it-8bit` ↔ `…-31B-it-assistant-bf16` | **✓ PASS** — engaged 46/46, passthrough=none, byte-identical-to-greedy in the 64-tok window, **1.52×** speedup |
| **MoE** | `gemma-4-26b-a4b-it-4bit` ↔ `…-26B-A4B-it-assistant-bf16` | **✓ PASS** — engaged 46/46, passthrough=none, coherent (speedup measure-per-use; NOTES MoE-batch-1 caveat stands) |
| **Effective (E-series)** | `gemma-4-e4b-it-4bit`, `gemma-4-e2b-it-4bit` | **✓ PASS** (on substrate `integration`) — E4B engaged 46/46 byte-identical-greedy, E2B engaged 34/34 |

**All three Gemma 4 families validated.** The drafter auto-paired + loaded from
the seeded map in every case; the `generate(mtpDrafter:)` drive is
target-agnostic.

The E-series needed **both** substrate fixes, present together only on
`integration`: `4acd179` (VLM Gemma4 KV-sharing target load — from Athena's
handoff) **and** `0bec134` (E-series drafter centroid masked-embedder forward).
Earlier E-series failures were a **wrong-branch** artifact — Athena's `Package.swift`
path-dep builds whatever the substrate has checked out, and we'd tested
`pr/gemma4-vlm-kv-sharing` (KV fix only). Rebuilt against `integration` → all pass.
Both earlier "E-series blocker" reports
(`~/Source/mlx/research/gemma4-eseries-*`) are resolved/false-alarm.

### E-series retest (2026-06-30, against substrate `4acd179`)

The KV-sharing target fix landed — **E-series targets now load** (E4B 985 ms). But
the first speculative decode **aborts the daemon** at a *second* substrate gap: the
E-series drafter's centroid LM head `Gemma4AssistantMaskedEmbedder.callAsFunction`
is an unimplemented `fatalError` (both E-series drafters ship
`use_ordered_embeddings=true`, `num_centroids=2048`; dense/MoE drafters use the
tied-`lm_head` path, which is why they work). Uncatchable from Athena (it's a
`fatalError` on the MLX worker), so the asks are upstream: implement the forward
(port mlx-vlm `masked_embedder.py`) **and** `throw` instead of `fatalError` so
consumers degrade. **UPDATE: this was a wrong-branch artifact** — the centroid
forward IS implemented on `integration` (`0bec134`); rebuilding Athena against
`integration` made E-series MTP pass (see the matrix above). Handoff retracted.

### E-series target-load blocker — root cause (FIXED upstream `4acd179`)

The E-series **target** fails to load — `speculative=false` fails identically, so
it is upstream of any MTP code (in the target `container.prepare`). Precise cause:

- E-series checkpoints use **KV-layer-sharing** (`num_kv_shared_layers`): the last
  N layers reuse an earlier layer's K/V and own no `k_proj`/`v_proj`. The failing
  layer index **exactly equals `num_layers − num_kv_shared_layers`**: E2B 35−20=**15**,
  E4B 42−18=**24** (observed `keyNotFound …layers.{15,24}.self_attn.k_proj.weight`).
- E-series carry `vision_config`, so Athena (correctly, ADR 010/012) routes them
  through `VLMModelFactory`. The substrate's **text-only** `MLXLLM/Gemma4Text.swift`
  **does** implement KV-sharing (`Gemma4SharedKVState`), but the **vision**
  `MLXVLM/Models/Gemma4.swift` backbone's weight loading does **not** — it builds a
  `k_proj` at the shared layers and demands a weight that the shared-layer model
  shouldn't have. Dense/MoE (`num_kv_shared_layers = 0`) are unaffected.
- **Fix is upstream** (the substrate VLM Gemma4 KV-sharing loader; one of the "few
  remaining PRs"). Athena's MTP wiring needs no change — it loaded the E-series
  *drafter* fine and will light up for E-series the moment the substrate loads the
  E-series *target*. A possible Athena-side stopgap (force the E-series down the
  text-only path that already handles KV-sharing) trades away the vision tower —
  an operator decision, not done here.

## Goal

Light up the per-request `speculative` knob for the **Gemma 4** family, which
currently has **no** speculative path (DFlash removed in ADR 028; Athena's MTP
path is Qwen-only). Lossless, opt-in, default-off — same guarantees as the
existing Qwen3.5 MTP path, but driven through the substrate's Gemma 4 MTP
machinery instead of Athena's own loop.

## Why this is NOT a parallel pipeline (CLAUDE.md "no parallel impl" rule)

Gemma 4 MTP is a **structurally different drafter mechanism** than Qwen3.5 MTP,
and the substrate — not Athena — owns its decode loop. So this is a **second
backend behind one knob**, precedent: ADR 018 (multi-backend diarization), ADR
020 (multi-backend transcription). It is not a second implementation of the
same pipeline.

| | Qwen3.5 MTP (Athena today) | Gemma 4 MTP (this change) |
|---|---|---|
| Drafter | `mtp.*` head **fused in the target checkpoint** | **separate `gemma4_assistant` model** |
| Decode loop | Athena `SpeculativeGeneration.swift` (vendored model) | substrate `MTPSpeculativeTokenIterator` |
| Hidden-state seam | Athena `logitsAndHidden` | substrate `emitDrafterState` |
| Athena's job | owns the whole loop | **load+pair drafter, drive the substrate iterator** |

## What the substrate already provides (verified, local fork)

- `MTPDrafterModelFactory.shared` — loads the drafter only, returns
  `MTPDrafterContainer`.
- `Gemma4AssistantDraftModel` (model_type `gemma4_assistant`) +
  `Gemma4AssistantRegistration.register()`.
- Capture seam: target emits post-final-norm hidden + per-layer shared KV via
  `mtpLastHiddenStatesKey` / `mtpSharedKVStatesKey`, gated by `emitDrafterState`.
- Ready-made loop: `MTPSpeculativeTokenIterator` (conforms
  `TokenIteratorProtocol`) + `generateLoopTask`, with auto-fallback to
  single-token "passthrough" if the target stops emitting drafter state.
- Telemetry: `MTPStatsCollecting` (roundCount, proposed/accepted, passthrough
  reason).
- One TODO (`Gemma4AssistantMaskedEmbedder`, `use_ordered_embeddings=true`):
  unused by current checkpoints (`false`), **not on our path**.

Athena already drives substrate generation via the substrate `TokenIterator`
([GuidedSubstrate.swift:81](GuidedSubstrate.swift), `container.generate`), so
the MTP iterator slots into the existing pattern — no hand-written loop.

## Precondition — CLEARED (2026-06-30)

Substrate ready on `integration` (HEAD `2c49472`, includes the Gemma 4 E-series
centroid-embedder merge — the earlier masked-embedder TODO is **resolved**, so
E2B/E4B work too). Verified: Athena already `import MLXVLM`
([MLXLLMModule.swift](../Sources/AthenaLLM/MLXLLMModule.swift)) so
`gemma4_assistant` self-registers; the `generate(… mtpDrafter:)` overload exists
([Evaluate.swift:1681]). No `Package.swift` change. Handoff:
[mtp-gemma4-handoff.md](mtp-gemma4-handoff.md). **Execution unblocked.**

## Operator decisions (from the gate interview)

1. **Drafter pairing = explicit `mtp_drafter` key (ground truth) + a seeded,
   operator-overridable default-drafter map (the happy path).** True auto-detect
   from the checkpoint is **impossible**: the Gemma 4 target `config.json` carries
   **no** drafter-advertising field (unlike Qwen's `mtp_num_hidden_layers`), and
   the drafter is a **separate HF repo**, not a subdir (verified: target
   `mlx-community/gemma-4-31b-it-8bit` ↔ drafter
   `mlx-community/gemma-4-31B-it-assistant-bf16`). So the "smartness" is a shipped
   map, not metadata sniffing — see §Drafter sourcing.
2. **Scope = all Gemma4 uniformly** (dense + 26B-A4B MoE), no per-variant gate.
   Honesty boundary: lossless always; **speedup not guaranteed** at batch-1 on
   the MoE variant (NOTES caveat) — measured, never claimed.
3. **Sequencing = block on the pending substrate PRs** (above).

## Decision (proposed — ADR 032 draft)

Wire Gemma 4 MTP as a **second speculative-decode backend behind the existing
`speculative` knob**, driven by the substrate's `MTPSpeculativeTokenIterator`.
The drafter is a **separate `gemma4_assistant` store model**, paired to a target
via an explicit `mtp_drafter` key **plus a seeded, operator-overridable
default-drafter map** (§Drafter sourcing) since the checkpoint cannot
self-advertise. Two resident models are accounted on the one Metal budget (ADR
011/023); the iterator drives target+drafter inside a single inference-execution
span (ADR 029). Default-off, opt-in, lossless.

**Rejected alternatives:** (a) vendor the drafter into Athena — rejected,
substrate-first per ADR 028 and the NOTES direction; (b) config-metadata
auto-detect — **impossible**, the Gemma4 target config carries no drafter field
(a real packaging fact, not a preference); (c) naming-convention derivation
(`<target>-assistant-*`) — rejected, the casing/dtype suffixes don't line up
(`31b-…-8bit` vs `31B-…-assistant-bf16`), so derivation guesses and mis-pulls;
(d) hard-coded map in code — rejected per ADR 021 (ships as config data instead);
(e) unify the two speculative loops — rejected, different drafter mechanisms and
the substrate owns Gemma's loop (multi-backend, not parallel pipeline).

## Drafter sourcing (pairing)

The checkpoint cannot self-advertise its drafter (facts above), so pairing is a
layered lookup — first hit wins:

1. **`mtp_drafter` config key** (per-target → drafter store id). Explicit, always
   authoritative, ADR-026 consistent.
2. **Seeded default-drafter map** — a shipped, **operator-overridable** TOML data
   file mapping known Gemma4 targets → their published drafter repo ids (sourced
   from Google's MTP docs / mlx-community). Auto-pairs the common case with no
   manual wiring; the operator can edit/extend it when repos move.
3. **Nothing matched** ⇒ knob inert (single-token), no error.

`athena pull <gemma4-target>` consults the map and, if a known drafter exists,
**reports it** and fetches both only under `--with-drafter` (no silent multi-GB
double-download — passive-oracle / operator-control posture). `pull --check`
surfaces the pairing without downloading.

### ADR 021 conflict (surfaced + resolved)

ADR 021's binding rule: *code must not hard-code a model id / HF repo (repos
move/vanish); docs may cite them, code must not.* A target→drafter map **is**
enumerated repo ids. **Resolution:** ship it as **seeded config data, not Swift
constants** — the same first-boot-seed pattern ADR 026 uses for default model
ids. The operator owns and can fix it without a recompile, which honors the
rule's intent (the prohibition targets rot-prone *code* strings and
*error/guidance* text, not operator-editable config seeds). Error/guidance
strings still name the structural requirement (e.g. "no drafter paired"), never
a repo id. **This refines ADR 026's per-module-key pattern; record it in ADR 032.**

## Surface (after the change)

- **No new route.** `/v1/chat/completions` `speculative` knob now also engages
  for Gemma4 **when a paired drafter is loaded**; inert (single-token) otherwise
  — same "inert when no head" semantics the knob has today on Qwen.
- New config (TOML, 5-touchpoint pattern): `mtp_drafter` (per-target pairing →
  drafter store id), `mtp_block_size` (draft block length; default from the
  drafter config), and a seeded **default-drafter map** (overridable data file).
  No pairing resolved ⇒ knob inert.
- `athena pull` gains `--with-drafter` (fetch the mapped drafter alongside the
  target; default does not silently double the download); `pull --check` reports
  the paired drafter.
- `OpenAPISpec.swift` `speculative` description updated to state Gemma4 needs a
  paired drafter. `SupportedModels.describe` advertises "MTP speculative (paired
  drafter)" for Gemma4 when one is configured.

## Slices (stacked, test-pinned — each a commit + semantic tag)

**S1 — classification + config + pairing map (MLX-free, can start pre-substrate).**
Add a `gemma4_assistant` **drafter modality** to `ModelSupport` (ADR 021) so pull
recognizes it and `convert` redirects it (it is not a generative target). Add the
`mtp_drafter` / `mtp_block_size` config keys, the **seeded default-drafter map**
(operator-overridable TOML data — NOT code constants, per ADR 021), and a pure
pairing-resolution function (`mtp_drafter` key → map → none). `athena pull`
gains `--with-drafter` and a `--check`/preflight line reporting the paired
drafter. Unit-pinned (ADR 008/009): classification, pairing resolution (key
overrides map; unknown target ⇒ none), eligibility predicate.

**S2 — load + pair + governor.** When a Gemma4 target loads with a resolved,
present drafter and speculative is enabled, load the drafter via
`MTPDrafterModelFactory.shared.load(...)` → `container.context.model` (an
`any MTPDrafterModel`) into a second resident slot. The drafter holds **no
target-derived state** (handoff §loading) → load once, reuse across requests.
Governor admission (ADR 023) counts both footprints; measure the drafter's
per-model footprint (G3) — `athena ps`/healthz stays honest. Inference-execution
gate (ADR 029): one gated op (the overload drives target+drafter sequentially).

**S3 — drive the substrate `generate(mtpDrafter:)` overload.** In `MLXLLMModule`
(inside `container.perform`), eligible Gemma4 requests call the substrate
`generate(input:cache:parameters:context:mtpDrafter:blockSize:)` overload
([Evaluate.swift:1681]) — **not** a hand-built iterator (substrate wraps it) —
consumed through the existing streaming/metering path. `blockSize` from
`mtp_block_size` (default 4). Eligibility: target is gemma4 family **and** a
drafter is paired+loaded **and** `speculative` resolves true. Greedy (temp 0) and
sampling (temp>0) both supported; correctness preserved via target-verify.
(Byte-identity to non-spec is bounded ~64 tokens by an MLX fused-SDPA quirk —
handoff §limits; the speculation-correctness guarantee still holds.)

**S4 — stats.** Bridge the substrate's `MTPStatsCollecting` → Athena's
`SpeculativeStats` observer (acceptance rate, passthrough reason) so Gemma4 MTP
is observable like Qwen.

**S5 — surface + docs.** `OpenAPISpec.swift` + `SupportedModels.describe`
updates; `docs/speculative-decoding.md` (or existing) documents the
target↔drafter pairing and which checkpoints carry a drafter. Formalize ADR 032
+ CLAUDE.md back-link in S1's commit window.

## Test bar / Definition of Done

- **MLX-free decision logic unit-pinned** (ADR 008/009): drafter classification,
  pairing resolution, eligibility — fail before, pass after, in `./deploy/test.sh`.
- **Heavy gated correctness (the real bar):** on a real Gemma4 target + its
  matching-size `gemma4_assistant` drafter (start with the E4B pair, ~78 MB
  drafter), `speculative=true` yields `proposedDraftTokens>0`,
  `acceptedDraftTokens>0`, `passthroughReason==nil`, coherent text, **and a
  measured tok/s speedup** vs `speculative=false`. Byte-identity to non-spec for
  the first ~64 tokens (the bounded MLX-SDPA window). Mirror the substrate's
  gated `MTPIteratorEndToEndDiagnosticTests.testMTPE4BPairProducesAcceptedDrafts`
  (`TEST_E4B_PAIR`).
- **Negative:** Gemma4 target with **no** paired drafter + `speculative=true` ⇒
  clean single-token decode, no error, telemetry shows inert.
- e2e RBAC green; pre-commit pipeline (Tests → Security → Quality → Refactor).

## Inherited limits (substrate; document, don't fight)

bf16 drafters only (no quantized drafter); single-stream (no B>1 MTP);
mid-stream KV-quant onset → transparent single-token passthrough
(`passthroughReason` set, correctness kept, speedup dropped) — so don't pair MTP
with mid-stream KV-quant when chasing the speedup. Pair each target with its
**matching-size** drafter (shared K/V geometry; mismatched pairs fail).

## Out of scope

- DSpark / non-MTP-target speculative decoding (separate NOTES item).
- Quantized drafters, batched MTP (substrate limits above).
- Any change to the Qwen3.5 MTP path (byte-unchanged).
