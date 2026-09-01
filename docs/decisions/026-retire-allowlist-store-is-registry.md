# ADR 026 — retire the model allowlist; the model store is the registry

**Status:** **Accepted — SHIPPED v0.10.202** (M80). The `model_allowlist` table, `/api/models/allow*` routes, WebUI mirrors, and the `athena allowlist` CLI are removed; selection is store-backed via `AthenaCore/ModelSelection.swift` (store-presence + class match; omit-model + >1 candidate ⇒ 400 `ambiguous_model`) and the per-module default lives in TOML (`athena default --module M <id>`). Plan: `docs/collapse-persistence-plan.md` S7. **Supersedes the allowlist portion of M42.** Part of the ADR-025 persistence cleanup; motivated by the operator's decision that a separate curated allowlist is redundant — *"the allow list should really just be which models are pulled into the store."*

## Context

The `model_allowlist` SQLite table (M42) currently does **three** jobs:

1. **The selectable set** per module class (llm / embedding / transcription / diarization / speaker-embedding) — a curated subset of the on-disk models.
2. **The per-module default** (`is_default`) used when a request omits `model`.
3. **The validation gate** — a requested `model` not in the allowlist → 400 `model_not_available` (the request path never downloads).

Two facts make the allowlist redundant:

- **The model store is already an enumerable registry.** `ModelStoreOps.list()` is a filesystem scan of the store dir (dirs with a `config.json`), independent of the allowlist — it backs `athena list`/`ls` today. The allowlist is a second, hand-maintained set layered on top.
- **ADR 021 (`ModelSupport`) already classifies each model dir by modality** from config-only metadata. So "available embedding models" = store dirs that classify as embedding — derivable, not something that needs its own table.

So jobs (1) and (3) are answerable from the store + `ModelSupport` with no persistence. Only job (2), the default, needs a home — and a config-based default **already partly exists**: `athena default` writes a TOML `model` key for the LLM (it just isn't generalized to the other classes).

Beyond redundancy, the allowlist table is a **persistence surface** that blocks ADR 025's stateless-loopback goal: as long as the allowlist lives in SQLite, the DB must exist even in loopback/no-auth mode. Removing it lets the DB be needed **only** for auth/audit/usage — so loopback can run with no DB at all.

## Decision

**Retire the allowlist. Availability = model-store contents (classified by `ModelSupport`); the per-module default lives in config.**

1. **Remove the allowlist entirely:** the `model_allowlist` table + all `AthenaStore` methods (`listModelAllowlist`/`defaultModelAllowlistID`/ `addModelAllowlist`/`removeModelAllowlist`/`setModelAllowlistDefault`/ `modelAllowlistCount`), the `/api/models/allow` GET/POST/DELETE + `/api/models/allow/default` PUT routes + handlers + WebUI mirrors (`/ui/allowlist`, `/ui/api/allowlist*`), the `athena allowlist` CLI **including the M43.5 offline `--data-dir` editing**, the allowlist DTOs, the `model.allow.{add,rm,default}` audit actions, and the allowlist tests/e2e/docs.

2. **Selectable set = store contents of the matching class.** Each module's `allowedModelIds()` returns the store dirs that `ModelSupport` classifies as that module's modality, replacing the DB-fed in-memory `allowedIds`. (Used for `/v1/models`, error messages, and the gate.)

3. **Validation gate = store presence + class match.** A requested `model` is resolved against the store via the existing case-insensitive basename matcher (`ModelAllowlist.swift`'s `canonicalByStoreIdentity`, retained as a store helper); absent or wrong-class → **400 `model_not_available`**. The request path still **never downloads** (unchanged invariant — only operator `pull` fetches).

4. **Per-module default in TOML.** Generalize `athena default` to `athena default --module M <name>`, writing a per-class default config key (LLM keeps the existing `model` key; embedding/transcription/diarization/ speaker get their own). The `--llm-model`/`--embedding-model`/… `load` flags become **first-boot config-default seeds** (write the default into TOML on install/init), not allowlist seeds. `athena init` aux-pull still pulls the configured defaults.

5. **Ambiguity → explicit 400.** When a request omits `model`: exactly one model of the class in the store ⇒ use it; a configured default ⇒ use it; **more than one and no configured default ⇒ 400 `ambiguous_model`** (cause-naming, tells the caller to specify `model=`); zero of the class ⇒ `model_not_available`. No silent auto-pick.

6. **M41 lifecycle stays, re-pointed at the store.** `/api/models/{load,unload, resident}` and inference-time rebind remain — they now select among store models (gated by #3), and the `model.rebind` audit + governor reconcile on rebind are unchanged.

### Honesty boundary / invariants preserved

- **No on-request download** — unchanged; the request path resolves against the store and 400s on a miss.
- **Case-insensitive store-identity matching** (basename, `Qwen/X` ≡ `X` ≡ `x`) — retained, applied to store entries instead of allowlist rows.
- **What changes is bookkeeping, not numeric behavior** — the same model loads and serves; only "which models are eligible / which is default" moves from a table to (store + config). Decision logic stays MLX-free + unit-pinned (ADR 008/009).

### Supersedes / amends

- **M42 (allowlist → SQLite + CRUD + live refresh + M43.5 offline editing)** — retired. The offline equivalent is now editing the TOML default + `pull`/`rm` of store models; no `--data-dir` allowlist surface.
- **M39 / M41** — the model-selection *gate* changes from allowlist-membership to store-presence; the single rebindable slot and lifecycle endpoints are otherwise unchanged.
- **ADR 025** — its allowlist treatment (decision #5: "config-seeded; SQLite when DB exists") is replaced by this ADR's full removal; ADR 025's stateless-loopback mode becomes unconditional (no allowlist reason to open the DB).
- **ADR 013** — model-selection semantics reference updated.
- **ADR 021 (`ModelSupport`)** — gains a second consumer (store-class enumeration), reinforcing the one-predicate-many-consumers intent.

### Rejected / deferred

- **Keep the allowlist as a default-off toggle** — rejected; it is redundant with the store and remains a persistence surface (same reasoning as ADR 025's delete-not-disable).
- **Deterministic auto-pick on ambiguity** — rejected (operator chose explicit 400); a served model that silently changes as the store changes is a footgun.
- **A single `[defaults]` config section** — not chosen; per-module keys (generalizing the existing `model` key) keep the precedent and are simpler to edit.

## Consequences

- Breaking removal of the `/api/models/allow*` surface, the `athena allowlist` CLI, and the WebUI allowlist console. No current consumer depends on them (operator-confirmed).
- The model-selection mental model simplifies to **"pulled = available; config names the default; otherwise say which one."**
- Advances ADR 025: the SQLite DB is now needed only for auth/audit/usage, so loopback/no-auth mode creates **no DB** unconditionally.
- `ModelSupport` (ADR 021) becomes load-bearing for availability — a model that classifies as `unsupported` is simply not offered, with the existing cause-naming refusal.

### Validation (on implementation)

- MLX-free decision logic — store-class enumeration, default resolution, the ambiguity → 400 rule — unit-pinned (ADR 008/009).
- e2e: pull two LLMs, omit `model` ⇒ 400 `ambiguous_model`; `athena default --module llm <name>` ⇒ that model served; `rm` a model then request it ⇒ 400 `model_not_available`; a loopback run creates no `athena.sqlite`.
- Regression: case-insensitive/basename resolution still matches (the `ModelAllowlistTests` coverage moves to the store-identity helper).

Plan + slices: `docs/collapse-persistence-plan.md` (on approval).
