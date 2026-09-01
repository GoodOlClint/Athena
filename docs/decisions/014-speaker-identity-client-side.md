# 014 — Cross-file speaker identity stays client-side; the daemon is not extended

**Status:** Accepted — decision; implementation is a client tool, not a daemon change
**Date:** 2026-06-17
**Milestone:** none (no daemon surface change; client-side workflow)
**Relates:** ADR 011 (governor-as-thesis; audio is a *tenant*), ADR 013 (audio division-of-labor: `/v1/audio/*` = analysis), ADR 006 (vector-store owner scoping).

## Context

Diarization labels (`speaker0/1/2…`) are **per-file and arbitrary** — `speaker2` in one recording is unrelated to `speaker2` in another. The operator wants **persistent speaker identity across files**: a small, mostly-fixed cast of recurring people mapped to stable names across a folder of `.m4a` recordings.

The daemon already ships every numeric primitive needed:

- `POST /v1/audio/diarizations` — Sortformer (+clustering) → per-file `{start,end,speaker}`.
- `POST /v1/audio/embeddings` — WeSpeaker ResNet34-LM → **one 256-d L2-normalized vector per requested time-range** (`segments` JSON). It accepts arbitrary segments, so diarization turns can be fed straight in.
- `POST /v1/vectors{,/query}` — owner-scoped cosine vector store.

What is missing is purely **glue**: diarize → embed each turn → average per local speaker → match against named reference voiceprints → relabel (or flag unknown). There is **no** persistent speaker/voiceprint/enrollment concept anywhere in the codebase today (grep: zero hits for `enroll`/`voiceprint`/`speaker_id`/`identity`-as-person), and adding one is a genuine fork: a documented **client-side workflow** vs. a **first-class daemon capability** (enrollment store + a server-side "identify speakers in this file" operation, new ADR + OpenAPI surface).

Two existing decisions bear directly on the fork:

- **ADR 011** — the unified Metal memory governor is the reason to exist; a capability earns in-daemon residency only when it must share the governed Metal pool. Voiceprint *storage + cosine matching* is **CPU-side bookkeeping over already-produced embeddings** — it consumes no Metal budget and multiplexes against no other tenant. It fails the governor test for absorption.
- **ADR 013** — `/v1/audio/*` is best-in-class audio *analysis*; new audio *inference* features go to `/v1` only. Speaker *embedding* (the analysis) already ships there. *Identity* (naming, persistence, thresholding policy) is an application built on top of that analysis, not a new analysis primitive — and unlike diarization output it has **no cross-vendor standard** to converge on (OpenAI ships transcript-side diarization, no voiceprint enrollment). Absorbing it would mint Athena-proprietary surface that every consumer would have to learn, against the "pay the standard, refuse the tail" posture.

## Decision

**Cross-file speaker identity is delivered as a hardened client-side workflow over the existing endpoints. The daemon is not extended — no enrollment table, no `/api/speakers`, no `/v1/audio/identifications`.**

Rationale, in one line each:

1. **It fails the governor test (ADR 011).** Matching is cosine over 256-d vectors on the CPU; it needs no Metal budget and coordinates with no tenant. A daemon that absorbs it grows maintained surface without strengthening the moat.
2. **It has no standard to honor (ADR 013).** The analysis primitive (WeSpeaker embeddings) is already canonical at `/v1/audio/embeddings`; identity policy on top is application glue, and a server-side version would be proprietary surface, i.e. tail to refuse.
3. **The scale doesn't warrant it.** A handful (≤10) of known voices → brute-force cosine over a local JSON store is trivially sufficient and outperforms the built-in vector DB for this job (see consequence below).
4. **It keeps the daemon a clean passive oracle.** Identity, naming, and threshold policy are operator concerns that change faster than the daemon should.

**Conservative-by-default matching:** a per-file speaker is named only when its centroid's cosine to a reference voiceprint clears a calibrated threshold; otherwise it is labelled `unknown-N`. The tool never emits a wrong name to avoid a missing one.

**Enrollment is hybrid:** unsupervised agglomerative clustering bootstraps a never-labelled folder (the operator names the resulting clusters), and supervised `enroll --name <clip>` pins or corrects a specific voice. Recurring `unknown` voices surface in the clustering view as enrollment candidates.

## Consequences

- **Daemon byte-unchanged.** No ADR-013 `/v1` slice, no `OpenAPISpec.swift` edit, no migration, no drift-guard churn. This ADR exists to record that the absorption was considered and **deliberately declined**, so it isn't re-litigated as a "gap."
- **The built-in vector DB is *not* used to store voiceprints** — deliberately. It is effectively single-dimension (text embeddings populate it at **2560-d**; WeSpeaker is **256-d**, and the two cannot coexist in one store), and it is owner-scoped (ADR 006). A local JSON voiceprint file sidesteps the dimension collision and the auth/owner surface entirely, and brute-force cosine over ≤10 identities is instant. If scale ever grows past ~hundreds of identities, *re-open this ADR* — that is the tripwire for reconsidering a server-side store, not before.
- **Threshold is data-dependent.** WeSpeaker cosine has no universal same-speaker cutoff; the tool ships a default plus a calibration mode and records the chosen value in the voiceprint file. Mis-calibration is an operator-tunable knob, not a daemon defect.
- **Short/overlapping turns are noisy.** The tool down-weights or drops sub-threshold- duration turns before averaging; overlap regions yield mixed-voice embeddings and are treated conservatively (they push toward `unknown`, never toward a confident wrong name).
- **The tool is not built in this repo.** It is delivered as a **self-contained handoff prompt** (`docs/speaker-identification-agent-prompt.md`) for a client-side coding agent to implement in whatever language/repo the operator chooses. Athena's tree carries only the design + the prompt — no tool code, no non-Swift artifact. This is the most literal reading of "the daemon is not extended": the spec lives here, the implementation lives client-side.
