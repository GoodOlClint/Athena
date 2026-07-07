# ADR 040 — Publish Athena as a personal-identity public monorepo

- **Status:** Accepted
- **Date:** 2026-07-07
- **Deciders:** operator + agent
- **Context source:** 2026-07-07 publication brainstorm session (decision record kept outside the repo — it names scrub terms; internal-repo bound); execution plan `docs/publication-plan.md`

## Context

The repo is going public. Two questions were entangled: (1) publication readiness — the tree and its 548-commit history contain private consumer-project references (35 tracked files, 10 shipping source files, 10 ADRs, one doc filename), no LICENSE, ~52 internal work-log docs, a private dependency (`AppleSiliconMetrics`), and an ad-hoc version at 0.10.263; (2) whether Linux support motivates splitting the monorepo into core / macOS / Linux / client repos. Coupling analysis showed the split fights the codebase: MLX itself is not macOS-only (mlx-swift builds Linux+CUDA upstream with the same `MLX.Memory.*` API the governor drives), so "Athena Linux" is the same daemon behind thin platform seams (media decode, logging sink, service manager, probes, hardening — see `docs/cuda-port-audit-2026-07-04.md`), while the core/macOS boundary does not exist in code (the HTTP handler layer in the executable imports the MLX modules directly). Identity was also unsettled: no LLC exists yet, an Apple organization account requires one, and the parent-project name is a working title for a future *consumer* of Athena, not this product.

## Decision

- **Repo shape: monorepo, no split.** Linux lands later as platform seams (conditional compilation / platform targets) inside this repo, gated on the CUDA-audit spike. No OS-based repo split, ever, in the current architecture; a client split only if the client earns an external audience.
- **Identity: personal, entity-free.** Publish under `GoodOlClint`; individual Apple Developer account; Developer ID signing under the operator's name; bundle id stays `me.goodolclint.athena`. The parent-project name is dropped from Athena entirely — it is a consumer reference, and consumers are never referenced (existing discipline, now generalized). `AppleSiliconMetrics` transfers to the personal account and flips public.
- **History: filter-repo string-scrub, timestamps untouched, filter → soak → flip.** The rewritten history (full 548-commit ledger, tags preserved) goes to a new private repo, soaks under normal development with automated scrub verification, then flips public. The original repo is archived read-only and never pushed again.
- **Development: public-primary after the flip**, with an explicit AI-authorship disclosure in the README (autonomous Claude agents under human direction; operator reviews/gates/approves). The disclosure is accurate and makes the commit-timestamp distribution self-explaining.
- **License: Apache-2.0** with a NOTICE file carrying the attribution pass for vendored/ported code.
- **Versioning: re-baseline to v0.11.0 at the flip; versions are release events** (semver: patch = fixes, minor = features, major/1.0 operator-reserved). The bump-every-slice ritual retires at the flip.
- **Docs: ADRs stay in the public repo** (scrubbed), joined by a retrospective `000-pre-adr-history.md` reconstructing the 2026-05-16 → 2026-06-11 pre-ADR decisions; plans/audits/handoffs/research move to a private internal repo.
- **Distribution: signed + notarized artifacts via a GitHub Actions release pipeline** (tag → build → test → sign → notarize → release). Nothing may hard-pin the Apple Team ID in future self-update verification (keeps the later individual→LLC account conversion non-breaking).

## Rejected alternatives

- **Four-repo split (core/macOS/Linux/client):** the core/macOS seam doesn't exist in code yet; an OS-split repo would institutionalize a parallel implementation of the canonical pipelines (a defect per house rules) for a per-OS delta that is a thin seam layer; multi-repo pinning tax already bit once (silent path-dep branch drift, 2026-07-02).
- **Fresh squashed public history:** was the earlier recommendation while commit timestamps looked like a liability; superseded by the AI-authorship disclosure, which makes the real ledger an asset. The scrub burden moved from "once at the cut" to "verified across all revisions," accepted.
- **Rewriting commit timestamps** (to move commits out of business hours): weak protection, detectable, and manufactures a false record; the disclosure answers the same concern honestly.
- **Org-placeholder identity now** (a placeholder GitHub org + org-domain bundle id): the LLC is neither named nor filed; a placeholder would force a second breaking rename. Personal identity is coherent with the individual signing cert; one coordinated migration happens if/when an entity exists.
- **Private-primary with a public release mirror:** solves timestamp optics but forfeits community development; unnecessary once the disclosure exists.
- **MIT license:** no patent grant; Apache-2.0 also matches the ported NeMo/Apache upstreams and its NOTICE mechanism is the attribution vehicle anyway.
- **Moving ADRs to a private repo:** hides the strongest public evidence of the agents-build/human-governs process and breaks the same-edit-window ADR discipline.

## Consequences

- Execution follows `docs/publication-plan.md` (P0 prerequisites → tree cleanup → filter/soak/flip → re-baseline → release pipeline), with operator gates at every repo-mutating and publishing step.
- The scrub list is maintained only in untracked scratch; its terms never appear in tracked files, commits, or tags. The parent-project name is now a forbidden reference in this repo, same as all consumer names.
- After the flip: versions are release events (CLAUDE.md and the release-workflow memory must be updated in the same window, S8); every commit is immediately public; internal planning artifacts live in the private internal repo.
- The Linux port, when it starts, enters this repo as platform seams behind the CUDA-audit spike — any future proposal to split by OS must supersede this ADR.
- The `Athena-archive` repo is the true-history record for provenance (IP evidence) and is never pushed to again.
