# Athena publication plan — brownfield change plan (2026-07-07)

**Status: operator-approved plan, ready to execute.** Decisions were made in the 2026-07-07 brainstorm session; the decision record is the brainstorm doc at `~/.athena-publication/publication-and-repo-split-brainstorm.md` (§2.5 — kept outside the repo because it deliberately names scrub terms) and ADR 040. This document is self-contained: an executing session needs only this file, the brainstorm doc, and the repo.

**Goal:** take Athena public under the operator's personal identity, with a scrubbed history, Apache-2.0 license, curated docs, re-baselined versioning, and a signed/notarized release pipeline — as a monorepo (no repo split).

**Executor guardrails (binding):**
- NEVER run `git filter-repo` (or any history rewrite) against the working repo at `~/Source/Athena` — fresh clones only (Phase 2).
- NEVER push anything to a public remote, flip repo visibility, create/rename/transfer GitHub repos, or publish a release without explicit operator confirmation at that step. All such steps are marked **[OPERATOR GATE]**.
- Every slice lands as a reviewed commit; per current house rules each slice still bumps `Athena.appVersion` + tags until S9 retires that ritual.
- The scrub terms themselves must never appear in commit messages, tags, or this plan's successor documents. Refer to them as "the scrub list" (P0.3 builds it in a **private, untracked** file).
- If any step contradicts `CLAUDE.md` or an ADR, stop and surface it; do not improvise.

## Ground truth (verified 2026-07-07 — re-verify anything load-bearing before acting)

- No LICENSE/NOTICE exists. No secrets in tree or in the 548-commit history (swept clean). README is near-publication-quality but describes the deprecated native inference API (removed by ADR 013/031) and markets the parent-project name.
- The primary private-name contaminant appears in **35 tracked files** (10 shipping source files, 10 ADRs, one doc *filename*) and throughout history. Additional contaminants: the parent-project name (now classed as a consumer reference), a sibling-project consumer named in ADR 027, three sibling codenames in `docs/backlog-hitlist.md`, the autonomous-bot identity + orchestrator internals in `docs/usability-audit-2026-07-02.md` (whole file must not be published), personal memory paths in 4 docs, `AUDIT_FINDINGS.md`/`AUDIT_FINDINGS_V2.md` at root.
- ~52 of 71 top-level `docs/*.md` are internal work logs; ~19 are user-facing; 40 ADRs under `docs/decisions/`.
- `Package.swift` pins: `GoodOlClint/mlx-swift-lm` (public ✓), `GoodOlClint/swift-huggingface` (public ✓), `GoodOlClint/AppleSiliconMetrics` (**was private + org-homed — build-breaking for public clones; transferred and flipped public 2026-07-07, P0.2 DONE**).
- Bundle id `me.goodolclint.athena` — **stays** (matches the individual signing identity; decided).
- git history: 548 commits since 2026-05-16; ADRs begin 2026-06-11 (ADR 001) — the pre-ADR era needs a retrospective record (P0.4).
- `deploy/build.sh` already carries a `NOTARIZE=1` hook (ADR 024 T1) and `deploy/verify-hardening.sh` exists.

## Phase 0 — prerequisites (start immediately; mostly parallel)

**P0.1 [OPERATOR] Apple Developer Program enrollment (individual).** $99/yr; produces Developer ID Application + Installer certs and an App Store Connect API key (for `notarytool`). Longest lead time — start first. Not blocking Phases 1–3.

**P0.2 AppleSiliconMetrics relocation — DONE 2026-07-07.** Transferred to `GoodOlClint/AppleSiliconMetrics` and flipped public (operator); `Package.swift` dependency URL updated. Residual DoD check: a clean clone of Athena with no GitHub auth resolves all dependencies.

**P0.3 Build the scrub list + verifier.** A private, **untracked** file (e.g. `~/…/scratch/scrub-list.txt`, never committed) enumerating every term to erase, with per-term match rules (word boundaries for short codenames to avoid false positives) plus the paths to delete from history outright (the usability audit; the consumer-integration doc whose filename leaks; `AUDIT_FINDINGS*.md`). Known term classes (spelled out in the brainstorm doc §1.1/§2.5 — do not spell them out in tracked files): the sibling consumer repo name and variants; the parent-project name and its GitHub org; the sibling consumer named in ADR 027; the three sibling-project codenames; the bot identity; personal memory paths. **The operator's own username/domain is NOT scrubbed** — it is the public identity. Write `deploy/scrub-verify.sh` (or a scratch script) that, given a repo path, greps every term across (a) the working tree and (b) **all revisions** (`git log -S<term> --all` + `git grep <term> $(git rev-list --all)` sampled at every tag) and exits non-zero on any hit. DoD: the verifier runs against the current repo and *finds* the known contamination (proves it discriminates), per the dod-verify discipline.

**P0.4 Retrospective decision record.** Write `docs/decisions/000-pre-adr-history.md`: the 2026-05-16 → 2026-06-11 foundational decisions reconstructed from the tag ledger and milestone plans — one entry each (date, context, decision, evidence pointer) for: Swift+MLX single native daemon; passive-oracle rule; unified-governor thesis origin; two-dialect API (/v1 + /api); port 7447; bearer/RBAC model; SQLite(-Cipher) store; substrate-fork strategy; vendored Qwen3.5/MTP approach. Explicitly marked **reconstructed, not contemporaneous**. Must be written scrub-clean from the start. DoD: document exists, every entry cites a commit/tag, verifier passes on it.

## Phase 1 — working-tree cleanup (normal commits in the current private repo)

**S0 De-vendor the substrate-duplicated model code. [DEFERRED — added 2026-07-07 while a goal session was already executing this plan; do NOT execute in that run. It becomes the first follow-up slice after the current goal session completes. If you are the current executor: skip S0 entirely and proceed with S1; the scrub/NOTICE slices must then still handle the AthenaModels files it would have deleted.]** The pinned mlx-swift-lm `integration` branch now exports what `Sources/AthenaModels/` vendors (verified 2026-07-07 in the checkout: `Qwen35`/`Qwen35MoE`/`Qwen35MTP` + MTP registrations in MLXLLM and MLXVLM, `GatedDelta` in MLXLMCommon, TriAttention wired into the substrate Qwen35 models) — the planned "vendored-MLX upstreaming collapse" has landed. Delete the `AthenaModels` target (AthenaQwen35*, GatedDelta, Qwen3NextHelpers, TriAttention/) and rewire consumers (`AthenaLLM/MLXLLMModule`, `KVCompression` in Core+LLM, `SupportedModels`, `Commands/Load`) to the substrate types. **Do a symbol/behavior parity check before deleting** — any Athena-local divergence in the vendored copies must be reconciled upstream-first (ADR 028 posture), not kept as a fork. The audio/diarization ports in AthenaTranscription STAY vendored (no upstream home). Gates: bit-identical-greedy checks, MTP speculative e2e (ADR 032 scripts), full 796-test tier, Release build. Also removes one contaminated source-file header outright (shrinks S1's NOTICE surface and S2's scrub set). Note the partial supersession of ADR 028's "TriAttention (Athena-vendored)" line in that ADR's status in the same edit window.

**S1 License + attribution.** Add `LICENSE` (Apache-2.0, operator as copyright holder) + `NOTICE`. Attribution audit of: vendored SQLCipher amalgamation (confirm upstream header intact; note its license in NOTICE); the ported model code — Whisper (OpenAI/MIT), Parakeet + Sortformer (NVIDIA NeMo/Apache-2.0), pyannote segmentation port (MIT), WeSpeaker (Apache-2.0), Qwen3.5/MTP port provenance, llguidance (rust-shim; add `license` field to `Cargo.toml`). Each ported source file gets/keeps a provenance header that does NOT name any consumer project. DoD: NOTICE covers every third-party derivation found by a `grep -ril "ported from\|vendored\|adapted from" Sources/ rust-shim/` sweep.

**S2 Consumer-reference scrub of the working tree.** Rewrite the 35+ contaminated files: source comments re-worded to preserve technical meaning without naming consumers ("the consuming application" / "a downstream client"); ADR 027's motivation line generalized; the consumer-integration doc deleted (its useful content, if any, folds into public API docs); `docs/usability-audit-2026-07-02.md` and `AUDIT_FINDINGS*.md` deleted from the tree (S3 moves them to the internal repo first); `Package.swift` header tagline and README re-written to stand alone (no parent project). `CLAUDE.md` scrubbed. DoD: P0.3's verifier passes on the **working tree** (history still fails — that's Phase 2's job).

**S3 Docs cull + internal relocation. [OPERATOR GATE: create `GoodOlClint/athena-internal` (private)]** Move the ~52 internal docs (plans/audits/handoffs/research/backlog/milestone docs), `AUDIT_FINDINGS*.md`, and root working notes into the new private repo, preserving them verbatim (they may keep consumer names — they're private). The public tree keeps: the ~19 user-facing docs, `docs/decisions/` (all ADRs, scrubbed), `docs/publication-plan.md` + brainstorm doc go internal too at the end. Trim `CLAUDE.md` to a public-appropriate version (build/test commands, canonical-pipeline rules, ADR discipline); the internal agent digest moves to the internal repo (re-linked for local sessions via `CLAUDE.local.md` or memory). DoD: `docs/` contains only user-facing docs + decisions; internal repo holds the rest; no tracked file matches internal-work-log patterns.

**S4 Docs validation pass.** Verify every remaining doc against the current code/OpenAPI spec: README's API table (native inference endpoints are gone — ADR 013/031), quickstart, cli-reference, feature docs (transcription/diarization/video/model-support/tool-calling/logging). Add the **AI-authorship section** to README: Athena is developed by autonomous Claude agents under human direction; the operator reviews, gates, and approves all changes (cite the ADR trail + operator-gate discipline as the mechanism; this also explains commit-timestamp patterns). DoD: a fresh reader can build, install, and call every documented endpoint exactly as written; no doc references a removed surface.

## Phase 2 — history rewrite + soak

**S5 Filter.** On a **fresh clone** in scratch space: `git filter-repo` with `--replace-text` (scrub list) + `--invert-paths` for the history-deleted files (P0.3's path list) + path-rename for the leaking doc filename in old revisions. Tags: filter-repo rewrites annotated tags automatically — confirm the full v0.1→v0.10.x ledger survives. Timestamps: **untouched** (decided — the ledger is part of the story). DoD: P0.3 verifier passes on the rewritten repo across **all revisions and tags**; `git log --oneline | wc -l` matches the original count; spot-check 5 rewritten commits' diffs for semantic damage from replace-text.

**S6 Stand up the soak repo. [OPERATOR GATE ×3: rename old repo, create new repo, remote swap]** Rename the existing GitHub private repo → `GoodOlClint/Athena-archive` and archive it (read-only) — it is the true-history record and is **never pushed again**. Push the rewritten repo to a **new PRIVATE** `GoodOlClint/Athena`. Repoint `~/Source/Athena`'s origin to the new repo (or re-clone). All development from this moment happens on the rewritten history. DoD: `git fetch` works against the new origin; archive repo is read-only; CI/local builds green on the rewritten clone (`./deploy/build.sh Release` + `./deploy/test.sh` — full 796-test tier).

**S7 Soak.** A validation window (operator sets length; suggest ≥1–2 weeks of normal development). During soak: (a) the verifier runs on every push (add it as a pre-push hook or CI job on the private repo); (b) clean-machine build check — fresh clone, no `ATHENA_LOCAL_DEV`, no GitHub auth: `rust-shim/build.sh` → `xcodebuild` Release → tests; (c) a final human read of README + top-level docs as a stranger would. DoD: verifier green for the whole soak window; clean-clone build documented.

## Phase 3 — re-baseline + flip

**S8 Version re-baseline.** Set `Athena.appVersion` = `0.11.0`; retire the bump-every-slice ritual: update `CLAUDE.md` (and the release-workflow feedback memory) so versions are **release events** — patch = fix-only releases, minor = features, major/1.0 operator-reserved; slices land as plain commits (optional `-dev.N` prerelease tags). Tag `v0.11.0`. DoD: `athena --version` reports 0.11.0; CLAUDE.md documents the new scheme; the ADR-040 consequence list matches.

**S9 Flip. [OPERATOR GATE]** Final verifier run + clean-clone build, then flip `GoodOlClint/Athena` visibility → **public**. Post-flip: clone from the public URL on a machine with no GitHub auth and build; confirm dependency resolution (P0.2); confirm no consumer term is searchable on the GitHub UI (search indexes lag — re-check after a day). DoD: public clean-clone builds; GitHub code search for each scrub term in the repo returns nothing.

## Phase 4 — release pipeline (post-flip; needs P0.1 certs)

**S10 Signing + notarization locally.** Wire the P0.1 certs into `deploy/build.sh`'s existing `NOTARIZE=1` path; `deploy/verify-hardening.sh` stays the fail-closed gate. Produce a signed, notarized, stapled `athena` binary + `.pkg` (pkgbuild/productsign). Design note (binding, from the brainstorm): nothing may hard-pin the Team ID in any future self-update verification. DoD: `spctl --assess` passes on a quarantined copy of the artifact on a second machine.

**S11 GitHub Actions release workflow.** Tag push (`v*`) → macOS runner: Metal toolchain download step (`xcodebuild -downloadComponent MetalToolchain` — known gotcha), rust-shim build, Release build, test tier, sign + notarize (App Store Connect API key in repo secrets), staple, attach `.pkg` + tarball to a GitHub Release. DoD: `v0.11.1` (or next release) publishes end-to-end from a tag push with no manual steps.

**S12 Contributor surface + PR CI.** (Can start during the soak — the files ride the rewritten repo.)
- **PR CI**: a GitHub Actions workflow on PRs/pushes running the MLX-free unit tier (`./deploy/test.sh`, 796 tests — these don't need Metal eval, so they're runner-friendly) + `swift build` type-check; the full Release xcodebuild (Metal toolchain) can be main-branch/nightly only to control macOS-runner cost. DoD: a PR with a failing unit test shows a red check.
- **Community files**: issue templates (bug — asks for `athena --version`, `athena doctor` output, model id, `/healthz`; feature request; model-support request pointing at `athena pull --check` first), PR template (test-bar + ADR-discipline checkboxes), `CONTRIBUTING.md` (build prereqs incl. full-Xcode/Metal-toolchain requirement, test tiers, the AI-authorship workflow — external PRs are reviewed by the operator like agent slices), `SECURITY.md` (private vulnerability reporting; passive-oracle posture). DoD: files render correctly on the GitHub UI; templates appear in the new-issue chooser.

**S13 Distribution tail (separate, post-first-release):** Homebrew cask; `athena update` (must honor the Team-ID rule); Linux enters later via the CUDA-audit spike, as platform seams in this same repo (ADR 040 / brainstorm §2) — explicitly out of scope for this plan.

## Deferred to its own planning session (explicitly NOT this plan)

**CLAUDE.md restructure — ADR digests → progressive disclosure.** CLAUDE.md is ~54 KB, dominated by inline ADR digests that grow with every decision; the operator wants to evaluate an OKF-style shape instead: a compact index (one line per ADR: number, title, one-hook summary, status) with sessions reading full ADRs on demand, mirroring the `~/okf/` progressive-disclosure pattern. Interacts with S3 (the public CLAUDE.md trim) — S3 should do the minimal public trim without pre-empting this design. Run as a separate planning session after the flip.

## Sequencing summary

P0.1 → (long lead) → S10/S11. P0.2/P0.3/P0.4 parallel, then S1→S2→S3→S4 (Phase 1 in order; S0 is DEFERRED to after the current goal run — see its banner), then S5→S6→S7 (Phase 2; S12 can run during the S7 soak), S8→S9 (Phase 3), S10→S11→S12 (Phase 4). Operator gates: P0.1, P0.2 (transfer), S3 (repo create), S6 (rename/create/swap), S9 (flip), each release.

## Standing rules for the executing session

- Read the decision record at `~/.athena-publication/publication-and-repo-split-brainstorm.md` and ADR 040 before starting.
- The scrub list lives only in untracked scratch; its terms never appear in tracked files, commits, or tags.
- Follow the repo's existing gates: tests green before any commit claim; `/ship` ritual for releases; ADR same-edit-window discipline for any new architectural decision surfaced mid-execution.
- Anything ambiguous or destructive: stop and ask the operator. The soak exists to catch mistakes — use it.
