---
name: pr-merge-loop
description: Take a working diff to state=MERGED — pre-submit gate (sibling sweep + subagent review + premise-check), verdict-gated merge (APPROVED merges immediately; folds only on CHANGES_REQUESTED), review-response policy (scoped fold-in, round cap), follow-up triage, inline-thread resolution, post-merge harvest. Invoke when code is ready to become a PR, and again whenever a review verdict or failed merge needs handling.
---

# PR merge loop

Opening a PR is the start of the job, not the end. Done means `state=MERGED`. Green tests + an open PR has shipped nothing. If the PR won't merge, triage why.

**Athena scope note:** operator slices may still land direct-to-main (house workflow); this loop applies whenever a change goes through a PR — larger features, anything wanting the automated review, and all non-operator contributions.

## Before opening the PR (pre-submit gate)

Four checks on the working diff, before the first push — a defect caught here costs one local fix; the same defect caught by CI costs a full round trip.

1. **Sibling sweep — no tunnel vision.** The instance you were asked to fix is rarely the only one. Before finalizing, hunt for similar code the same defect or pattern lives in: code-intel MCP `search` (semantic — "where else does X happen") plus grep for the exact symbol/pattern, and graphify blast-radius ("what calls the function I changed") when the change alters behavior callers depend on. In-scope siblings get fixed in this PR; out-of-scope ones get a `gh issue create` before you open the PR.
2. **Adversarial pre-review — two models, not one.** This gate is deliberately cross-model: the measured gain from a second reviewer comes from *context separation*, and a different vendor buys different blind spots (ADR-referenced research, 2026-08-03).

   **(a) House reviewer.** Spawn the repo's **`athena-pre-reviewer`** agent (`.claude/agents/athena-pre-reviewer.md`; fall back to correctness-reviewer or /code-review where unavailable) on the working diff. Its definition carries the house instructions — correctness pass, attack-the-claims, the mutation check on test-pinning claims, and the premise check — so don't re-derive them in the spawn prompt; just point it at the diff and name the issues it implements.

   **(b) Cross-model reviewer.** Run Codex over the same diff:

   ```
   codex exec review --base <default-branch> < /dev/null     # work is committed on a branch
   codex exec review --uncommitted < /dev/null               # work is still in the working tree
   ```

   **What the separation actually is.** Codex *does* read `AGENTS.md` — it auto-discovers project instructions, so it sees the same house rules (a) does. The separation this gate buys is therefore (i) a **fresh process with no working-session context** — it has not watched you write the diff, argue yourself into an approach, or accumulate the reasoning that makes a mistake feel settled — and (ii) a **different model family**, with different blind spots. That is the measured effect: cross-*context* review beat same-session self-review with the same model. Do not claim a fresh-reader asymmetry the tool does not provide. Typical cost is well under a minute.

   **Reconcile before pushing.** Merge both finding sets, drop duplicates, and fix what is real. Where the two disagree, the house reviewer wins on repo-constraint questions and Codex wins on nothing automatically — adjudicate it yourself and say which you followed. Codex is advisory here and holds **no approval authority**; it never gates the merge.

   **Never skip it silently.** If Codex is unavailable, unauthenticated, or errors, say so in your response ("cross-model gate down: <error>") and proceed with (a) alone — the same discipline as the semantic-lane rule. A silent skip hides a broken gate.

3. **Premise check — risk-class fixes only.** Covered by `athena-pre-reviewer` (its pass 3) when the diff is risk-class (resource bound, cap/limit, security guard, concurrency, escaping) — make sure the spawn prompt says the diff is risk-class so the pass runs. Where the agent is unavailable, the one-sentence procedure: what quantity does this fix bound and at what point in the flow; what quantity did the finding say was unbounded; do they match; and what concrete input or interleaving still evades the fix. A mismatch or working evasion is fixed before push. (Lesson from a sibling repo: three follow-up fixes shipped with wrong premises, and each cost a later merge cycle.)
4. **Green gate — tests, last.** After all pre-submit fixes land (steps 1–3 can change code, so this runs last), run the tiers the diff touches: `./deploy/test.sh` (unit tier) always; `./deploy/build.sh Release` when the change affects the MLX-linked targets or deploy scripts; the relevant `deploy/e2e-*.sh` when an HTTP surface or store behavior changed (model-gated e2e needs real hardware — say so if skipped). A red result never gets pushed. Docs-only diffs may skip the run; say so explicitly.

## Open the PR — merge is verdict-gated (policy 2026-07-31)

- Push branch → `gh pr create` (base main). Do NOT arm auto-merge: the loop's job doesn't end at merge — the post-merge harvest and doc reconciliation need the agent present at merge time, and auto-merge can land the squash while the loop is between polls.
- The PR body carries `Closes #N` for every issue it resolves — never close issues by hand with `gh issue close`; the merge closes them.
- Squash on merge: per-PR commits collapse to one curated commit on main, matching the house one-commit-per-slice history.
- Never bump `Athena.appVersion` or touch `v*` tags in a PR — versions are release events (ADR 040 S8), cut by the operator via `/ship`.
- Merging is an explicit act after the verdict — see APPROVED below.

## The review loop

The automated `claude-review` GitHub Action reviews on every push and submits exactly one formal review per run: APPROVED, CHANGES_REQUESTED, or COMMENTED (deferral to the operator).

### CHANGES_REQUESTED

- Fix every blocker.
- In-scope follow-ups the reviewer wrote in the review body may be fixed in the same round as the blockers. Issues the reviewer filed are out-of-scope by construction (the review prompt files issues only for findings outside the diff; if a legacy-prompt review files an in-scope issue anyway, it waits for end-of-cycle triage all the same) — they stay open for end-of-cycle triage. If the operator directs an issue fold-in anyway, add `Closes #N` to the PR body (`gh pr edit <N> --body`); never close issues by hand.
- **Fold-ins spend the PR's single fold budget — one round per PR.** A PR that has already folded once has spent it: every later round fixes exactly what the reviewer flagged and nothing else. (Unconditional fold-in once spiraled a PR through three review rounds in a sibling repo — this cap is the lesson.)
- Push; the reviewer re-reviews automatically. Loop.

### APPROVED

- **Merge immediately**: `gh pr merge <N> --squash`. Then the post-merge harvest.
- **Do NOT fold anything on an approval** (policy 2026-08-01). If a finding truly had to land before merge, the reviewer's verdict would have been CHANGES_REQUESTED — trust the verdict. Every post-approval push costs a full CI run plus a full re-review round; in the first remediation cycle the fold-after-APPROVED path added 9 extra rounds across 7 of 14 PRs for items that could all have waited one PR.
- Follow-ups the reviewer noted in the review body survive via the post-merge harvest; issues the reviewer filed stay open for **end-of-cycle triage, before starting the next plan item** — report each with a priority and when it will be resolved. Don't let them silently become backlog.

### COMMENTED (deferral)

- The reviewer flagged something needing human judgment (architecture, security posture, passive-oracle boundary, dependencies, release/versioning). Surface it to the operator and stop the loop — do not try to satisfy it yourself.

### No review arrives

- The action self-skips PRs that modify `.github/workflows/` (its anti-tamper gate), draft PRs, and fork PRs (no secrets). No automated verdict will come — request review from the operator (`gh pr edit <N> --add-reviewer GoodOlClint`), report why, and stop the loop.

## Inline threads

If `required_conversation_resolution` is enabled on the repo, unresolved inline threads keep the PR `BLOCKED` even after approval. After fixing what a thread raised, resolve it via GraphQL `resolveReviewThread(input:{threadId})` (thread ids from `pullRequest.reviewThreads`). An unresolved thread is an unmerged PR.

## Exit conditions

- Loop until `gh pr view <N> --json state` shows `MERGED`.
- COMMENTED deferral or no-review-arrives → the loop suspends pending the operator; report and stop (these are valid exits short of MERGED).
- PR approved but not merging → triage in order: unresolved review threads, required status checks (`gh pr checks <N>`), branch protection. Report what you find; fix what you can.
- Stacked PRs: merge sequentially, one at a time, rebasing each onto the new main after the prior lands.

## After merge (harvest — the loop's last act)

On `state=MERGED`, before end-of-cycle triage: (1) if a tracked document (ADR, plan doc, issue) describes this change, reconcile it against the squashed diff — the record describes the code that merged, never the first cut; (2) sweep every review body on the PR for observations that exist nowhere else — "minor, not filed" / "notes, not blocking" items become issues labeled `review-note` (create the label first if the repo lacks it: `gh label create review-note --description "harvested from a PR review body"`), and anything the reviewer addressed to the operator directly is surfaced to the operator rather than left in the review. Review bodies are the only place this material exists; the harvest is what makes it survive the merge. Those two sentences are the whole procedure — the operator's environment carries it as a `doc-reconcile` skill, but no skill is required.
