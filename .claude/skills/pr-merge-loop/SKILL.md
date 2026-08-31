---
name: pr-merge-loop
description: Take a working diff to state=MERGED — pre-submit gate (sibling sweep + subagent review + premise-check), verdict-gated merge (APPROVED hands ready-to-merge to the operator; folds only on CHANGES_REQUESTED), review-response policy (scoped fold-in, round cap), follow-up triage, inline-thread follow-up, post-merge harvest. Invoke when code is ready to become a PR, and again whenever a review verdict or failed merge needs handling.
---

# PR merge loop

Opening a PR is the start of the job, not the end. Done means `state=MERGED`. Green tests + an open PR has shipped nothing. If the PR won't merge, triage why.

**Athena scope note:** operator slices may still land direct-to-main (house workflow); this loop applies whenever a change goes through a PR — larger features, anything wanting the automated review, and all non-operator contributions.

## Before opening the PR (pre-submit gate)

Four checks on the working diff, before the first push — a defect caught here costs one local fix; the same defect caught by CI costs a full round trip.

1. **Sibling sweep — no tunnel vision.** The instance you were asked to fix is rarely the only one. Before finalizing, hunt for similar code the same defect or pattern lives in: code-intel MCP `search` (semantic — "where else does X happen") plus grep for the exact symbol/pattern, and graphify blast-radius ("what calls the function I changed") when the change alters behavior callers depend on. In-scope siblings get fixed in this PR; out-of-scope ones get an issue (github MCP `issue_write`) before you open the PR.
2. **Adversarial pre-review — two models, not one.** This gate is deliberately cross-model: the measured gain from a second reviewer comes from *context separation*, and a different vendor buys different blind spots.

   **(a) House reviewer.** Spawn the repo's **`athena-pre-reviewer`** agent (`.claude/agents/athena-pre-reviewer.md`; fall back to correctness-reviewer or /code-review where unavailable) on the working diff. Its definition carries the house instructions — correctness pass, attack-the-claims, the mutation check on test-pinning claims, and the premise check — so don't re-derive them in the spawn prompt; just point it at the diff and name the issues it implements.

   **(b) Cross-model reviewer.** Hand the same diff to Codex through the plugin's own runtime:

   ```
   node "$(ls -d ~/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs | sort -V | tail -1)" \
     review --wait --base origin/main
   ```

   **Use this, not a hand-rolled `codex exec review`.** That is the whole lesson of this step's history. Shelling out directly hits a hard limitation: an MCP tool call in non-interactive `codex exec` is auto-cancelled because nothing can approve it, so the review **hangs, or exits 0 with no verdict**. Nothing lifts it — not `approval_policy` at any value, not a blanket auto-approve `PreToolUse` hook, not `--dangerously-bypass-hook-trust`. Two separate PRs were then spent arguing about a `-c mcp_servers` flag, and one shipped a form that never worked at all. The plugin runtime does not take that path and needs no flags. Verified on a real code diff: exit 0, zero MCP calls attempted, real verdict.

   `/codex:review` and `/codex:adversarial-review` are the same runtime and are the right entry point when the OPERATOR is driving. Both are `disable-model-invocation`, so an agent must call the companion script directly, as above. For a risk-class diff swap `review` for `adversarial-review` — it challenges the approach and its assumptions rather than hunting defects, which is where the highest-value findings have come from.


   **Commit first — the intended diff must be committed.** `--base` reviews commits only and `--uncommitted` reviews the working tree only, so on a branch holding both, *neither* sees the whole proposed change and the gate silently reviews a fraction of it. Commit **the intended diff** before running this (you are about to push anyway) — not `git add -A`. Unrelated staged, unstaged, or untracked work in the checkout must be left alone or stashed first; sweeping it in to satisfy the commit-first rule contaminates the PR with work the operator did not intend to ship. Verify with `git status --porcelain` that what remains uncommitted is only that unrelated work, and say so when you report the gate result.

   **What the separation actually is.** Codex *does* read `AGENTS.md` — it auto-discovers project instructions, so it sees the same house rules (a) does. The separation this gate buys is therefore (i) a **fresh process with no working-session context** — it has not watched you write the diff, argue yourself into an approach, or accumulate the reasoning that makes a mistake feel settled — and (ii) a **different model family**, with different blind spots. That is the measured effect: cross-*context* review beat same-session self-review with the same model. Do not claim a fresh-reader asymmetry the tool does not provide. Typical cost is well under a minute.

   **Reconcile before pushing.** Merge both finding sets, drop duplicates, and fix what is real. Where the two disagree, the house reviewer wins on repo-constraint questions and Codex wins on nothing automatically — adjudicate it yourself and say which you followed. Codex is advisory here and holds **no approval authority**; it never gates the merge.

   **Then commit again.** Fixes made during reconciliation land in the working tree, and step 4's green gate would test a tree the PR does not contain — the push would carry only the earlier, defective commit. Commit the reconciliation fixes, then confirm `git status --porcelain` shows **nothing from the intended diff** — only the unrelated work you deliberately set aside — before step 4 runs. It cannot require an empty tree: the rule above tells you to leave unrelated work alone, which makes the tree non-empty by construction, and the only way to force it empty is `git add -A` — the contamination that rule exists to prevent.

   **One pass, not a loop.** Run the gate once, fix what it found, and move on — do **not** re-run it after each fix hunting for a clean verdict. A capable reviewer will keep finding narrower things indefinitely, and each fix legitimately exposes the next layer; this change's own introduction went four rounds before the findings turned from defects into wording. The CI review is the next gate and exists to catch what this one missed. Re-run only if the fixes were substantial enough to be a different diff.

   **Never skip it silently.** If Codex is unavailable, unauthenticated, or errors, say so in your response ("cross-model gate down: <error>") and proceed with (a) alone. A silent skip hides a broken gate.

3. **Premise check — risk-class fixes only.** Covered by `athena-pre-reviewer` (its pass 3) when the diff is risk-class (resource bound, cap/limit, security guard, concurrency, escaping) — make sure the spawn prompt says the diff is risk-class so the pass runs. Where the agent is unavailable, the one-sentence procedure: what quantity does this fix bound and at what point in the flow; what quantity did the finding say was unbounded; do they match; and what concrete input or interleaving still evades the fix. A mismatch or working evasion is fixed before push. (Lesson from a sibling repo: three follow-up fixes shipped with wrong premises, and each cost a later merge cycle.)
4. **Green gate — tests, last.** After all pre-submit fixes land (steps 1–3 can change code, so this runs last), run the tiers the diff touches: `./deploy/test.sh` (unit tier) always; `./deploy/build.sh Release` when the change affects the MLX-linked targets or deploy scripts; the relevant `deploy/e2e-*.sh` when an HTTP surface or store behavior changed (model-gated e2e needs real hardware — say so if skipped). A red result never gets pushed. Docs-only diffs may skip the run; say so explicitly.

## Open the PR — merge is verdict-gated (policy 2026-07-31)

- **All GitHub writes go through the `github` MCP tools (`mcp__github__*`), acting as the `goodolclint-claude` App** — `Bash(gh …)` is denied, and the bot identity is what lets the automated review run on agent-authored PRs (`allowed_bots` in `claude-code-review.yml`, PR #154). Setup + rationale: `~/Source/homelab/docs/github-agent-identity.md`.
- Push the branch per **AGENTS.md "Agent pushes go through the GitHub MCP"** (operator decision 2026-08-31): `create_branch` + `push_files` as the App (commit message goes to the tool; no `Co-Authored-By` trailer — the App is the author), then the byte-verify — commit the identical change locally, `git fetch`, and `git diff <local-commit> origin/<branch> --` must be empty. `push_files` cannot express deletions or renames; a diff needing them fails the verify and waits for an operator-attended local `git push`, as do `.github/workflows/*` changes (the App has no `workflows` permission). Then `create_pull_request` (base main). Do NOT arm auto-merge: the loop's job doesn't end at merge — the post-merge harvest and doc reconciliation need the agent present at merge time, and auto-merge can land the squash while the loop is between polls.
- The PR body carries `Closes #N` for every issue it resolves — never close issues by hand; the merge closes them.
- Squash on merge: per-PR commits collapse to one curated commit on main, matching the house one-commit-per-slice history.
- Never bump `Athena.appVersion` or touch `v*` tags in a PR — versions are release events (ADR 040 S8), cut by the operator via `/ship`.
- Merging is the operator's explicit act after the verdict — see APPROVED below.

## The review loop

The automated `claude-review` GitHub Action reviews on every push and submits exactly one formal review per run: APPROVED, CHANGES_REQUESTED, or COMMENTED (deferral to the operator).

### CHANGES_REQUESTED

- Fix every blocker.
- In-scope follow-ups the reviewer wrote in the review body may be fixed in the same round as the blockers. Issues the reviewer filed are out-of-scope by construction (the review prompt files issues only for findings outside the diff; if a legacy-prompt review files an in-scope issue anyway, it waits for end-of-cycle triage all the same) — they stay open for end-of-cycle triage. If the operator directs an issue fold-in anyway, add `Closes #N` to the PR body (`update_pull_request`, body); never close issues by hand.
- **Fold-ins spend the PR's single fold budget — one round per PR.** A PR that has already folded once has spent it: every later round fixes exactly what the reviewer flagged and nothing else. (Unconditional fold-in once spiraled a PR through three review rounds in a sibling repo — this cap is the lesson.)
- Push; the reviewer re-reviews automatically. Loop.

### APPROVED

- **The verdict is the gate; the merge is the operator's.** Under the App identities the bot never merges (house rule, and the `main` ruleset requires a code-owner review no bot can give) — the operator reviews and merges as themself. On APPROVED: report the PR ready-to-merge, request the operator's review (`update_pull_request`, `reviewers`), and suspend the loop; when the merge lands, run the post-merge harvest. (Lineage: the 2026-08-01 "merge immediately" policy predates the identity split — the agent then merged on the operator's own credentials, which no longer exist in the loop. The verdict-gating and no-fold rules below are unchanged.)
- **Do NOT fold anything on an approval** (policy 2026-08-01). If a finding truly had to land before merge, the reviewer's verdict would have been CHANGES_REQUESTED — trust the verdict. Every post-approval push costs a full CI run plus a full re-review round; in the first remediation cycle the fold-after-APPROVED path added 9 extra rounds across 7 of 14 PRs for items that could all have waited one PR.
- Follow-ups the reviewer noted in the review body survive via the post-merge harvest; issues the reviewer filed stay open for **end-of-cycle triage, before starting the next plan item** — report each with a priority and when it will be resolved. Don't let them silently become backlog.

### COMMENTED (deferral)

- The reviewer flagged something needing human judgment (architecture, security posture, passive-oracle boundary, dependencies, release/versioning). Surface it to the operator and stop the loop — do not try to satisfy it yourself.

### No review arrives

- The action self-skips when the PR's copy of **`.github/workflows/claude-code-review.yml`** differs from the default branch (its anti-tamper gate) — that one file, not `.github/workflows/` broadly; a PR touching an unrelated workflow is still reviewed. It also skips draft PRs and fork PRs (no secrets).
- The causes do **not** behave alike, so name which one you hit:
  - **Anti-tamper, PR edits the file** — the job runs and fails closed, so `claude-review` goes **red**. Being a required check, the PR then needs an operator **admin merge**. Say that plainly rather than reporting "no review arrived": the red check, not the missing review, is what blocks the merge.
  - **Anti-tamper, stale branch** — the PR does *not* touch the file, but branched before a main-side change to it, so its copy differs anyway and the same red fail-closed check fires. Different fix: `update_pull_request_branch` (merges main in), and the review runs on the resulting push (verified on PR #153: red → success). Try this before reporting a dead loop.
  - **Draft or fork** — the job's own `if:` is false, so `claude-review` is **skipped**, which reads as passing (#77) and blocks nothing. Do not report a red check here; there isn't one.
- For the non-recoverable causes (true tamper, draft, fork) no automated verdict will come — report which one and stop the loop; after a stale-branch refresh the verdict arrives normally, so keep polling instead. Requesting the operator's review (`update_pull_request`, `reviewers`) works on bot-authored PRs (GitHub refuses the request when the requestee authored the PR, which no longer applies; verified on #153) — it's the right signal on a true-tamper PR, since the operator's admin merge is the only way it lands.

## Inline threads

If `required_conversation_resolution` is enabled on the repo, unresolved inline threads keep the PR `BLOCKED` even after approval. After fixing what a thread raised, resolve it. The github MCP toolset carries no resolve-thread mutation (GitHub exposes it via GraphQL only, and `gh api` is denied) — reply on the thread (`add_reply_to_pull_request_comment`) saying what fixed it, then ask the operator to resolve it in the web UI. An unresolved thread is an unmerged PR.

## Exit conditions

- Loop until `pull_request_read` (get) shows the PR merged.
- COMMENTED deferral, no-review-arrives, or awaiting the operator (ready-to-merge on APPROVED, or inline-thread resolution) → the loop suspends pending the operator; report and stop (these are valid exits short of MERGED).
- PR approved but not merging and the operator asks why → triage in order: unresolved review threads, required status checks (`pull_request_read`, get_check_runs), branch protection. Report what you find; fix what you can.
- Stacked PRs: merge sequentially, one at a time, rebasing each onto the new main after the prior lands.

## After merge (harvest — the loop's last act)

On `state=MERGED`, before end-of-cycle triage: (1) if a tracked document (ADR, plan doc, issue) describes this change, reconcile it against the squashed diff — the record describes the code that merged, never the first cut; (2) sweep every review body on the PR for observations that exist nowhere else — "minor, not filed" / "notes, not blocking" items become issues labeled `review-note` (create the label first if the repo lacks it: `label_write` — `color` is required on create, e.g. `ededed`; description "harvested from a PR review body"), and anything the reviewer addressed to the operator directly is surfaced to the operator rather than left in the review. Review bodies are the only place this material exists; the harvest is what makes it survive the merge. Those two sentences are the whole procedure — the operator's environment carries it as a `doc-reconcile` skill, but no skill is required.
