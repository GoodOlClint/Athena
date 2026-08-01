# Remediation-loop retro (cycle 2026-07-31 → 08-01)

Scope: the 14 PRs (#1–#39) and 26 issues (#2–#42) of the issue-remediation loop, reviewed against the goals: less issue churn, less token burn, fewer runner minutes, same code quality.

## Scoreboard

| Metric | Value |
|---|---|
| PRs merged | 14 in ~26 h; open→merge 9–24 min each |
| Review-bot runs | 27 runs / 131 runner-min, 31 formal reviews — **~2 full reviews per PR** |
| PRs needing ≥2 full review rounds | 10 of 14 |
| CI runs | 46 / 188 wall-min; macOS unit job in every one (10× billing multiplier — repo is private) |
| Post-merge main CI runs (cache-seed) | 14 — each re-runs lint + full unit tier on a tree that just passed as the PR head |
| Issues filed by the loop | 20 (vs 6 planned) — 2.5 filed per issue closed |
| Issues folded into the PR that surfaced them | 9, with lifetimes of **5–21 minutes** |
| claude.yml (@claude) runs | 68, all skipped (trigger noise; ~0 billed) |

## What worked — keep it

- **The pre-submit subagent gate found the only two live bugs** (#20, #38 — real ADR 029 exclusivity violations on `main`). Neither CI nor the review bot found them. This is the highest-value spend in the loop.
- **The one-fold-budget cap held.** No PR spiraled; the worst (#26) stopped at 2 CHANGES_REQUESTED rounds.
- **Verdict-gated merge + end-of-cycle triage** worked exactly as designed — the parked #8 stayed open, the triage table is honest.
- **Quality outcome is good**: every merged diff is small (median ~130 LOC), test-pinned, and the review bot's four catches (overclaiming prose) were real.

## The churn engine — two policies interlocking

The cost is not any one policy; it's the interaction of two:

1. `claude-code-review.yml` instructs (pre-move inline prompt; since relocated to `.github/review-prompt.md`): every FOLLOW-UP finding → `gh issue create`, and "when a finding sits between NIT and FOLLOW-UP, file the FOLLOW-UP" — a deliberate bias toward filing.
2. `pr-merge-loop` instructs: on APPROVED, decide fold-in NOW; in-scope follow-ups fold into the open PR.

Result: the bot **approves** a PR and files issues → the agent folds them → push → `synchronize` retriggers **both** a full macOS CI run and a full review-bot run → second APPROVED → merge. That second round happened on ~9 of 14 PRs, and the issues it was mediated through lived 5–21 minutes. GitHub issues were used as an IPC channel between two agents already looking at the same diff — each one costing a full spec body (tokens), a skipped claude.yml trigger, `Closes #N` bookkeeping, and once (PR #26) a wrong-issue auto-close via a negated closing keyword.

## Recommendations, ranked by savings

### 1. APPROVED merges immediately; fold only on CHANGES_REQUESTED

Edit `pr-merge-loop` APPROVED section: follow-ups filed with an approval go straight to end-of-cycle triage — no fold, no second round. The triage step already proved it works (this cycle's table is exactly that). If a follow-up is truly blocking, the reviewer should have said CHANGES_REQUESTED — trust the verdict.

Saves: ~1 macOS CI run + ~1 review-bot run on ~9 of 14 PRs ≈ **–35% CI minutes, –40% review-bot runs/tokens**. Risk: minor items land one PR later. That's what a triage queue is for.

### 2. In-scope findings stay in the review body — issues are for out-of-scope only

Edit the review prompt: a FOLLOW-UP that touches lines in this PR's diff is written **in the review body** (as a blocker if it matters, a note if not), never as a `gh issue create`. Only findings outside the diff's scope become issues. Also drop the "between NIT and FOLLOW-UP, file the FOLLOW-UP" bias for in-diff findings.

Saves: the 9 five-minute-lifetime issues, their spec-body tokens, the `Closes #N` bookkeeping, and the #26-style closing-keyword hazard. Issue list becomes signal (real backlog) instead of chat log.

### 3. Post-merge main CI: seed the cache without re-running the test tier

The 14 push-to-main runs exist only to save a main-scoped SPM cache (the `on:` comment says so), but they re-run lint + the full macOS unit tier on a squashed tree identical to the PR head that just passed. Make the main-push path a build-only job (`swift build --build-tests`, then cache save) — or skip test execution when `git rev-parse HEAD^{tree}` matches the merged PR head's tree.

Saves: ~40% of macOS billed minutes (the 10× multiplier makes this the biggest raw-dollar item).

### 4. Slim the review bot's mandatory reading

The prompt says "Read the repo's CLAUDE.md and docs/decisions/" — re-read on all 27 runs. The prompt already inlines the hard constraints. Change to: rely on the inlined constraints; open a specific ADR only when the diff names it or touches its subsystem.

Honesty note (from this retro's own review): the realizable saving is the `docs/decisions/` bulk-read — the repo-root CLAUDE.md is auto-loaded by the action's Claude session regardless of what the prompt says, so that cost stays until CLAUDE.md itself slims down or the action grows a way to suppress it.

Saves: the per-run ADR sweep; compounds with rec 1's run-count cut.

### 5. Amend "one issue per PR" to "one concern per PR"

Each PR carries ~2 CI runs + ~2 review runs + merge bookkeeping of fixed overhead. The triage table itself recommends #42 + #41 together (same subsystem, same cause). Let mechanical, same-subsystem siblings share a PR; keep the invariant for anything with an argument to make.

### 6. Prompt reviewers to attack claims, not measurements

The cycle's own lesson (plan doc, "What this run says"): all four CI-review catches were prose overclaiming what code delivered, and pre-submit subagents verified evidence but not conclusions. Add one line to both the pre-review subagent prompt and the CI review prompt: "check every claim in comments/docs/PR body against what the diff actually guarantees; an overclaim is a finding."

### 7. Not worth chasing

- The 68 skipped claude.yml runs bill ~nothing (job-level `if:` short-circuits before a runner boots). Log noise only; dropping the `issues: opened` trigger is optional cosmetics.
- Review-bot wall time (4–7 min/run, ubuntu 1×) is fine once run count halves.

## Projected next-cycle profile (recs 1–3 applied)

| Metric | This cycle | Projected |
|---|---|---|
| Review-bot runs per PR | ~2 | ~1.1 |
| macOS CI runs per PR (incl. main seed) | ~3.3 | ~1 + cheap seed |
| Loop-filed issues | 20 (9 ephemeral) | ~11, all real backlog |
| Live-bug catch rate | pre-submit gate | unchanged — gate untouched |
