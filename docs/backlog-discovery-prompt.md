# Backlog-discovery prompt — "what did we forget to implement?"

A self-contained prompt to hand a fresh agent (or run yourself). It audits every
decision record, plan, and memory against the **actual code** and produces a
ranked, classified hitlist of unfinished / forgotten / deferred work — what to
do now, what to defer, what to drop. It **discovers and recommends; it does not
implement.**

Paste everything below the line as the task.

---

## Role

You are auditing the Athena repo for **unfinished, forgotten, or quietly-dropped
work**. Your job is to reconcile what the project *said it would do* against what
the code *actually does*, and produce a prioritized hitlist. You do **not** write
implementation code; you produce `docs/backlog-hitlist.md` for operator review.

## Inputs (read all of them)

1. **ADRs** — every `docs/decisions/*.md`. The status line + the
   "Rejected / deferred", "Consequences", and "Tripwire" sections are gold.
2. **Plans** — every `docs/*-plan.md` and any `docs/m*-*.md` milestone doc. Read
   the slice lists, "Out of scope", "Open decisions", and test-bar sections.
3. **Memories** — `~/.internal/projects/-Users-goodolclint-Source-Athena/memory/
   MEMORY.md` and the topic files it indexes. These record deferrals, tripwires,
   and "open follow-up" items not always written into the repo.
4. **CLAUDE.md** — the binding rules + the ADR index (each line's status).
5. **`LESSONS.md`** if present; the audit/readiness docs
   (`docs/audit-remediation-plan.md`, `docs/commercial-readiness.md`,
   `docs/m35-readiness-loose-ends`-style ledgers).

## Signal vocabulary — what counts as a candidate item

Flag any of: `deferred`, `defer`, `follow-up`, `fast-follow`, `TODO`, `FIXME`,
`open decision`, `open question`, `not yet`, `out of scope` *(with a "revisit"
clause)*, `gated on …`, `gate locked`, `tripwire` / `retire-tripwire`, `revisit
when`, `next milestone`, `deferred with gate`, `spike` (un-acted), `wart`,
`known issue`, `should-have`, `M__.5`-style trailers, `left open`, a `Proposed`
ADR with no shipped follow-through, a plan slice never tagged.

## Method (do NOT trust status labels — verify against code)

For each candidate item:

1. **Locate the claim** (file + line/section + the exact deferral text).
2. **Verify the real state in the code** — this is the crux. A doc that says
   "shipped" may be partial; one that says "deferred" may have quietly landed.
   Use `grep`/read on `Sources/` (and `Tests/`, `deploy/`) to confirm whether the
   thing exists, is wired to a route/CLI, and is tested. Prefer
   `graphify query "<concept>"` for relationship/where-implemented questions.
   Record the evidence (path:line) for the verdict.
3. **Detect drift / contradictions explicitly** — e.g. "ADR 0XX claims X is
   Accepted+shipped, but no `Sources/**` symbol implements it" or "memory says Y
   is deferred, but `Sources/.../Y.swift` ships it." Drift is the highest-value
   output.
4. **Check for supersession** — an item may be obsoleted by a later ADR. Don't
   resurrect something a later decision rejected; note the supersession instead.
5. **Respect the guardrails** — never recommend anything that violates the
   passive-oracle rule, the `/v1` compatibility rule, the canonical-pipeline
   rule, or that re-proposes an explicitly *rejected* option. Anything
   substantial must route through the change gate (design doc + ADR), not be
   "just done".

## Classification rubric

Assign each verified-open item exactly one:

- **DO NOW** — small, high-value, low-risk, no unmet dependency, and either a
  correctness/security/operability gap or a cheap truth/visibility win. (e.g. a
  one-file fix with an obvious test.)
- **DO SOON** — valuable but needs a design pass / a slice plan / a build cycle,
  or has a soft dependency. Worth scheduling.
- **DEFER** — genuinely gated (on an upstream/substrate, a consumer requirement,
  a tripwire condition) or low ROI right now. State the **gate condition** that
  would un-defer it.
- **DROP** — superseded, rejected by a later decision, or no longer relevant.
  State why; recommend deleting the stale claim so it stops resurfacing.

## Output — write `docs/backlog-hitlist.md`

1. **Summary table**, sorted DO NOW → DO SOON → DEFER → DROP:

   | # | Item | Source | Verified state | Class | Effort | Risk | Gate / dependency |
   |---|------|--------|----------------|-------|--------|------|-------------------|

2. **Per-item card** for every DO NOW / DO SOON:
   - **What** (one line) · **Why it matters** · **Evidence** (the doc claim + the
     code reality, with paths) · **Proposed first slice** · **Test** · **Effort**
     (S/M/L) · **Risk** · **Dependencies / gate**.
3. **Drift & contradictions** section — every doc↔code mismatch found, with the
   recommended reconciliation (fix the code, or correct the doc).
4. **Don't-touch list** — items confirmed correctly deferred/dropped, so a future
   audit doesn't re-flag them (with the gate that would change that).

## Scope discipline

- Be exhaustive over the *inputs*, conservative in the *recommendations*: a long
  DO-NOW list is a red flag — most discovered items are correctly DEFER/DROP.
- Quote evidence; don't assert. Every verdict cites a doc line and a code path.
- If the repo is large, fan out the reading (one pass per ADR cluster / plan
  family / memory section) and dedup before classifying — but keep the final
  hitlist single and ranked.

When done, report the counts per class and the top 3 DO-NOW items in your final
message; the full detail lives in `docs/backlog-hitlist.md`.
