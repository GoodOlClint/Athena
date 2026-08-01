# Filing an out-of-scope follow-up issue

Only for findings OUTSIDE the PR's diff scope — pre-existing code the diff doesn't touch, or work beyond the PR's stated goal. In-scope follow-ups belong in the review body, not here.

## Never create duplicates

Before creating, `gh issue list --state open --search "<keywords>"` and check whether the same finding already has an open issue. If so, reference it in the review summary instead of duplicating.

## Issue format

```
Title: <short imperative description>

Body:
## What to fix
<the gap — what's missing or wrong>

## Why it matters
<impact — what this costs if unfixed>

## Acceptance criteria
<how to verify the fix worked>

## Out of scope
<anything that is NOT part of this fix>

---
Surfaced from PR #<PR> review.
```

Label with `bug` or `enhancement` (this repo uses the stock label set): `gh issue create --title "..." --body "..." --label enhancement`

Never put a closing keyword (`closes`/`fixes`/`resolves`) next to an issue number anywhere in the body — GitHub's matcher ignores negation, so even "do not close #N" closes #N on merge.

After creating, capture the issue number and list it in the review summary as `#<num> — <title>`.
